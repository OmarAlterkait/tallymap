# Design decisions

The decisions behind tallymap and the evidence for each.

Where a number appears, it was measured with the benchmarks in this repository: `benchmarks/toy_bench.py` and
`benchmarks/rev_bench.py` (the "toy walk": one item takes a data-dependent number of random steps and adds a
weight into a 256-bin histogram), and `benchmarks/perf.py`, which checks the recorded rows in
`benchmarks/perf_baseline.json`. Unless stated otherwise those timings are an NVIDIA L40S GPU, JAX 0.5.3 and
`cuda_math='approx'` (the default), reported as `jax.jit` wall time per call. `docs/performance.md` holds the
comparisons against plain JAX. Terms are defined in `docs/glossary.md`.

tallymap was built for two in-house Monte Carlo engines, one for optical photon transport and one for
electromagnetic particle showers. Those engines are not public, so measurements made on them are not quoted here;
where a decision was driven by one, this page says so and gives the repository's own evidence for the same point.

Two conventions used throughout. A **jaxpr** is JAX's own intermediate representation of a traced function; the
library reads and rewrites jaxprs rather than Python. **JVP** (Jacobian-vector product) is forward-mode
differentiation and **VJP** (vector-Jacobian product) is reverse mode; `jax.jvp` and `jax.vjp` are JAX's names for
them.

## 1. The user contract

### 1a. Emits are threaded, not effects

`fn(tally, *args) -> tally | (tally, y)`; `t = t.add(name, idx, val)`.

An argument-less `emit(name, idx, val)` callable anywhere would need JAX's effect system (private registries) or a
zero-output primitive that a future dead-code elimination pass may delete, silently losing emits. Threading the
Tally like a pseudo-random-number-generator key costs one loop carry and buys:

- a data dependency: no effect machinery, nothing can be eliminated as dead code;
- a plain dense meaning: outside tallymap a Tally is an array and an emit is `t.at[idx].add(val)`. So
  `jax.vmap(fn)` over items followed by a sum IS the definition (`tm.reference`), and JAX's own JVP, batching and
  transpose rules for that dense program define what every transformation must return;
- backends implement tallies virtually: a write-through atomic add, or a masked scatter into one shared
  accumulator. This equals the dense meaning if and only if tallies are write-only and used linearly, which is
  checked statically on the jaxpr (`_rewrite.check_tallies`): forked or dropped tallies, reads, tallies closed
  over by loops, and branches that disagree raise `TallyError` at trace time.

The only item-level primitives are `tallymap_emit` and `tallymap_spawn` (`_tally.py`).

### 1b. No init / step / done for the user

The CUDA kernel builder finds the outer loop itself. Two scheduling strategies were measured against each other:
running a whole item on a persistent thread (the item's loops become C loops), and refilling a lane with a new
item at loop-STEP granularity. Step refill won on a production photon-transport workload, by about a third,
because a lane whose item has finished otherwise idles until the longest item in its warp ends. So an item with
exactly ONE top-level while loop is split into pre / cond / body / post
(`backends/cuda/kernel.py::_split`); anything else runs whole-item. With spawn the order reverses and whole-item
is the faster of the two, by about half again, so the rule is: step refill unless the program spawns. (A warp is
the group of 32 GPU threads that execute together on NVIDIA hardware.)

That rule rests on the one production measurement. On five of the `benchmarks/vs_jax` problems at 262,144 items
(L40S), whole-item mode was equal to or faster than step refill, forward and gradient alike (transport 0.90 vs
0.96 ms forward), so the rule is due to be re-derived from a wider measurement rather than trusted.

`max_steps` is applied by re-tracing the item jaxpr: every top-level while loop gets a counter and the item
returns `truncated` (`Result.truncated`, `stats['n_truncated']`). It is one rewrite, upstream of every backend and
every automatic-differentiation transform, so the JVP, VJP and batched versions of the capped program are
consistent automatically.

Semantics, which are deliberately strict: a tallymap call means exactly `jax.vmap(item)` followed by a sum. It
deviates only where a tallymap resource limit intervenes, and then it says so by making every float tally that
includes an affected item NaN. There are two such limits, `max_steps` and the spawn `Queue` capacity, and they
behave identically. `max_steps=None` is the default, so by default there is no limit and no deviation, including
the consequence that an item which never terminates does not terminate.

### 1c. Spawn

`t = t.spawn(child_args, when)`: the child is a new item with the same per-item input structure. The lockstep
backend uses a ring buffer processed in later blocks (children first); the CUDA backend uses a bounded
multiple-producer multiple-consumer ticket queue. Overflow drops the child, counts it (`stats['dropped']`) and
makes the float tallies NaN. There is no stall: a stalled mid-item step cannot be replayed. Forward-mode
derivatives with spawn give children tangents for every float per-item leaf. Reverse mode uses the tree adjoint
(section 4).

### 1d. Float semantics, fallback, sizes

- One CUDA emitter: IEEE round-to-nearest arithmetic without fused-multiply-add contraction, transcendental
  functions from the CUDA math library, and the `cuda_math` choice for f32 division, square root and reciprocal
  square root (`docs/float_semantics.md`). The CUDA and lockstep backends agree statistically, not bitwise;
  `tm.compare` is the contract (`docs/validation.md`).
- `backend='auto'`: CUDA on a GPU; a program the emitter does not support raises `UnsupportedError` there unless
  `allow_fallback=True`, in which case lockstep runs with a warning, because a silent change to the reference
  interpreter is a silent factor-of-100 slowdown. On every other platform it is lockstep with a one-time warning.
  `backend='cuda'` raises instead of falling back. Compile errors from NVRTC (NVIDIA's runtime CUDA compiler) are
  bugs and raise. Effects, such as `jax.debug.print` or an input/output callback, raise on both backends.
- N, the number of items, is a runtime kernel argument: neither the compiled kernel nor the in-memory kernel cache
  key contains it. One call takes fewer than 2^31 items.
- `Sum(exact=True)`: int64 fixed point, `round(v * scale)`, with one scale per output. Unrepresentable values,
  meaning `|v * scale| >= 9.2e18`, are dropped AND counted (`stats['exact_overflow']`) on both backends. A fixed
  2^32 scale was tried first and wraps silently for tangents of order 1e11, so the scale is per output.
- Keys: tallymap invents no randomness. Typed and raw uint32 keys both work. Results do not depend on scheduling.
- `n_active`: `f(*args, n_active=k)` runs only items `i < k`, so a fixed capacity with a varying count never
  recompiles.
- The lockstep backend's results may depend on its block width at the level of one unit in the last place,
  because XLA (JAX's compiler) compiles the same body differently for a different lane count. `lockstep_block` is
  therefore fixed by default, at 2^18.

## 2. One primitive, so every transformation composes

The first integration made the user pick `ad='forward'` (a `custom_jvp`) or `'reverse'` (a `custom_vjp`) per
function. `grad` of a forward-mode call, `hessian`, JVP-of-grad and grad-of-JVP were all impossible. It was
replaced by ONE JAX primitive (`_primitive.py`):

```
tallymap_call(*per_item, *broadcast, node=Node) -> [sum tallies] + [stats] + [per-item results]
```

A **node** is an ItemProgram (the item jaxpr plus its operand and output layout) together with how it was
derived. Every transformation rule builds a NEW ItemProgram by applying `jax.jvp`, `jax.vjp` or `jax.vmap` to an
ITEM function and binds the primitive again, so the rules recurse to any order of differentiation.

- **JVP** (any node): the primal is the same node; the tangent is a `tangent` node whose operands are the primal
  operands plus the non-zero tangents and whose outputs are the tangent part of `jax.jvp(item)`. It is linear in
  its trailing operands and takes the primal operands as inputs, so JAX's DEFAULT partial evaluation keeps the
  primal node known and stages the tangent node with the operands as residuals, which is O(1) per item. A test
  asserts no residual exceeds 4N elements. `linearize`, `grad` and `vjp` need no custom partial-evaluation rule on
  any tested JAX version.
- **transpose**: a tangent node becomes an `adjoint` node (the replay adjoint, `_reverse.vjp_prog`); an adjoint
  node becomes the tangent of its base; any other linear node (`dual_lanes`, `folded_batch`) becomes the generic
  replay adjoint of its own item at zero linear inputs.
- **batching**: when only operands the node is linear in are batched, which is the `jacfwd`, `jacrev` and
  vmap-of-JVP case, the result is a `dual_lanes` node: one kernel with T dual lanes per item. Otherwise it is a
  `folded_batch` node, which runs B*N items in ONE launch with member b's emits offset to the slice
  `[b*S, (b+1)*S)` by a retrace emit hook. Folding requires that there is no spawn and no exact output, that
  every batched broadcast operand is small (at most 4096 elements per member) and, on CUDA, that no tally leaves
  shared-memory privatisation. Otherwise it falls back to `lax.map`, one launch per member.
- **lowering** is per platform: `'cuda'` goes to the CUDA backend for `backend='auto'` and `'cuda'`, every other
  platform to lockstep. `backend='cuda'` lowered for another platform raises.
- **impl** (eager, outside `jit`): one `jax.jit` per node STRUCTURE, in a 64-entry cache, so repeated eager calls
  with new values do not recompile. `disable_jit`, `debug_nans` and `debug_infs` run the backend operation by
  operation.
- **shard_map**: operands are cast to the union of their varying axes (`lax.pvary` / `lax.pcast`) at every bind;
  the abstract evaluation returns that union; the lowering traces the backends with plain abstract values.
- **fused JVP**: `jax.jvp` would otherwise run two kernels, the primal node and the tangent node. The JVP rule
  instead binds a `fused_jvp` node, a second primitive `tallymap_jvp` holding `jax.jvp(item)`, which is ONE
  kernel. Wherever JAX partially evaluates it (`linearize`, `grad`, `vjp`, `jax.checkpoint`, and partial
  evaluation inside `scan`, `cond` or `pjit`) two rules split it back into a known primal node plus a staged
  tangent node, and its batching rule unfuses it. This uses two legacy JAX registries,
  `pe.custom_partial_eval_rules` and `pe.partial_eval_jaxpr_custom_rules`. If a JAX release drops them,
  `FUSE_JVP` turns off by feature detection and the two-node design runs instead: the library is correct either
  way, and the full test matrix passes with the fusion on and off on every tested version
  (`tests/run_all.py --unfused`). One known simplification: a fully known fused equation under `jax.checkpoint`
  is treated as saveable for every policy, which costs memory and is never wrong.
- **dual-lanes guard**: on CUDA, a `dual_lanes` program whose loop body carries a first-order intermediate of 512
  elements or more is split into chunks of fewer lanes, because the kernel keeps T copies of it in per-thread
  local memory. Measured on the toy walk at 1M items, `jax.hessian` went from 1629 ms to 13.5 ms with the guard.
  At most MAX_LANES = 16 lanes go in one kernel: `jacrev` of the 256 outputs of the toy walk at 1M items took
  229 / 191 / 175 / 176 / 325 ms in chunks of 4 / 8 / 16 / 32 / 64 lanes, and 4.3 s with all 256 at once.
- A `fused_jvp` node whose primal operands are unbatched is batched into ONE `dual_lanes` kernel that computes the
  primal once plus T tangent lanes (`dual_lanes_prog(shared=...)`). Before that fix, `jacfwd` ran two launches.

Measured on the toy walk, 1M items, 256 bins, against the old `custom_jvp` / `custom_vjp` integration: forward
0.62 vs 0.63 ms; JVP with primal 0.68 ms in one launch, against 1.22 ms in two launches unfused; `jacfwd` with
T=8 4.58 vs 4.70 ms; `grad` 4.16 vs 4.30 ms. Tracing one call takes 5.9 ms instead of 21 ms. Folding against
`lax.map` at B=8: folding is 1.2 to 2.3 times faster at 65k items on CUDA and 3.4 times on lockstep, equal at 1M
items with 4096 bins or more, and 4.5 times SLOWER when it pushes a tally out of shared memory, which the rule
above excludes.

`tests/transforms.py` is the composability matrix: a big function with pre- and post-processing, two chained
calls, and calls inside `scan`, `fori_loop`, `cond` and `while_loop`, every cell compared against the same
transformation of the dense reference. Negative controls (`TEST_NEG=tangent|cotangent|fold`, which inject a 0.1%
error) turn every affected cell into a failure, which is how the matrix is known to have power.

## 3. Reverse mode (`_reverse.py`)

- **Checkpointed loops.** `retrace(rev=...)` replaces every while loop by `checkpointed_while`. The forward pass
  stores the first C = `rev_checkpoints` (default 16) states; if the trip count K exceeds C, a checkpoint pass
  stores every m-th state with m = ceil(K/C); the backward pass walks segments newest first, refilling a buffer
  of at most C states from each segment's checkpoint. That is O(K) work for K <= C^2 and 2C+1 states of memory
  per item, with no bound on K required anywhere. Toy walk at 65k items, the ratio of grad to forward: 3.5x at
  K=100 and 3.8x at K=1000. The scheme this replaced, replaying from the start for every step, was 13x at K=100
  and over 100x at K=1000. A single fused reverse loop (one VJP instance for the whole loop) was also tried and
  was 2 to 5 times slower, because it defeats in-place updates; it was reverted.
- **Activity.** Only float loop carries that can depend on a perturbed constant, carry or table token are
  differentiated in a step VJP. The set is a fixpoint probed with symbolic zeros. On a production
  photon-transport workload it removed half of the float carries and about a quarter of the reverse-mode time.
- **Sparse table cotangents.** A perturbed broadcast operand of 32 elements or more that is read ONLY by
  `gather`, `dynamic_slice` or `slice` becomes a `custom_jvp` read whose linear part (`tallymap_tread_lin`)
  transposes into ONE emit into a cotangent token, which is a virtual tally threaded through the item and
  flushed by an atomic scatter-add into a single accumulator. Any other use of that operand makes it dense for
  ALL of its uses; the detection happens at trace time. The negative control is dropping the detection, which
  gives a relative error of 0.79 to 0.89, so the test has power. This is the difference between a large
  interpolation table costing nothing extra in reverse mode and a dense per-item copy of it, which runs out of
  memory on a GPU and takes seconds and gigabytes on the lockstep backend.
- **In-place updates.** The CUDA emitter updates a dead per-item array in place in `dynamic_update_slice`, using
  liveness over all active scopes, with loop constants, pass-through carries, predicate inputs and function
  parameters frozen. Checkpoint stores were O(C * state) copies before that: 3.3 ms to 0.47 ms at K=100.
- **Replay fingerprint.** The forward ItemProgram carries a hidden per-item uint32 hash of the final carries of
  every top-level loop, computed with ordinary jaxpr operations. The adjoint receives the forward kernel's value
  as an operand (the forward node is re-bound and XLA eliminates the common subexpression, which costs a few
  percent) and compares it against its own replay. A mismatch makes every float cotangent of that call NaN. The
  adjoint uses the forward node's options, so the arithmetic is identical and there is no contraction anywhere.
  The negative control (a private option that runs the adjoint in `approx` math against an `ieee` forward) flags
  every item whose final state changes, 832 of 4096 on the toy walk. In every test and full-program run the
  count is 0.
- **Host reporting is opt-in** (`rev_stats=True`, read through `tm.diagnostics()['reverse']`). Any
  `jax.debug.callback` in a jitted program takes `jax.jit` off its C++ dispatch path: the toy-walk gradient takes
  42 ms instead of 2.3 ms, and it does so even when the callback sits inside a `lax.cond` that is never taken
  (41.5 ms). So a failed check poisons the gradient with NaN, which is never silently wrong and needs no host
  round trip, and counting is opt-in.
- **Second order** composes: `hessian`, Hessian-vector products, `jacrev(jacfwd)` and grad-of-grad. The primal
  token of a sparse table read inside an adjoint is a fresh zero tally that is only passed through, so its
  tangent is identically zero, and forward-mode tracing of an adjoint (`_reverse.forward_tangents`) drops that
  term. Toy walk: Hessian-vector product 44 ms, a 2x2 Hessian 51 ms, grad-of-grad 199 ms.
- **When reverse mode wins.** Reverse mode costs about 3.5 to 10 times the forward pass, independent of the
  number of parameters P, while forward dual lanes cost about forward x (1 + alpha P). On the toy walk reverse
  beats `jacfwd(10)` once the trip count reaches about 30, and loses to a plain JVP for P <= 2. On the
  photon-transport workload, where alpha was about 0.3, the break-even was near P = 30. The rule of thumb is in
  the README.

## 4. Reverse mode through spawn: the tree adjoint (`_reverse_spawn.py`)

A spawn is an EMIT of the child's initial state into a child-slot output. The cotangent of that output is the
child's initial-state cotangent, and the transpose of an emit hands it to the parent as the cotangent of the
emitted value AT THE SPAWN STEP, inside the parent's ordinary replay adjoint. No new adjoint machinery is needed.

- **Recording program** (a retrace with `spawn=`, `emit=` and `gate=` hooks): a spawn writes the child's float
  leaves into a cotangent-carrier tally and the uint32 words of every leaf into an exact payload tally, advances
  the slot counter and folds the words into a hash. Every emit also folds its value into the hash. Items at or
  beyond the active count have their top-level loops gated off and their effects masked. This is a gate and not
  a `cond` because JAX's partial evaluation of an item-level `cond` forwards the item's constant tables as
  residual outputs, which the CUDA emitter cannot hold.
- **Forward**, per generation: a count pass and a write pass, with payloads in (parent, ordinal) order in one
  flat buffer. The two passes' count and hash must agree.
- **Reverse**, deepest generation first, ONE adjoint launch per generation. The roots' initial-state cotangents
  are the cotangents of the inputs. Generation launches use capacity buckets Cg, Cg/2, and so on, down to 256.
- **Never silently wrong**: a generation, tree or depth overflow, a total child count above the Queue capacity
  (which means the forward MAY have dropped children), or a replay or forward mismatch all make the gradient
  NaN, with the counts in `tm.diagnostics()['reverse']['tree']` when `rev_stats=True`. The recorded tree holds at
  most the Queue capacity in non-root items, which is exactly the bound under which the forward provably dropped
  nothing. Second order through it raises, because the payloads are bit-cast.
- **Mechanics.** `tests/reverse_spawn.py` runs toy cascades: 128 roots, 25 to 26 generations, 6.2k to 7.4k items,
  with a theta-dependent split probability scored into an inherited causal score, compared against `jax.grad` of
  a dense plain-JAX reference. Agreement: 9 units in the last place on CPU in f32 (98% of items bitwise),
  relative 1.7e-15 on CPU with 64-bit enabled, 2.6 to 8 units in the last place on an L40S; against `jacfwd` and
  `jvp` at the same keys, relative 2.5e-14 in 64-bit. The negative controls (detached children, an adjoint replay
  perturbed by one unit in the last place, a generation bucket smaller than the largest generation, a Queue
  smaller than the tree) all fail as they must.
- **Cost.** The tree adjoint is expensive: on the shower workload it cost on the order of 70 to 100 times the
  forward pass, and the break-even against `jacfwd(P)` was at P in the high tens on an L40S and far lower on a
  2080 Ti. The cost is the adjoint KERNEL, not the tree bookkeeping: that step VJP compiled to the full 255
  registers with a 13 KB stack frame and several kilobytes of spill stores, where the forward step used about a
  hundred registers and 56 bytes of stack.

## 5. Fewer knobs

Unknown option names raise. The public options are the eight in the README; every measurement and
negative-control switch is a private option whose name starts with an underscore (they are listed in
`_api._OPTS`), part of the node and of every cache key, never global state.

Removed, with their measurements: 20 environment switches; the replay-from-start reverse scheme (3.4 to 13 times
slower wherever it was measured); turning activity analysis off; the CUDA block size, shared-memory budget and
schedule options (fixed rules instead: a block of 128 threads is within 1% of 64 and 256, and step refill unless
the program spawns); the replay-check raise mode; and the tree adjoint's own capacity, depth and on-mismatch
options. The fold rule calls the CUDA kernel's own privatisation rule rather than duplicating its logic.

Dual-lanes chunking, measured on the shower-shaped program: one kernel with T lanes cost 1.34, 1.99, 2.27 and
2.57 times the forward pass for T = 1 to 4, but 14, 23 and 32 times for T = 6, 8 and 10. Chunking into the
largest size allowed by the guard gave 4.5, 4.5, 6.8 and 9.1 times for T = 4, 6, 8 and 10. That is why the guard
exists.

## 6. Why reverse mode is slow, and what to do per case

- **Host dispatch** (fixed): see section 3, opt-in host reporting.
- **Step-VJP residuals spill.** The step VJP of one thread holds ALL of the step's residuals between
  linearisation and transposition. Beyond the register file they spill to local memory, at low occupancy. For a
  single step the ratio of VJP to forward was about 3 for a transport-shaped step and about 12 for a
  shower-shaped one (800-odd residuals, several kilobytes per item); the same step vmapped by plain XLA costs
  about 1.2. The remedy is `rev_step_segments=S`, which recomputes the step in S checkpointed segments, cutting
  at small live sets, saving loop, branch and call outputs (recomputing a geometry walk doubled the cost in one
  case). On the toy walk it took the ratio from 31x to 4.2x at S=8. On the shower-shaped step it went from 10.4x
  to 5.2x at S=16, with the spills gone. On the transport-shaped one it was slightly worse, because there were
  few spills to remove and only recomputation to add. Hence it is opt-in, default 0.
- **Refuted levers**: contention on parameter-cotangent atomics; raising occupancy through launch bounds (the
  extra spills offset the occupancy); carrying one segment buffer instead of allocating a fresh one per segment
  (slower); a flat K <= C sweep under a `lax.cond` (slower). One real bug was found while measuring these:
  uniform-constant folding was allocating pools of tens of megabytes of zeros.
- **Where the adjoint time goes**, decomposed on a transport workload whose adjoint kernel cost about 8 times its
  forward: scheduling 1.26x, halving the warp occupancy 1.37 to 1.46x, and the rest split between the forward
  replay with its local buffer, sweeping the primal steps, and the transposes, in roughly equal thirds with the
  sweep largest. A user-level two-kernel prototype (the forward writes the first C states to a global buffer; one
  reverse kernel does step refill over active carries only) was about twice as fast as the library's adjoint. It
  is not in the library: it needs the K > C path and a residual output of the forward node at partial-evaluation
  time.
- **Case taxonomy** (cost per item, with the forward pass as 1):

  | the parameter enters as | what it costs, and what to use |
  |---|---|
  | a factor of an emitted value only | a direct term, at most 2x; either mode is fine |
  | a multiplicative weight, an additive log-weight or a score | forward mode on the small active sub-state is exact and needs no buffer |
  | a pathwise quantity (positions, directions, energies) | reverse mode, bounded by store plus recompute plus transpose: 3 to 4x in arithmetic, 4 to 6x in wall time |
  | a large read-only table | reverse mode with sparse atomic cotangents (section 3) |
  | a spawn tree | the tree adjoint: the same step-VJP problem plus a critical path (section 4) |

## 7. Measured against hand-written kernels

For the public, reproducible comparison against plain JAX, see `docs/performance.md`: on a GPU the library's
kernel is far faster than both a dense `jax.vmap` of the item and a hand-written masked `while_loop` with a
shared tally, and on CPU its forward pass matches the hand-written masked loop. What plain JAX cannot do at all is
reverse-differentiate a `lax.while_loop`.

Against a hand-written, hand-tuned CUDA kernel for the same program there is one private data point, on the
photon-transport engine: tallymap's generated kernel was close to the hand-tuned one for both the forward pass
and a JVP. That measurement is not reproducible from this repository, so no number from it is quoted. Two things
from it are worth recording, because they are reproducible here:

- IEEE division and square root were the whole gap. `cuda_math='approx'`, which uses the PTX instructions (PTX is
  NVIDIA's assembly-level intermediate language) that XLA:GPU itself emits for f32 division, square root and
  reciprocal square root, closes it, and became the default after a pre-registered validation
  (`docs/validation.md`). The `ieee` mode costs about 1.5 times the forward pass and 1.65 times a JVP.
- `cuda_min_blocks`, the `__launch_bounds__` minimum-blocks-per-multiprocessor register cap, is worth single-digit
  percentages and the best value is per kernel, which is why it is an option with a default of 0 rather than a
  fixed rule. Block size 64, 128 and 256 are within 1% of each other, which is why that one is fixed.

## 8. The maintenance pass

Everything in this section has a test.

- **No silent failure modes left on the default path.** `backend='auto'` raises on a GPU instead of falling back
  to the roughly 100 times slower lockstep backend (`allow_fallback=True` opts back in). A dropped spawn child or
  an item truncated at `max_steps` makes the float tallies NaN by multiplying them by
  `where(affected > 0, nan, 1)`, which is exact when nothing was affected. Integer tallies cannot be NaN, so for
  those the counter in `stats` is the only signal. Out-of-range emit indices are dropped, exactly as JAX's own
  scatter drops out-of-bounds updates, so this is not a deviation from the dense meaning, but they are counted in
  `stats['dropped_emits']`. Error messages name the item function and the source line of the offending operation,
  using `jax.extend.source_info_util`; re-traced equations keep the user's source information.
- **One reporting channel**: `tm.diagnostics()` gives versions, the GPU, the kernels with their cached CUDA
  source, the item jaxprs and the reverse-mode counters. `tm.assert_arms_differ` is the A/B rule
  (`docs/validation.md`).
- **The JAX surface in two modules**: `_compat.py` holds every version shim (batching registration, dropped
  variables, typed-key pseudo-random generators through the public implementation methods, the version policy)
  and `_jaxpr.py` is the only reader of control-flow and call equation parameters. No `jax._src` import is left.
- **The operation table is measurable, not smaller.** No emission rule was deleted: real programs reach complex
  arithmetic, sorting, cumulative sums and `argmin`, and the programs that automatic differentiation derives
  reach operations the forward pass never emits. Sugar primitives are rewritten by a canonicalisation pre-pass
  (`backends/cuda/canon.py`) into the same IEEE operations. `tm.supported_primitives()` and `docs/primitives.md`
  list what is covered, and a test checks every table name against `jax.extend.core.primitives`.
- **Bugs the tests of this pass found**: reverse mode of a table read in a while loop inside a `cond` inside a
  `scan` raised a `TallyError`, because an inert discard tally became a stacked `scan` residual; such residuals
  are now allowed and stay virtual. The eager-call cache key did not identify host callbacks, because JAX 0.11
  prints only their name, so a stale program could be reused. The CUDA operation table emitted
  `bitcast_convert_type` elementwise even between types of different widths, which changes the shape: silently
  wrong words, found by the first GPU run with 64-bit enabled.

## 9. The lockstep backend: cost that follows the active lanes

The lockstep backend is the reference implementation and the only backend off CUDA, and masking alone makes it
pay for lanes that have nothing to do: a `while_loop` runs its body on all P lanes until the LAST lane finishes,
so it costs P * max(trip count) rather than sum(trip counts), and a `cond` or `switch` with a per-lane index
runs every branch that ANY lane takes on all P lanes, so a step costs the SUM of the branches. The cuda backend
solves the same problem per thread. Here the lanes that need the work are gathered instead.

- **A while loop narrows its lane axis.** The masked loop stops as soon as at most W = P >> 2 lanes are still
  running, gathers those lanes into a W-wide lane axis (one gather per carry) and runs the SAME loop there,
  where a round costs a quarter as much; at the end one scatter puts the survivors back. A lane that stops never
  restarts, because its carry is frozen and its predicate stays false, so nothing is ever gathered back and the
  bookkeeping is one `cumsum` per level, not per round. Nothing has to be known in advance: a loop whose lanes
  all run to the end narrows once, near the end, and pays two gathers for it.
- **A cond runs each branch on the lanes that take it.** One permutation sorts the lanes by branch, then each
  branch runs on its own contiguous range in chunks of P >> 2 lanes with a dynamic chunk count -- when that
  pays: `_pays` compares the branch work the masked interpreter throws away (the sum of the branches minus their
  average, since compaction costs the average branch) against what moving the lanes costs (the permutation, the
  gather of the operands, the scatter of the results), both estimated from the jaxpr with the measured
  per-element cost of each operation, in units of one add. Nothing in it depends on the problem. On a switch over
  K phases inside a divergent loop it is 1.3 to 9.9x faster forward and 1.0 to 6.3x in reverse mode; the case it
  declines is branches of two flops and no emit, where compacting measured 1.3 to 1.5x SLOWER.
- **How much.** Both arms in one process on one milano node, JAX 0.5.3, the `benchmarks/vs_jax` problems at
  N = 65,536 (the full table is in the lockstep REPORT): forward 1.3 to 2.1x faster on the divergent problems
  (walk on spheres 264 -> 126 ms, transport 288 -> 156 ms, random walk 238 -> 137 ms, interacting md 100 ->
  76 ms), gradients 1.1 to 1.8x faster, and 0.94 to 1.01x -- within noise -- on the problems whose lanes stay
  active together (barrier, uniform, dense state, particle filter forward). Trace and compile time grows by up
  to 30%, because the loop body is instantiated once more.
- **Results do not change.** On a CPU every benchmark problem and every test gives BITWISE identical tallies,
  per-item results and stats with compaction on and off, in f32 and f64, forward, jvp, grad and vmap. That is
  not luck: a scatter-add runs in index order there, the gather is stable so the live lanes keep their relative
  order, and the lanes that drop out contributed nothing to the accumulator in the first place. On a GPU the
  hardware orders the updates of one scatter itself, so a different lane count sums them differently: the
  forward pass measured bitwise on a 2080 Ti and a jvp 9.7e-08 relative, which is the same one-ulp caveat the
  block width already carries (section 1d). Integer tallies and stats are identical on both.
  `tests/edge.py` asserts it (`_lockstep_compact=False` is the A/B control, with `_lockstep_cond` for the cond
  half).
- **Two designs measured and rejected.** (1) Chunking EVERY round into ceil(active / W) chunks, which is what
  the per-element cost model says to do: on a CPU an HLO operation costs a thread fork-join whatever its width,
  so chunked rounds cost 1.1 to 4.1x the masked loop on the problems whose lanes stay active, for about 2x on
  the divergent ones. (2) Letting every nested loop narrow, and narrowing more than once: the compiled program
  then grows with the loop nesting, and the shower's reverse-mode program took 899 s to compile instead of 56 s
  for 1.3x per call. Hence the rule: only a loop the interpreter is not already inside a loop for narrows, and
  only once.
