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

That rule was re-measured on every `benchmarks/vs_jax` problem (private `_schedule`, forward and gradient, `div`
high), as the ratio of step-refill time to whole-item time (below 1: step refill is faster):

| | 2080 Ti, 262k items | 2080 Ti, 4M items | L40S, 262k items |
|---|---|---|---|
| gillespie | 0.71 / 0.71 | 0.60 / 0.60 | 0.98 / 0.98 |
| barrier | 0.72 / 0.96 | 0.74 / 0.95 | 0.99 / 1.01 |
| random walk | 0.73 / 0.93 | 0.84 / 1.00 | 1.13 / 1.05 |
| transport | 0.84 / 0.91 | 0.82 / 0.98 | 1.07 / 1.01 |
| walk on spheres | 0.96 / 1.04 | 0.88 / 0.99 | 1.11 / 1.03 |
| ode ensemble | 0.81 / 1.21 | 0.97 / 1.00 | 1.05 / 1.01 |
| particle filter | 0.94 / 1.03 | 1.14 / 1.04 | 1.09 / 1.04 |
| uniform | 1.17 / 1.05 | 1.25 / 1.02 | 1.11 / - |
| interacting md | 1.28 / 1.04 | 1.56 / 1.15 | 1.13 / 1.06 |
| shower (spawns) | 1.30 / 1.02 | 1.19 / - | 0.91 / 1.00 |

(forward / gradient; identical kernels rerun differ by up to about 10%.)  Step refill wins by up to 1.7x where
trip counts diverge and each resident thread runs several items (the 2080 Ti holds about 70k resident threads);
it loses by up to 1.6x on uniform or interaction-bound items, and by 1-13% on the L40S at 262k items, where its
218k resident threads leave about one item each and there is nothing to refill.  No structural rule picks the
faster mode on both GPUs, and a rule on items per resident thread would need a fitted threshold (occupancy and
divergence both move it), so the rule stays: step refill unless the program spawns, which is right on the
divergent workloads the library is for.

Event scheduling (`_schedule='events'`) is the same kernel as step refill unless the STEP BODY has a top-level
`cond` worth cutting; it then cuts the step there and lets each warp run ONE branch per round (`kernel._cut`,
`kernel._events_main`).  No `benchmarks/vs_jax` problem has such a `cond`, so on all of them it compiles to the
identical kernel; `benchmarks/events_bench.py` is the measurement of what it buys, on the item shape it is for
-- a divergent loop whose body is a lopsided three-way switch, one branch W times more expensive and taken with
probability P (2080 Ti, 4M items, median of 9 / min):

| | `'step'` | `'events'` | `'item'` |
|---|---|---|---|
| forward, P = 0.2, W = 64 | 21.91 / 16.45 ms | **13.73 / 11.89** | 22.00 / 21.94 |
| forward, P = 0.05, W = 256 | 45.64 / 44.98 ms | **14.60 / 12.68** | 46.27 / 46.11 |
| gradient, P = 0.2, W = 64 | 176.4 ms | 170.5 | 176.4 |
| gradient, P = 0.05, W = 256 | 2978 ms | 2943 | 2978 |

So the rarer and costlier the branch, the bigger the win: 3.1x on the forward pass at P = 0.05, W = 256, and
1.6x at P = 0.2, W = 64; the gradient is unchanged (1.01-1.03), because an adjoint program has three top-level
while loops and so never gets the step split at all (section 6) -- it runs whole-item either way.  Event
scheduling nevertheless stays OPT-IN and is not a default: nothing in the item jaxpr says how lopsided a branch
is at run time, and a short branch gains nothing (a toy with a 256-step branch: 0.90 ms step against 0.86
events at P = 0.14, 0.93 against 1.07 at P = 0.5).

Only the step loop is cut.  Event-scheduling an INNER while loop (at any depth, by the same ski-rental rule) was
implemented and is removed: it never fires on any `benchmarks/vs_jax` problem either, and on four toys built to
be its best case -- a lopsided switch inside an inner while loop of 8 or 64 trips, long branch 256 or 2048
steps, 1M items, 2080 Ti -- it was 1.82x, 1.36x, 1.22x and 1.03x SLOWER than the plain loop, never faster.  It
was also what made `_schedule='events'` cost 3.1x in reverse mode (an adjoint program's three top-level loops
were each event-scheduled): with it gone the gradient row above is a wash.

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
  2^32 scale was tried first and wraps silently for tangents of order 1e11, so the scale is per output.  A
  tangent tally uses its primal output's scale; a separate `tangent_scale` was an unused knob and was removed.
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
  every batched broadcast operand is small (at most 4096 elements per member) and, on CUDA, that ONE member's
  launch of N items could not fill the device by itself (N below the multiprocessor count times 1024, the
  smallest maximum-resident-threads-per-multiprocessor of any tested architecture, so the bound is a floor).
  Otherwise it falls back to `lax.map`, one launch per member.
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
  At most MAX_LANES = 32 lanes go in one kernel: `jacrev` of the 256 outputs of the toy walk at 1M items took
  229 / 191 / 175 / 176 / 325 ms in chunks of 4 / 8 / 16 / 32 / 64 lanes, and 4.3 s with all 256 at once.
- A `fused_jvp` node whose primal operands are unbatched is batched into ONE `dual_lanes` kernel that computes the
  primal once plus T tangent lanes (`dual_lanes_prog(shared=...)`). Before that fix, `jacfwd` ran two launches.

Measured on the toy walk, 1M items, 256 bins, against the old `custom_jvp` / `custom_vjp` integration: forward
0.62 vs 0.63 ms; JVP with primal 0.68 ms in one launch, against 1.22 ms in two launches unfused; `jacfwd` with
T=8 4.58 vs 4.70 ms; `grad` 4.16 vs 4.30 ms. Tracing one call takes 5.9 ms instead of 21 ms.

The fold rule used to be "fold while every privatised tally stays privatised", and it got both ends wrong,
because what folding costs and buys has nothing to do with the tally: it makes B copies of every per-item
operand and runs ONE launch of B*N items, which is a win exactly while one member's N items cannot fill the
GPU by themselves. Measured on the toy walk, B=8, 2080 Ti (68 multiprocessors, so the rule folds below
69,632 items), each cell timed in the same position of the same script under the old and the new rule
(`vmap over theta B=8`, minimum of 7, ms):

| items | bins | old rule | new rule | |
|---|---|---|---|---|
| 65,536 | 64 | fold 1.21 | fold 1.22 | both fold |
| 65,536 | 256 | `lax.map` 3.05 | **fold 2.31** | 1.32x faster |
| 65,536 | 1,024 | `lax.map` 3.03 | **fold 1.37** | 2.21x faster |
| 65,536 | 4,096 | fold 1.20 | fold 1.20 | both fold |
| 262,144 | 64 | fold 3.72 | `lax.map` 5.08 | 1.37x slower |
| 262,144 | 256 | `lax.map` 5.10 | `lax.map` 5.09 | both `lax.map` |
| 262,144 | 1,024 | `lax.map` 5.20 | `lax.map` 5.15 | both `lax.map` |
| 262,144 | 4,096 | fold 3.76 | `lax.map` 5.07 | 1.35x slower |
| 1,048,576 | 64 | fold 33.3 | **`lax.map` 13.6** | 2.46x faster |
| 1,048,576 | 256 | `lax.map` 13.7 | `lax.map` 13.6 | both `lax.map` |
| 1,048,576 | 1,024 | `lax.map` 13.6 | `lax.map` 13.6 | both `lax.map` |
| 1,048,576 | 4,096 | fold 34.5 | **`lax.map` 13.9** | 2.48x faster |
| 4,194,304 | 64 | fold 277.6 | **`lax.map` 40.5** | 6.85x faster |
| 4,194,304 | 4,096 | fold 282.6 | **`lax.map` 48.4** | 5.83x faster |

The old rule refused the 2.2x at 65k items and 1,024 bins (that tally never fits shared memory, so folding
could not change its status) and allowed a loss that grows with N -- 2.5x at a million items and 5.8 to 6.9x
at four million (a 64-bin tally is privatised folded or not, so it passed the old rule at every N). The new
rule removes both, and gives up 1.35 to 1.37x at 262k items, which is inside the bar. On lockstep there is no occupancy and no such reversal -- folding is one masked pass over B*N lanes
instead of B passes, 1.27 to 1.44 times faster at 16k and 65k items (CPU, 8 cores) -- so there it is always
taken.

`tests/transforms.py` is the composability matrix: a big function with pre- and post-processing, two chained
calls, and calls inside `scan`, `fori_loop`, `cond` and `while_loop`, every cell compared against the same
transformation of the dense reference. Negative controls (`TEST_NEG=tangent|cotangent|fold`, which inject a 0.1%
error) turn every affected cell into a failure, which is how the matrix is known to have power.

## 3. Reverse mode (`_reverse.py`)

- **Checkpointed loops.** `retrace(rev=...)` replaces every while loop by `checkpointed_while`. The forward pass
  stores nothing and only counts the steps K; the backward pass replays the loop once storing every m-th state,
  then walks the segments newest first, refilling a buffer of at most C = `rev_checkpoints` (default 16) states
  from each segment's checkpoint. That is O(K) work for K <= C^2 and 2C states of memory per item, with no bound
  on K required anywhere. Toy walk at 65k items, the ratio of grad to forward: 3.0x at K=100. The scheme this
  replaced, replaying from the start for every step, was 13x at K=100 and over 100x at K=1000. A single fused
  reverse loop (one VJP instance for the whole loop) was also tried and was 2 to 5 times slower, because it
  defeats in-place updates; it was reverted.
- **The checkpoint stride is C, not ceil(K/C).** The segment buffer is allocated once per segment and that costs
  C stores whatever the segment's length turns out to be, so the sweep's local-memory traffic is (number of
  segments) x C states, not K. A stride of ceil(K/C) gives one state per segment whenever K <= C, which wastes
  C-1 slots of every buffer; a stride of C fills every buffer, keeps the number of segments at most C, and falls
  back to ceil(K/C) only beyond K = C^2, where the sub-segment path splits a segment into full buffers too. So
  the traffic is ~K states for every K. Measured on a 2080 Ti at the `benchmarks/vs_jax` sizes, this took the
  gradient of transport from 15.6 to 7.4 ms, walk-on-spheres from 9.9 to 5.0 ms, and the toy walk at K=100 from
  1.70 to 1.47 ms, with no problem slower. It is also why `rev_checkpoints` used to look so sharply
  non-monotone: at C = 32 the same transport gradient cost 44.9 ms before and 10.7 ms after.
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

Unknown option names raise. The public options are the six in the README; every measurement and
negative-control switch is a private option whose name starts with an underscore (they are listed in
`_api._OPTS`), part of the node and of every cache key, never global state.

Removed, with their measurements: 20 environment switches; the replay-from-start reverse scheme (3.4 to 13 times
slower wherever it was measured); turning activity analysis off; the CUDA block size, shared-memory budget and
schedule options (fixed rules instead: a block of 128 threads is within 1% of 64 and 256, and step refill unless
the program spawns); the replay-check raise mode; the tree adjoint's own capacity, depth and on-mismatch
options; `cuda_min_blocks` (at most 1.08x on any cell measured) and `rev_step_segments` (nothing on four
independent workloads), both covered below; `Sum(tangent_scale=)`, which nothing ever set; the inner-loop half
of event scheduling (never faster, up to 1.82x slower, section 1b); and the two reverse-mode
cost-decomposition options (`_rev_phase`, `_tree_phase`), which produced wrong gradients on purpose and which no
test used -- the decompositions they produced are quoted in section 6. The fold rule is one comparison against
the device's resident-thread count and no longer consults the kernel's privatisation rule at all (section 2).

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
  about 1.2. There was an opt-in remedy, `rev_step_segments=S`, which recomputed the step in S checkpointed
  segments, cutting at small live sets and saving loop, branch and call outputs; it is REMOVED, because no
  public workload reproduces a benefit (see "Refuted: step segmentation" below). The only evidence for it was
  the two engines (a shower-shaped step 10.4x to 5.2x at S=16, with the spills gone; a transport-shaped one
  slightly worse), which is not reproducible here.
- **Refuted levers**: contention on parameter-cotangent atomics; raising occupancy through launch bounds (the
  extra spills offset the occupancy); carrying one segment buffer instead of allocating a fresh one per segment
  (slower); a flat K <= C sweep under a `lax.cond` (slower). One real bug was found while measuring these:
  uniform-constant folding was allocating pools of tens of megabytes of zeros.
- **Refuted: one kernel per phase, with a bounded global buffer.** Every adjoint program has exactly three
  top-level while loops (the replay, the checkpoint pass, the sweep), so the step split, which needs exactly one,
  never applies to an adjoint: it always runs whole-item and always gets one register allocation for all three
  phases. Cutting the adjoint jaxpr at its top-level loops into one ItemProgram per loop, with the values that
  cross a cut as per-item results of one piece and per-item operands of the next (items in chunks of M, so the
  buffer is O(M * crossing state), 584 bytes per item on the toy walk), was implemented and measured on a
  2080 Ti. Each phase does get its own registers, and the sweep's drop sharply (63 to 50), but the gradient was
  SLOWER on every one of the six problems: toy 2.47 against 1.66 ms, transport 24.0 against 15.6, walk-on-spheres
  14.2 against 10.2, ode 34.3 against 25.4, barrier 19.3 against 15.4, particle filter 31.0 against 22.1, at 20
  times the peak memory (264 MB against 12 MB on transport). Two reasons, both measured: the crossing state is a
  write and a read of the whole checkpoint buffer per item through global memory, plus a copy of it in the piece
  that updates it; and under the step split the buffer becomes per-lane state that lives in the main kernel,
  where ptxas puts it in registers -- phases 0 and 1 compiled to 248-255 registers with spills, and forcing
  whole-item pieces instead (which costs the scheduling the split was for) only brought them back to 64. The
  prototype this came from was faster because its FORWARD kernel wrote the checkpoints, so it had no replay phase
  at all; that needs a residual output of the forward node, which is O(N * C * state) and so is the one thing the
  chunking cannot bound.
- **Refuted: let the forward store the checkpoints.** With a stride of C the forward pass of a checkpointed loop
  can store the stride-C states itself (slot C as a scratch sink makes the store unconditional), which removes
  the backward pass's checkpoint pass entirely while K <= C^2. It does remove it -- the replay phase
  (measured with a temporary phase-skip instrumentation) drops from 3.23 to 2.81 ms on transport -- but it costs one local store per step and
  takes per-item memory back to 2C+1 states, and the gradient is faster on only two of the six problems
  (walk-on-spheres 4.07 -> 3.61 ms, barrier 13.6 -> 12.3) and slower on three (transport 6.03 -> 6.84, toy walk
  1.44 -> 1.51, particle filter 15.1 -> 15.7), with the adjoint kernel's stack frame back up from 544 to 832
  bytes. Which side wins depends on K, which is not known at trace time, so there is no rule that would pick it.
- **Refuted: step segmentation.** Recomputing each loop step in S checkpointed segments inside its VJP was an
  option (`rev_step_segments`) and is gone. A rule could have read the spill bytes out of the compiled kernel
  and picked S, but there was nothing to pick: ptxas reports ZERO spill bytes for the adjoint kernel of all six
  problems (stack frames 400 to 832 bytes, which is the checkpoint buffers, not spills). Four independent
  re-measurements on a 2080 Ti then found no benefit anywhere, and the option is removed:
  the six `benchmarks/vs_jax` gradient cells at their sweep sizes, S = 0 / 4 / 8, were within 1.00-1.05
  (job 41231627); the toy walk at 1M items, S = 0 / 4 / 8 / 16, gave 6.32 / 6.36 / 6.45 / 6.43 ms
  (job 41231673); a toy built to spill -- a step holding 128 live nonlinear intermediates, 168 registers and
  1,328 bytes of local memory per thread, grad/forward 8.1x -- gave 32.4 / 31.2 / 32.1 ms at S = 0 / 2 / 4
  (job 41232260); and the diffrax ensemble of `docs/equinox.md` gave 4,936 ms at S = 0 against 4,968 at S = 4.
  A step that really spills may still exist (the two engines), but nothing public shows one, and an option
  cannot be kept on evidence nobody can rerun.
- **Where the adjoint time goes**, decomposed on a transport workload whose adjoint kernel cost about 8 times its
  forward: scheduling 1.26x, halving the warp occupancy 1.37 to 1.46x, and the rest split between the forward
  replay with its local buffer, sweeping the primal steps, and the transposes, in roughly equal thirds with the
  sweep largest. A user-level two-kernel prototype (the forward writes the first C states to a global buffer; one
  reverse kernel does step refill over active carries only) was about twice as fast as the library's adjoint. It
  is not in the library: it needs the K > C path and a residual output of the forward node at partial-evaluation
  time.
- **The same decomposition on the public problems** (`benchmarks/vs_jax` at its sweep sizes, 2080 Ti, with the
  private `_rev_phase` options: `skip` is the replay and the checkpoint pass with no sweep, `primal` runs one
  PRIMAL body step in place of each step VJP). Reading it as replay / sweep / transposes:

  | problem | K (mean) | forward | grad | grad/fwd | `skip` | `primal` |
  |---|---|---|---|---|---|---|
  | toy walk, K=100, 65k | 100 | 0.49 | 1.47 | 3.0x | 0.80 | 1.05 |
  | transport, 2^19 | 16 | 0.97 | 6.07 | 6.3x | 3.36 | 4.71 |
  | walk on spheres, 2^19 | 8.8 | 0.75 | 4.14 | 5.5x | 2.81 | 4.00 |
  | ode ensemble, 2^18 | 227 | 3.48 | 25.3 | 7.3x | 15.3 | 21.4 |
  | barrier, 2^19 | 222 | 3.75 | 13.4 | 3.6x | 6.65 | 14.2 |
  | particle filter, 2^16 | 45 | 5.36 | 17.0 | 3.2x | 11.3 | 15.1 |

  The replay and the checkpoint pass are half of the adjoint and the transposes about a third; on barrier and the
  particle filter a primal step costs as much as the step VJP, so `primal` is not below `grad` there and the
  transposes are free. The replay alone is 3 to 4 times the forward kernel for the same arithmetic, because the
  adjoint runs whole items where the forward refills lanes at step granularity.
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
- The `__launch_bounds__` minimum-blocks-per-multiprocessor register cap was an option (`cuda_min_blocks`) and is
  gone: re-measured on the toy walk at 4M items on a 2080 Ti, values 0 / 1 / 2 / 4 gave forward 6.22 / 6.25 /
  6.22 / 6.17 ms and a JVP 7.16 / 7.25 / 7.28 / 7.17 ms, i.e. nothing outside rerun noise, which matches the
  single-digit percentages it was worth when it was added. Block size 64, 128 and 256 are within 1% of each
  other, which is why that one is fixed too.

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
  and `_jaxpr.py` is the only reader of control-flow and call equation parameters. One guarded `jax._src` import
  is left: JAX's own pure-JAX LU, for the cuda lowering of `jnp.linalg.solve` (`_compat.jax_generic_linalg`).
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

## 9. Compile time and code size

The cuda emitter inlines the whole item into one kernel: call bodies (`pjit`, `custom_jvp`), `cond` branches and
loop bodies are emitted in place, and arrays of up to 32 elements become straight-line code. For a large item this
is what compile time is made of. The measurements are on the photon-transport engine's reconstruction item
(forward: 24,407 jaxpr equations, counting sub-jaxprs; 4.2 MB of CUDA; the `jacfwd` program with 6 lanes: 5.1 MB),
because the repository's own problems compile in seconds.

- **Where the time goes.** About 85% is NVRTC's front end (C++ to PTX), and ptxas most of the rest. Long lazily
  inlined expressions were the first cause found: capping the inlined lazy nodes per element at 32
  (`codegen.LAZY_MAX_COST`, was 256) took that forward kernel from 1543 to 834 s of compile on a 2080 Ti and
  from 9.04 to 6.87 s of run time, with bitwise identical outputs.
- **Repeated sub-programs are not the bulk.** Call bodies hold 10,333 of the 24,407 equations when every call site
  is counted, and 2,630 when each distinct body is counted once. Emitting each distinct body ONCE, as a
  `__noinline__` device function called from every site (deduplicated on the generated text), was built and
  measured: 134 call sites became 25 functions, but the source shrank only 13% (4.16 to 3.64 MB), because most of
  the source is straight-line element code, not repeated calls.
- **Splitting does make NVRTC faster, and costs as much at run time.** NVRTC is superlinear in function size:
  making only the kernel's four phase functions (`sj_pre` / `sj_cond` / `sj_body` / `sj_post`) `__noinline__`
  took the forward compile from 550 to 406 s on a CPU node, with identical source, and outlining every call body of
  4 equations or more gave 416 s. On the GPU node, end to end (2080 Ti, compile s / run s): base forward 842 /
  6.78, `jacfwd` 1005 / 7.07; phases not inlined 793 / 7.23 and 893 / 7.68; phases plus call bodies of 16 or more
  equations 799 / 7.12 and 902 / 7.60. Compile time drops 5-11% and run time rises 5-9%. A kernel is compiled once
  and then called many times, so neither is the default, and the mechanism is not in the library.
- **Branch stack frames do not add up.** The concern behind outlining branches was that NVVM keeps every branch's
  local arrays in the caller's frame at once. A switch whose branches each fill their own array with dynamic
  indexing (2080 Ti, local bytes per thread, inlined / outlined): 512 floats, 2 branches 4,176 / 4,496, 8 branches
  4,176 / 4,528; 2048 floats, 2 branches 32,800 / 16,464, 8 branches 16,416 / 16,496. The frame does not grow with
  the number of branches; the one doubled case (2048 floats, 2 branches) is a compiler choice for that shape,
  which outlining happens to undo, and outlining costs run time (1-8%) and, at 8 branches, occupancy.

What remains is the volume of straight-line code itself. A large item's stack frame and compile time are
properties of the program; the levers that worked were in the program (fewer generic call sites, a static set of
geometry types, a bounded stack depth), and on the library side the lazy-inline cap.
