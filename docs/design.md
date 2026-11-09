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
the group of 32 GPU threads that execute together on NVIDIA hardware.) Under the split, a loop carry is normally rebuilt by
`sj_body` into a temporary and copied back once per step; an ARRAY carry of at least `STEP_INPLACE_MIN` (64)
elements (a checkpoint buffer, a state history) is instead passed to `sj_body` by mutable reference and, when the
body's result is that parameter updated in place, it has no temporary and no copy. Smaller carries keep the copy, so
kernels without a large carry are unchanged.

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

Event scheduling of the step loop was implemented and is removed. It cut the step body at its most expensive
top-level `cond` and let each warp run ONE branch per round, chosen online by a ski-rental rule. It won only on
the synthetic item built for it, a divergent loop around a lopsided three-way switch (one branch W times more
expensive, taken with probability P; 2080 Ti, 4M items): 1.6x forward at P = 0.2, W = 64 and 3.1x at P = 0.05,
W = 256, with the gradient unchanged. No `benchmarks/vs_jax` problem has a step body it would cut. On the
photon-transport engine at the same commit it was 0.81x (0.052 against 0.043 s), and its general form in the
particle-shower engine (the step cut into phases) was 0.50x and 0.77x on a 2080 Ti and hung on one
configuration. An earlier inner-loop variant was 1.03-1.82x SLOWER on four toys built as its best case.

A lane claims ONE item per claim, warp-aggregated: the needy lanes of a warp take one contiguous run of item
indices between them, so the claim counter sees one atomic per warp rather than one per lane. Claiming a RUN of
up to 16 items per lane (a chunk sized from the item count and the grid) was in the kernel and is removed: it
leaves the last claims holding several items each, which is a tail every other lane waits for. Measured on a
2080 Ti over all 18 `benchmarks/vs_jax` forward and gradient cells, one item per claim is 1.46x faster on
random_walk's forward pass, 1.14 to 1.18x on transport forward, random_walk gradient and uniform gradient,
1.09x on ode, and never slower (worst cell 1.002).

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
  the `compare` protocol is the contract (`docs/validation.md`).
- `backend='auto'`: CUDA on a GPU; a program the emitter does not support raises `UnsupportedError` there, with
  the advice to pass `backend='lockstep'`, because a silent change to the reference interpreter is a silent
  factor-of-100 slowdown. On every other platform it is lockstep with a one-time warning. `backend='cuda'` raises
  the same way. Compile errors from NVRTC (NVIDIA's runtime CUDA compiler) are
  bugs and raise. A compiled kernel whose per-thread local memory is over CUDA's 512 KiB per-thread limit raises
  `UnsupportedError` with the bytes and the limit before it is launched (`compile.check_frame`, read from the
  cubin's `.nv.info` records); the driver would refuse it with `invalid argument`. Effects, such as `jax.debug.print` or an input/output callback, raise on both backends.
- N, the number of items, is a runtime kernel argument: neither the compiled kernel nor the in-memory kernel cache
  key contains it. One call takes fewer than 2^31 items.
- `Sum(exact=True)`: int64 fixed point, `round(v * scale)`, with one scale per output. Unrepresentable values,
  meaning `|v * scale| >= 9.2e18`, are dropped AND counted (`stats['exact_overflow']`) on both backends, and an
  overflow makes every exact float output of the call NaN (the resource-limit rule; the counter is per call). A fixed
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
  At most MAX_LANES = 16 lanes go in one kernel: `jacrev` of the 256 outputs of the toy walk at 1M items took
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
  counts the steps K and stores the states before steps 0..C-1 in a HEAD buffer; an item of K <= C steps is swept
  straight off the head, with no replay. Otherwise the backward pass replays the loop once storing every m-th
  state, then walks the segments newest first, refilling a tape of at most C = `rev_checkpoints` states from each
  segment's checkpoint (the first C states come from the head). That is O(K) work for K <= C^2 and 3C+1 states of
  memory per item, with no bound on K required anywhere. Only the carries the step or predicate reads are stored
  (`_rewrite._live_carries`, read off the loop's jaxprs). A counting pass that stored nothing made every K <= C
  item replay its whole loop once more: 1.7x slower on lumen SK's gradient and 1.4x on SBND (lumen/perf
  8c92ee8 / 762da14). The head is a read-only residual, not the segment tape: reusing it as the tape (a carry of
  the segment loops) made the emitter copy it, a 28 KB against a 12 KB frame on lumen SK, 17% slower.
- **C is chosen from the state size** (`rev_checkpoints='auto'`, `_reverse.auto_checkpoints`): 16 below 128 B of
  stored state, else the largest of 4 ... 64 whose 3C+1 states fit 64 KiB of LOGICAL bytes. lumen (306-318 B
  states, warp-maximum trip counts 16 to 60) is 1.3-1.6x faster at C = 64 than at 16; the `vs_jax` problems
  (8-25 B) are 1.5-3x slower at 64. The 64 KiB budget ignores the device (the driver reserves the compiled
  frame for every resident thread): sizing C from the compiled frame against free memory is still to do. An
  adjoint that is differentiated again (hvp, grad-of-grad) is rebuilt at C = 16 (`_primitive._lean`): its
  memory goes as C1 x C2, and at C1 = 64 the toy's rev-over-rev kernel needed 290 KB per thread (launch OOM). Toy walk at 65k items, the ratio of grad to forward: 3.0x at K=100. The scheme this
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
- **Checkpoint buffers are uninitialised.** The head, the checkpoint buffer and the segment tape are
  `tallymap_uninit` buffers (`_reverse.uninit_p`): every slot is written before it is read, by construction. The cuda
  backend declares them with no fill, and a copy of one that has not been written yet emits nothing (a copy of an
  undefined value is undefined; the step split and loop-carry adoption would otherwise load and keep a second copy
  of garbage). Every other backend gets zeros. Zero-filling 3C+1 states per item at C = 64 made the cheap-step
  `benchmarks/vs_jax` gradients up to 5x slower at 2^19 items (random_walk 0.80 -> 3.98 ms; lumen/perf d6cb9fc). The
  contract is tested, not assumed: `backends/cuda/codegen.FILL` (test-only module constant) fills every
  uninitialised per-thread declaration with a byte, and tests/reverse.py requires bitwise-equal per-item results
  and per-item cotangents for 0x00 and 0xFF.
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
- **In-place updates.** The CUDA emitter updates a dead per-item array in place in `dynamic_update_slice` and in
  `scatter` (`x.at[i].set(v)` with a traced `i` is a scatter), using liveness over all active scopes, with loop
  constants, pass-through carries, predicate inputs and function parameters frozen, and never when the update or
  the indices read the operand's own storage (an element-by-element write would read elements it had already
  overwritten). Checkpoint stores were O(C * state) copies before that: 3.3 ms to 0.47 ms at K=100. Scatter
  copied its operand every time before (Unreleased): on DTRAX's mode-2 item (g0102, 2080 Ti) that was 17 copies of its
  float64[34,23] step carry per step and a 178,160 B frame (11.6 GiB of local reservation: out of memory on the
  2080 Ti), 71,440 B in place (the private engine line: 70,160 B); `sEM1_lar_e` 557,248 B (over the 512 KiB
  limit) -> 156,496 B. Digests unchanged (`tab=85bce7a5ff627c27`, `97ceb69adae09286`). LUCiD's two kernels
  (fwd, jac6) are byte-identical with and without it (same `.cu` sha256; job 43897628).
- **Local memory is reserved for the device, not the launch.** The CUDA driver reserves `local bytes per thread x
  SMs x max resident threads per SM` at the first launch, whatever the grid: measured on a 2080 Ti (job
  43897685) 2,382 MiB at 36,864 B and 750 MiB at 12,288 B for n_items = 1, 128, 8,192 and 69,632, spawning or not,
  equal to that formula. The persistent grid is already `min(occupancy x SMs, ceil(n_items / block))` blocks for a
  non-spawning program (a spawning one needs the whole grid for its children), but bounding the grid cannot shrink
  the reservation: only a smaller frame does. (The private engine line's `TALLYMAP_GRID=1` shrank its grid-sized
  global arena, not this reservation; dev has no arena.)
- **Replay fingerprint.** Reverse mode uses a variant of the forward ItemProgram that carries a hidden per-item
  uint32 hash of the final carries of every top-level loop, computed with ordinary jaxpr operations (a
  forward-only call does not compute it: section 10). The adjoint receives the forward kernel's value as an
  operand (the forward node is re-bound and XLA eliminates the common subexpression, which costs a few percent)
  and compares it against its own replay. A mismatch makes every float cotangent of that call NaN. The
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

Unknown option names raise. The public options are the five in the README; every measurement and
negative-control switch is a private option whose name starts with an underscore (they are listed in
`_api._OPTS`), part of the node and of every cache key, never global state.

Removed, with their measurements: 20 environment switches; the replay-from-start reverse scheme (3.4 to 13 times
slower wherever it was measured); turning activity analysis off; the CUDA block size, shared-memory budget and
schedule options (fixed rules instead: a block of 128 threads is within 1% of 64 and 256, and step refill unless
the program spawns); the replay-check raise mode; the tree adjoint's own capacity, depth and on-mismatch
options; `cuda_min_blocks` (at most 1.08x on any cell measured) and `rev_step_segments` (nothing on four
independent workloads), both covered below; `Sum(tangent_scale=)`, which nothing ever set; event scheduling
(section 1b); and the two reverse-mode
cost-decomposition options (`_rev_phase`, `_tree_phase`), which produced wrong gradients on purpose and which no
test used -- the decompositions they produced are quoted in section 6. The fold rule is one comparison against
the device's resident-thread count and no longer consults the kernel's privatisation rule at all (section 2).

Removed from the public surface before the first release (API decisions, so no measurement is involved):

- `tm.compare`, with `criteria` and the square mode of the re-tracer that produced its floor run. Its frozen
  thresholds are a test protocol rather than an API, and one test used it; it is now `tests/compare_protocol.py`,
  where the floor run is the same item with squared emits, written by the caller.
- `tm.assert_arms_differ` (and its `artifacts`), the A/B rule: measurement hygiene for the library's own
  developers, built entirely on `tm.diagnostics`. It is now `tests/ab_rule.py`, used by the tests and by
  `benchmarks/perf_negative_control.py`.
- The option `allow_fallback` (run lockstep with a warning when `backend='auto'` on a GPU cannot compile a
  program), with `_lockstep_cannot_run`, which kept its error from suggesting a fallback that could not run
  either. `backend='lockstep'` is the same thing, explicit, and the cuda error now says to pass it.

Dual-lanes chunking, measured on the shower-shaped program: one kernel with T lanes cost 1.34, 1.99, 2.27 and
2.57 times the forward pass for T = 1 to 4, but 14, 23 and 32 times for T = 6, 8 and 10. Chunking into the
largest size the `LANES_MAX_LOCAL` guard allowed (129 lane-dependent elements per lane: at most 3 lanes per
chunk) gave 4.5, 4.5, 6.8 and 9.1 times for T = 4, 6, 8 and 10. That is why the guard exists. It never fires on
the photon-transport engine (54 / 117 elements per lane at T = 6 / 13), but a DTRAX parameter tangent makes
essentially the whole float carry lane-dependent (about 1,400-5,000 f64 per lane), so the guard stays; replacing
it with chunking by the local-memory budget is open (2026-11-04).

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
- **The same decomposition on the public problems** (`benchmarks/vs_jax` at its sweep sizes, 2080 Ti, measured
  with a temporary phase instrumentation that is no longer in the library: `skip` is the replay and the
  checkpoint pass with no sweep, `primal` runs one PRIMAL body step in place of each step VJP; both give wrong
  gradients, so they were a measurement and not an option). Reading it as replay / sweep / transposes:

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
  to the roughly 100 times slower lockstep backend (`backend='lockstep'` asks for it explicitly). A dropped spawn
  child or an item truncated at `max_steps` makes the float tallies NaN by multiplying them by
  `where(affected > 0, nan, 1)`, which is exact when nothing was affected. Integer tallies cannot be NaN, so for
  those the counter in `stats` is the only signal. Out-of-range emit indices are dropped, exactly as JAX's own
  scatter drops out-of-bounds updates, so this is not a deviation from the dense meaning, but they are counted in
  `stats['dropped_emits']`. Error messages name the item function and the source line of the offending operation,
  using `jax.extend.source_info_util`; re-traced equations keep the user's source information.
- **One reporting channel**: `tm.diagnostics()` gives versions, the GPU, the kernels with their cached CUDA
  source, the item jaxprs and the reverse-mode counters. The A/B rule is `tests/ab_rule.py`
  (`docs/validation.md`), built on it.
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
  and then called many times, so neither is the default, and the mechanism is not in the library. Note that this
  needs the `__noinline__` attribute written out: merely dropping `__forceinline__` changes nothing on that
  kernel (430.0 s against 431.4 s, identical cubin size, 255 registers and an 89,920-byte frame either way),
  because a function with one call site is inlined straight back.
- **Several TRANSLATION UNITS, compiled in parallel, is a much bigger lever -- and costs occupancy.** The
  superlinearity is per translation unit, not per function, so compiling the item's functions *apart* beats
  `__noinline__`, which only hides them inside the same module. Measured on one `milano` node with the generated
  source split by hand, NVRTC `-rdc=true` per unit in separate processes and `nvJitLink` without LTO: the 4.16 MB
  forward kernel takes **38.2 s as 40 units in 32 processes against 456 s monolithic (11.9x**, and 189 s of CPU
  against 452 s), `compile_scaling` at 200 stages goes 485 s -> 6.5 s as 34 units in 16 processes (74x), and the
  link itself is free (0.00 s). The units must be separate PROCESSES: in threads NVRTC inflates the CPU time 5x
  and recovers little wall. What it costs is registers -- each unit is allocated on its own, so the kernel's
  register count rises (56 -> 94, 72 -> 165, 64 -> 254) and occupancy goes with it: **1.8-2.7x of run time** on
  the `compile_scaling` items and **2.2-15.2x** on `benchmarks/vs_jax` at 4M items (`transport` forward is the
  worst), with `-maxrregcount` recovering only a quarter of it. The one exception is an item whose kernel ptxas
  has ALREADY capped at 255 registers -- a 16x16 dense solve per step -- where the split costs **6.6%**, which is
  this kernel's own regime and why the `__noinline__` arm above costs only 5-9% here. The application's own
  production kernels, compiled split (128-KB chunks, 32 processes) against monolithic on a 2080 Ti, fresh caches,
  results bitwise equal between the arms: the 6-column `jacfwd` (255 registers either way) goes from a 4,126 s to a
  264 s first call (144 units, 77 s compile) for **+5.2%** run time (0.634 -> 0.667 s); the 26-column one (255
  either way) from 4,369 s to 302 s for **+17.2%** (1.919 -> 2.250 s); the forward kernel, which monolithic sits
  at 213 registers, rises to 255 and pays **+53%** (0.169 -> 0.259 s). The split shrinks the frame (114,560 ->
  81,616 bytes, 7.43 -> 5.29 GiB of driver reservation) but that buys no speed. One job per arm, so 5% against
  17% is indicative, but it settles the rule: being at the register ceiling is necessary and not sufficient, and
  no measurable property of the monolithic kernel predicts which side of 10% the split lands on without
  compiling it the slow way first. So the split is not in the library; a 15x first-call saving against a run-time
  cost of 5-53% is a trade only the application can make, and the persistent disk cache already pays the
  monolithic compile once per kernel.
- **Device LTO does not give the inlining back.** `-dlto` + `nvrtcGetLTOIR` + `nvJitLink -lto` moves the whole
  cost into one single-threaded link (371.1 s of the 4.16 MB kernel's 372.5 s compile, 1.22x against monolithic
  where `-rdc` gives 11.9x) and produces a kernel with the outlined arm's registers and frame, not the
  monolithic arm's. NVRTC's own `--split-compile=N` does nothing here (it splits the backend per function, and
  the kernel is one function), and `--split-compile-extended` is not accepted through the NVRTC API even in 12.9.
  NVRTC **12.9 does compile the same source 1.19-1.21x faster than 12.1** for an unchanged run time, and
  `compile.nvrtc()` already prefers the `nvidia-cuda-nvrtc-cu12` wheel when it is importable: that one is free.
- **Branch stack frames do not add up.** The concern behind outlining branches was that NVVM keeps every branch's
  local arrays in the caller's frame at once. A switch whose branches each fill their own array with dynamic
  indexing (2080 Ti, local bytes per thread, inlined / outlined): 512 floats, 2 branches 4,176 / 4,496, 8 branches
  4,176 / 4,528; 2048 floats, 2 branches 32,800 / 16,464, 8 branches 16,416 / 16,496. The frame does not grow with
  the number of branches; the one doubled case (2048 floats, 2 branches) is a compiler choice for that shape,
  which outlining happens to undo, and outlining costs run time (1-8%) and, at 8 branches, occupancy.

- **How it scales, reproducibly.** `benchmarks/compile_scaling.py` is the same curve on a program this
  repository can run: an item whose loop body is M stages of 8-element array arithmetic, so each stage
  becomes straight-line code (2080 Ti, forward kernel, fresh cubin cache):

  | stages | source | NVRTC | registers | local bytes |
  |---|---|---|---|---|
  | 25 | 98 KB | 17 s | 56 | 32 |
  | 50 | 184 KB | 53 s | 56 | 32 |
  | 100 | 354 KB | 178 s | 56 | 32 |
  | 150 | 528 KB | 424 s | 53 | 32 |
  | 200 | 706 KB | 825 s | 51 | 32 |

  The same items as a 6-column `jacfwd` (one kernel, 6 dual lanes):

  | stages | source | NVRTC | forward NVRTC | ratio | local bytes |
  |---|---|---|---|---|---|
  | 25 | 152 KB | 52 s | 17 s | 3.1x | 2528 |
  | 50 | 252 KB | 164 s | 53 s | 3.1x | 4992 |
  | 100 | 489 KB | 667 s | 178 s | 3.7x | 9792 |

  Compile time grows as source size to the power 1.8 to 2.3 over that range, and the exponent rises with size:
  the same superlinearity the `__noinline__` experiment above exposed from the other side. The `jacfwd` rows show
  the other multiplier: 3.1 to 3.7 times the forward compile for 1.4 to 1.5 times the source, because `jax.vmap`
  of the item makes every lane-dependent array T times longer and the emitter unrolls it -- which is also what
  takes the per-thread local memory from 32 bytes to 9,792.

What remains is the volume of straight-line code itself. A large item's stack frame and compile time are
properties of the program; the levers that worked were in the program (fewer generic call sites, a static set of
geometry types, a bounded stack depth), and on the library side the lazy-inline cap.

## 10. Fixed rules ported from the `wavefront-state` branch (consolidation, 2026-11)

`wavefront-state` was a private experimental branch driven by the shower engine's exact Geant4 port (DTRAX), with
its levers behind environment variables. The ones that cleared the bar were re-implemented here as fixed rules: no
environment variable, no option. Each paragraph gives the measurement that justified it and the `wavefront-state`
commit it came from; the measurements are on that private engine.

- **A structural digest, not `str(jaxpr)`, as the cache key** (wavefront `290f137`). The cuda kernel cache and the
  eager-call cache keyed programs on the printed jaxpr, and JAX's pretty-printer builds a document tree per
  equation: on DTRAX's mode-2 item that key cost 95.0 s per build on an A100 and 133.7 s on a 2080 Ti, more than
  the trace and the emitter together. `_jaxpr.digest` hashes the same information into a sha256 without
  materialising anything (4.22 s and 5.26 s; the cubin was unchanged): primitive names, operands numbered by first
  appearance (alpha-equivalent jaxprs hash equal, as their printed forms are equal), output avals, every parameter
  by value, sub-jaxprs recursively (each hashed once), and the exact bytes of every literal and of every nested
  closed jaxpr's consts, which the printer rounds or omits and the emitter bakes into the kernel. The top-level
  consts are keyed as before (by value in the kernel key, by identity in the eager key). `tests/invariants.py`
  [key] checks that the real keys separate every pair, that a key made of the digest alone is caught by the same
  pairs as the printed text, and the digest's own separation (alpha-equivalence, primitive, operand order, shape,
  dtype, parameter, cond / while / scan structure, a literal one ulp apart).
- **Diagnostics print the item jaxpr when read, not when bound** (wavefront `283b9e0`). `record_program` built
  `str(jaxpr)` at every `tm.vmap` bind for `tm.diagnostics()['programs']`; on DTRAX that one call was 102 s of a
  260 s fully cached build (cProfile, A100). The record now holds the jaxpr object, which the program being bound
  keeps alive anyway, and `tm.diagnostics()` prints it.
- **The replay fingerprint only where reverse mode reads it** (wavefront `501b2bd`, made automatic in `1f098b3`).
  Every call computed the per-item fingerprint, a hash of every leaf of every top-level loop carry, although its
  only consumer is reverse mode's replay check. Forward with it off, on an A100: a 16.8 KB per-item carry (the
  faithful engine, 65,536 events) 4.354e7 -> 7.680e7 steps/s (+76%); DTRAX mode 3 +10.6%, with the cubin's local
  frame 44,144 -> 29,136 bytes; 1.19-1.31x on cheap items. Now `tm.vmap` builds the program without it, and
  `_primitive._vjp_call` asks for `ItemProgram.with_fp()`, which traces the fingerprinted variant once (derived
  programs, `fold_prog` and `dual_lanes_prog`, derive it the same way from the base's variant). A call that is being
  linearized (its operands are tracers of `jax.grad` / `jax.vjp`) is built with the fingerprint up front, so its
  primal launch is the one the replay check re-binds and XLA merges the two; elsewhere reverse mode pays one extra
  forward launch. A spawning program never builds it (its reverse mode, the tree adjoint, has its own check).
  `tests/reverse.py`'s replay negative control (`_adj_math`) still makes the gradient NaN.
- **A serial chain of 64-bit multiplies is cut every 8 links** (wavefront `41140a0`, `8337bb7`, merged in `58db0f2`).
  NVVM is exponential in the depth of a dependency chain of 64-bit integer multiplies inside one function: on the
  identical arithmetic (`cicc` alone) 0.2 s at depth 8, 0.7 s at 12, 127.3 s at 16, > 300 s at 20, while the same
  multiplies without the dependency, a chain of 32-bit multiplies or a shift/xor mixer of the same size are free;
  DTRAX's MixMax reseed is a 17-link chain. The emitter gives every value its chain depth (longest path counting one
  per 64-bit multiply; a cond / loop body / call adds its own input-to-output depth after its own cuts) and passes a
  scalar whose depth would exceed `codegen.MULCHAIN_K = 8` through `sj_brk`, an empty `asm` with the value tied in
  and out, then restarts its depth. Cut every 1, 4 or 8 links the compile is 0.2 s at depths 16, 24 and 48; every
  12 it is 2.2 s, every 16 125.6 s. The barrier is zero instructions (identical SASS, 928 == 928) and returns its
  operand, so results are bitwise unchanged; the definitions are emitted only into kernels that use them, so every
  other kernel's source is unchanged. Array-valued chain carriers are not cut (a chain through array slots did not
  cliff). Measured on this tree through tallymap's own compiler path (NVRTC 12.1, sm_75, job 43057989): the spawning
  item of wavefront's `test_mulchain.py` compiles in 60.7 s at chain depth 16 and 60.8 s at 24 without the cut, 1.7 s
  with it (nvcc: 52.6 / 53.5 s -> 0.9 / 1.0 s); depth 12 is 5.3 -> 1.7 s; results bitwise equal with and without the
  cut and to lockstep. A flat item with the same chains does not cliff (1.6-1.8 s either way), so the CI test
  `tests/mulchain.py` is structural only (cuts present, bitwise vs lockstep, compile time a sanity bound).
- **A lazy value used more than once is materialised when its expression is long** (prototype `TM_MAT_MULTI` in
  the GDML calibration study's codegen patch, 2026-10-18). The emitter keeps one-use elementwise results and cheap
  ops on large table gathers lazy (`op_table._ew_rule`, rules (a) and (b)), and a lazy value is re-inlined into
  every consumer whatever its use count: on the GDML geometry kernel one equation became 543 kB of source.
  Rule: a `Lazy` (or a `View` of one) used more than once in its jaxpr is computed into a local array when one
  element's C expression is longer than `codegen.MAT_MULTI_CHARS = 200` characters. Measured with that threshold:
  the GDML kernel's source 13.0 -> 7.5 MB. One-use values and short expressions stay lazy, so a large row gather
  read in two places with a short expression keeps loading only the elements it reads.
- **Pooled constant buffers are concatenated once** (wavefront `3732513`). `Emitter.fold_table` re-concatenated
  its whole per-dtype pool at every folded table, O(K x S) in the number and total size of the folded tables,
  although the pool is read once, when the kernel's buffers are laid out; `Emitter.flush_pools` now builds it
  there. Same bytes in the same order, so the source and the cubin are unchanged; worth about a second of a 29 s
  emitter on DTRAX's mode-2 item, which folds into few pools; a program that folds hundreds of tables pays the
  quadratic term.
- **Buffer assignment at loop and cond boundaries** (wavefront "v3", base commit `82d6ee2`, and "v5", `fbf2c5b`, as
  ONE rule). v3: a `while` loop's initial carry array that is dead after the loop (the in-place liveness rule,
  `Emitter.can_mutate`) and that no other operand of the loop reads IS the carry storage, so the init copy
  disappears; a `cond` output array is written into the storage of a dead operand of the same shape and dtype
  (destination passing), so a branch that passes the operand through or updates it in place copies nothing, and a
  destination whose branch result reads another written destination falls back to a fresh array. v5: the same
  when the value is lazy and reads that storage only at its own index (`Emitter.elementwise_self`, e.g. a
  whole-array `jnp.where(p, a, x)`), which is then evaluated into the storage element by element; and a loop body
  result of that kind is written straight back into its carry. Placement only: every element's expression is
  unchanged, so results are bitwise unchanged. Measured on DTRAX's tape gate, L40S (M4, wavefront `a3b5107`):
  with every buffer-assignment rule on 0.399 s and a 41,056-byte frame; v4, v5, v6 and scan-carry aliasing off
  1.23x (v5 alone 1.21x) and 49,840 bytes; v3 off as well 1.59x (0.640 s) and 58,496 bytes, with the A/A spread
  at most 1.8%. v4 (materialise the small lazy readers that block an in-place update) and scan-carry aliasing gave
  an identical cubin and v6 (an elementwise result overwrites a dead operand) 0.98x, so they were not ported. On a
  2080 Ti the same rules are 1.016x on DTRAX, and on the photon-transport engine (LUCiD) 0.97x (v3 never fires
  there; +7 KB of frame). Not ported either: a cond output that every branch returns as the same live operand's
  storage, which no switch turned off and the measurement therefore did not isolate. (`scatter` in place, the
  other unswitched rule, was ported (Unreleased) once DTRAX's frame isolated it: see "In-place updates".) The rule acts at a `while` / `cond` equation the emitter sees: an item whose only
  loop is its single top-level `while` is split into step functions (section 1b) whose parameters are never
  mutated, and there it does not fire, in `wavefront-state` either (checked on the generated source). `tests/edge.py`
  therefore nests the loop, and compares against the rule patched off.
