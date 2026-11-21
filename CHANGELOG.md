# Changelog

All notable changes. Versions follow [SemVer](https://semver.org/) 0.x: until 1.0 a minor release may change the
API, and only after one minor release in which the old form still works and warns. Patch releases never change
the API. The release policy for JAX versions is in `docs/maintenance.md`.

## Unreleased

### Added

- `tm.output_memory(output, backend, derivative, cells=, platform=, x64=)` -> `(per_cell, fixed)` (peak memory of an
  output at most `per_cell * cells + fixed`), `tm.call_memory(outputs, ...)` (a call's outputs, launch counted once),
  `tm.cell_bytes(output, backend, derivative, cells=, platform=, x64=)` and `tm.max_cells(output, backend,
  derivative, memory=)`: the device memory one cell of an output holds while a call runs (accumulator, the copies
  XLA keeps in its loops, the readout) for a `Sum` (exact, fixed point, plain) or `Records`, forward / jvp / vjp /
  grad, per backend and platform, and the largest cell count that fits a budget. Upper bounds of measured peak
  memory (tests/cellbytes.py, GPU and CPU). A `Sum` of more than 2^31 - 1 cells now raises at `tm.vmap` (cells are
  int32-indexed on every backend).
- `min_blocks_per_sm=N` (kernel backend on a CUDA GPU, opt-in): the kernel's `__launch_bounds__(block, N)`, so ptxas caps
  registers for N resident blocks per SM (more warps to hide latency; it may spill to local memory). The default
  source is unchanged; N is part of the kernel and cubin cache keys; outputs are bitwise those of the default (a
  float tally's atomic order aside); ignored on the CPU target and the xla backend. tests/invariants.py [lbounds].
- `tm.Records(fields, capacity=None)` outputs and `t.append(name, {field: value}, when=...)`: a variable number of
  records per item, in the deterministic (item, append order) layout (with `spawn`: the spawn tree's nodes in
  breadth-first order, children by spawn ordinal). Implemented as count-then-fill inside `tm.vmap` (a count pass,
  an offset prefix sum, a fill pass of ordinary Sum tallies), so every transformation, both backends, `spawn`,
  `n_active`, `max_steps` and sharded `jax.jit` work unchanged. Overflow past the capacity and a count / fill
  disagreement poison the call (float outputs NaN, integer record fields -1), counted in
  `stats['records_dropped']` and `stats['records_mismatch']`; `stats['appended']` is the true total. Not supported:
  `t.append` inside a `jax.vmap` within an item. README "Records"; tests/records.py, tests/sharding.py.
- `Records` run in ONE pass by default (given a capacity, without `spawn`): each append takes a raw slot from an
  atomic counter and writes its record and key (item, k) there, and the launch canonicalises its own output into
  the same (item, append order) layout, so the forward is one run of the item loop instead of two and a jvp is one
  launch. Derivatives that replay in another launch (grad, vjp, linearize, `jax.checkpoint`) use the primal's
  per-item counts as their record and write each record's slot directly; a replay that disagrees poisons that
  derivative. Outputs, stats and every derivative are bitwise those of count-then-fill, which remains the
  implementation with `spawn` or `capacity=None`. `stats['appended']` is exact on overflow (the count report for
  sizing a rerun).

### Fixed

- Kernel backend: value facts (integer ranges, NaN / finite, magnitude floors) were attributes of a Python object
  that in-place writes reused for a NEW value (an in-place `dynamic_update_slice` / scatter, a while / scan carry in
  its initial value's storage, a cond destination), so a NaN guard `where(x != x, c, x2)` on a value `x2` written
  into `x`'s storage since was taken for a guard and a later `maximum` / `minimum` dropped the NaN (default options;
  tests/edge.py "facts per SSA value").  Every write into an existing storage now returns a new value, so one object
  is one SSA value with its own facts; facts are no longer cleared by primitive name (a cond destination kept the
  operand's magnitude floor under `cuda_math='fast'`), and the magnitude-floor pattern compares values, not
  rendered text.  Aliasing and destination decisions ask one function, `Emitter.storages_read`.
- Kernel backend, `cuda_math='fast'`: a `lax.cond` output (a new value since the entry above) took no magnitude
  floor, so a division by a guarded divisor that passed through a cond lost its div.approx; it now takes the floor
  every branch proves (none when a branch may make it subnormal, e.g. a destination scaled by 2^-130 in one
  branch: IEEE division there). tests/edge.py "cuda_math='fast': a guarded divisor overwritten in place by a cond".
- `Records` with `spawn`: a call with more children in all than the `Queue` capacity, none dropped (the ring reused
  slots: at most `capacity` were pending at once), gave silently wrong records. The count pass reads the spawn tree
  back from the ring, which had lost nodes, so their children took node id 0 and added into root 0's slots (3 chains
  of 9 children, `Queue(4)`: `appended` 3 of 30, slot 0 the sum of every record, not poisoned). It now poisons, as
  the tree adjoint already did: one rule, `_limits.queue_lost`, for every path that reads the tree from the ring
  (`spawned > capacity` or a dropped child) and for the forward (a dropped child). tests/records.py "spawn: Records
  with more children in all than the Queue holds".
- Reverse mode with JAX 0.6.2 - 0.11 through a while loop that reads a table (any broadcast operand) inside ONE
  branch of a `lax.switch` / `lax.cond` (lumen SBND: a facet table read in one outcome branch): the checkpointed loop
  keeps its body consts as residuals, and JAX's cond partial eval returns each branch's residuals as cond outputs
  (a filler in the branches without them), so the table became a per-item cond output: the kernel backend raised
  "per-item array of shape (257, 65) is too large for a local array" (smaller tables silently became per-thread
  local copies: 1105 / 4160 floats for 17 x 65 / 64 x 65), the xla backend held items x table.  The same for a
  perturbed table, for a JAX op's own residual (the product in `dot(tab, f(w))` in one branch), through nested
  conds, the single-loop adjoint's sweep cond and a `lax.scan` (there stacked per iteration).  The adjoint's
  dead-code pass (`_reverse._dce`) now proves GUARDED aliases (a value equals the table whenever given switch
  indices take given branches) and reads the table itself wherever the guards hold; every other read is unchanged,
  so the values are the same.  JAX 0.12 forwards such residuals itself, and there the pass leaves the program
  unchanged (checked).  lumen SBND `jax.grad` on the kernel backend now runs.  tests/reverse.py "table ... kept as
  a residual in ONE branch of a lax.switch" (top, nested, cond2, scan, jaxop; a negative control below 0.12), "a
  USER cond returning the table or zeros".
- Reverse mode (grad, vjp, jacrev) through more than one gated `t.append` per item outside a
  loop, or more than one per loop step: the transpose of each `lax.cond` returned the output's whole
  cotangent (capacity-sized for `Records`) as a cond output and chained it into the previous one, a per-item array of
  the output's size: the xla backend allocated items x capacity (out of memory: 128 GiB for a 2^22-slot `Records`),
  the kernel backend raised "per-item array ... too large" from 2^16 slots. The adjoint now forwards every cond (and
  jit) output that each branch passes through unchanged to its operand, so the cotangent is read in place: Records
  grad / vjp memory equals the forward's (2^22 slots, 2080 Ti: kernel 84 MB grad vs 67 MB forward compiled; xla
  152 vs 151 MB). tests/records.py "large capacity (2^22)", "memory: Records grad / vjp".
- xla backend: an exact (or, without x64, fixed-point) `Sum` above 2^31 / 19 cells raised `OverflowError` (its flat
  word index W * S passed int32), and the f32 and x64 fixed-point paths wrapped above 2^30 cells; every scatter now
  indexes (word or row, cell) with two int32 indices. Checked to 2^31 - 1 cells on both backends (the kernel backend
  already indexed with int64).
- xla backend memory of an exact `Sum` (x64 off, RTX 2080 Ti, per cell at 2^24 cells): forward 151 -> 92 B, jvp 290
  -> 180, vjp 155 -> 96 (x64 on: 196 -> 101, 298 -> 179). The readout runs `READ_CHUNK` (2^20) cells at a time, and a
  large tally's carry step is a loop of row updates in place: the `lax.cond` it replaces copied the accumulator
  (under jvp two accumulators per tally were held). Results are bitwise unchanged.
- xla backend without `jax_enable_x64`: an exact `Sum` (`Sum(fixed_point=True)` of float32) and its tangents no
  longer raise: they keep the same integer in 19 int32 words of 16-bit digits (80 B per bin with a per-bin count),
  bitwise the x64 result and the kernel backend's (overflow and NaN poisoning identical). Cost against x64 (2080
  Ti): forward 1.9-2.7x, jvp 2.2-2.9x, grad 1.1-1.25x, compiled temporaries 1.25-1.4x.
  `fixed_point=True` of float16, bfloat16 and (u)int8/16/32 with a power-of-two `scale` works there too (8 int32
  words; the float64 readout reproduced exactly); another scale or dtype still raises without x64. tests/edge.py "exact Sum without x64", "fixed_point=True float16 / bfloat16 /
  integer outputs without x64".
- With `jax_enable_x64`, `spawn` with `Records` (count-then-fill) raised "while_loop carry types differ": the tree
  order's loop summed an int32 mask with `jnp.sum`, which widens to int64 under x64. `stats['n_truncated']` was
  int64 under x64 for the same reason; it is int32 always. tests/records.py "spawn + Records ... x64 toggled".
- Kernel backend: `lax.max` / `lax.min` with a signed-zero tie returned the first operand; XLA (every platform, f32
  and f64) returns +0 for max(-0, +0) and -0 for min(+0, -0), as the kernel does now (seen as tests/edge.py "float
  facts min / max" failing under x64). tests/edge.py "signed-zero ties and NaN of lax.max / lax.min".
- xla backend with `jax_enable_x64` on a CPU: an exact `Sum` whose total is subnormal read 0 (XLA:CPU flushes the
  float64 -> float32 conversion); its bits are now taken from the integer, as without x64.
- tests/invariants.py [tdep] CG under x64: the tolerance (1e-10) was below plain JAX's own jacfwd vs stacked-jvp
  difference for this CG (1.1e-7 in f64); now 1e-6.
- xla backend without `jax_enable_x64`: reverse-mode cotangents of f32 broadcast operands (a parameter's or a
  table's gradient) were added in f32 by a scatter-add, which is an atomic on a GPU, so `grad`, `jacrev` and
  `hessian` changed in their last bits from run to run (seen as the Records cell "one pass == count-then-fill
  BITWISE" failing at random on GPU). They are now the exact sum there too, kept in int32 words and rounded like
  the int64 format: bitwise the same in any order, run to run, with or without x64 and on both backends.
  `docs/float_semantics.md` item 4; tests/edge.py "exact f32 reverse cotangents", tests/records.py "derivatives BITWISE run to run".

- Subnormal f32 values were flushed to zero by `Sum` (fixed_point=False) and therefore by `Records` float fields,
  contradicting "a float field stores each value exactly": lumen's record readout lost every deposit whose q
  relative to the row weight was subnormal (4 x 1.4e-44 emitted gave 0; XLA's multiply keeps them). The cause is
  not a compile flag (`--ftz=false` was already set, in every `cuda_math` mode): a GPU's f32 atomic add into global
  memory (`red` / `atom.add.f32`) flushes subnormal inputs and results to zero by definition. The kernel backend
  now emits |v| >= 2^-102 with that atomic (provably bitwise IEEE there), adds a smaller nonzero v with a
  compare-and-swap loop of the IEEE add, and skips +-0 (a no-op); the block flush of privatised tallies too. The
  xla backend (XLA's scatter-add is the same atomic) adds values below 2^-102, times 2^100, into one more row of
  the tally. And the NaN poison of a call that can hit a resource limit (`Records`, `max_steps`, `spawn`, the
  adjoint's truncation) multiplied every float output by 1.0, which XLA:CPU flushes: it is now a select (tangent:
  times NaN only when poisoned). `Sum(fixed_point=True)` of f32 was always exact for subnormals. The xla backend on
  a CPU still follows XLA:CPU, which flushes subnormals in its own arithmetic. `docs/float_semantics.md` item 6;
  tests/edge.py "subnormal f32 emits".

- `Sum(fixed_point=True)` of float32 is now EXACT, forward, in every tangent and in reverse mode: each bin is the
  correctly rounded f32 of the exact sum of every finite value emitted into it, with no scale, floor or cap
  (`scale` is no longer used for float32; other dtypes keep int64 fixed point). The 2^32 fixed point it replaces
  rounded every emit to 2^-32 of the output's scale, tangents included, so a tangent below that quantum was
  silently 0: lumen's SBND d/d(TPB WLSABSLENGTH) was 0 against an exact 3.368e-8, and a d/dlog of it lost 6% of its
  sum. A bin is the integer sum of the emits in units of 2^-149 (every finite f32 is one, below 2^277), kept in
  five signed int64 words of weight 2^(63 j) (a word that overflows carries +-2 into the next): integer adds, so
  the result is bitwise the same in any order, run to run, in every AD mode and on both backends (the xla backend
  keeps the same integer in nine int64 words and needs `jax_enable_x64`; without it it raises instead of summing
  in float). Device memory: 44 B per bin with the output (2^32 fixed point: 20 B). A non-finite emit makes its bin
  NaN and is counted in `stats['fixed_point_overflow']`; so is a bin whose running sum passes the accumulator's
  top word (about 2^166): a bin is exact or NaN, never silently wrong.
  Measured on a 2080 Ti: lumen SK forward 1.01x and jvp 1.03x the time of 2^32 fixed point, SK jvp at 16 charge
  replicas 6.4 GB peak. `docs/float_semantics.md` item 4; tests/edge.py "exact".
- Reverse mode: the cotangent of an f32 broadcast operand (a parameter's or a table's `jax.grad`) was added in f32
  atomics, one contribution per item into a few bins, so every add lost its part below the bin's ulp, always in
  the same direction and in a different order each run (lumen's WCTE gradient at 524288 photons was 1.1e-3 low
  against jvp and not repeatable). These sums are now the exact sum above: bitwise repeatable, the same on both
  backends. The xla backend needs `jax_enable_x64` for this and otherwise adds them in f32 as before (the one
  documented exception to its x64 rule: an exact Sum raises there without x64).
  tests/reverse.py "broadcast cotangents".
- Kernel backend: a `lax.cond` whose operand array filled two operand slots could write two outputs of that
  array's shape into its one storage, the second overwriting the first (the in-place destination rule tracked
  slots, not storages). Seen as `jax.vmap` of a jvp over exactly one tangent being wrong; any cond with that
  pattern was exposed, on CPU and GPU. tests/transforms.py.
- Kernel cache on a directory shared by several nodes: the host kernel, host shim and cuda shim were compiled to
  a temporary named only by the pid, so two nodes with the same pid could write the same file. The temporary now
  names the host, the pid and a random token (still renamed into place atomically). tests/edge.py.
- CPU target: a host kernel whose `-fstack-usage` file was missing ran without its stack-size check. A cached
  kernel without that file is now rebuilt, and a build that does not write it is an error. tests/edge.py.
- CPU target: the host kernel build took any `-fstack-usage` file for its hash from the cache directory, so it
  could take another node's half-written one for the same kernel. It now builds in a private directory and
  takes only its own file. tests/edge.py.
- Cuda target: a cubin from which no frame size could be read passed the per-thread local-memory check. NVRTC
  writes a record for every function (a 0-byte frame included), so this is now an error. tests/edge.py.
- Kernel backend: a loop constant that is a lazy view of a per-item array (e.g. `jax.random.key_data` of a key
  array, read at its own index) did not protect that array from in-place reuse: a loop nested in the body could
  start from it and take its storage as its carry, and the next iteration read the overwritten array (59% / 76%
  off in a scan / a nested while). Loop constants now freeze every storage they read, not only their own.
  tests/edge.py, "alias: loop constant that is a lazy view ...".

### Changed

- Kernel backend: one destination-aliasing rule (`op_table._may_overwrite`) replaces the four separate analyses
  (dynamic_update_slice / scatter in place, while initial-carry alias, lazy initial carry, cond destinations): a
  storage may be overwritten only if no read of its old contents remains on any path, the equation's own reads
  counted per value (one array in two operand slots is two reads), not per slot. Every kernel source of the test
  suite is byte-identical to before except the case above. tests/edge.py, six "alias:" hazard cases (in place ==
  copying == xla; jvp, vmap of one / two tangents, grad == xla).

- Kernel backend: no whole-array copies around in-place updates. A live read of an array element (`x = a[j]`
  used after `a = a.at[k].set(...)`) is captured before the update, which is then done in place, instead of the
  array being copied twice per update (a 1024-float loop carry: 37x on an L40S, 35x on a 2080 Ti). A loop carry
  larger than a thread's 255 registers that is rebuilt from other elements of itself (a pivot step of an
  elimination) gets two buffers swapped per iteration instead of a fresh array copied back: LUCiD's forward 1.23x
  and its 6-column Jacobian 1.20x on an L40S (1.02x on a 2080 Ti), bitwise; lumen unchanged (gate: 650/650
  arrays bitwise). `docs/design_notes.md` section 3, "In-place updates".
- Kernel backend, spawn: the queue's five control words sit on separate 128-byte lines: every thread without work
  polls the queue, and those polls queued every atomic on the same line behind them. The shower forward 1.3-1.8x
  faster and a tree grown from a few roots 1.6-1.8x on both GPUs; a DTRAX-like segment 1.0-1.3x; spawn counts
  exact. `docs/design_notes.md` 1c.
- Reverse mode through spawn (the tree adjoint): the forward launch records the tree (each item's queue slot,
  parent, ordinal and generation), so a gradient takes 1 + G launches instead of 1 + 3G: grad and value_and_grad
  1.2-1.7x faster on an L40S, a 2080 Ti and the CPU target, 1.6-2.1x on the xla backend, memory -20%. Generation
  launches are sized by `n_active` instead of a gate, and under `jax.vmap` every member shares one capacity bucket
  (vmap of a tree gradient 3.3-17.5x faster, now 0.6-1.5x the cost of the members' gradients run one by one).
  The replay check compares child counts (`replay_mismatch`); the value fingerprint and its `fwd_mismatch` entry in
  `tm.diagnostics()['reverse']['tree']` are gone. Works under `jax.checkpoint` (tests/reverse_spawn.py 2b).
  `docs/design_notes.md` section 4.
- Simplifications, no change in results (the structure review's verified list). Dead code removed: unused kernel
  helpers (`sj_atomic_max_f32`, `sj_maximum` / `sj_minimum`, the host atomicMax / Min family and max / min
  templates), the cuda shim's `grid` attribute, `tallymap_host_threads`, never-read attributes, `emit_p_bind`,
  `_tally._isfloat`, call-like primitive names that exist on no tested JAX. One copy of each layout:
  `ItemProgram.out_tallies` / `float_outs` / `float_ys`, named stats slots (`_api.STATS`), the kernel's result
  buffers written once (`Layout.res`). The two loop adjoints share their common code (the single-loop adjoint still
  builds tally cotangents inside its micro-step). `_rewrite` no longer imports `_reverse` (the re-tracer reaches
  reverse mode through `RevOpts`), and in-function imports are at module level except the ten that would close an
  import cycle. The reverse-mode negative controls live in the tests, which patch one function each; the
  instrumentation is one store, `_diagnostics.REVERSE`. On the CPU target the NVVM multiply-chain cut is the
  identity. Generated kernel sources change only by the deleted prelude lines (and, on the CPU, by the cut).

### Added

- `tm.history()`: the last programs' `closed_over_floats` and the kernel records (tag, target, schedule, hash,
  source) without `tm.diagnostics()`'s cost (no jaxpr rendered, nothing probed). tests/edge.py.

## 0.7.1 - 2026-11-12

### Fixed

- Wrong results on the kernel backend (`backend='auto'`, GPU and CPU target) in 0.6.0rc1, 0.6.0 and 0.7.0: when one
  array was both a while loop's initial carry and another operand of the same loop (a loop constant or a second
  carry) and dead after the loop, its storage was reused as the carry, so the loop body read the updated value
  where it should have read the original. The single-loop reverse adjoint (steps of >= 1000 equations) passes its
  initial state this way, so its gradient was wrong for every item running more steps than `rev_checkpoints` (auto
  or explicit) when the loop state holds an array; a forward program with the pattern was wrong too. The xla
  backend was correct. The conflict check now excludes only the initial carry's own operand slot. Regression tests
  in `tests/reverse.py` (array state, K on both sides of C) and `tests/edge.py`.
- `Sum(fixed_point=True)`: a bin whose running total passed `2^63 / scale` (2^31 at the default scale 2^32) wrapped
  silently, with no `fixed_point_overflow`; a large tangent bin (a derivative of a large sum) was the usual victim.
  Each bin now keeps a signed carry word, the net carry out of its int64 word, counted from each atomic add's old
  value (in the emits and in the flush of the shared-memory accumulators; on the CPU target the same code; on the
  xla backend as separate sums of each value's low and high 32 bits). The readout is the exact total rounded once
  to float64, then cast: bitwise today's value wherever no bin carries, and order-independent where one does. The
  per-emit range check and its count are unchanged; a bin whose exact total does not fit the output dtype is now
  counted in `fixed_point_overflow` too, and is NaN in a float output (it was `inf` or a wrapped integer).

### Added

- JAX 0.12 is tested (`TESTED_JAX` adds 0.12.0.1; the full suite passes on CPU and L40S). JAX 0.12 wraps some
  `jax.numpy` functions (`searchsorted`) in a new `call_hi_primitive`; the kernel backend inlines its expansion.

## 0.7.0 - 2026-11-11

### The kernel backend runs on the CPU

- One compiled backend for both CUDA GPUs and the CPU: the emitter that produced CUDA C now also targets the host
  (`tallymap/backends/cuda` is now `tallymap/backends/kernel`). JAX's device placement picks the target, so
  `jax.default_device(jax.devices("cpu")[0])` on a GPU machine runs the CPU kernel; tallymap no longer consults
  `jax.default_backend()` anywhere.
- On the CPU: the system C++ compiler (`-O3 -march=native -ffp-contract=off`, cached per compiler, CPU and source),
  one precompiled `jax.ffi` CPU handler running an OpenMP team, one thread per GPU thread (`OMP_NUM_THREADS`, else
  the process's CPU set). IEEE arithmetic (subnormals kept) with libm; `cuda_math` has no effect. Per-item arrays live
  on the worker threads' stacks (`OMP_STACKSIZE`, set to 16M when unset); a kernel whose stack bound is over half of
  it is refused before it runs, as an over-limit CUDA kernel is. Without a C++ compiler with OpenMP, `backend='auto'`
  on the CPU raises `UnsupportedError` naming the fix and `backend='xla'`.
- Every feature runs on both targets; the whole test suite, including the op-parity suite (`tests/ops.py`), now runs
  the CPU kernel against the xla reference, so CI exercises the emitter on every push.
- CPU, one 32-core EPYC 7713 socket against the best plain-JAX program: forward 2.8-5.1x on the divergent problems
  and at parity (1.0-1.1x) on long uniform loops, gradients 8.6-54x; against plain `jax.jit` on one device, forward
  27-129x (README "Backends", `docs/performance.md`).

### Several devices: sharded `jax.jit`

- Shard the item axis with ordinary JAX sharding (`jax.device_put(xs, NamedSharding(mesh, P("d")))` and `jax.jit`)
  and each device runs the call on its own items: tallies and stats are summed over the devices (`psum`), per-item
  results stay split like the items. Before, XLA all-gathered every operand and ran the whole call on every
  device. The rule is JAX's `custom_partitioning`, so it covers jvp, grad, jacfwd, spawn and the tree adjoint
  with no tallymap API; GSPMD and Shardy both work. Measured at 2^20 items, 4 devices: L40S forward 0.22 ms
  (0.93 ms before; one device 0.25 ms: this toy does not fill one GPU), CPU 4 forced host devices 25 ms against
  39 ms on one, with half the CPU time.
- On the CPU the devices running a call share the process's CPUs: each gets `OMP_NUM_THREADS` (else the CPU
  set) divided by their number, instead of each oversubscribing the node.
- A spawn `Queue(capacity)` is per device under sharding (the total capacity is capacity x devices).
- New suite `tests/sharding.py` (4 forced CPU devices when there is one GPU or none), with a negative control.

### Removed

- `backend='cuda'` raises `ValueError` saying to use `backend='auto'`, which runs the same kernel on a GPU.

### Changed

- The cache directory variable is `TALLYMAP_CACHE` (it holds CPU kernels too); `TALLYMAP_CUDA_CACHE` still works.
- Nothing a program writes is in git any more: the benchmark plots, the measured rows, the demos' plots and results,
  and the `tests/ops.py` and `benchmarks/perf.py` baselines (now recorded per machine in `$TALLYMAP_CACHE`). The
  docs keep the numbers as tables; `make_tables.py` no longer draws figures.
- `docs/performance.md` is the summary (method, headline, CPU); every per-problem table moved to
  `docs/performance_detail.md`. `docs/design.md` is a short map of why each part exists; the full record of
  measurements, including the removed and refuted alternatives, is `docs/design_notes.md`.
- `tm.diagnostics()['kernels'][i]` has `target` (`'cuda'` | `'cpu'`); CPU kernels report `resources` as their
  stack bound. `tm.diagnostics()['host']`: the CPU compiler, its `-march=native` identity, cache, thread count.
- The one-time xla-backend warning names the platform it is for (`'rocm'`, ...).
- `examples/demos/`: each demo is one short `run.py` (`python run.py [--n N]`: the arms, the agreement checks, a
  timing) and a short README with the measured numbers unchanged. The sweep driver (`--full`/`--one`/`--plots`,
  results JSON, plots) is gone; it is at tag `v0.6.0rc1` for reproducing the numbers. A missing incumbent library
  is reported and skipped, not a failure.

## 0.6.0 - 2026-11-11

0.6.0rc1 plus the fixes below. Accepted by lumen (forward bitwise, reverse 0.82-0.93x the rc1 time, gradients within
the rerun spread).

### Fixed

- `jax.jacrev` of a call with `max_steps` raised a broadcast error on the xla backend: truncated children are now
  counted from a laned program's per-lane flags.
- A typed PRNG key as a mapped input of a spawning call raised (`np.dtype` on a key dtype).
- The xla backend interprets each `cond` branch once (traced with `make_jaxpr`, replayed by `lax.cond` /
  `lax.switch`): 79 -> 0.99 binds per equation at nesting depth 4, first call 84 -> 3.6 s.
- The xla backend's f32 `Sum` tallies accumulate into 64 partial rows summed at the end: error against f64
  0.177 -> 0.0031 (plain JAX 0.0021), run time unchanged.

### Changed

- CI pins the per-push newest-JAX cells to the newest tested version, not an unpinned latest.

## 0.6.0rc1 - release candidate

### Renamed (a clean break: the old spellings raise)

0.5.0 was tagged but never published, so these renames break no released code and get no deprecation release:
every old spelling below raises like any unknown option, `Sum` field or backend, or is simply absent (a stats or
diagnostics key). The entries further down use the new names.

| old | new |
|---|---|
| `tm.Sum(..., exact=True)` | `tm.Sum(..., fixed_point=True)` ("exact" promised no error; it rounds every value to a multiple of `1/scale`) |
| `Result.stats['exact_overflow']` | `Result.stats['fixed_point_overflow']` |
| `Result.stats['dropped']` (spawn Queue full) | `Result.stats['children_dropped']` |
| `backend='lockstep'` | `backend='xla'` |
| option `lockstep_block` | option `xla_batch_size` |
| module `tallymap.backends.lockstep` | `tallymap.backends.xla` |
| `tm.diagnostics()['programs'][i]['baked_floats']` | `['closed_over_floats']` |
| `tm.diagnostics()['reverse']['tree']['particles']` | `['children']` |
| `tm.diagnostics()['reverse']['tree']['overflow']` | `['generation_overflow']` |
| `tm.diagnostics()['kernels'][i]['mode']` | `['schedule']` |
| private option `_schedule` | public option `schedule` (see below) |

Error messages name the call the user wrote (`t.add(...)`, `t.spawn`) instead of "emit". The docs use "lane" only
for a tangent (or cotangent) direction of a dual-lanes kernel, "reference" only for `tm.reference`, one name pair
for the schedules ("item schedule" / "step schedule"), and the spelling "materialise".

### Removed (private options)

- Every private option: `_cuda_inplace`, `_lanes`, `_fold`, `_rev_sparse_min` and `_rev_nodense` (and, above,
  `_schedule`, `_rev_gen_capacity`, `_tree_neg`). They were A/B and negative-control switches for the tests and
  benchmarks. They are now TEST-ONLY module constants that the library never sets: `codegen.INPLACE`,
  `_primitive.LANES`, `_primitive.FOLD`, `_reverse.SPARSE_MIN`, `_reverse.NODENSE` (the tests set them with
  `tests/ab_rule.patched`, which also clears the eager-call cache). `tm.vmap` takes the six public options only;
  any other name raises.

### Changed (pre-rc1: the round-3 reviews)

- `n_active` is the launch's item count, not program structure. It used to wrap the whole item in
  `lax.cond(item_id < n_active, item, skip)` with an extra per-item id operand; that cond hid the item's loop, so
  `n_active` and `schedule='step'` excluded each other (and the single-loop adjoint never applied). Now the count is
  a broadcast operand the item does not read (`ItemProgram.nact`, kept by every derived program): the cuda kernel
  stops running items at `min(n_items, n_active)` and writes zero per-item results for the rest, the xla backend
  masks `ids < n_active`. Semantics unchanged: items `i >= n_active` emit nothing, are never truncated, and their
  per-item results (and tangents / cotangents) are zero. `n_active` with `schedule='step'` now runs.
- cuda: the kernel parameter struct has exactly the program's buffer count (`struct SjParams { void* p[r]; }`),
  replacing the two-tier 504 / 2048 rule. The limits are the driver's own: up to 511 buffers within the classic
  4 KB parameter block, up to 4,094 (32,764 bytes) with CUDA 12.1+ large kernel parameters; more raises naming
  the limit. Every kernel's source (and cubin) changes once.
- Both backends implement JAX's while contract for a batched (per-lane) predicate: the loop runs while ANY entry of
  the predicate holds, each entry keeps its carries once its own predicate is false (per-entry select), and its
  emits inside the body are dropped. `jax.vmap` produces such a loop for a condition that reads a batched value;
  under `jacfwd` with dual lanes that is any loop whose condition reads the TANGENT (a hand-written CG inside a
  `custom_jvp` whose tangent rule iterates on the tangent residual). It used to fail to build: a `bool[T]`
  predicate where the cuda rule wrote a scalar test, a shape error on the xla backend. A loop with a batched
  predicate is never split by the step schedule (it runs the item schedule). Test: invariants `[tdep]` hand CG,
  `jacfwd` with lanes against `jax.jacfwd` of plain JAX and against stacked jvp columns, both backends.
- REMOVED: the batch fold (`folded_batch` node, `fold_prog`, `_can_fold`, `FOLD_MAX_BC`, the test-only
  `_primitive.FOLD`, the device-occupancy rule). `jax.vmap` of a call over a non-linear operand is `lax.map`, one
  launch per member, as it already was whenever the fold did not apply; vmap over tangent / cotangent operands
  stays dual lanes. No application path reached the fold (round2/simplicity), and it paid only while one member's
  launch could not fill the device. `tests/transforms.py` loses its `TEST_NEG=fold` control.
- REMOVED: shard_map support (casting operands to their varying axes at every bind, the varying-axes abstract
  evaluation and lowering adapter: `_varying`, `_with_varying`, `_plain`, `_cast_varying`). shard_map's typing is
  experimental and moved on every JAX minor from 0.7 to 0.11. `jax.pmap` over devices still works.
- JAX floor 0.6.2 (was 0.5.3; round3/jaxfloor): `jax>=0.6.2`, `jax[cuda12]>=0.6.2` (which brings NVRTC itself,
  so the separate `nvidia-cuda-nvrtc-cu12` requirement goes), `TESTED_JAX` from 0.6.2. Removed with it: the
  host-callback address scrubber of `tm.diagnostics(...)['stablehlo_sha256']` (0.6.2+ lowers no address),
  `platform_index`'s `has_default`, the `custom_vjp_call_jaxpr` names, and three guards dead on every tested
  version (the plain batching-registry fallback, the key-impl name lookup, the partial-eval feature test).
  `tests/ops_baseline.json` still holds 0.5.3 entries only, so `tests/ops.py` checks bounds alone on 0.6.2
  until it is re-recorded.

### Changed (tree adjoint)

- Its private options `_rev_gen_capacity` and `_tree_neg` are now the test-only module constants
  `_reverse_spawn.GEN_CAPACITY` and `_reverse_spawn.TREE_NEG` (like `codegen.FILL`); passing either option raises.
- Its replay fingerprint (`fwd_mismatch`, `replay_mismatch`) is kept: unlike the per-item replay check, no
  deterministic gate covers it yet (`docs/design.md` section 4).
- The docs give the measured rule for reverse versus forward mode through spawn (L40S, job 43935229): with dual
  lanes `jacfwd` wins up to about 64 parameters; when lanes are refused the tree adjoint wins from about 4
  parameters (4-24x at 16-64). The tree adjoint costs 2.4-4.1 times the forward pass on the four programs
  measured.

### Changed (schedule)

- The cuda schedule is a public option, `schedule='item' | 'step'`, and the default is `'item'` (a thread runs
  one whole item at a time). It replaces the private `_schedule` and the rule "step unless the program spawns".
  LUCiD's forward is 3.0x slower under the step schedule (L40S, job 43935228) while lumen's value, jvp and
  jacfwd are 1.01-1.39x faster under it (job 43935227), so the step schedule is an explicit request.
  `schedule='step'` on an item that cannot be split raises `tm.UnsupportedError` saying why (the number of
  top-level while loops, or the loop constant that cannot cross the split); it used to fall back to the item
  schedule silently. Kernels derived from the item (jvp, lanes, an adjoint) use the step schedule wherever they
  split. Spawn works under both schedules. `_schedule` now raises like any unknown option.
- `tm.diagnostics()['kernels'][i]['schedule']` (was `'mode'`) is the schedule the kernel ran.
- A kernel frame over CUDA's per-thread limit, and a launch that fails for lack of local memory, now name the
  schedule that ran; when the item schedule ran on an item the step schedule can split, the error says that
  `schedule='step'` gives a smaller frame. Nothing switches schedule on its own.

### Changed (revfit: reverse mode for lumen's acceptance)

- Reverse mode: the single-loop adjoint (T3, from lumen/perf 6a70383 / b38db3f / 22e615d / e58e216, re-implemented
  on dev) for an item whose one top-level loop has a step of >= 1000 equations (`_reverse.T3_MIN_EQNS`): ONE
  while loop of micro-steps, so the step schedule can split it and the step is instantiated
  once. lumen/perf measured grad / value 2.66 / 2.78 / 3.31x against 3.99 / 5.78 / 8.46x nested (SK / SBND /
  ANNIE, L40S). Not taken from lumen/perf: the custom-rule guard (the single loop advances states with the
  linearized step's primal, which is what `jax.grad` itself evaluates), the `_rev_single` option and the
  `TALLYMAP_REV_SINGLE` env (tests set `_reverse.SINGLE`; `run_all.py --t3`), the fingerprint in the probe.
- Reverse mode: the runtime replay check is REMOVED (the fingerprint, `ItemProgram.fp / with_fp`, the linearize
  guess, `tm.diagnostics()['reverse']['replay']`). Replay is bitwise by construction; `tests/reverse.py` "REPLAY
  GATE" gates it (reverse == forward-mode derivative per item on an ulp-sensitive trajectory; the `_adj_math`
  negative control must fail it). `rev_stats` now concerns the tree adjoint only.
- Truncation (`max_steps`) poisons the derived kernels themselves (lumen T6b, re-implemented): the primal outputs'
  poison has an identity tangent, a tangent kernel emits NaN into its tangent sums for a truncated item, and the
  adjoint poisons its cotangents from its own replay's flags. `jax.grad` of a loss linear in the outputs is ONE
  launch (it was a forward launch + the adjoint).
- Exact overflow poisons its own bins in every kernel (see Fixed).

### Fixed

- `Sum(fixed_point=True)`: a TANGENT-only overflow was silently dropped on every unfused jvp path (FUSE_JVP off,
  linearize, jacfwd with lanes refused), and the fused jvp's single counter made the correct PRIMAL fixed-point outputs
  NaN. Each kernel now flags the bins an unrepresentable emit was dropped from and writes them as NaN (cuda and
  xla); the call-level fixed-point poison is gone. A primal overflow no longer poisons other fixed-point outputs or the
  (correct) tangent.

### Added

- `tm.diagnostics(...)["programs"][i]["closed_over_floats"]`: shape, dtype and size of every FLOAT constant the item
  function closed over -- the values a derivative cannot reach (a captured parameter's tangent is silently
  zero). Built when diagnostics are read, not at bind. From lumen/next b8fabe4 (= the private engine line's
  7abbc71), which built it at every bind.
- `benchmarks/compile_scaling.py`: NVRTC compile time, generated source size, registers and per-thread
  local memory against the SIZE of one item, with and without `jacfwd` lanes. The cuda emitter inlines the
  whole item into one kernel, so for a large item the compile time is the size of that one function, and
  `docs/design.md` section 9 had only the private engine's numbers for it. On a 2080 Ti compile time grows
  as source size to the power 1.8 to 2.3 between 99 KB and 706 KB of generated source
  (18 s to 725 s).

- `tm.diagnostics()["kernels"]` reports each kernel's compile time and, once it has run, its measured
  resources: registers and local memory per thread, resident blocks per SM, and the local memory the CUDA driver
  reserves outside JAX's pool (what an out-of-memory error at launch is about).
- A program that fails to lower because a `custom_jvp` closes over a traced value (a JAX staging limit) gets an
  error that names the cause and the fix (pass the value as an argument).

### Removed

- The nvcc fallback compiler (`compile.find_nvcc`, `_nvcc_compile`, the `NVCC` environment variable and
  `compile.compiler()`): the cuda backend compiles with NVRTC only, and raises an error naming the
  `nvidia-cuda-nvrtc-cu12` wheel when no libnvrtc is found. The `cuda` extra now requires that wheel, since
  `jax[cuda12]` brings it directly only from JAX 0.6.2. NVRTC was the compiler in use in both tested environments
  (the JAX 0.5.3 container: wheel 12.1.105; JAX 0.11.2: 12.9.86), whose nvcc wheels hold only `ptxas`.
  `tm.diagnostics()['cuda']['compiler']` is the NVRTC build.
- Constant folding in the cuda emitter (`codegen.FOLD_MIN`, `Emitter._try_fold`, `fold_table`, `flush_pools`):
  array equations whose inputs were all compile-time constants were evaluated on the host and hoisted into
  read-only tables, pooled into one kernel buffer per dtype. With it off, lumen's value and jvp outputs were
  bitwise identical on SK, SBND and ANNIE (L40S, job 43946370), as were LUCiD's forward and Jacobian and every
  DTRAX table; the largest cost was 1.14x (lumen SK value, 2080 Ti) and the pools alone were worth at most 1.02x.
  Such equations are now emitted like any other. Scalar constants still fold into literals, and the lazy gather
  rule (`op_table.LAZY_GATHER`) stays, since without it LUCiD and DTRAX fail to launch.
- Equinox-family support inside an item: the expansion of Equinox's own primitives (`unvmap_*`,
  `select_if_vmap`, `maybe_set`, `nonbatchable`, `nontraceable`, `nondifferentiable_backward`) at the item trace,
  the late expansion of Lineax's `linear_solve` and Equinox's `vprim`, the rule that ran a program holding a
  batched `vprim` one tangent lane at a time, and the Equinox-specific advice in error messages.
  `diffrax.diffeqsolve`, `optimistix`, `lineax` and `quadax` calls inside a `tm.vmap` item are no longer
  supported: on cuda their primitives raise `tm.UnsupportedError` like any other primitive without a rule.
  `docs/equinox.md`, `examples/diffrax_ensemble.py`, `tests/equinox_compat.py` and the `equinox` extra go with
  it. The rewrite engine `_sugar.py` now lives in `backends/cuda/canon.py`, its only remaining user. diffrax and
  quadax stay as the plain-JAX baselines of `benchmarks/vs_jax` and `examples/demos/`.
- Linear algebra inside an item on cuda: the emission rules for `lu`, `lu_pivots_to_permutation`,
  `triangular_solve`, `cholesky`, `geqrf` and `ormqr` and their straight-line algorithms
  (`backends/cuda/dense.py`, with its size limits `MAXD` / `MAXB`). `jnp.linalg.solve`, `inv`, `det`,
  `cholesky`, `qr` and every other `lax.linalg` primitive inside an item now raise `tm.UnsupportedError` on cuda,
  like any other primitive the emitter has no rule for. The xla backend is unchanged: XLA runs these
  primitives there. Write a small solve out by hand (Cramer's rule, an unrolled elimination), as
  `examples/demos/stiff_ode/kv3.py` does, or run the item with `backend='xla'`.
- The public option `rev_step_segments` (recompute each loop step in S checkpointed segments inside its VJP).
  Passing it now raises, like any unknown option. Four independent measurements on a 2080 Ti found no benefit:
  the six `benchmarks/vs_jax` gradient cells (0.98-1.05 at S = 4 and 8), the toy walk at 1M items
  (6.32 / 6.36 / 6.45 / 6.43 ms at S = 0 / 4 / 8 / 16), a toy built to spill (168 registers, 1,328 bytes of
  local memory per thread, grad/forward 8.1x: 32.4 / 31.2 / 32.1 ms at S = 0 / 2 / 4) and a diffrax ensemble
  inside an item (4,936 against 4,968 ms). `docs/design.md` section 6 already recorded that ptxas reports
  zero spill bytes for every public adjoint kernel, so no rule could have set S either. -96 lines.
- Event scheduling (the private `_schedule='events'`, `kernel._cut`, `kernel._events_main`) and
  `benchmarks/events_bench.py`; `_schedule` takes `'step'` | `'item'`. It won only on the synthetic lopsided
  switch it was built for (1.6-3.1x forward, gradient unchanged), never fired on a `benchmarks/vs_jax` problem,
  and was 0.81x on the photon-transport engine at the same commit; its general form in the shower engine was
  0.50x / 0.77x on a 2080 Ti and hung once (`docs/design.md` section 1b). The inner-loop half (1.03-1.82x
  slower on four toys built for it) had already gone. No other kernel changes.
- `tm.compare` (with `criteria`, the private `_api._square_mode` and the re-tracer's `square=` hook). The frozen
  comparison is a test protocol, not an API: it moved to `tests/compare_protocol.py`, where `floor_fn` (the same
  item with every emitted value squared) is a required argument instead of a global trace mode. `tm.reference`
  is unchanged.
- `tm.assert_arms_differ` (and `_diagnostics.artifacts`): the A/B rule is developer measurement hygiene built on
  `tm.diagnostics`, and moved to `tests/ab_rule.py` (used by `tests/` and `benchmarks/perf_negative_control.py`).
- The public option `allow_fallback` (and `_primitive._lockstep_cannot_run`). Passing it now raises, like any
  unknown option. `backend='xla'` is the explicit equivalent, and the `UnsupportedError` for a program the
  cuda backend cannot compile now says to pass it, for `backend='cuda'` as well as `'auto'`.
- `tm.Sum(..., tangent_scale=)`, the fixed-point scale of the TANGENT tally of an `fixed_point=True` output. Nothing
  in the tests, benchmarks, examples or documentation ever set it and no number was ever recorded for it; a
  tangent tally now always uses its primal output's `scale`, which is what every program that did not set it
  already got, so no result changes. Passing it raises `TypeError`, as any unknown `tm.Sum` field does.
- The public option `cuda_min_blocks` (the cuda `__launch_bounds__` minimum-blocks-per-SM register cap). Passing
  it now raises, like any unknown option. Re-measured on the toy walk at 4M items on a 2080 Ti: 0 / 1 / 2 / 4
  gave forward 6.22 / 6.25 / 6.22 / 6.17 ms and a JVP 7.16 / 7.25 / 7.28 / 7.17 ms, all inside rerun noise.
  Every kernel is now compiled with `__launch_bounds__(128)` alone.
- Three private kill-switches of the cuda emitter that were permanently on and that nothing set:
  `codegen.FOLD` (constant folding), `op_table.LAZY_FUSE` and `op_table.LAZY_FUSE_MAXUSE` (one-use lazy
  fusion). The generated CUDA source and the cubins are unchanged.
- The two private reverse-mode cost-decomposition options, `_rev_phase` (`'primal'` | `'skip'`) and
  `_tree_phase` (`'fwd'` | `'roots'` | `'children'`). Both deliberately computed WRONG gradients to attribute the
  adjoint's cost to its phases; no test used either, and the decompositions they produced are recorded in
  `docs/design.md` section 6. No behaviour change on any supported option value.

### Changed

- `Sum(fixed_point=True)`: an emit too large for the int64 fixed point (`stats['fixed_point_overflow'] > 0`) makes every
  fixed-point float output of the call NaN, tangents and cotangents included, like every other resource limit
  (`max_steps`, the spawn Queue). It was dropped and only counted, so an overflowed sum came back finite and
  wrong. The counter is per call, so all fixed-point outputs are poisoned, not only the one that overflowed;
  other outputs are unchanged.
- Reverse mode (from lumen/perf 8c92ee8, 762da14, 830c071): the forward pass of a checkpointed loop stores the
  states before steps 0..C-1 in a head buffer, so an item of K <= C steps is never replayed; the first segment
  tape reads the head. Only the carries the step or predicate reads are stored (liveness read off the loop's
  jaxprs, no re-trace). `rev_checkpoints` defaults to `'auto'`: 16 for a state under 128 B, else the largest of
  4 ... 64 whose 3C+1 states fit 64 KiB (logical bytes); an adjoint differentiated again is rebuilt at C = 16.
  Per-item memory is 3C+1 states (was 2C). The replay check stays on. lumen's grad / value ratio was 10.1 / 18.0 /
  19.5x on SK / SBND / ANNIE with the old scheme (L40S); see `docs/design.md` section 3.
- Reverse-mode checkpoint buffers are uninitialised on cuda (`tallymap_uninit`): zero-filling them cost the
  cheap-step `vs_jax` gradients up to 5x. Tested by filling every uninitialised declaration with 0x00 and 0xFF
  (bitwise-equal results required).
- cuda step split: an array loop carry of >= 64 elements is updated in place instead of being rebuilt and
  copied back every step.

- The cuda kernel cache and the eager-call cache key a program on a structural sha256 digest of its jaxpr
  (`_jaxpr.digest`) instead of its printed text. JAX's pretty-printer cost 95 s per build on a large engine item
  (4.2 s with the digest); the digest also separates literal values the printer rounds (`docs/design.md`
  section 10).
- `tm.diagnostics()` builds the item jaxpr's text when it is called, not at every bind (102 s of a 260 s cached
  build on a large engine item). The returned record is unchanged.
- A forward-only call (primal, `jvp`, `jacfwd`, a folded batch) no longer computes the per-item replay
  fingerprint; reverse mode builds the fingerprinted program when it needs it, so the replay check is unchanged.
  +76% forward on an engine item with a 16.8 KB loop carry, 1.19-1.31x on cheap items.
- cuda: a serial chain of 64-bit integer multiplies is cut every 8 links by a zero-instruction barrier, because
  NVRTC's front end is exponential in that chain's depth (127 s at 16 links, 0.2 s cut). Results are bitwise
  unchanged; `tests/mulchain.py` checks it.
- cuda: a lazily inlined value used more than once is computed once into a local array when its element expression
  is longer than 200 characters, instead of being re-inlined into every consumer (a geometry kernel's source
  13.0 -> 7.5 MB).
- cuda buffer assignment: a `while` loop's dead initial carry array becomes the carry storage, and a `cond` output
  is written into a dead operand of the same shape (also when the value is lazy and reads that storage only at its
  own index), so neither is copied. Placement only, results bitwise unchanged; 1.59x on a large engine item on an
  L40S (`docs/design.md` section 10).

- The cuda kernel claims ONE item per warp-aggregated claim. It used to claim a run of up to
  `clamp(N / (16 * grid * block), 1, 16)` items per thread, which left the last claims holding several items each
  — a tail the other threads wait for. Measured over all 18 `benchmarks/vs_jax` forward and gradient cells on a
  2080 Ti: one item per claim is 1.46x faster on random_walk forward, 1.14-1.18x on transport forward,
  random_walk gradient and uniform gradient, and never slower (worst cell 1.002). Three fitted constants and
  two kernel variables go with it.

- The rule that decides whether `jax.vmap` over a tallymap call folds the batch into the item axis (one launch
  of B*N items) or falls back to `lax.map` (one launch per member). On cuda it was "fold while every privatised
  tally stays privatised"; it is now "fold while one member's N items cannot fill the device" (N below
  multiprocessors x 1024, the smallest maximum-resident-threads-per-multiprocessor of any tested architecture).
  Folding costs B copies of every per-item operand and buys one saturated launch, which has nothing to do with
  the tally size. Measured on the toy walk at B=8 on a 2080 Ti: at 65,536 items the new rule folds two more
  cells and is 1.32x and 2.21x faster (256 and 1,024 bins); at a million items it stops folding two cells and
  is 2.46x and 2.48x faster (64 and 4,096 bins) and at four million 6.85x and 5.83x; at 262,144 items it gives
  up 1.35 to 1.37x. `lax.map` cells
  and xla are unchanged (`docs/design.md` section 2). `kernel.privatised()` loses its `B=` parameter.

- cuda compile time: lazily inlined element expressions are capped at 32 nodes (was 256). On a large
  photon-transport item this took the forward compile from 1543 to 834 s and the run from 9.04 to 6.87 s on a
  2080 Ti, with bitwise identical results.
- cuda dual lanes (`jacfwd` / `jacrev` / vmap of `jvp`): the lane-memory guard measures the lane-dependent loop
  state on the UNBATCHED program (that state is linear in the lane count) and picks the lane width before the
  T-lane program is built, instead of tracing the item over 16 lanes, measuring it, and rebuilding at the
  chunked width or discarding it. Same width chosen; on a large engine item the discarded 16-lane trace was
  3-4 h of compile time. From the private engine line (8af307d), with the measurement moved ahead of the build
  rather than added as a second check.
- A "reads a Tally" refusal names each tally operand: its position, aval, tally slot and kind, and the
  primitive and source line that produced it. Raised inside a call-like sub-jaxpr (`jit`, `custom_jvp`), it
  also shows the call's operands against the sub-jaxpr's invars, with the slot each position carries; inside a
  reverse program, the program's slot legend (user tallies, cotangent tallies, cotangent tokens). Text only.
  From the private engine line (0863aa1, c033ea8).
- cuda: a program may use up to 2048 buffers (operand leaves, constant tables, tallies, results); the limit was
  504, a hard failure for an engine item with ~540 constant tables. A program with at most 504 buffers emits
  exactly the source it did (one 4 KB by-value parameter block); a larger one gets a 16 KB block, which needs
  CUDA 12.1+ large kernel parameters (compute capability 7.0+, a 12.1+ toolkit and driver). MEASURED on a
  2080 Ti (7.5) and an L40S (8.9), driver API 12.9, with NVRTC 12.1 (JAX 0.5.3) and 12.9 (JAX 0.11.2): 600 and
  2000 broadcast tables run and match the analytic sum; 2100 raise naming the limit.
- `jacfwd` / `jacrev` with more tangent lanes than one kernel takes split them evenly across kernels (26 lanes:
  13 + 13, not 16 + 16 with 6 padded); every chunk recomputes the primal, so padding cost a primal's worth of
  lanes.

- Reverse mode: the checkpoint stride of a `checkpointed_while` is now `max(C, ceil(K/C))` rather than
  `ceil(K/C)`, and the forward pass of the checkpointed loop no longer stores states. The backward sweep
  allocates one segment buffer of C states per segment, and that costs C stores whatever the segment's length is,
  so a stride of `ceil(K/C)` -- one state per segment for every loop with K <= C steps -- spent most of the
  adjoint kernel zeroing slots it never read. MEASURED on a 2080 Ti at the `benchmarks/vs_jax` sizes,
  `jax.grad` wall time: transport 15.6 -> 6.07 ms, walk on spheres 9.91 -> 4.14 ms, particle filter 22.5 ->
  17.0 ms, toy walk at K=100 2.9 -> 2.5x its forward pass; barrier and the ode ensemble, whose loops run about
  220 steps and so already filled their buffers, are unchanged (13.4 -> 13.4 ms, 24.8 -> 25.3 ms). Peak memory
  is unchanged, and per-item memory drops from 2C+1 to 2C states. `rev_checkpoints` is much less sensitive as a
  result: over C = 8, 16, 32, 64 the six problems now spread by at most 1.6x, against up to 20x before.
  Gradients are unchanged except for the float summation order of a cotangent accumulated by atomics, which
  already varies between runs; per-item cotangents are bitwise identical.
### Fixed

- cuda: `scatter` (`x.at[i].set(v)` / `.add` with a traced index) updates a dead operand in place, as
  `dynamic_update_slice` already did; it copied the operand every time. On DTRAX's mode-2 item (2080 Ti) the
  per-thread frame goes from 178,160 to 71,440 B (`g0102`: it ran out of memory on the 2080 Ti, now launches) and
  from 557,248 to 156,496 B (`sEM1_lar_e`: over the 512 KiB limit, now launches), digests bitwise unchanged; the
  private engine line's frames are 70,160 / 155,200 B. LUCiD's kernels are byte-identical. Neither rule writes in
  place when the update or the indices read the operand's own storage.
- cuda: a kernel whose per-thread local memory (the item's frame: per-item arrays and register spills) is over
  CUDA's 512 KiB per-thread limit raises `UnsupportedError` before its first launch, naming the bytes, the limit
  and the item function. The frame is read from the cubin's own `.nv.info` records, so no launch is needed. It
  failed in the launch with `CUDA driver error 1: invalid argument` (DTRAX `sEM1_lar_e`, 557,248 B per thread).
- An error re-raised with the item function and source line appended (tally misuse, unsupported operations)
  keeps every attribute of the original exception; it kept only the message. No library exception carries one
  today; the private engine line lost `CodegenError.nbytes` this way (8fe2295).
- `jax.jvp` / `jacfwd` of a spawning program in which a `spawn` site's float child leaves are all constants (no
  tangent reaches it): JAX binds that spawn without calling its jvp rule, so it carries only the primal leaves.
  The xla backend raised a `while_loop` carry-structure TypeError; the cuda backend wrote nothing into the
  child's tangent slots, so the child read whatever an earlier child had left in that ring slot (zero only until
  the ring wraps). Both now write those tangents as zero. Found by the DiCE tests ported from the private engine
  line.
- `tm.diagnostics(f, ...)["stablehlo_sha256"]` (and `tests/ab_rule.py`, which compares it for the xla
  backend) is the same on every lowering of a program with a host callback. On JAX < 0.7 a callback lowers
  with the address of its Python callable, new every time, so the raw text never hashed the same twice; the
  descriptors are now replaced by their ordinal before hashing. From the private engine line (48eea82).

## 0.5.0 - first public release

tallymap runs divergent per-item Monte Carlo loops as a single JAX primitive. You write one ordinary JAX function
of ONE item, with its own `lax.while_loop`, its own trip count, its own gathers and its own `jax.random` draws,
and `tm.vmap` maps it over N items whose results go into commutative sums ("tallies"). On an NVIDIA GPU the item
compiles to one CUDA kernel in which a lane takes a new item as soon as its current one finishes; everywhere else
it runs the lockstep backend, a masked plain-JAX interpreter that is also the reference implementation. Because it
is one primitive, every JAX transformation composes: `jit`, `vmap`, `jvp` / `jacfwd` / `linearize`, `grad` /
`vjp` / `jacrev`, `hessian`, `checkpoint`, `shard_map`, and control flow around the call.

The reason to use it rather than plain JAX is in `README.md` and measured in `docs/performance.md`. The short
version: plain JAX cannot reverse-differentiate a `lax.while_loop` at all, and the padded masked `lax.scan` you
have to write instead costs the longest item for every item, plus memory proportional to the step cap.

### What a 0.4 user will notice

The library had no public users before this release, so these changes carry no deprecation period.

- **`max_steps` is now a resource limit with a visible failure.** When `max_steps` is set and an item hits it,
  every float tally of that call becomes NaN, exactly as a dropped spawn child already did. Before, the call
  returned a finite, plausible, too-small number with only `stats['n_truncated']` set. `Result.truncated` and
  `stats['n_truncated']` are unchanged. Integer tallies cannot be NaN, so for those the counter remains the only
  signal. There is no opt-out: a step budget you impose on purpose belongs in the loop condition,
  `(alive) & (step < K)`, which is exact JAX semantics and not a tallymap limit at all. The default is still
  `max_steps=None`, which means no limit and exactly `jax.vmap` semantics, including the fact that an item which
  never terminates does not terminate.
- **Out-of-range emit indices are counted.** `stats['dropped_emits']` counts emits whose index was negative or at
  least the output size, on both backends. They are still dropped, which matches what JAX's own scatter does with
  an out-of-bounds update, so this is not a deviation from the dense meaning; it is an off-by-one in your binning
  that used to be invisible. Nothing is poisoned.
- **`tm.compare` is a function, not a module.** The module moved to `tallymap/_compare.py`; the public names are
  `tm.compare` and `tm.reference`. `from tallymap.compare import reference` no longer works; use `tm.reference`.
  There is no compatibility shim.
- **The version policy is stated and tested.** `jax>=0.5.3`, Python >= 3.10, no upper bound.
  `tm.TESTED_JAX = ("0.5.3", "0.6.2", "0.7.2", "0.8.3", "0.9.2", "0.10.2", "0.11.2")`, every one of which passes
  the full CPU suite; importing tallymap warns once only when JAX is newer than the newest tested version, so
  nothing in the middle of the range warns any more. A release found to be broken gets a `!=` exclusion in
  `pyproject.toml`. `check_jax_version` is private.

### Also in this release

- **Equinox-based solvers run inside an item.** `diffrax.diffeqsolve`, `optimistix` root finds, `lineax` solves
  and `quadax` quadrature can be called inside a `tm.vmap` item. Equinox's own primitives are expanded at the
  item trace, and small dense linear algebra (`lu`, `cholesky`, `triangular_solve`, QR) has cuda rules. See
  `docs/equinox.md` for the recommended diffrax adjoint inside an item and for `EQX_ON_ERROR`.
- **Two cuda emitter fixes for long items.**
  - Generated identifiers could collide with the kernel's own integer typedefs (`i8` ... `u64`), and the kernel
    then failed to compile.
  - A lazily fused value used twice inside another fused expression doubled per stage: a 40-stage `jnp.roll`
    chain exhausted host memory. Fusion is now capped at 256 inlined nodes per element.

  Programs that compiled before produce identical CUDA source.
- **Measured comparisons.**
  - `benchmarks/vs_jax/` is a reproducible suite against the best plain-JAX program for 13 problems: seven
    natural fits, four bad fits and two interacting systems. Results and figures are in `docs/performance.md`.
  - `examples/demos/` holds eight head-to-head demos against BlackJAX, gymnax, crn-jax, Ray-trax, diffrax, quadax
    and plain-JAX baselines.
- **New examples:** walk-on-spheres, an adaptive ODE ensemble, Gillespie, interacting particles, a particle filter,
  `gradient_pitfall` and `diffrax_ensemble`. The two adaptive-step examples and their benchmark problems
  stop-gradient the step-size factor, as diffrax does. Through the controller, the ODE ensemble's f32 gradient
  had the opposite sign to its f64 one. `grad` and `jacfwd` then agreed only where XLA rounded both programs
  identically, which XLA:CPU from JAX 0.8 does not.

- **Licence: Apache-2.0** (`LICENSE`, `NOTICE`), with `CITATION.cff` for citation.
- **Packaging.** `[project.urls]`, keywords and classifiers; the `[cuda]` extra is `jax[cuda12]`, which already
  brings NVRTC, so the explicit `nvidia-cuda-nvrtc-cu12` dependency is gone. The wheel holds only the package;
  the sdist also carries `tests/`, `examples/`, `benchmarks/` and `docs/`, because the commands the README
  documents live there.
- **CI anyone can see.** `.github/workflows/ci.yml` runs ruff and the short suite per push and pull request at
  both ends of the supported JAX range, plus the whole CPU suite on the newest release.
  `.github/workflows/weekly.yml` runs the whole CPU suite on every tested minor, plus a JAX nightly cell that is
  allowed to fail.
- **Documentation for an outside reader.** A rewritten `README.md`: what the library is, a regime table for
  whether it will help you, the honest comparison against `jax.vmap`, Dr.Jit, Warp and Pallas, the mental model,
  the correctness rules you must know, a section on gradients that are exactly computed and still not what you
  want (including why an adaptive step-size controller must be stop-gradient-ed), a NaN diagnosis order, and the
  support boundary. `docs/performance.md` holds the measured comparison
  against plain JAX. `CONTRIBUTING.md` explains how to run the tests and how to add a primitive.
- **NaN diagnosis.** The docs now give the order: if `jvp` or `jacfwd` is NaN too, it is your own arithmetic; if
  only reverse mode is NaN, check `stats` for a truncated item or a dropped spawn child; only then use
  `rev_stats=True` and `tm.diagnostics()`.
- **`tests/ops.py` baselines tallymap's own output.** Its per-(GPU, JAX) baseline now stores a hash of tallymap's
  result plus the fields judged against the float64 reference, not the comparison with XLA. XLA's side is not
  reproducible across processes: on L40S with JAX 0.11.2, about 1 process in 6 compiles the `cumsum8` row reduction
  with a different emitter, which failed 2 of 172 rows while tallymap's output stayed bitwise identical.
- **A first-contact fix on CPU.** The one-time backend warning says there is no CUDA GPU on this platform,
  instead of naming XLA's internal platform string, and it is attributed to the caller.
- **A first-contact fix on GPU.** A kernel launch that runs out of memory now says why: the per-thread local
  memory the CUDA driver reserves outside JAX's pool, how much, and the `XLA_PYTHON_CLIENT_*` setting to change.
  Found by installing the wheel in a fresh environment, where JAX's default 75% preallocation made three
  forced-dense reverse tests fail; `tests/run_all.py` now turns preallocation off unless you set it.

### Known limits

These raise rather than returning something wrong: `grad` through a `lax.while_loop` AROUND a call (JAX itself
cannot reverse-differentiate a while loop); second-order derivatives through the spawn tree adjoint;
`jax.export` of a CUDA program; effects inside the item; a primitive the CUDA emitter does not know
(`UnsupportedError`, with `backend='lockstep'` as the workaround that always works); reading, forking or dropping
a tally (`TallyError` at trace time); `n_active` together with spawn. TPU and ROCm are not supported. A100 and
H200 are untested. The full list is in the README.
