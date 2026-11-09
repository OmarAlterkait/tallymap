# Changelog

All notable changes. Versions follow [SemVer](https://semver.org/) 0.x: until 1.0 a minor release may change the
API, and only after one minor release in which the old form still works and warns. Patch releases never change
the API. The release policy for JAX versions is in `docs/maintenance.md`.

## Unreleased

### Changed (revfit: reverse mode for lumen's acceptance)

- Reverse mode: the single-loop adjoint (T3, from lumen/perf 6a70383 / b38db3f / 22e615d / e58e216, re-implemented
  on dev) for an item whose one top-level loop has a step of >= 1000 equations (`_reverse.T3_MIN_EQNS`): ONE
  while loop of micro-steps, so the cuda backend refills lanes at step granularity and the step is instantiated
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

- `Sum(exact=True)`: a TANGENT-only overflow was silently dropped on every unfused jvp path (FUSE_JVP off,
  linearize, jacfwd with lanes refused), and the fused jvp's single counter made the correct PRIMAL exact outputs
  NaN. Each kernel now flags the bins an unrepresentable emit was dropped from and writes them as NaN (cuda and
  lockstep); the call-level exact poison is gone. A primal overflow no longer poisons other exact outputs or the
  (correct) tangent.

### Added

- `tm.diagnostics(...)["programs"][i]["baked_floats"]`: shape, dtype and size of every FLOAT constant the item
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
  like any other primitive the emitter has no rule for. The lockstep backend is unchanged: XLA runs these
  primitives there. Write a small solve out by hand (Cramer's rule, an unrolled elimination), as
  `examples/demos/stiff_ode/kv3.py` does, or run the item with `backend='lockstep'`.
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
  unknown option. `backend='lockstep'` is the explicit equivalent, and the `UnsupportedError` for a program the
  cuda backend cannot compile now says to pass it, for `backend='cuda'` as well as `'auto'`.
- `tm.Sum(..., tangent_scale=)`, the fixed-point scale of the TANGENT tally of an `exact=True` output. Nothing
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

- `Sum(exact=True)`: an emit too large for the int64 fixed point (`stats['exact_overflow'] > 0`) makes every
  exact float output of the call NaN, tangents and cotangents included, like every other resource limit
  (`max_steps`, the spawn Queue). It was dropped and only counted, so an overflowed sum came back finite and
  wrong. The counter is per call, so all exact outputs are poisoned, not only the one that overflowed;
  non-exact outputs are unchanged.
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
  `clamp(N / (16 * grid * block), 1, 16)` items per lane, which left the last claims holding several items each
  — a tail the other lanes wait for. Measured over all 18 `benchmarks/vs_jax` forward and gradient cells on a
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
  and lockstep are unchanged (`docs/design.md` section 2). `kernel.privatised()` loses its `B=` parameter.

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
  The lockstep backend raised a `while_loop` carry-structure TypeError; the cuda backend wrote nothing into the
  child's tangent slots, so the child read whatever an earlier child had left in that ring slot (zero only until
  the ring wraps). Both now write those tangents as zero. Found by the DiCE tests ported from the private engine
  line.
- `tm.diagnostics(f, ...)["stablehlo_sha256"]` (and `tests/ab_rule.py`, which compares it for the lockstep
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
