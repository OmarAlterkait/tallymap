# Changelog

All notable changes. Versions follow [SemVer](https://semver.org/) 0.x: until 1.0 a minor release may change the
API, and only after one minor release in which the old form still works and warns. Patch releases never change
the API. The release policy for JAX versions is in `docs/maintenance.md`.

## Unreleased

### Added

- `benchmarks/compile_scaling.py`: NVRTC compile time, generated source size, registers and per-thread
  local memory against the SIZE of one item, with and without `jacfwd` lanes. The cuda emitter inlines the
  whole item into one kernel, so for a large item the compile time is the size of that one function, and
  `docs/design.md` section 9 had only the private engine's numbers for it. On a 2080 Ti compile time grows
  as source size to the power 1.8 to 2.3 between 99 KB and 706 KB of generated source
  (18 s to 725 s).

- `benchmarks/events_bench.py`: the measurement of event scheduling (`_schedule='events'`) against the default
  step refill, on the item shape it is for (a divergent loop whose body is a lopsided switch). No `vs_jax`
  problem has a top-level `cond` with a dynamic index in its step body, so on all of them the two schedules
  compile to the same kernel and the mechanism's benefit was undocumented. It is 3.1x on the forward pass of
  a rare, 256-times-more-expensive branch at 4M items on a 2080 Ti, and a wash in reverse mode
  (`docs/design.md`, section 1b).
- `jnp.linalg.solve`, `inv`, `det` and `lu_factor` / `lu_solve` run inside an item on cuda: `lu`,
  `lu_pivots_to_permutation`, `custom_linear_solve` and `triangular_solve` are lowered after autodiff to JAX's own
  pure-JAX implementations (and a substitution loop), so the derivative is JAX's own. `supported_primitives()`
  lists them as `expanded`.
- `tm.diagnostics()["kernels"]` reports each kernel's compile time and, once it has run, its measured
  resources: registers and local memory per thread, resident blocks per SM, and the local memory the CUDA driver
  reserves outside JAX's pool (what an out-of-memory error at launch is about).
- A program that fails to lower because a `custom_jvp` closes over a traced value (a JAX staging limit) gets an
  error that names the cause and the fix (pass the value as an argument).

### Removed

- The public option `rev_step_segments` (recompute each loop step in S checkpointed segments inside its VJP).
  Passing it now raises, like any unknown option. Four independent measurements on a 2080 Ti found no benefit:
  the six `benchmarks/vs_jax` gradient cells (0.98-1.05 at S = 4 and 8), the toy walk at 1M items
  (6.32 / 6.36 / 6.45 / 6.43 ms at S = 0 / 4 / 8 / 16), a toy built to spill (168 registers, 1,328 bytes of
  local memory per thread, grad/forward 8.1x: 32.4 / 31.2 / 32.1 ms at S = 0 / 2 / 4) and the diffrax ensemble
  of `docs/equinox.md` (4,936 against 4,968 ms). `docs/design.md` section 6 already recorded that ptxas reports
  zero spill bytes for every public adjoint kernel, so no rule could have set S either. -96 lines.
- Event scheduling of INNER while loops. `_schedule='events'` now cuts only the step body of the outer loop
  (`kernel._cut` / `kernel._events_main`); an inner `while` is lowered as a plain C loop again. On four toys
  built to be the inner-loop scheduler's best case it was 1.82x / 1.36x / 1.22x / 1.03x slower than the plain
  loop and never faster, and it never fired on any `benchmarks/vs_jax` problem. It was also the reason
  `_schedule='events'` cost 3.1x in reverse mode (every top-level loop of an adjoint program was
  event-scheduled); the gradient is now within 3% of the default, while the forward win (1.6-3.1x on
  `benchmarks/events_bench.py`) is unchanged.
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

- The cuda kernel cache and the eager-call cache key a program on a structural sha256 digest of its jaxpr
  (`_jaxpr.digest`) instead of its printed text. JAX's pretty-printer cost 95 s per build on a large engine item
  (4.2 s with the digest); the digest also separates literal values the printer rounds (`docs/design.md`
  section 11).
- `tm.diagnostics()` builds the item jaxpr's text when it is called, not at every bind (102 s of a 260 s cached
  build on a large engine item). The returned record is unchanged.
- A forward-only call (primal, `jvp`, `jacfwd`, a folded batch) no longer computes the per-item replay
  fingerprint; reverse mode builds the fingerprinted program when it needs it, so the replay check is unchanged.
  +76% forward on an engine item with a 16.8 KB loop carry, 1.19-1.31x on cheap items.

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
- **The lockstep backend's cost now follows the ACTIVE lanes.** A `while_loop` used to run its body on all P
  lanes until the last lane finished. A loop now hands its survivors to the same loop over a quarter as many
  lanes as soon as they fit (`docs/design.md` section 10). Measured on one CPU node, both arms in one process, the `benchmarks/vs_jax`
  problems at N = 65,536: forward 1.3 to 2.1x faster on the divergent problems, gradients 1.1 to 1.8x, and
  within noise (0.92 to 1.04x) on the problems whose lanes stay active together. Trace and compile time grows by
  up to 30%, except the shower's gradient (the tree adjoint's loop body), where it went from 75 to 219 s for
  1.24x per call; peak memory grows by the W-wide copy of the loop carries (4 to 76 MB at N = 65,536). On a CPU
  every tally, per-item result and stat is BITWISE unchanged, which `tests/edge.py` asserts; on a GPU the
  hardware picks the order of a scatter-add, so a float tally can move by an ulp, exactly as the block width
  already could. There is no new public option.

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
