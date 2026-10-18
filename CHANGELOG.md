# Changelog

All notable changes. Versions follow [SemVer](https://semver.org/) 0.x: until 1.0 a minor release may change the
API, and only after one minor release in which the old form still works and warns. Patch releases never change
the API. The release policy for JAX versions is in `docs/maintenance.md`.

## Unreleased

### Changed

- **The lockstep backend's cost now follows the ACTIVE lanes.** A `while_loop` used to run its body on all P
  lanes until the last lane finished, and a `cond` / `switch` with a per-lane index used to run every branch
  any lane took on all P lanes. A loop now hands its survivors to the same loop over a quarter as many lanes as
  soon as they fit, and a cond runs each branch on the lanes that take it when that is cheaper than moving them
  (`docs/design.md` section 9). Measured on one CPU node, both arms in one process, the `benchmarks/vs_jax`
  problems at N = 65,536: forward 1.3 to 2.1x faster on the divergent problems, gradients 1.1 to 1.8x, and
  within noise (0.94 to 1.01x) on the problems whose lanes stay active together. Trace and compile time grows by
  up to 30%. Tallies, per-item results and stats are BITWISE unchanged, which `tests/edge.py` asserts; there is
  no new public option.

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
