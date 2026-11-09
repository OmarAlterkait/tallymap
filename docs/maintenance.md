# Maintenance

What a JAX upgrade can break, what to do when it does, the constants that were tuned, how the library is tested,
and how a release is made.

## Supported

- Python >= 3.10. CUDA 12 with NVRTC, NVIDIA's runtime compiler, the only compiler the cuda backend uses (the
  `cuda` extra installs the `nvidia-cuda-nvrtc-cu12` wheel; a CUDA toolkit's `libnvrtc` works too), plus a C++17
  compiler once, to build the `jax.ffi` launch shim
  (`backends/cuda/shim.cc`, cached per toolchain and jaxlib version).
- GPUs tested: compute capability 7.5 (RTX 2080 Ti) and 8.9 (L40S). A100 and H200 are untested. The spawn queue
  needs compute capability 7.0 or newer, for `__nanosleep`.
- Every other platform runs the xla backend: `backend='auto'` warns once, `backend='cuda'` raises. TPU and
  ROCm are untested. The xla backend is the portable interpreter, slow by design, about 100 times the CUDA
  backend.

## Version policy

The floor is **JAX 0.5.3 and Python 3.10**, with **no upper bound**, as equinox, optax and diffrax do. A release
found to be broken gets a `!=` exclusion in `pyproject.toml` and a patch release; there are none so far.

**Every minor release from the floor to the newest is tested**, not only the two ends, because the parts of JAX
this library leans on (the legacy transformation registries, the jaxpr equation parameter names, the shard_map
varying-axes type) change between minors and not between patches. Each minor is tested at its latest patch, on
CPU, with the full suite:

| JAX | 0.5.3 | 0.6.2 | 0.7.2 | 0.8.3 | 0.9.2 | 0.10.2 | 0.11.2 |
|---|---|---|---|---|---|---|---|
| Python | 3.10 | 3.10 | 3.11 | 3.11 | 3.11 | 3.11 | 3.12 |

`tm.TESTED_JAX` is that list. Importing tallymap warns once only when JAX is **newer than the newest tested
version**: everything between the floor and the newest is tested, so nothing in the middle warns. That matters
because the untested middle is where a default `pip install` can land: on Python 3.10 pip resolves `jax>=0.5.3`
to 0.6.2, not to the floor and not to the newest.

### Raising the floor

Raise it **deliberately, never by accident**. Each raise deletes compatibility code in `_compat.py` and
`_jaxpr.py`, which is the point, and drops users who cannot move, which is the cost.

The scientific-Python convention (SPEC 0) is to keep a core dependency for 24 months after its release. JAX 0.5.3
is inside that window until **2027-03**. Revisit the floor then. Do not raise it before there is something to
gain: the planned hijax port (see "Watch item" below) is the first such thing, and it needs a floor of 0.11.

### When a new JAX release appears

1. Run the full matrix on it (CPU, plus the GPU cells if a GPU is available).
2. If it passes, add the version to `TESTED_JAX`, with the Python version it needs.
3. Publish a patch release, aiming for within about two weeks of the JAX release. The point of the deadline is
   that the warning on an untested-newer JAX is useless if it fires for months.
4. If it fails and the fix is not quick, add a `!=` exclusion for it in `pyproject.toml` and say so in the
   changelog.

The weekly CI job runs the whole CPU matrix plus a JAX nightly cell (allowed to fail), which is the early
warning.

## The JAX surface, by risk

| tier | what tallymap uses | where |
|---|---|---|
| public (3-month deprecation policy) | `make_jaxpr`, `jvp` / `vjp` / `vmap`, `custom_jvp` / `custom_vjp` (with symbolic zeros), `checkpoint`, `typeof`, `tree_util`, `debug.callback`, `lax` control flow, `lax.pcast` / `pvary`, `jax.random.{key_data, wrap_key_data, key_impl}`, `jax.ffi.*`, `jax.export.DisabledSafetyCheck`, `jax.test_util.check_grads` (tests only) | everywhere |
| `jax.extend` | `core.{Primitive, ClosedJaxpr, Literal, jaxpr_as_fun, DropVar, primitives}`, `random.{threefry,rbg,unsafe_rbg}_prng_impl`, `source_info_util.{summarize, user_context, register_exclusion}` | `_api`, `_jaxpr`, `_compat` |
| legacy modules: where a JAX upgrade would break us | `jax.core.{ShapedArray, Tracer}`; `jax.interpreters.ad.{primitive_jvps, primitive_transposes, Zero, instantiate_zeros, is_undefined_primal}`; `batching.fancy_primitive_batchers`; `mlir.{register_lowering, lower_fun}`; `partial_eval.{custom_partial_eval_rules, partial_eval_jaxpr_custom_rules}` (the fused JVP only, feature-detected) | `_tally`, `_primitive`, `_reverse`, `_compat` |
| experimental | the shard_map varying-axes type (`aval.vma` on 0.7-0.10, `aval.manual_axis_type.varying` on 0.11) | `_primitive` (feature-detected) |
| private | `aval.dtype._impl` of a typed pseudo-random key (with a name lookup as the fallback); the jaxpr equation parameters read by `_jaxpr.py` | `_compat`, `_jaxpr` |
| jaxlib / XLA | the foreign-function-interface C++ headers (api_version 4), hashed into the shim's cache key | `backends/cuda/shim.cc`, `compile.py` |

No `jax._src` import is left. Every interface that differs between tested versions goes through
`tallymap/_compat.py` (version shims) or `tallymap/_jaxpr.py` (jaxpr structure). Keeping it to two modules is
deliberate: it is what makes an upgrade a bounded job.

## When X breaks, do Y

| symptom | cause | fix |
|---|---|---|
| `UnsupportedError: primitive 'foo' is not supported` on cuda, for a program that used to run | a new or renamed JAX primitive | add a rule to `backends/cuda/op_table.py`, or a canonical form to `canon.py` if it is shorthand for primitives that already have rules; add a row to `tests/ops.py`; regenerate `docs/primitives.md` |
| the edge cell "op table: names are JAX primitives" fails | a table name no longer exists in `jax.extend.core.primitives` (a rename, such as `pjit` to `jit`) | keep the old name if a tested version still uses it, add the new one, and explain the old one in the table's own exclusion list |
| `KeyError` or `AttributeError` on a control-flow equation parameter (`cond_nconsts`, `body_jaxpr`, `num_consts`, `branches`, `call_jaxpr`, ...) | JAX changed how a control-flow or call equation stores its sub-jaxprs | fix the accessor in `_jaxpr.py` (`while_parts`, `cond_branches`, `scan_parts`, `call_jaxpr`); nothing else in the library reads those parameters |
| `run_all.py --canary` fails with a DeprecationWarning from tallymap's own code | JAX deprecated an API the library uses | move to the replacement in `_compat.py`; if it is one of the legacy transformation registries, start the hijax port |
| every JVP suddenly runs two kernels (`tallymap._primitive.FUSE_JVP` is False), while the numbers stay correct | the partial-evaluation registries are gone | nothing is wrong: that is the fallback design, and `run_all.py --unfused` already tests it. Regaining one kernel needs the hijax port |
| batching rules not found, or a wrong batcher signature | the batching registry changed | adapt `_compat.register_batcher`, which is one wrapper |
| the shard_map or pmap cells fail on a new JAX | the varying-axes type moved again | adapt `_varying` / `_with_varying` / `_plain` in `_primitive.py`; meanwhile users can pass `shard_map(..., check_vma=False)` |
| typed-key randomness fails on cuda (`random_bits`, `split`, `fold_in`) | the pseudo-random implementation object or `dtype._impl` changed | adapt `_compat.key_impl` and the `random_*` shims; the RNG rows of `tests/ops.py` check it bitwise; meanwhile users can pass raw uint32 keys |
| the shim does not build or does not load | a new jaxlib FFI header, or a different host compiler | `shim.cc` is about 200 lines and its cache key contains the headers, so a stale shared object is never reused; check `$CXX` first |
| `tests/ops.py` reports CHANGED rows | the CUDA emitter, NVRTC or the CUDA math library changed the numbers | if the change was intended: rerun the `compare` protocol (`docs/validation.md`), then re-record `tests/ops_baseline.json` in a commit of its own |
| `stats['children_dropped'] > 0` and the float tallies are NaN | the spawn Queue is too small | size it (README, "Sizing the Queue") |
| `stats['n_truncated'] > 0` and the float tallies are NaN | items hit `max_steps` | raise it, or put the budget in the loop condition if it is deliberate (README, `max_steps`) |
| `stats['dropped_emits'] > 0` | an emit index was negative or at least the output size | an off-by-one in your binning; clip the index |
| a NaN gradient | in this order: your own arithmetic, a resource limit, then a reverse-mode check | if `jvp` or `jacfwd` is NaN too it is your arithmetic (look for overflow); else check `stats` for truncation and dropped children; through spawn, rerun with `rev_stats=True` and read `tm.diagnostics()['reverse']['tree']` |
| a perf row is more than 1.5x slower | a lost `jax.jit` fast path (usually a host callback in the jitted program), a changed kernel, or a busy machine | rerun once; compare `tm.diagnostics(f, *args)['kernels']` against the baseline commit's; then profile |

## Tuning constants (and where they were measured)

| constant | value | where | measured on |
|---|---|---|---|
| `MAX_LANES` | 16 | `_primitive.py` | L40S: `jacrev` of 256 outputs in chunks of 4 / 8 / 16 / 32 / 64 lanes: 229 / 191 / 175 / 176 / 325 ms |
| `LANES_MAX_LOCAL` | 512 elements | `_primitive.py` | L40S: `jacfwd` lane counts; the toy `jacfwd(grad)` went from 95 ms to 2.3 ms once a 6 KB stack frame was split |
| `FOLD_MAX_BC` | 4096 elements | `_primitive.py` | L40S: a folded batch against `lax.map`, 1.2 to 3.4x (the fold rule itself is the device's resident-thread count, not a constant: `docs/design.md` section 2) |
| `BLOCK` (cuda) | 128 threads | `backends/cuda/kernel.py` | L40S: 64, 128 and 256 within 1% |
| `PRIV_BYTES` | 4096 bytes per block | `backends/cuda/kernel.py` | 2080 Ti: random walk 7.9 ms against 36.4 and barrier 14.4 against 56.6 (forward, 4M items) with privatisation off |
| `UNROLL` / `MAX_LOCAL` | 32 / 16384 elements | `backends/cuda/codegen.py` | L40S and 2080 Ti: compile time against local memory |
| `LAZY_GATHER` | 16 elements | `backends/cuda/op_table.py` | L40S and 2080 Ti: at 0 (every gather materialised) LUCiD's forward needs 108,160 B of local memory per thread instead of 68,048 and DTRAX's 255,840 B, and both fail to launch (out of memory) |
| `rev_checkpoints` default (`T3_MIN_EQNS`, `AUTO_C_SMALL`, `AUTO_CKPT_BYTES`, `LEAN_C`) | 'auto': 16 for a step under 1000 equations, else up to 64 within 64 KiB; 16 at second order | `_reverse.py` | L40S: lumen SK / SBND / ANNIE (306-318 B states) 1.3-1.6x faster at 64; `vs_jax` (8-25 B) 1.5-3x slower at 64 |
| `sparse_min` | 32 elements | `_reverse.py` | L40S, toy walk |
| `MIN_BUCKET` (tree adjoint) | 256 | `_reverse_spawn.py` | L40S |
| xla `BLOCK` (`xla_batch_size` default) | 2^18 items | `backends/xla.py` | CPU and L40S, memory bound |
| `EAGER_CACHE` | 64 programs | `_primitive.py` | the eager-call jit cache size |

None was measured on A100 or H200. The CUDA grid is the occupancy-derived persistent grid of the device, not a
constant.

## Caches

- **tallymap's own cubin cache** (`$TALLYMAP_CUDA_CACHE`, else `~/.cache/tallymap`): NVRTC output keyed by the
  CUDA source, the compiler build, the architecture, the options and the tallymap, jax and jaxlib versions. It
  also holds the built FFI shim. Safe to delete. The cached `.cu` next to each `.cubin` is what
  `tm.diagnostics()` points at.
- **JAX's persistent compilation cache** (`jax_compilation_cache_dir`) caches XLA executables, and is
  complementary: the CUDA kernels are registered with the FFI shim at lowering time, so a process that loads a
  cached XLA executable still needs tallymap's cache, or it rebuilds.

## Known limits

- **`jax.export` of a CUDA program is not portable.** Its custom call is not on JAX's export allow-list
  (`DisabledSafetyCheck.custom_call("tallymap_cuda_launch")` exports it anyway) and the cubin registry is per
  process, so a deserialised CUDA program only runs in a process that built the same kernels. Programs on the xla backend
  export normally.
- **shard_map**: the primitive follows shard_map's varying-axes typing. If a JAX release changes that type before
  the adapter does, `shard_map(..., check_vma=False)` (`check_rep=False` on 0.5.3) is the escape hatch.

## Profiling

```python
with jax.profiler.trace("/tmp/tm-trace"):            # a TensorBoard or Perfetto trace of one call
    with jax.profiler.TraceAnnotation("grad"):
        jax.block_until_ready(g(theta))
```

The kernels appear as launches of the `tallymap_cuda_launch` custom call. For a kernel itself, use `nsys profile`
or `ncu` on a script that calls the jitted function a few times; `tm.diagnostics(f, *args)['kernels']` gives the
cached CUDA source of each kernel. The perf safety net measures `jax.jit` WALL time on purpose: only wall time
shows a lost C++ fast path, which is worth an order of magnitude (the toy gradient goes from 2.3 ms to 42 ms with
one host callback in the program).

## The A/B rule

Before comparing two arms of an option, for timings or for numerics, assert that they compiled to different
programs: `assert_arms_differ(f_a, f_b, *args)` from `tests/ab_rule.py` (on CUDA it compares the cubin hashes,
on xla the StableHLO). Comparing two identical programs measures noise. Before a negative control, measure that the injected
defect changes the observable at all.

## Tests and CI

`python tests/run_all.py` is the entry point. `pytest` is not the runner: the suites run as separate processes,
because each needs fresh JAX state.

| flag | what it adds |
|---|---|
| `--gpu` | the CUDA backend as well as xla, the spawn tree adjoint, `tests/ops.py` (per-operation parity) and `tests/mulchain.py` |
| `--x64` | `jax_enable_x64` |
| `--unfused` | the two-node JVP design, which is what runs if JAX drops the partial-evaluation registries |
| `--canary` | DeprecationWarning and FutureWarning become errors: the JAX-upgrade canary |
| `--only a,b` | a subset of the suites |

| cadence | where | cells |
|---|---|---|
| every push and pull request | GitHub Actions, `.github/workflows/ci.yml` | ruff; the short suite on Python 3.10 + jax 0.5.3 and on Python 3.12 + the newest jax; the full CPU suite on the newest |
| weekly | GitHub Actions, `.github/workflows/weekly.yml` | the full CPU suite on every minor in `TESTED_JAX`; a JAX nightly cell, allowed to fail |
| weekly, and before a release | a GPU machine | the full suite on each supported GPU at the floor and at the newest JAX, once with `--x64`, plus `--unfused` and `--canary` on CPU |
| weekly, and before a release | a GPU machine | `benchmarks/perf.py` against `benchmarks/perf_baseline.json` (more than 1.5x slower is a failure; `benchmarks/perf_negative_control.py` is its negative control) |

GitHub Actions has no GPU, so the GPU cells run wherever the maintainer has one; each cell is one run of
`tests/run_all.py --gpu` on a checkout of one commit. Run them from a snapshot of a commit, not from a live
working tree, so that editing the tree cannot change a queued run.

## Release checklist

1. The full matrix is green on ONE commit: the CPU minors of `TESTED_JAX`, the GPU cells, `tests/ops.py` and
   `benchmarks/perf.py`.
2. For anything in the release declared "no behaviour change": prove it rather than asserting it. Identical
   StableHLO, identical generated CUDA sources and identical cubins between the previous release and this commit,
   with outputs bitwise equal wherever the previous release's own reruns are bitwise equal. Two reruns are a weak
   control for atomics-ordered float sums, so for those identical programs are the proof, not identical numbers.
3. If `backends/cuda` math or the operation table changed (`tests/ops.py` reports CHANGED rows): rerun the
   `compare` protocol unchanged (`docs/validation.md`), then re-record `tests/ops_baseline.json` in a commit of
   its own. Re-record `benchmarks/perf_baseline.json` only in a commit of its own, and only when a change in
   speed is intended.
4. Update `CHANGELOG.md`: API changes, behaviour changes, and the measured cost of anything that got slower.
5. Bump `tallymap.__version__` and `CITATION.cff`; tag.
6. Build and check the distributions (`python -m build`, then `twine check dist/*`), confirm the sdist carries
   `tests/`, `examples/`, `benchmarks/` and `docs/`, and publish.
7. State in the release notes which GPU and JAX cells passed, so a user can tell what "tested" covered.

## Watch item: hijax

`jax.experimental.hijax.HiPrim` (JAX 0.11 and later) maps one-to-one onto this library's node design (expand,
jvp, lin, transpose, batch, dce, remat, nospec) and would replace the legacy registries. Port only when it
removes BOTH the partial-evaluation registries and the transpose plumbing, and when the floor has reached 0.11
and hijax has left experimental, or when the deprecation canary fires. Budget one to two days, and keep one
implementation rather than two.

## Adopting the library in a larger program

Keep the adapter in the calling program, never here: an optional import, a flag that defaults to off, goldens
unchanged by default, and a shadow mode in continuous integration that runs both paths and compares per item with
the `compare` criterion against a FIXED reference. Validate gradients as per-output vectors, away from the truth,
with common-random-number finite differences. Time with `jax.jit` wall time, not ahead-of-time compilation.
