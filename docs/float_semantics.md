# Float semantics

Every backend runs the same per-item program.  This page states what "the same result" means across backends,
math modes, schedules and JAX versions.

## Guarantees

1. **RNG is bit-exact.**  `jax.random` inside an item (threefry2x32, typed and raw keys) is emitted bitwise by the
   cuda backend; the lockstep backend runs JAX's own ops.  `tests/ops.py` checks bits / uniform / split / randint
   bitwise against `jax.jit(jax.vmap(op))`.
2. **Per-item results do not depend on scheduling.**  Each item is a pure function of its inputs; step refill, block
   size, the item count N and the claim order do not change an item's trajectory on the cuda backend.
3. **f32 sums agree within float reassociation.**  Sum outputs are accumulated with atomics (cuda) or a scatter-add
   (lockstep; on a GPU also atomic), so their summation order, and their last bits, change from run to run on a GPU.
   CPU runs are deterministic.
4. **`Sum(exact=True)` is order-independent and exact.**  Each value is rounded to int64 fixed point
   (`round(v * scale)`) before an integer add: bitwise results across schedules, `vmap`, and the two backends.  On the
   lockstep backend it needs `jax_enable_x64`.  Unrepresentable values are dropped and counted
   (`stats['exact_overflow']`), and the exact outputs of that call are NaN (a resource limit, like `max_steps`).
5. **Forward, tangent and adjoint share one item program.**  The adjoint replays the same jaxpr with the same options
   (same math), so a replay is bit-exact on the same backend; the replay fingerprint checks it on every reverse call.

## Arithmetic per backend

- **lockstep**: whatever XLA emits for the platform (it is plain JAX).
- **cuda, `cuda_math='approx'` (default)**: f32 division, sqrt and rsqrt use the PTX that XLA:GPU itself emits
  (`div.full.f32`, `sqrt.approx.f32`, `rsqrt.approx.f32`).  Everything else is as in `ieee`.  The justification for
  the default is that sentence: it is the arithmetic a plain JAX program on the same GPU already gets, so choosing
  `ieee` makes the cuda backend differ from XLA:GPU, not agree with it.  It became the default after the
  pre-registered `compare` validation (`validation.md`), because IEEE division and square root were the whole gap
  to a hand-tuned kernel.
- **cuda, `cuda_math='ieee'`**: correctly rounded f32 add / sub / mul / div / sqrt (round to nearest), rsqrt as
  rcp(sqrt) (two roundings), the portable reference.  Use it when portability matters more than speed.
- **Both cuda modes**: no FMA contraction (`-fmad=false`), no flush-to-zero (`--ftz=false`); transcendentals call the
  CUDA math library (libdevice) with its documented maximum ulp errors; f64 arithmetic is IEEE.  No attempt is made
  to reproduce XLA's own algebraic rewrites or polynomial approximations, so cuda and lockstep agree statistically,
  not bitwise.
- `cuda_math` is an option only (no environment default: a process-global numerics switch is a hidden input).  A
  driver frozen with the old default passes `cuda_math='ieee'` explicitly.

`tests/ops.py` states the per-operation bounds (bitwise for exact ops; <= 1 ulp for ieee arithmetic; 2-3 ulp for
the approx PTX; the CUDA Programming Guide bounds for libdevice) and records the measured numbers per GPU and JAX
version in `tests/ops_baseline.json`.

## Rules for users

- Set `jax_default_matmul_precision='highest'` for any GPU parity test: JAX's default TF32 matmul on A100 / L40S
  biases results.
- For run-to-run bitwise sums on a GPU, use `Sum(exact=True)`.
- Compare per-output vectors, never scalar ratios.  Rerun the same keys as a control before blaming a difference.
- `grad` and `jacfwd` (or two backends) agree to rounding only when the derivative is well conditioned.  Through
  an adaptive step-size controller it is not.  An ODE ensemble's f32 gradient had the opposite sign to its f64
  one, and the two modes agreed only where XLA happened to round both programs identically.  Stop-gradient the
  controller's factor (README, "An adaptive step-size controller").  An f64 run is the test.
- Pass the evaluation point as a `jit` argument (`validation.md`).
