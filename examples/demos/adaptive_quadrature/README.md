# Adaptive quadrature, one integral per item

N integrals of `w / (1 + a (x - c)^2)` over `[0, 1]` to a tolerance of 1e-6, with the peak sharpness `a` drawn
per item, so the number of subdivisions varies.  The incumbent is [quadax](https://github.com/f0uriest/quadax)
**0.3.0**, `quadax.quadgk` under `jax.vmap` at its default `max_ninter=50`.

`run.py` runs the tallymap item (locally adaptive Simpson with an explicit stack of depth 24), the identical
function under `jax.vmap` + `lax.while_loop`, and `quadax.quadgk` when quadax is installed.  It checks them
against each other and against the closed form, and times them at one size.

## Measured

One NVIDIA L40S, JAX 0.11.2, quadax 0.3.0, N = 16,384 to 1,048,576 integrals:

- forward **2.8-18.1x** against the better of `quadax.quadgk` and `vmap` + `while_loop`;
- the gradient with respect to the integrand amplitude **12.4x** faster than quadax's, in **98-380x less** device
  memory (67 MB against 25.5 GB at 262,144 integrals).

Where it loses, or the margin shrinks: these are two different algorithms (locally adaptive Simpson against
globally adaptive Gauss-Kronrod) compared at the same tolerance, so read it as an execution-model result, not an
integration-quality one.  The forward margin is 5.5x at 16,384 integrals and grows mostly because the baselines
degrade.  tallymap's gradient costs about 13x its own forward pass.

## Run

```
pip install -r requirements.txt   # quadax; without it the quadax arm is skipped
python run.py                     # CPU, N = 4096: checks and a timing, prints PASS/FAIL
python run.py --n 262144          # on a GPU
```
