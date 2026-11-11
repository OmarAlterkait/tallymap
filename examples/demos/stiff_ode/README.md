# A stiff ODE ensemble, Kvaerno3 per item

N trajectories of a forced Van der Pol oscillator with a per-trajectory stiffness, solved with Kvaerno3 (an
implicit ESDIRK with a chord root finder and a PI step-size controller), and a smooth path functional tallied and
differentiated with respect to three parameters.  The incumbent is [diffrax](https://github.com/patrick-kidger/diffrax)
**0.7.2** (with optimistix 0.1.0), `diffrax.diffeqsolve(..., Kvaerno3(), PIDController(rtol=1e-4, atol=1e-6))`
under `jax.vmap`.

The item is `kv3.py`, a re-implementation of what diffrax does for one trajectory, **not diffrax's own code**; it
agrees with diffrax to the solver tolerance but takes a different number of steps.  `run.py` runs it as a tallymap
item and under `jax.vmap` + `lax.while_loop` (same code, same step counts), and diffrax when it is installed; it
checks the tally and the gradient against diffrax (or against `jacfwd` without it) and times the arms at one size.

## Measured

One NVIDIA L40S, JAX 0.11.2, diffrax 0.7.2: **96.6-274x** per item-solver-step against `diffrax.diffeqsolve` with
Kvaerno3 (**42.5-138x** raw wall time), and **19.5-126x** on the reverse-mode gradient.

Where it loses, or the margin shrinks: the item takes 0.73x diffrax's step count, which is why the per-step figure
is the one to quote.  tallymap's gradient costs 83x its own forward pass; the memory margin is only 2-17x, because
diffrax's adjoint is good.

## Run

```
pip install -r requirements.txt   # diffrax + optimistix; without them the diffrax arm is skipped
python run.py                     # CPU, N = 256: checks and a timing, prints PASS/FAIL
python run.py --n 262144          # on a GPU
```
