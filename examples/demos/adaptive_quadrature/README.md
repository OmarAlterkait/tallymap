# Adaptive quadrature, one integral per item

N one-dimensional integrals of `w / (1 + a (x - c)^2)` over `[0, 1]` are evaluated to an absolute and relative
tolerance of 1e-6, with the peak sharpness `a` drawn log-uniformly per item.  A sharper peak needs more
subdivisions, so the work per integral varies by a factor of a few, and the gradient with respect to the
integrand amplitude `w` is the quantity a calibration would want.

## The incumbent

[quadax](https://github.com/f0uriest/quadax) **0.3.0**, pinned in `requirements.txt`.  The demo imports it and
calls `quadax.quadgk` as it ships, under `jax.vmap`, at its default `max_ninter=50`.

`quadgk` is **globally** adaptive Gauss-Kronrod: it keeps a mesh of up to `max_ninter` intervals with their
errors and repeatedly bisects the worst one, which means four length-50 arrays of per-item state
(`quadax/adaptive.py:217-224`), about 200 floats.

## The three arms

| arm | what it is |
|---|---|
| `quadax` | `quadax.quadgk` under `jax.vmap`, `epsabs = epsrel = 1e-6`, `max_ninter=50` (its default) |
| `tm` | one integral = one tallymap item: **locally** adaptive Simpson with an explicit stack of depth 24, so 48 floats of per-item state |
| `jax_while` | the identical local adaptive Simpson under `jax.vmap` + `lax.while_loop` |

## Why the comparison is fair, and where it is not

- quadax is imported and called as documented.  Nothing is vendored.
- `jax_while` runs the **identical Python function** as the tallymap item, so that column isolates the
  execution model from the algorithm.
- **These are two different algorithms, and that is the honest caveat of this demo.**  Globally adaptive
  Gauss-Kronrod is more robust on a nasty integrand than locally adaptive Simpson: it can put its effort
  wherever the error is worst rather than refining one bracket at a time.  They are compared at the same
  requested tolerance, and `--quick` reports how far apart their answers land.  A fair reading of the numbers
  below is "the locally adaptive algorithm fits in registers and the globally adaptive one does not", not
  "tallymap integrates faster than quadax".
- quadax's 1e-6 tolerance is per integral and both arms meet it.  `tm` and `jax_while` agree bitwise; `tm`
  agrees with `quadax` to the tolerance, not to the last bit, because they are different algorithms.

## Measured

Measured on one **NVIDIA L40S (gpu)**, JAX **0.11.2**, Python 3.12.14, jax 0.11.2, quadax 0.3.0, tallymap **0.4.0**, an unreleased development build (commit `2061bb4b8695` of the private pre-release history).

Median `jax.jit` wall time per call after warm-up, one measurement per process so the peak-memory counter is clean.  Every number below is a row of `results/` in this folder.

### Forward

| integrals N | tallymap (ms) | vmap + while_loop (ms) | quadax.quadgk (ms) | x vs best baseline | subdivisions mean / max (item) | tallymap peak MB | quadax peak MB |
|---|---|---|---|---|---|---|---|
| 16,384 | 0.6 | 3.59 | 3.29 | **5.47x** | 64.6 / 121 | 64.2 | 64.4 |
| 65,536 | 2.08 | 5.81 | 8.3 | **2.79x** | 64.4 / 125 | 64.8 | 258 |
| 262,144 | 5.81 | 28.5 | 57.8 | **4.91x** | 64.4 / 123 | 67 | 901 |
| 1,048,576 | 20.3 | 368 | 403 | **18.1x** | 64.4 / 125 | 76 | 3,609 |

![forward time](plots/fwd_time.png)

### Gradient with respect to the integrand amplitude

| integrals N | tallymap (ms) | quadax.quadgk (ms) | x vs best baseline | tallymap peak MB | quadax peak MB | x less memory |
|---|---|---|---|---|---|---|
| 65,536 | 17.5 | 216 | **12.4x** | 64.8 | 6,363 | **98.3x** |
| 262,144 | 76.4 | 952 | **12.5x** | 67 | 25,491 | **380x** |

![gradient time](plots/grad_time.png)

![gradient peak memory](plots/grad_mem.png)

quadax's `max_ninter=50` default is what makes its gradient need 24.9 GB at 262,144 integrals: it carries a global interval mesh per item and reverse mode keeps it. A user can lower `max_ninter`, which would cut that memory and change the algorithm's robustness; this demo leaves the library at its default rather than tuning it down to flatter the comparison.

### The divergence sweep

`--div` sets the range the per-item peak sharpness is drawn from, which is what makes the subdivision count vary.

| div (at N = 262,144) | subdivisions mean / max | fwd tm / while / quadax (ms) | fwd x vs best | grad tm / quadax (ms) | grad x |
|---|---|---|---|---|---|
| low | 45.9 / 83 | 4.2 / 19.3 / 46.4 | **4.6x** | 68.4 / 931 | **13.6x** |
| mid | 64.4 / 123 | 5.81 / 28.5 / 57.8 | **4.91x** | 76.4 / 952 | **12.5x** |
| high | 81.3 / 145 | 7.27 / 33.7 / 75 | **4.64x** | 84.9 / 987 | **11.6x** |

## Where tallymap loses, or the margin shrinks

- **These are two different algorithms, and that is the biggest caveat here.**  quadax is globally adaptive Gauss-
  Kronrod and the item is locally adaptive Simpson. They are run at the same requested tolerance and their answers
  agree to that tolerance, not to the last bit. The work counters are not comparable either: the item records 64.4
  subdivisions on average and quadax reports 69.3 function evaluations, which is a different quantity. Read the
  numbers as "the locally adaptive algorithm fits in registers and the globally adaptive one does not", not as
  "tallymap integrates faster than quadax".
- **The forward margin is small at small N.**  5.47x at 16,384 integrals against 18.1x at 1,048,576. At the small end
  this is a modest win over `vmap` + `while_loop` running the identical function, and the headline number at the large
  end comes mostly from the baselines degrading, not from tallymap improving.
- **tallymap's own gradient is expensive relative to its forward pass.**  76.4 ms against 5.8 ms forward at 262,144
  integrals, about 13x. The replay adjoint re-executes the subdivision loop; the margin over quadax is 12.5x, which is
  far less than the memory margin of 380x.
- **Compile time, which happens not to be a penalty here.**  tallymap's gradient lowers in 0.87 s (trace plus the
  NVRTC build of the item) plus 0.18 s of XLA, against 1.69 s plus 4.05 s for quadax: 1.05 s against 5.74 s. The NVRTC
  build is still a cost tallymap pays once per program shape and the baseline does not pay at all, and in a demo with
  a larger item it dominates; see `nuts_blackjax` and `stiff_ode`.

## How to rerun

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install tallymap

python run.py --quick     # CPU, under a minute: all three arms, the agreement checks, PASS/FAIL
python run.py --full      # one GPU, regenerates results/adaptive_quadrature.json and plots/
```
