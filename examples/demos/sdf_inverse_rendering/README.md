# A differentiable sphere tracer, and its inverse-rendering gradient

A CSG scene of eight spheres in a ring, combined with the standard polynomial smooth-min union plus a ground
plane, is rendered by sphere tracing from a pinhole camera: each ray marches `t <- t + max(d(p), 1e-4)` until
it hits, misses or runs out of steps.  The tally is an inverse-rendering residual against a target image
rendered at a different set of scene parameters, so `jax.grad` of its total is the gradient a practitioner
actually wants from a differentiable sphere tracer.

Most rays converge in about 20 steps.  A ray that grazes a silhouette takes hundreds.  That ratio, up to 24 at
the widest camera setting, is what this demo is about.

## The incumbent

**There is no incumbent library here, and that is the honest framing.**  Differentiable sphere tracers in JAX
are written by hand, and what they are written as is a `lax.scan` padded to a fixed step budget, because
`lax.while_loop` has no reverse-mode rule in JAX at all.  So the baselines are the two programs a practitioner
writes, and the padded ones are padded to the observed maximum march length rather than to a guess.

(If you want the head-to-head against a real published JAX ray tracer, that is the `radiative_transfer` demo,
which runs Ray-trax on Ray-trax's own problem.)

## The four arms

| arm | what it is | which modes |
|---|---|---|
| `tm` | item = one ray, a `lax.while_loop` that stops at the hit or the miss | forward, grad, jacfwd |
| `jax_while` | `jax.vmap` of the identical march | forward only: JAX cannot reverse-differentiate a `while_loop` |
| `jax_scan` | `jax.vmap` of the same march rewritten as a masked `lax.scan` of the observed maximum length | all |
| `jax_ckpt` | `jax_scan` with a nested, `jax.checkpoint`-ed inner scan (recursive checkpointing) | all |

## Why the comparison is fair

- The SDF, the shading, the camera and the target image are written once in `run.py` and every arm calls the
  same functions.  The only thing that differs between arms is the loop.
- `jax_while` runs the identical Python function as the tallymap item, so its column isolates the execution
  model.
- The scan length is the **observed maximum** march length at the largest N and the same divergence level,
  measured by `--mode calib` in the same sweep and recorded in `results/`.  The value for each divergence level
  is in `results/sdf_inverse_rendering.json` under `meta.scan_lengths`.  `--quick` re-measures it from the
  `while_loop` arm and sizes the padded arms from that measurement, so you can check it yourself.
- The forward pass agrees **bitwise** across all four arms: the residual tally, the hit flag and the step count
  per ray are identical.  The gradient agrees to about one part in 1e6, not bitwise, because each arm
  accumulates the five parameter cotangents in a different order.

## Estimator caveat, stated and not fixed

The hit point of a sphere-traced ray is a discontinuous function of the scene parameters at a silhouette, so
these pathwise gradients are **biased** estimators of the true shape gradient.  tallymap returns the exact
derivative of the program as written, and all three differentiable arms agree, so this is a property of sphere
tracing and not of the library.  The standard fixes (edge sampling, reparameterisation) sit in the user's
renderer, exactly as they do for the padded-scan arm.

## Measured

Measured on one **NVIDIA L40S (gpu)**, JAX **0.11.2**, Python 3.12.14, jax 0.11.2, tallymap **0.4.0**, an unreleased development build (commit `7df17110d12f` of the private pre-release history).

Median `jax.jit` wall time per call after warm-up, one measurement per process so the peak-memory counter is clean.  Every number below is a row of `results/` in this folder.

### Forward

Scan length 241 steps, which is the longest march measured at 1,048,576 rays (the `calib` rows in `results/`).

| rays | tallymap (ms) | vmap + while_loop (ms) | vmap + padded scan (ms) | padded scan + remat (ms) | x vs best baseline | march steps mean / max | tallymap peak MB |
|---|---|---|---|---|---|---|---|
| 16,384 | 0.19 | 2.9 | 2.1 | 2.17 | **11.1x** | 20.7 / 166 | 64.5 |
| 65,536 | 0.264 | 4.85 | 1.98 | 2.12 | **7.48x** | 20.6 / 212 | 66.5 |
| 262,144 | 0.462 | 5.12 | 2.44 | 2.64 | **5.27x** | 20.6 / 206 | 59 |
| 1,048,576 | 1.13 | 8.33 | 5.44 | 5.8 | **4.8x** | 20.6 / 241 | 100 |

![forward time](plots/fwd_time.png)

### Inverse-rendering gradient with respect to five scene parameters

| rays | tallymap (ms) | vmap + padded scan (ms) | padded scan + remat (ms) | x vs best baseline | tallymap peak MB | padded scan peak MB | + remat peak MB | x less memory than + remat |
|---|---|---|---|---|---|---|---|---|
| 16,384 | 0.395 | 66.6 | 71 | **169x** | 64.5 | 1,488 | 129 | **1.99x** |
| 65,536 | 0.679 | 72 | 74.3 | **106x** | 66.5 | 5,957 | 515 | **7.74x** |
| 262,144 | 1.65 | 174 | 178 | **106x** | 59 | 23,846 | 1,651 | **28x** |
| 1,048,576 | 5.4 | **OOM** | 852 | **158x** | 100 | **OOM** | 6,597 | **66x** |

![gradient time](plots/grad_time.png)

![gradient peak memory](plots/grad_mem.png)

### The divergence sweep

`--div` sets (camera half-angle, hit tolerance).  A wider cone means more rays that graze a silhouette or miss
entirely; a tighter tolerance means more small steps at the end of a grazing march.  The scan length is the
observed maximum at each level, so the padded arms are sized correctly at every row.

| div (at N = 262,144) | scan length | march steps mean / max | fwd tm / while / scan (ms) | grad tm / remat (ms) | grad peak tm / remat (MB) |
|---|---|---|---|---|---|
| low | 114 | 23.5 / 112 | 0.444 / 2.91 / 1.26 | 1.48 / 82.7 | 59 / 1,149 |
| mid | 241 | 20.6 / 206 | 0.462 / 5.12 / 2.44 | 1.65 / 178 | 59 / 1,651 |
| high | 467 | 19.3 / 377 | 0.511 / 9.19 / 4.55 | 2.01 / 340 | 59 / 2,254 |

## Where tallymap loses, or the margin shrinks

- **The forward margin is modest, and it shrinks before it grows.**  11.1x at 16,384 rays falling to 4.8x at
  1,048,576. The padded scan is a good program for this item: the march is only 21 steps on average, so the pad costs
  a factor of a few rather than a factor of hundreds. The forward pass is not the argument for this demo; the gradient
  is.
- **`vmap` + `while_loop` is slower than the padded scan at every N here.**  8.33 vs 5.44 ms at 1,048,576 rays. "Just
  use `while_loop`" is not the baseline to beat, which is why the padded scan is the one the "x vs best baseline"
  column uses.
- **The gradient is biased, in every arm.**  The hit point of a sphere-traced ray is discontinuous in the scene
  parameters at a silhouette, so none of these pathwise gradients is an unbiased estimator of the true shape gradient.
  All three differentiable arms agree, so this is a property of sphere tracing; it is not fixed by any of them.
- **Compile time is a cost tallymap pays and the baseline does not, though on this item it is not larger.**  The
  gradient lowers in 1.86 s (trace plus the NVRTC build of the item) plus 0.44 s of XLA, 2.31 s in total, against 0.63
  s plus 4.80 s, 5.43 s, for the checkpointed scan. The NVRTC build still grows with the size of the item, and a cold
  cubin cache adds seconds on top of the first call.
- **The padded scan's memory is the real result, and it is a memory result rather than a time result.**  At 1,048,576
  rays the unchecked padded scan ran out of memory, the checkpointed one needed 6,597 MB, and tallymap needed 100 MB.
  If your step cap is small and your ray count is small, the padded scan is fine and far simpler.

## How to rerun

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install tallymap

python run.py --quick     # CPU, under a minute: all four arms, the agreement checks, PASS/FAIL
python run.py --full      # one GPU, regenerates results/sdf_inverse_rendering.json and plots/
```
