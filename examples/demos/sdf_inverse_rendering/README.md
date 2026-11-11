# Sphere tracing a signed distance field, with an inverse-rendering gradient

N camera rays sphere-trace a smooth-min union of eight spheres; the tally is a per-tile residual against a target
image, and its gradient with respect to five scene parameters is the inverse-rendering gradient.  The march length
varies strongly from ray to ray (grazing rays take many small steps).

**There is no incumbent JAX library**: differentiable sphere tracers in JAX are written by hand as a `lax.scan`
padded to a step budget, because `lax.while_loop` has no reverse-mode rule.  The baseline is that program, padded to
the observed maximum march length.

`run.py` runs one ray per tallymap item and the padded scan, checks that the forward pass agrees bitwise and the
gradient to f32 rounding (and `jacfwd` against `grad`), and times the gradient at one size.

## Measured

One NVIDIA L40S, JAX 0.11.2, N = 16,384 to 1,048,576 rays: **108-180x** on the inverse-rendering gradient against a
checkpointed padded scan; the unchecked padded scan runs out of memory at 1,048,576 rays, where tallymap needs
100 MB.

Where it loses, or the margin shrinks: forward is only **4.8-11.1x**, because the march is 21 steps on average,
so the pad is cheap.  The gradient is a biased estimator of the shape gradient in every arm, because a silhouette
is a discontinuity.

## Run

```
python run.py               # CPU, N = 4096: checks and a timing, prints PASS/FAIL
python run.py --n 1048576   # on a GPU
```
