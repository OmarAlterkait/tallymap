# BlackJAX samplers over many chains, and a gradient through one

Two of BlackJAX's kernels have a data-dependent inner loop, so under `jax.vmap` across chains every chain pays
the slowest chain's work at every draw.  This demo runs those kernels unmodified, once the way BlackJAX's own
multi-chain how-to shows and once with one chain per tallymap item, and measures both.  It also takes
`jax.grad` through a short elliptical-slice chain, which plain JAX cannot do at all.

## The incumbent

[blackjax](https://github.com/blackjax-devs/blackjax) **1.7.1**, pinned in `requirements.txt`.  The demo
imports it and calls `blackjax.nuts(...).step`, `blackjax.elliptical_slice(...).step` and
`blackjax.window_adaptation`; nothing is vendored or re-implemented.

- **NUTS.** One transition is two nested `lax.while_loop`s, the doubling loop
  (`blackjax/mcmc/trajectory.py:719`) and the sub-trajectory integration inside it (`trajectory.py:368`), plus a
  third in the iterative U-turn check (`blackjax/mcmc/termination.py:95`).  The number of leapfrog steps per
  draw is between 1 and `2^max_num_doublings - 1` and is data dependent.  NumPyro's `build_tree` has the same
  structure (`numpyro/infer/hmc_util.py:1063,1179`); BlackJAX's file header credits it.
- **Elliptical slice sampling.** Its kernel samples a point on a latent ellipse and then shrinks the angle
  bracket in a `lax.while_loop` until the proposal is inside the slice
  (`blackjax/mcmc/elliptical_slice.py:211-237`).

BlackJAX's own multi-chain how-to states the consequence of `vmap`-ing an adaptive kernel across chains: each
step costs as much as the slowest chain.  Its recommendation is `jax.pmap`, which is limited to the device
count.  This demo runs a single GPU, so `pmap` has one device to work with; the `bj_pmap` arm runs it anyway and
reports what it gives, rather than leaving the recommendation untested.

## The arms

For `--prog nuts` and `--prog ess`, both arms call the **same** `alg.step`:

| arm | what it is |
|---|---|
| `bj_vmap` | the documented pattern: `lax.scan` over draws of `jax.vmap(alg.step)` across chains |
| `tm` | one chain = one tallymap item; the item runs the `lax.scan` over draws of the same `alg.step` |
| `bj_pmap` | the how-to's recommendation, `jax.pmap` over devices with `vmap` inside.  One device here |

For `--prog essgrad`, the four arms are three loop lowerings of one program plus tallymap:

| arm | what it is | `jax.grad` |
|---|---|---|
| `tm` | `tm.vmap` of the item written with `lax.while_loop` | works |
| `while` | the identical `jax.vmap` + `lax.while_loop` program | **raises** |
| `scan` | the rewrite: a masked `lax.scan` of the observed maximum shrinkage count per draw | works |
| `ckpt` | `scan` with a `jax.checkpoint`-ed body | works |

## Why the comparison is fair

- Step size and inverse mass matrix come from **one** `blackjax.window_adaptation` run, shared by both arms and
  not timed, so both arms sample the same kernel.
- Both arms take the same number of draws per chain and produce the same outputs.  `--mode store` keeps every
  draw in a `Sum((chains, draws, dim))`; `--mode expect` keeps only the first two moments.  Storing every draw
  turns out to cost nothing, which disposes of the idea that MCMC does not fit because the output is a
  trajectory.
- Both arms report the kernel's own work counter (`num_integration_steps` for NUTS, `subiter` for elliptical
  slice), so you can see that neither arm is taking a shortcut.  The tables include it.
- **The agreement contract here is statistical, not bitwise, and that is a real limitation.**  The two arms
  consume different random streams: `bj_vmap` splits one key per draw across chains, the item folds its own
  per-chain key.  So `--quick` pools each chain's draws into a chain mean, compares the two ensembles with a
  z score, compares the marginal standard deviations, and checks that the two arms' total inner-loop step counts
  agree to Monte-Carlo noise.  It cannot check more than that, and this README does not claim more.
- For `essgrad` the contract **is** mechanical: all four arms are the same program, the padded scan is padded to
  the observed maximum shrinkage count (`--quick` checks that the padded arm takes exactly as many shrinkage
  steps as the `while_loop` arm), and tallymap's gradient equals the padded rewrite's.
- The chain in `essgrad` is written out in `run.py` rather than taken from BlackJAX.  That is deliberate: the
  point of that program is the comparison between three loop lowerings of one function, and `--prog ess` is the
  one that runs BlackJAX's kernel.  The README of a reader's own sampler is `--prog ess`.

## Measured

Measured on one **NVIDIA L40S (gpu)**, JAX **0.11.2**, Python 3.12.14, blackjax 1.7.1, jax 0.11.2, tallymap **0.4.0**, an unreleased development build (commit `2061bb4b8695` of the private pre-release history).

Median `jax.jit` wall time per call after warm-up, one measurement per process so the peak-memory counter is clean.  Every number below is a row of `results/` in this folder.

### NUTS, forward

8-dimensional target, 32 draws per chain, `max_num_doublings=8`.

| chains | tallymap (ms) | blackjax + jax.vmap (ms) | x vs best baseline | leapfrogs, tallymap | leapfrogs, blackjax | leapfrog ratio |
|---|---|---|---|---|---|---|
| 256 | 0.598 | 16.6 | **27.8x** | 50,966 | 51,268 | 0.9941 |
| 1,024 | 0.604 | 16.8 | **27.8x** | 204,400 | 204,876 | 0.9977 |
| 4,096 | 0.608 | 20.2 | **33.3x** | 818,604 | 817,818 | 1.0010 |
| 16,384 | 1.65 | 23.9 | **14.5x** | 3,275,854 | 3,274,950 | 1.0003 |

The how-to's `jax.pmap` recommendation, at 4,096 chains on this single GPU: 20.2 ms against `jax.vmap`'s 20.2 ms. With one device there is nothing for it to spread over, which is exactly why it is not an answer on one GPU.


### NUTS: where the per-chain state stops fitting

NUTS carries two `(max_num_doublings, dim)` U-turn checkpoint buffers
(`blackjax/mcmc/termination.py:49-53`) plus about nine dim-vectors, so the per-chain state grows as roughly
`(2 x doublings + 9) x dim` floats.  tallymap's cost is driven by per-item state size, so this sweep is the
boundary of the whole demo.  **Read the ratio column only**: each dimension is adapted on its own target, so the
absolute times are not comparable across rows.

| dimension | approx. floats of per-chain state | tallymap (ms) | blackjax + jax.vmap (ms) | x | tallymap lower + kernel build (s) |
|---|---|---|---|---|---|
| 2 | 42 | 0.216 | 9.93 | **46x** | 4.6 |
| 8 | 168 | 0.38 | 10.3 | **27.2x** | 9.8 |
| 16 | 336 | 1.01 | 11 | **11x** | 22.8 |
| 32 | 672 | 5.2 | 12.1 | **2.33x** | 5.6 |
| 64 | 1,344 | 13.2 | 13.9 | **1.05x** | 12.7 |

The compile column is not monotone in the dimension, because the CUDA kernel cache is warm for some of these
item shapes and cold for others; read it as "seconds, not milliseconds", not as a trend.


### Elliptical slice sampling, forward

10-dimensional target, 64 draws per chain.

| chains | tallymap (ms) | blackjax + jax.vmap (ms) | x vs best baseline | shrinkage steps, tallymap | shrinkage steps, blackjax | ratio | tallymap peak MB | blackjax peak MB |
|---|---|---|---|---|---|---|---|---|
| 1,024 | 0.558 | 20.8 | **37.2x** | 409,409 | 410,407 | 0.9976 | 64.1 | 80 |
| 4,096 | 0.566 | 32.2 | **56.8x** | 1,641,678 | 1,644,725 | 0.9981 | 64.2 | 80.2 |
| 16,384 | 0.594 | 35.7 | **60.1x** | 6,569,600 | 6,569,247 | 1.0001 | 64.8 | 80.8 |
| 65,536 | 1.69 | 39.2 | **23.3x** | 26,293,879 | 26,276,707 | 1.0007 | 67.3 | 179 |


### A gradient through the sampler

The capability result first, verbatim from the sweep (`--quick` runs the same check):

> ```
> jax.grad of the `vmap` + `while_loop` arm
> ValueError: Reverse-mode differentiation does not work for lax.while_loop or lax.fori_loop with dynamic start/stop values. Try using lax.scan, or using fori_loop with static start/stop.
> ```

So a BlackJAX user who wants this gradient today must rewrite the kernel as a padded, masked scan.  Here is what
tallymap costs against that rewrite.

Forward:

| chains | tallymap (ms) | vmap + while_loop (ms) | padded scan (ms) | padded scan + remat (ms) | x vs best baseline | shrinkage steps (all arms) |
|---|---|---|---|---|---|---|
| 4,096 | 0.356 | 7.17 | 4.06 | 4.09 | **11.4x** | 360,493 |
| 16,384 | 0.367 | 7.35 | 4.02 | 4.03 | **11x** | 1,445,040 |
| 65,536 | 0.408 | 9.67 | 4.17 | 4.22 | **10.2x** | 5,782,570 |

Reverse-mode gradient:

| chains | tallymap (ms) | vmap + while_loop (ms) | padded scan (ms) | padded scan + remat (ms) | x vs best baseline | tallymap peak MB | padded scan peak MB | + remat peak MB |
|---|---|---|---|---|---|---|---|---|
| 4,096 | 0.636 | **ERROR** | 5.64 | 8.8 | **8.87x** | 64.1 | 80.2 | 80.2 |
| 16,384 | 0.798 | **ERROR** | 6.7 | 9.49 | **8.4x** | 64.5 | 129 | 80.6 |
| 65,536 | 2.06 | **ERROR** | 9.65 | 11 | **4.68x** | 66 | 514 | 82.5 |

The `vmap` + `while_loop` column is the error above, not a number: JAX cannot reverse-differentiate that program at all.




**Estimator caveat, stated and not fixed.**  The slice acceptance is a hard threshold, so this pathwise
derivative misses a boundary term.  What is demonstrated is the mechanics: tallymap's gradient is the exact
gradient of the program as written, and equals the padded rewrite's.  Whether that is an unbiased estimator of
`d E[loss] / d theta` is the user's problem, exactly as it is for the padded rewrite.

## Where tallymap loses, or the margin shrinks

- **The margin is gone by dimension 64, and this is the boundary of the whole demonstration.**  1.05x at dimension 64
  against 46x at dimension 2. tallymap's cost is driven by per-item state size, and NUTS carries two
  `(max_num_doublings, dim)` U-turn buffers plus about nine dim-vectors. The practical rule for this domain: tallymap
  is for **many chains on a small-to-moderate model** (hierarchical models with a handful of global parameters,
  calibration problems, the many-short-chains regime), not for a 500-parameter neural-network posterior, where
  `jax.vmap` is the right tool and the per-step gradient is dense linear algebra anyway.
- **Raising `max_num_doublings` costs tallymap even when the extra depth is never used.**  NUTS's two U-turn
  checkpoint buffers are shaped `(max_num_doublings, dim)` (`blackjax/mcmc/termination.py:49-53`), so raising the cap
  grows the per-item state whether or not the sampler ever reaches that depth. Set the cap from the depths you
  actually observe rather than leaving it at its default of 10 just in case.
- **Compile time is the honest price of the NUTS numbers.**  tallymap lowers the NUTS item in 1.8 s (trace plus the
  NVRTC build of the whole sampler) plus 0.13 s of XLA, against 0.34 s plus 1.09 s for blackjax. The dimension-16 item
  took 22.8 s to lower. A cold cubin cache adds more. If you are drawing a few thousand samples once, that dominates;
  if you are calling the same shape in a loop, it does not.
- **Memory is not the argument for the two forward programs.**  At 16,384 chains NUTS is 80 MB against 97 MB, both
  near the device allocator's floor. The memory result in this folder is the gradient table, where the padded
  rewrite's peak grows with the chain count and tallymap's does not.
- **The agreement contract is statistical.**  The two arms consume different random streams, so `--quick` can only
  check that the posteriors agree to Monte-Carlo error and that the two arms do the same number of inner-loop steps.
  It cannot check the draws one by one, and this README does not claim it can.
- **The gradient program is not blackjax's kernel.**  `--prog essgrad` writes its elliptical-slice chain out in
  `run.py`, because the point there is the comparison between three loop lowerings of one function. The speed-up for
  blackjax's own kernel is `--prog ess`, which is forward only.

## How to rerun

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install tallymap

python run.py --quick     # CPU, about a minute: both kernels, the statistical agreement checks, PASS/FAIL
python run.py --full      # one GPU, regenerates results/nuts_blackjax.json and plots/
```
