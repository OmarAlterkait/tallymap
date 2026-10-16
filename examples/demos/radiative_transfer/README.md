# Radiative transfer: Ray-trax's own problem, and a volumetric Monte-Carlo walk

Two programs in one folder, because radiative transfer is done both ways.

`--prog ray` is the head-to-head: the mean intensity J on a G^3 grid from one point source, computed by marching
`num_rays` Fibonacci-lattice directions with a semi-analytic per-step update.  It is a quadrature, so there is no
RNG, every arm computes the same deterministic function, and the derivative is exact and unbiased: there is no
estimator question here at all.

`--prog mc` is a volumetric Monte-Carlo transport walk through a scattering and absorbing medium, which is the
shape of every Monte-Carlo transport code.  A photon packet's number of collisions has a geometric tail, so the
batch maximum sits far above the mean.

## The incumbents

**`--prog ray`: [Ray-trax](https://github.com/lorenzobranca/Ray-trax)** (arXiv:2511.09389), a GPU-oriented,
fully differentiable 3-D ray tracer in JAX for the time-dependent emission-absorption transfer equation on
regular grids.  The demo imports the library's own `compute_radiation_field_from_source` and its own
`sample_sphere` direction set and calls them unmodified.

Ray-trax has no packaging (no `pyproject.toml`, no PyPI release), so it cannot go in `requirements.txt`.  Clone
it at the commit the numbers were measured against and put it on `PYTHONPATH`:

```
git clone https://github.com/lorenzobranca/Ray-trax.git
git -C Ray-trax checkout 89347946fba4eb47817ea6e7ab76ec8ff9117614
export RAYTRAX_PATH=$PWD/Ray-trax        # or pass --raytrax <dir>
```

Its per-ray kernel (`ray_trax/core.py:126-195`) is a `lax.fori_loop` of a **static** number of steps; per step it
interpolates the emissivity trilinearly, looks up the opacity at the nearest voxel, updates (I, tau)
semi-analytically and deposits I trilinearly onto a grid.  **The deposit target is a full grid allocated per
ray** (`core.py:191`); rays are batched with `jax.vmap` and summed over the ray axis (`core.py:202-240`), with
an optional `ray_batch_size` micro-batching knob whose stated purpose is to cap that peak memory.  So the
library's memory is `O(rays x voxels)` by default, and that, rather than its arithmetic, is what the numbers
below are mostly about.

**`--prog mc`: no incumbent, stated plainly.**  The codes that do this in production (MCX and MMC for
biomedical photon transport, MYSTIC, SMART-G and gCMCRT for atmospheric and astrophysical transfer, Mitsuba 3's
`volpath` and `prbvolpath` for graphics, gPMC and moqui for proton dose) are none of them JAX libraries, so a
head-to-head would measure CUDA against JAX rather than two execution models.  The baselines there are a
faithful plain-JAX port of the same item function.

## The arms

`--prog ray`:

| arm | what it is |
|---|---|
| `raytrax` | the library's public `compute_radiation_field_from_source` with `ray_batch_size=None`, its default: one `vmap` over all rays |
| `raytrax_b` | the same call with the library's own `ray_batch_size` set so one micro-batch of per-ray grids stays under 1 GB, which is the setting its docstring recommends when memory is tight |
| `jax_scan_shared` | the best plain-JAX rewrite: ONE shared grid in a `lax.scan` carry with the rays vectorised inside the step, so `O(grid)` memory instead of `O(rays x grid)`, still padded to the horizon |
| `jax_ckpt` | `jax_scan_shared` with a recursively checkpointed (nested, remat'd) scan |
| `tm` | item = one ray, tally = `Sum((G^3,))`, and the ray's own `while_loop` stops when it leaves the box |
| `tm_exact` | the same with `Sum(..., exact=True)`, the int64 fixed-point accumulator |

`--prog mc`: `tm`, `jax_while` (forward only), `jax_scan` (padded to the observed maximum), `jax_ckpt`.

## Why the comparison is fair

- The two library arms call the library.  The march step for every other arm is written once in `run.py` and
  all of them call it, so the only difference between those columns is the loop.
- The library's `4 pi / N` normalisation is divided out, so every arm returns the same quantity.
- The tallymap item stops at the box exit.  Every arm's deposits are identically zero after that, so all arms
  compute the same function.  `--quick` checks that against an **f64 run of the same program**, which is the
  arbiter rather than any one f32 arm.
- The horizon is swept.  `--div low` pads to the observed longest exit, which is the fairest possible pad;
  `mid` is 2x and `high` is 4x that.  All three are below the library's own shipped default of
  `max_steps=500` on a 32^3 grid.
- Both library arms are reported, including the micro-batched one, so the library's own memory remedy is in the
  table rather than omitted.
- For `--prog mc`, the plain-JAX arms use ONE shared tally in the loop carry with the photons vectorised inside
  the step.  That is the fair shape: `jax.vmap` of an item with its own dense `(Nvox,)` tally would need
  `N x Nvox` floats, 16 GB at N = 1M and Nvox = 4,096, and is not a program anybody would write.
- `--prog mc` agreement is bitwise on the deposition tally and the escaping flux, and the per-photon collision
  count is identical for 100% of photons.  The physics is a ratio-tracking scheme with a fixed majorant and a
  fixed roulette survival probability, so the path does not depend on the medium and the pathwise derivative is
  both the exact derivative of the estimator and an unbiased estimator of the derivative.

## Measured

Measured on one **NVIDIA L40S (gpu)**, JAX **0.11.2**, Python 3.12.14, jax 0.11.2, tallymap **0.4.0**, an unreleased development build (commit `2061bb4b8695` of the private pre-release history).

Median `jax.jit` wall time per call after warm-up, one measurement per process so the peak-memory counter is clean.  Every number below is a row of `results/` in this folder.

Ray-trax at commit `89347946fba4` (github.com/lorenzobranca/Ray-trax).

### Ray-trax's problem, forward

32^3 grid (32,768 voxels), horizon padded to the observed longest exit.

| rays | tallymap (ms) | tallymap, exact=True (ms) | Ray-trax (shipped) (ms) | Ray-trax (micro-batched) (ms) | plain-JAX shared scan (ms) | shared scan + remat (ms) | x vs best baseline | x vs Ray-trax | march steps mean / max |
|---|---|---|---|---|---|---|---|---|---|
| 1,024 | 0.125 | 0.145 | 1.27 | 1.27 | 0.86 | 0.875 | **6.89x** | 10.2x | 38.4 / 54 |
| 4,096 | 0.159 | 0.163 | 2.62 | 2.62 | 0.953 | 0.967 | **6x** | 16.5x | 38.4 / 54 |
| 16,384 | 0.382 | 0.314 | 7.5 | 8.48 | 1.5 | 1.52 | **3.91x** | 19.6x | 38.4 / 54 |
| 65,536 | 1.11 | 0.784 | 27.1 | 33 | 3.9 | 3.92 | **3.52x** | 24.4x | 38.4 / 54 |

Peak device memory on the same runs:

| rays | tallymap (MB) | tallymap, exact=True (MB) | Ray-trax (shipped) (MB) | Ray-trax (micro-batched) (MB) | plain-JAX shared scan (MB) | shared scan + remat (MB) | x less than Ray-trax |
|---|---|---|---|---|---|---|---|
| 1,024 | 64.5 | 64.5 | 273 | 273 | 0.794 | 0.795 | **4.23x** |
| 4,096 | 64.5 | 64.5 | 545 | 545 | 0.923 | 0.924 | **8.44x** |
| 16,384 | 64.7 | 64.7 | 2,113 | 1,073 | 1.44 | 1.44 | **32.7x** |
| 65,536 | 65.3 | 65.3 | 8,305 | 1,073 | 5.5 | 5.5 | **127x** |

![forward time](plots/ray_fwd_time.png)

![forward peak memory](plots/ray_fwd_mem.png)

### Ray-trax's problem, reverse-mode gradient

A weighted least-squares loss against a target field, differentiated with respect to **both** 3-D fields, so
65,536 parameters at G = 32.

| rays | tallymap (ms) | tallymap, exact=True (ms) | Ray-trax (shipped) (ms) | Ray-trax (micro-batched) (ms) | plain-JAX shared scan (ms) | shared scan + remat (ms) | x vs best baseline | x vs Ray-trax | tallymap peak MB | Ray-trax peak MB | shared scan peak MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1,024 | 0.338 | 0.361 | 3.25 | 3.2 | 2.34 | 2.98 | **6.92x** | 9.62x | 64.5 | 273 | 9.01 |
| 4,096 | 0.386 | 0.395 | 5.53 | 5.53 | 2.75 | 3.35 | **7.12x** | 14.3x | 64.5 | 545 | 33 |
| 16,384 | 0.749 | 0.697 | 14.6 | 18 | 4.94 | 5.61 | **6.6x** | 19.5x | 64.7 | 2,192 | 129 |
| 65,536 | 2.77 | 2.45 | 48.1 | 71.7 | 11 | 11.1 | **3.96x** | 17.3x | 65.3 | 8,735 | 294 |

![gradient time](plots/ray_grad_time.png)

![gradient peak memory](plots/ray_grad_mem.png)

### The voxel-count sweep: the O(rays x voxels) wall

16,384 rays.

| voxels | fwd tm / Ray-trax / batched / shared scan (ms) | grad tm / Ray-trax / batched / shared scan (ms) | fwd peak tm / Ray-trax (MB) | x less memory |
|---|---|---|---|---|
| 16^3 = 4,096 | 0.365 / 1.93 / 1.95 / 1.95 | 0.724 / 6.86 / 6.83 / 5.73 | 64 / 304 | **4.74x** |
| 32^3 = 32,768 | 0.382 / 8.52 / 10.3 / 2.95 | 0.746 / 19.5 / 26.3 / 9.94 | 65 / 2,113 | **32.7x** |
| 64^3 = 262,144 | 0.463 / 52.9 / 106 / 5.02 | 1.15 / 87.9 / 243 / 14.4 | 69 / 16,789 | **243x** |

![forward time vs voxels](plots/ray_grid_fwd.png)

### The horizon sweep

16,384 rays, 32^3 grid.  tallymap's time should not move, because the item stops at the box exit; the padded
arms pay the pad.

| horizon | pad (steps) | tallymap (ms) | Ray-trax (ms) | plain-JAX shared scan (ms) |
|---|---|---|---|---|
| 1x (the observed longest exit) | 55 | 0.382 | 7.5 | 1.5 |
| 2x | 109 | 0.382 | 8.52 | 2.95 |
| 4x | 216 | 0.382 | 10.5 | 7.17 |

### Volumetric Monte-Carlo transport, forward

16^3 grid, 4,096 voxels.

| photons | tallymap (ms) | vmap + while_loop (ms) | padded scan (ms) | padded scan + remat (ms) | x vs best baseline | collisions mean / max | pad (steps) | items truncated |
|---|---|---|---|---|---|---|---|---|
| 65,536 | 0.26 | 5.03 | 13.7 | 13.8 | **19.3x** | 9.5 / 94 | 384 | 0 |
| 262,144 | 0.655 | 13.6 | 43.5 | 43.6 | **20.7x** | 9.5 / 99 | 384 | 0 |
| 1,048,576 | 1.86 | 48 | 165 | 166 | **25.8x** | 9.5 / 105 | 384 | 0 |
| 4,194,304 | 6.68 | 330 | 1,000 | 1,000 | **49.4x** | 9.5 / 124 | 384 | 0 |

![Monte-Carlo forward time](plots/mc_fwd_time.png)

### Volumetric Monte-Carlo transport, reverse-mode gradient

With respect to both fields (8,192 parameters) and the three global parameters.

| photons | tallymap (ms) | padded scan (ms) | padded scan + remat (ms) | x vs best baseline | tallymap peak MB | padded scan peak MB | + remat peak MB | x less memory than the padded scan |
|---|---|---|---|---|---|---|---|---|
| 65,536 | 0.934 | 36.2 | 39.3 | **38.8x** | 66.8 | 1,136 | 131 | **17x** |
| 262,144 | 9.29 | 97.6 | 97.5 | **10.5x** | 74.1 | 4,542 | 522 | **61.3x** |
| 1,048,576 | 54.8 | 384 | 427 | **7.01x** | 72.1 | 18,172 | 1,780 | **252x** |

![Monte-Carlo gradient time](plots/mc_grad_time.png)

![Monte-Carlo gradient peak memory](plots/mc_grad_mem.png)

## Where tallymap loses, or the margin shrinks

- **Against the plain-JAX shared scan, rather than against the library, the margin is single digits.**  At 65,536 rays
  and 32^3 voxels the shared scan takes 3.90 ms forward against tallymap's 1.11 ms (3.52x) and 11.0 ms for the
  gradient against 2.77 ms (3.96x). The large numbers against Ray-trax come mostly from its per-ray grid allocation,
  which a rewrite removes. Both arms are in every table above for exactly that reason, and the "x vs best baseline"
  column uses the shared scan, not the library.
- **The shared scan uses LESS device memory than tallymap in the forward pass.**  5.5 MB against tallymap's 65 MB at
  65,536 rays. One shared `(G^3,)` grid in a scan carry is about as small as this problem gets, and tallymap's peak
  here does not grow with the ray count either (65 MB at 1,024 rays). The memory argument against Ray-trax, which
  allocates a grid per ray, holds at hundreds of times; the memory argument against a careful rewrite does not.
- **f32 atomics are the least accurate arm.**  `--quick` compares every arm against an f64 run of the same program.
  tallymap's global f32 atomic sums and the plain-JAX shared scan are the least accurate arms there, and Ray-trax's
  tree-reduced per-ray grids are the most accurate, which is the other side of not materialising a per-ray grid.
  `Sum(exact=True)` matches Ray-trax's accuracy, and here it costs nothing: 0.784 ms against 1.11 ms forward at 65,536
  rays.
- **The Monte-Carlo gradient margin falls as N grows.**  7.8x at 1,048,576 photons against 42.1x at 65,536. The memory
  margin goes the other way and is the durable one.
- **What the Ray-trax numbers are not.**  Its per-ray grid is a deliberate design that buys a tree-reduced, accurate
  sum and clean vectorisation over rays, sources and frequency bins, and the library solves the time-dependent problem
  and multi-frequency, multi-species opacities, which this demo does not touch. What is measured here is two execution
  models on one of its configurations, not a verdict on the library.
- **No incumbent for the Monte-Carlo program.**  The production codes for volumetric transport are CUDA, not JAX, so
  the `--prog mc` baselines are a plain-JAX port of the same item. Those numbers say what tallymap buys over plain
  JAX, not what it buys over MCX or Mitsuba.

## How to rerun

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install tallymap
git clone https://github.com/lorenzobranca/Ray-trax.git
git -C Ray-trax checkout 89347946fba4eb47817ea6e7ab76ec8ff9117614
export RAYTRAX_PATH=$PWD/Ray-trax

python run.py --quick     # CPU, about a minute: both programs, every arm, the f64 arbiter, PASS/FAIL
python run.py --full      # one GPU, regenerates results/radiative_transfer.json and plots/
```
