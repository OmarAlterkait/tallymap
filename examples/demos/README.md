# Demos: tallymap against the libraries people use

Each folder takes a real library's own code or its own problem, runs it as that library ships it, and runs the
same computation through tallymap.  Nothing is vendored: every demo imports the incumbent from a pinned
release.  Where a demo re-implements an algorithm instead of calling it, or has no incumbent library at all,
its README says so in its first section.

Demos are optional.  They are not in the default test suite, they are not installed with the package, and each
folder carries its own `requirements.txt`.

## The demos

| demo | incumbent, pinned | measured on one L40S, JAX 0.11.2 | where it loses, or the margin shrinks |
|---|---|---|---|
| [`nuts_blackjax`](nuts_blackjax/) | blackjax 1.7.1 | NUTS **14.5-33.3x** and elliptical slice **23.3-60.1x** against the documented `jax.vmap`-over-chains pattern, with the same kernel and the same number of inner-loop steps. `jax.grad` through an elliptical-slice chain, which plain JAX cannot take at all, and **4.7-8.9x** against the padded rewrite you have to write instead. | The margin is **1.05x by dimension 64** and 2.3x at 32: NUTS's per-chain state outgrows the register budget. Compile time is seconds, up to 23 s for the dimension-16 item. |
| [`gymnax_rollouts`](gymnax_rollouts/) | gymnax 1.0.0 | **17.7-25.9x** against the better of gymnax's own `RolloutWrapper` and a `vmap` + `while_loop` rewrite (**29-39x** against the wrapper itself), with bitwise-identical returns, length histogram and per-episode outputs. | **9.3x** under a policy where every episode runs the full horizon, which is the no-divergence floor. Memory is not a win here; both arms sit near the allocator's floor. |
| [`gillespie_crn`](gillespie_crn/) | crn-jax 0.3.1 | **24.6-56.3x** against the library as shipped, running the library's own `simulate_interval` inside the item, with the mean trajectory agreeing to one f32 ulp and the terminal state bitwise equal for 100% of replicates. **44-128x less device memory**. | **16.7x** at zero extrinsic noise. There is no pathwise gradient to win (it is exactly zero for an SSA); the gradient table is the +25-35% cost of the score estimator instead. |
| [`radiative_transfer`](radiative_transfer/) | Ray-trax @ 8934794 (source, no release) | On Ray-trax's own steady-state problem: **10.2-24.4x** forward and **9.6-19.5x** for the inverse-problem gradient, **114x** forward and **243x less memory** at 64^3 voxels. A volumetric Monte-Carlo walk is **19.3-49.4x** forward with **17-252x less** gradient memory than a padded scan. | Against a careful plain-JAX rewrite rather than the library it is **3.5-6.9x**, and that rewrite uses *less* forward memory than tallymap. tallymap's f32 atomics are the least accurate arm; `Sum(exact=True)` fixes it for free. |
| [`stiff_ode`](stiff_ode/) | diffrax 0.7.2 + optimistix 0.1.0 | **96.6-274x** per item-solver-step against `diffrax.diffeqsolve` with Kvaerno3 (**42.5-138x** raw wall time), and **19.5-126x** on the reverse-mode gradient. | The item re-implements Kvaerno3 rather than calling diffrax, and takes 0.73x its step count, which is why the per-step table is the one to quote. tallymap's gradient costs 83x its own forward pass; the memory margin is only 2-17x, because diffrax's adjoint is good. |
| [`sdf_inverse_rendering`](sdf_inverse_rendering/) | none (plain JAX; no JAX sphere tracer ships) | **108-180x** on the inverse-rendering gradient against a checkpointed padded scan, and the unchecked padded scan runs out of memory at 1,048,576 rays where tallymap needs 100 MB. | Forward is only **4.8-11.1x**: the march is 21 steps on average, so the pad is cheap. The gradient is a biased estimator of the shape gradient in every arm, because a silhouette is a discontinuity. |
| [`adaptive_quadrature`](adaptive_quadrature/) | quadax 0.3.0 | **2.8-18.1x** forward against `quadax.quadgk`, **12.4x** on the gradient with respect to the integrand amplitude, and **98-380x less** device memory for that gradient. | Two different algorithms (locally adaptive Simpson against globally adaptive Gauss-Kronrod), compared at the same tolerance: read it as an execution-model result, not an integration-quality one. |
| [`branching_epidemic`](branching_epidemic/) | none (plain JAX; the R tools are single-lineage CPU codes) | A spawn tree against the best plain-JAX generation-frontier program: **3.8-13.5x** forward and **4.8-7.6x** on `jacfwd` over four parameters, with the incidence tally bitwise equal. | **Reverse mode is a tie** (1.2-1.7x) and uses 2-4x *more* memory than the frontier scan, so use `jacfwd` here. Two of the four parameters have no pathwise derivative at all. |

Every number above is a row of that demo's `results/` and is reproduced in its README next to the arms it compares. Two of the eight have no incumbent library, because none exists in JAX for that problem; those compare against the best plain-JAX program instead and say so.

The demos were measured at a few different commits of this branch, because two of them were re-measured after a fairness fix. Each `results/<demo>.json` records the commit its own numbers were taken at; the commits after it change plotting and driver plumbing, not the measured programs.

## Running a demo

Every folder has the same entry points:

```
python run.py --quick     # CPU, about a minute: runs every arm, checks their answers agree, prints PASS/FAIL
python run.py --full      # the GPU sweep that regenerates results/*.json and plots/*.png
python run.py --plots     # redraw plots/ from the committed results/, no GPU and no measurement
```

`--quick` is the one to run first: it is the honesty check.  It runs every arm on the same inputs and compares
their answers, so you can see for yourself that the comparison is between two programs that compute the same
thing.  `--full` re-execs itself once per measurement so that each one gets a fresh process and the device
peak-memory counter means something.

## What all the demos share

`_demo.py` holds the measurement protocol, the results bookkeeping and the plotting, so that every demo
measures the same way:

- median `jax.jit` wall time per call after a warm-up, with `jax.block_until_ready` on the result;
- compile time measured separately and reported as `lower_s` (trace, plus any CUDA kernel build) and
  `xla_compile_s`;
- peak device memory from the device allocator's own `peak_bytes_in_use` counter, one measurement per process;
- one JSON row per measurement in `results/`, with the GPU, the JAX version, the incumbent versions and the
  tallymap commit in the same file;
- one plot per mode.  Forward time, gradient time and peak memory are never mixed in one axes, there is no
  second y axis anywhere, and identity is carried by colour and marker and a direct end label.

## How the comparisons are kept fair

- The incumbent is imported from a pinned release and called the way its own documentation says to batch.
  Where its docs recommend `jax.pmap` instead of `jax.vmap`, the demo runs that too and reports what it gives.
- Where an arm has to be padded to a fixed trip count, it is padded to the **observed maximum** at the largest
  N and the same divergence level, measured by a calibration run in the same sweep and recorded in
  `results/<demo>.json` under `meta.scan_lengths`.  Nothing is padded to an inflated cap, which is the easiest
  way to make a baseline look bad.
- Every demo carries a control arm that runs the **identical Python function** as the tallymap item under
  `jax.vmap`, so there is one column where the only difference is the execution model.
- Every "x vs best baseline" column takes the **better** of the baselines at that row, never the worse one, and
  never another tallymap arm.
- Every demo's `--quick` compares the arms' answers, and its README says what kind of agreement to expect:
  bitwise, one f32 ulp, the solver tolerance, or statistical where the two arms consume different random
  streams.
- Every README has a section on where tallymap loses or the margin shrinks, with the measurement behind it.

## What these numbers are not

- They are one GPU, one JAX version and one problem size per row.  Read the ranges, not the best row.
- Several demos have no incumbent library, because none exists in JAX for that problem.  Those compare against
  plain JAX, and their READMEs say so rather than implying a library comparison.
- A speed-up against a library is not a verdict on that library.  Ray-trax's per-ray grid, quadax's global
  interval mesh and crn-jax's scan-of-while are deliberate designs that buy accuracy, robustness and
  composability; what is measured here is one configuration of each against a different execution model.
