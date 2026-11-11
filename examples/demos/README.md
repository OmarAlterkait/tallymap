# Demos: tallymap against the libraries people use

Each folder runs a real library's own code or its own problem as that library ships it, and the same computation
through tallymap.  Nothing is vendored: the incumbent is imported from a pinned release (`requirements.txt` in
each folder).  Two of the eight have no incumbent library, because none exists in JAX for that problem; those
compare against the best plain-JAX program and say so.

Demos are optional: they are not in the test suite and not installed with the package.

## The demos

| demo | incumbent, pinned | measured on one L40S, JAX 0.11.2 | where it loses, or the margin shrinks |
|---|---|---|---|
| [`nuts_blackjax`](nuts_blackjax/) | blackjax 1.7.1 | NUTS **14.5-33.3x** and elliptical slice **23.3-60.1x** against the documented `jax.vmap`-over-chains pattern, with the same kernel and the same number of inner-loop steps. `jax.grad` through an elliptical-slice chain, which plain JAX cannot take at all, and **4.7-8.9x** against the padded rewrite you have to write instead. | The margin is **1.05x by dimension 64** and 2.3x at 32: NUTS's per-chain state outgrows the register budget. Compile time is seconds, up to 23 s for the dimension-16 item. |
| [`gymnax_rollouts`](gymnax_rollouts/) | gymnax 1.0.0 | **17.7-25.9x** against the better of gymnax's own `RolloutWrapper` and a `vmap` + `while_loop` rewrite (**29-39x** against the wrapper itself), with bitwise-identical returns, length histogram and per-episode outputs. | **9.3x** under a policy where every episode runs the full horizon, which is the no-divergence floor. Memory is not a win here; both arms sit near the allocator's floor. |
| [`gillespie_crn`](gillespie_crn/) | crn-jax 0.3.1 | **24.6-56.3x** against the library as shipped, running the library's own `simulate_interval` inside the item, with the mean trajectory agreeing to one f32 ulp and the terminal state bitwise equal for 100% of replicates. **44-128x less device memory**. | **16.7x** at zero extrinsic noise. There is no pathwise gradient to win (it is exactly zero for an SSA); the gradient table is the +25-35% cost of the score estimator instead. |
| [`radiative_transfer`](radiative_transfer/) | Ray-trax @ 8934794 (source, no release) | On Ray-trax's own steady-state problem: **10.2-24.4x** forward and **9.6-19.5x** for the inverse-problem gradient, **114x** forward and **243x less memory** at 64^3 voxels. A volumetric Monte-Carlo walk is **19.3-49.4x** forward with **17-252x less** gradient memory than a padded scan. | Against a careful plain-JAX rewrite rather than the library it is **3.5-6.9x**, and that rewrite uses *less* forward memory than tallymap. tallymap's f32 atomics are the least accurate arm; `Sum(fixed_point=True)` fixes it for free. |
| [`stiff_ode`](stiff_ode/) | diffrax 0.7.2 + optimistix 0.1.0 | **96.6-274x** per item-solver-step against `diffrax.diffeqsolve` with Kvaerno3 (**42.5-138x** raw wall time), and **19.5-126x** on the reverse-mode gradient. | The item re-implements Kvaerno3 rather than calling diffrax, and takes 0.73x its step count, which is why the per-step table is the one to quote. tallymap's gradient costs 83x its own forward pass; the memory margin is only 2-17x, because diffrax's adjoint is good. |
| [`sdf_inverse_rendering`](sdf_inverse_rendering/) | none (plain JAX; no JAX sphere tracer ships) | **108-180x** on the inverse-rendering gradient against a checkpointed padded scan, and the unchecked padded scan runs out of memory at 1,048,576 rays where tallymap needs 100 MB. | Forward is only **4.8-11.1x**: the march is 21 steps on average, so the pad is cheap. The gradient is a biased estimator of the shape gradient in every arm, because a silhouette is a discontinuity. |
| [`adaptive_quadrature`](adaptive_quadrature/) | quadax 0.3.0 | **2.8-18.1x** forward against `quadax.quadgk`, **12.4x** on the gradient with respect to the integrand amplitude, and **98-380x less** device memory for that gradient. | Two different algorithms (locally adaptive Simpson against globally adaptive Gauss-Kronrod), compared at the same tolerance: read it as an execution-model result, not an integration-quality one. |
| [`branching_epidemic`](branching_epidemic/) | none (plain JAX; the R tools are single-lineage CPU codes) | A spawn tree against the best plain-JAX generation-frontier program: **3.8-13.5x** forward and **4.8-7.6x** on `jacfwd` over four parameters, with the incidence tally bitwise equal. | **Reverse mode is a tie** (1.2-1.7x) and uses 2-4x *more* memory than the frontier scan, so use `jacfwd` here. Two of the four parameters have no pathwise derivative at all. |

The numbers are one GPU, one JAX version and one problem size per row: read the ranges, not the best row.  They
were measured with the full sweep driver (per-measurement processes, peak device memory, results JSON and plots)
that these folders carried up to tag `v0.6.0rc1`; check out `examples/demos/` at that tag to reproduce them.  A
speed-up against a library is not a verdict on it: Ray-trax's per-ray grid, quadax's global interval mesh and
crn-jax's scan-of-while are deliberate designs that buy accuracy, robustness and composability.

## Running a demo

```
cd examples/demos/<demo>
pip install -r requirements.txt     # the incumbent
python run.py                       # CPU, small N: runs the arms, checks they agree, times them, prints PASS/FAIL
python run.py --n 262144            # the same on a GPU at a larger size
```

`run.py` is the honesty check: it runs the tallymap item and the incumbent (or plain-JAX) arm on the same inputs
and compares their answers, bitwise, to the solver tolerance, or statistically where the arms consume different
random streams.  Where the item itself calls the library (`gillespie_crn`, `gymnax_rollouts`) a missing library
makes the demo exit with a message saying which one it needs; elsewhere only the library's arm is skipped, and the
tallymap item is still checked against the identical function under `jax.vmap` or a plain-JAX rewrite.
