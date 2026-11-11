# Radiative transfer, one ray per item

Ray-trax's own steady-state problem: rays from a point source march through a 3-D grid, solving the
emission-absorption transfer equation and depositing the intensity trilinearly onto the grid; the inverse problem
differentiates a loss on that field with respect to the emissivity and opacity fields.  The incumbent is
[Ray-trax](https://github.com/lorenzobranca/Ray-trax) at commit `8934794` (a source repository with no release),
whose per-ray kernel is a fixed-length `lax.fori_loop` that deposits into a full grid allocated per ray.

`run.py` runs one ray per tallymap item (a `while_loop` that stops when the ray leaves the box), the best plain-JAX
rewrite (one shared grid in a `lax.scan` carry, padded to the longest exit path), and Ray-trax's
`compute_radiation_field_from_source` when Ray-trax is importable.  It checks the field and the gradient with
respect to both fields against each other and times the forward pass at one size.

## Measured

One NVIDIA L40S, JAX 0.11.2: on Ray-trax's own problem **10.2-24.4x** forward and **9.6-19.5x** for the
inverse-problem gradient, **114x** forward and **243x less memory** at 64^3 voxels.  A volumetric Monte-Carlo walk
(in the full sweep at tag `v0.6.0rc1`, see `../README.md`) is **19.3-49.4x** forward with **17-252x less**
gradient memory than a padded scan.

Where it loses, or the margin shrinks: against the careful plain-JAX rewrite rather than the library it is
**3.5-6.9x**, and that rewrite uses *less* forward memory than tallymap.  tallymap's f32 atomics are the least
accurate arm; `Sum(fixed_point=True)` fixes that at no cost.

## Run

```
git clone https://github.com/lorenzobranca/Ray-trax.git
git -C Ray-trax checkout 89347946fba4eb47817ea6e7ab76ec8ff9117614
export RAYTRAX_PATH=$PWD/Ray-trax   # without it the Ray-trax arm is skipped
python run.py                       # CPU, 2048 rays on a 16^3 grid: checks and a timing, prints PASS/FAIL
python run.py --n 65536             # on a GPU
```
