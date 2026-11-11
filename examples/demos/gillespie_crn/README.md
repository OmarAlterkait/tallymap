# A Gillespie SSA ensemble, with crn-jax's own hot loop inside the item

A bistable gene-regulatory circuit (the Gardner-Cantor-Collins / Lugagne-2017 LacI-TetR toggle switch) is
simulated exactly with the Gillespie stochastic simulation algorithm, for 64 observation intervals of half a
minute, over N independent replicates.  The quantity wanted is the ensemble mean trajectory plus a terminal
two-species histogram, and the number of reactions in an interval is Poisson, so the per-replicate work varies
both within a replicate and across replicates.

## The incumbent

[crn-jax](https://pypi.org/project/crn-jax/) **0.3.1** (`pip install crn_jax`), a pure-JAX GPU Gillespie
library for chemical reaction networks, pinned in `requirements.txt`.  The demo imports the library's own
`gillespie.simulate_interval` and `gillespie.simulate_trajectory` and calls them unmodified.

Its shipped batch entry point is `crn_jax.models._common.make_vmap_simulator` (`models/_common.py:43-89`):

```
jax.jit(jax.vmap(simulate_one))              # simulate_one -> gillespie.simulate_trajectory
  -> lax.scan over n_steps fixed intervals   (gillespie.py:309 / 333)
     -> gillespie.simulate_interval          (gillespie.py:194)
        -> gillespie.simulate_until          (gillespie.py:75)
           -> lax.while_loop over reactions  (gillespie.py:185)
```

A batch of N replicates therefore pays, **for every interval**, the maximum reaction count over the N
replicates, and the outer scan stacks the whole `(N, n_steps, n_species)` trajectory tensor.

## The four arms

| arm | what it is |
|---|---|
| `tm` | the tallymap item: a `lax.fori_loop` over intervals calling crn-jax's `simulate_interval`, tallying the mean into `Sum((64, 2))` |
| `crnjax` | the library **as shipped**: `jax.vmap(simulate_trajectory)`, stacking the trajectory tensor and reducing it |
| `crnjax_lean` | the best plain-JAX program: `jax.vmap` of a scan that calls the same `simulate_interval` and accumulates only the running sum, with no stacked tensor |
| `jax_scan` | `crnjax_lean` with the inner `while_loop` replaced by a masked `lax.scan` padded to the observed maximum reactions per interval.  This is what you must write for any gradient, since `lax.while_loop` has no transpose |

## Why the comparison is fair

- Three of the four arms call the library's own `simulate_interval` or `simulate_trajectory`.  Nothing is
  vendored.
- crn-jax's models close over a frozen dataclass of kinetic parameters, so differentiating with respect to them
  needs them as traced arrays.  `props_toggle` in `run.py` rebuilds the same propensity expression
  (`models/toggle_switch.py:89-103`) with theta as an argument, and `--quick` checks it against the library's
  own closure **bit for bit**.  That is the only piece of the model this demo writes itself.
- `jax_scan` is padded to the maximum reactions per interval **measured** at the same N and divergence level
  (`--mode calib` does the measuring and the number is in `results/`), not to crn-jax's own `max_steps` guard
  of 512.  Padding it to the guard would make it look far worse than it is.
- The divergence knob is the library's own per-replicate `input` channel, used for a lognormal production
  multiplier.  That is how cell-to-cell (extrinsic) variability is modelled, not a synthetic knob.
- Agreement: the mean trajectory agrees to one f32 ulp and the terminal state is bitwise equal for 100% of
  replicates against all three plain-JAX arms.  `--quick` checks it.

## Measured

Measured on one **NVIDIA L40S (gpu)**, JAX **0.11.2**, Python 3.12.14, crn_jax 0.3.1, jax 0.11.2, tallymap **0.4.0**, an unreleased development build (commit `3848ef41fe0b` of the private pre-release history).

Median `jax.jit` wall time per call after warm-up, one measurement per process so the peak-memory counter is clean.  Every number below is a row of `results/` in this folder.

### Forward

Padded-scan length 185 reactions, which is the batch maximum per interval measured at 1,048,576 replicates (the `calib` rows in `results/`). The mean is 12.6 reactions per interval, so the pad costs about 15x the mean work.

| replicates N | tallymap (ms) | crn-jax as shipped (ms) | crn-jax loop, sum only (ms) | padded scan (ms) | x vs best baseline | x vs crn-jax as shipped |
|---|---|---|---|---|---|---|
| 16,384 | 4.42 | 249 | 258 | 457 | **56.3x** | 56.3x |
| 65,536 | 6.26 | 308 | 308 | 516 | **49.2x** | 49.3x |
| 262,144 | 15.1 | 445 | 446 | 736 | **29.4x** | 29.4x |
| 1,048,576 | 51.1 | 1,257 | 1,277 | 2,301 | **24.6x** | 24.6x |

Peak device memory on the same runs:

| replicates N | tallymap (MB) | crn-jax as shipped (MB) | crn-jax loop, sum only (MB) | padded scan (MB) | x less than crn-jax |
|---|---|---|---|---|---|
| 16,384 | 0.504 | 64.4 | 64.4 | 64.4 | **128x** |
| 65,536 | 2 | 129 | 97.5 | 97.5 | **64.5x** |
| 262,144 | 8 | 356 | 293 | 293 | **44.5x** |
| 1,048,576 | 32 | 1,400 | 624 | 628 | **43.7x** |



### The divergence sweep

`--div` sets the lognormal spread of the per-replicate production multiplier, which is what makes the batch
maximum reaction count pull away from the mean.

| extrinsic spread (at N = 262,144) | reactions per interval, mean / batch max | padded-scan length | tm / crn-jax / padded scan (ms) | x vs best baseline |
|---|---|---|---|---|
| 0 | 13.3 / 41 | 41 | 9.96 / 173 / 166 | **16.7x** |
| 0.25 | 13.3 / 47 | 47 | 10.5 / 203 / 190 | **18.1x** |
| 0.8 | 12.6 / 185 | 185 | 15.1 / 445 / 736 | **29.4x** |
| 1.6 | 10.8 / 1,547 | 1547 | 45.6 / 3,043 / 6,083 | **66.8x** |

### Gradients: what an SSA ensemble's gradient actually is

**The pathwise gradient of an SSA ensemble is exactly zero, in every arm.**  `--quick` measures it rather than
asserting it.  That is correct, not a bug: the state only ever changes by an integer stoichiometry vector, so
the trajectory is a piecewise-constant function of the rates.  A padded-scan arm "wins" that comparison only
because XLA proves the output is a constant and deletes the whole program.  Nobody should call `jax.grad` on a
Gillespie simulator.

The estimator that does work is the score function (likelihood ratio).  crn-jax's loop cannot accumulate it,
because the score needs the propensity *derivatives* at every event and the library's carry holds only the
propensities.  `interval_scored` in `run.py` is `gillespie.py:138-191` with one extra 4-vector in the carry, and
`--quick` confirms it produces the bitwise-identical trajectory to the library's own loop and that its estimate
agrees with a common-random-number central difference of `E[obs]`.

So the gradient table here is not tallymap against crn-jax.  It is the cost of carrying the score:

| replicates N | item without the score (ms) | item with the 4-parameter score (ms) | overhead |
|---|---|---|---|
| 65,536 | 6.43 | 8.06 | **+25%** |
| 262,144 | 15.3 | 19.9 | **+30%** |
| 1,048,576 | 51.6 | 69.4 | **+35%** |


## Where tallymap loses, or the margin shrinks

- **There is no pathwise gradient to win, and the score function is the only gradient on offer.**  `--quick` measures
  the pathwise gradient of the ensemble as exactly zero in every arm, so the gradient table in this folder is the cost
  of the score estimator rather than a comparison against crn-jax. Carrying the four-parameter path score costs +25%
  of the forward pass. The score estimator's variance is its own problem: `--quick` checks it against a common-random-
  number finite difference and the agreement is within Monte-Carlo error, but the finite difference is the more
  precise estimator per pass.
- **The headline is against crn-jax's own loop, not against the padded scan.**  At `--div mid` the batch maximum is
  185 reactions per interval against a mean of 12.6, so a masked scan padded to that maximum takes 2,301 ms at
  1,048,576 replicates against crn-jax's own scan-of-while at 1,257 ms. crn-jax's loop structure is the better plain-
  JAX program and the "x vs best baseline" column uses it. The padded arm is in the table because it is what you must
  write for any gradient.
- **The margin at zero extrinsic noise is the floor.**  16.7x at spread 0 against 66.8x at spread 1.6. Even with no
  cell-to-cell variability the reaction count within a replicate is still Poisson, so some divergence remains; what is
  left on top of that is register residency.
- **crn-jax's own safety guard is shared by every arm.**  Every arm passes crn-jax's default `max_steps=10,000`. At
  `--div high` the measured batch maximum is 1,547 reactions per interval, well under the guard, so nothing is
  truncated; at a wider setting still it would be, in every arm equally.
- **The propensity function is rewritten, and only that.**  crn-jax's models close over a frozen dataclass of kinetic
  parameters, so the demo rebuilds the same propensity expression with theta as a traced argument. `--quick` checks it
  bit for bit against the library's own closure. Everything else in the hot loop is the library's code.
- **Compile time is a cost tallymap pays and the baseline does not, though on this item it is not larger.**  The
  forward pass lowers in 0.63 s (trace plus the NVRTC build of the item) plus 0.15 s of XLA, 0.78 s in total, against
  0.11 s plus 1.06 s, 1.17 s, for crn-jax as shipped. The NVRTC build still grows with the size of the item, and a
  cold cubin cache adds seconds on top of the first call.

## How to rerun

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install tallymap

python run.py --quick     # CPU, under a minute: all four arms, the agreement checks, PASS/FAIL
python run.py --full      # one GPU, regenerates results/gillespie_crn.json and plots/
```
