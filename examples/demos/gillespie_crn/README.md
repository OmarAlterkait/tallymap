# A Gillespie SSA ensemble, with crn-jax's own hot loop inside the item

A bistable gene-regulatory circuit (the LacI-TetR toggle switch) simulated exactly with the Gillespie algorithm
for 64 observation intervals over N replicates; the number of reactions per interval varies within and across
replicates.  The incumbent is [crn-jax](https://pypi.org/project/crn-jax/) **0.3.1**, whose batch entry point
`jax.vmap(simulate_trajectory)` pays, for every interval, the batch-maximum reaction count and stacks the whole
trajectory tensor.

`run.py` runs crn-jax as shipped and a tallymap item that calls the library's own `simulate_interval`,
unmodified, checks that the mean trajectory agrees and the terminal state is bitwise equal per replicate, and
times both at one size.  The only code the demo writes itself is the propensity function (theta as an argument),
checked bit for bit against the library's closure.

## Measured

One NVIDIA L40S, JAX 0.11.2, crn-jax 0.3.1, N = 16,384 to 1,048,576 replicates: **24.6-56.3x** against the
library as shipped, with the mean trajectory agreeing to one f32 ulp and the terminal state bitwise equal for
100% of replicates, in **44-128x less device memory**.

Where it loses, or the margin shrinks: **16.7x** at zero extrinsic noise.  There is no pathwise gradient to win
(it is exactly zero for an SSA); the score-function estimator costs +25-35% of the forward pass.

## Run

```
pip install -r requirements.txt   # crn_jax; without it the demo exits saying so
python run.py                     # CPU, N = 256: checks and a timing, prints PASS/FAIL
python run.py --n 262144          # on a GPU
```
