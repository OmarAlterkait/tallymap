# gymnax episodic rollouts

CartPole-v1 episodes are run to termination for a linear policy, and the summed return, a 64-bin histogram of
episode lengths and the per-episode (return, length) are collected: exactly what an evolution-strategies or CEM
fitness evaluation needs.  A CartPole episode ends when the pole falls, which is somewhere between 10 and 500
steps depending on the policy, so the per-episode work varies a lot.

## The incumbent

[gymnax](https://github.com/RobertTLange/gymnax) **1.0.0**, pinned in `requirements.txt`.  The demo imports it
and calls its own `env.reset` and `env.step`; nothing is vendored or re-implemented.

`gymnax.experimental.rollout.RolloutWrapper.single_rollout` (`gymnax/experimental/rollout.py:72-108`) runs one
`lax.scan` of `env_params.max_steps_in_episode` steps per episode whatever happens, and multiplies the reward
by a `valid_mask` that goes to zero after `done`.  `batch_rollout` and `population_rollout` are `jax.vmap` of
that over keys and over policy parameters.  On CartPole-v1 `max_steps_in_episode` is 500, so an episode that
falls over after 20 steps still costs the full 500-step horizon.

## The three arms

| arm | what it is |
|---|---|
| `gx_scan` | gymnax's own structure: `jax.vmap` over episodes of a fixed 500-step `lax.scan` with the valid mask |
| `gx_while` | the obvious plain-JAX improvement: `jax.vmap` of a `lax.while_loop` that stops on `done`.  Still lockstep, so it costs the batch maximum episode length |
| `tm` | the same `while_loop` as a tallymap item: one episode is one item |

## Why the comparison is fair

- All three arms call the same `env.reset` and `env.step` from the installed gymnax, with the same policy and
  the same per-episode PRNG key, consumed in the same order.
- `gx_while` is the control that matters: its loop body is character for character the body of the tallymap
  item, so the only difference between those two columns is the execution model.
- There is no padding to choose here.  The fixed horizon is gymnax's own `max_steps_in_episode`, and the two
  `while_loop` arms use it as their stopping condition, so all three arms compute the same function.
- gymnax's rollout wrapper is `vmap`-based and its README shows no `pmap` path for the batched rollout, so
  there is no documented alternative batching to show.
- The three arms agree **bitwise**: the summed return, the length histogram, every per-episode return and every
  per-episode length are identical.  That is possible here because an episode is an integer-step program with
  no float reduction order to disagree about.  `--quick` checks it at every divergence level.

## Measured

Measured on one **NVIDIA L40S (gpu)**, JAX **0.11.2**, Python 3.12.14, gymnax 1.0.0, jax 0.11.2, tallymap **0.4.0**, an unreleased development build (commit `2061bb4b8695` of the private pre-release history).

Median `jax.jit` wall time per call after warm-up, one measurement per process so the peak-memory counter is clean.  Every number below is a row of `results/` in this folder.

### Forward

| episodes | tallymap (ms) | gymnax RolloutWrapper (ms) | vmap + while_loop (ms) | x vs best baseline | x vs gymnax | episode length mean / max |
|---|---|---|---|---|---|---|
| 4,096 | 0.193 | 7.24 | 3.91 | **20.2x** | 37.5x | 28.7 / 140 of 500 |
| 16,384 | 0.211 | 8.25 | 5.47 | **25.9x** | 39.1x | 28.8 / 187 of 500 |
| 65,536 | 0.336 | 9.86 | 6.56 | **19.5x** | 29.3x | 28.8 / 187 of 500 |
| 262,144 | 0.661 | 21.9 | 11.7 | **17.7x** | 33.2x | 28.8 / 212 of 500 |

Peak device memory on the same runs:

| episodes | tallymap (MB) | gymnax RolloutWrapper (MB) | vmap + while_loop (MB) |
|---|---|---|---|
| 4,096 | 64 | 80.1 | 80.1 |
| 16,384 | 64.1 | 80.3 | 80.3 |
| 65,536 | 64.5 | 81 | 81 |
| 262,144 | 66 | 85 | 85 |



### The policy sweep, including a zero-divergence control

`--div none` is a policy under which **every** episode runs the full 500-step horizon, so there is no variation
in episode length at all.  It separates the two effects that make up the margin: with zero divergence the only
thing left is register residency, a per-thread kernel keeping the episode state out of HBM.

| policy | length mean / max | tallymap (ms) | gymnax RolloutWrapper (ms) | vmap + while_loop (ms) | x vs best baseline |
|---|---|---|---|---|---|
| none (every episode runs the full horizon) | 500.0 / 500 | 1.06 | 9.8 | 17 | **9.25x** |
| low | 86.5 / 500 | 0.774 | 9.79 | 17 | **12.6x** |
| mid | 28.8 / 187 | 0.336 | 9.86 | 6.56 | **19.5x** |
| high | 22.3 / 154 | 0.279 | 9.84 | 5.38 | **19.3x** |


### Gradients

This demo has no gradient table, and the reason is the model rather than the library.  The policy's action is a
hard threshold on a logit, and the episode length is an integer, so the pathwise derivative of the return with
respect to the policy weights is zero almost everywhere; the estimators people actually use here are
evolution strategies and policy gradients, both of which difference or score the *forward* pass.  That forward
pass is what this demo measures.

## Where tallymap loses, or the margin shrinks

- **A policy that never terminates early.**  At `--div none` every episode runs the full 500-step horizon, so there is
  no divergence at all, and the margin falls to 9.25x against the better baseline. That is the floor of this
  demonstration. What is left there is register residency, a per-thread kernel keeping the episode state out of HBM;
  the rest of the margin at `--div mid` is scheduling.
- **The margin is not monotone in N.**  It is 20.2x at 4,096 episodes, peaks at 25.9x at 16,384, and falls to 17.7x at
  262,144. Neither end of this sweep is an asymptote, so quote the range rather than the best row.
- **Memory is not the story here.**  Every arm sits near the device allocator's floor at these sizes (66 MB against 85
  MB at 262,144 episodes), so the memory table above says little. The memory argument for tallymap is in the demos
  with a per-item tally or a gradient, not in this one.
- **Compile time is a fixed cost, and here it is not a penalty.**  tallymap lowers in 0.41 s (trace plus the NVRTC
  build of the item) and XLA takes 0.13 s, a total of 0.54 s at 262,144 episodes, against 0.09 s plus 0.82 s, 0.91 s,
  for gymnax's own scan. The CartPole item is small enough that the kernel build is cheap. The demos with a large item
  (`nuts_blackjax`, `stiff_ode`) are where compile time does become a real cost, and their READMEs report it.
- **`vmap` + `while_loop` is not automatically the baseline to beat.**  At `--div none` and `--div low` it is *slower*
  than gymnax's fixed scan (17.0 vs 9.8 ms at `none`), because a data-dependent XLA `while_loop` costs more than a
  static scan at the same trip count. Every "x vs best baseline" column in this folder takes the better of the two
  baselines, never the worse one.

## How to rerun

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install tallymap

python run.py --quick     # CPU, under a minute: every arm, the agreement checks, PASS/FAIL
python run.py --full      # one GPU, regenerates results/gymnax_rollouts.json and plots/
```

`--full` re-execs `run.py --one ...` once per measurement so that each one gets a fresh process and the device
peak-memory counter is clean.  `results/gymnax_rollouts.json` records the GPU, the JAX version, the gymnax
version and the tallymap commit alongside the numbers.
