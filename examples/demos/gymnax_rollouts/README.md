# gymnax rollouts, one episode per item

Episodic rollouts of gymnax's CartPole-v1 under a noisy linear policy, as an evolution-strategies or CEM fitness
evaluation needs them: the summed return, a histogram of episode lengths and the per-episode (return, length).
The incumbent is [gymnax](https://github.com/RobertTLange/gymnax) **1.0.0**, whose
`RolloutWrapper.single_rollout` runs a fixed `max_steps_in_episode`-step `lax.scan` per episode with a validity
mask, so a short episode still costs the full 500-step horizon.

`run.py` runs gymnax's own scan structure, the obvious `vmap` + `while_loop` that stops on `done`, and the same
`while_loop` as a tallymap item, all stepping gymnax's own `env.step`; it checks that all three give bitwise
identical outputs and times them at one size.

## Measured

One NVIDIA L40S, JAX 0.11.2, gymnax 1.0.0, N = 4,096 to 262,144 episodes: **17.7-25.9x** against the better of
gymnax's own `RolloutWrapper` and a `vmap` + `while_loop` rewrite (**29-39x** against the wrapper itself), with
bitwise-identical returns, length histogram and per-episode outputs.

Where it loses, or the margin shrinks: **9.3x** under a policy where every episode runs the full horizon, which is
the no-divergence floor.  Memory is not a win here; both arms sit near the allocator's floor.

## Run

```
pip install -r requirements.txt   # gymnax; without it the demo exits saying so
python run.py                     # CPU, N = 512: checks and a timing, prints PASS/FAIL
python run.py --n 262144          # on a GPU
```
