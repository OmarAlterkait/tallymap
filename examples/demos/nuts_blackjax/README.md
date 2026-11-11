# BlackJAX samplers over many chains, and a gradient through one

Two of BlackJAX's kernels have a data-dependent inner loop (NUTS's tree doubling, elliptical slice sampling's
bracket shrinking), so under `jax.vmap` across chains every chain pays the slowest chain's work at every draw.
The incumbent is [blackjax](https://github.com/blackjax-devs/blackjax) **1.7.1**, called unmodified.

`run.py` runs `blackjax.nuts` the way BlackJAX's multi-chain how-to shows (`jax.vmap(alg.step)` across chains)
and with one chain per tallymap item, and checks that the posteriors agree to Monte-Carlo error and that both arms
take the same number of leapfrog steps (skipped without blackjax).  It also takes `jax.grad` through a short
elliptical-slice chain, which plain JAX cannot do, and checks it against the padded masked-scan rewrite you have to
write instead.  That part needs only JAX.

## Measured

One NVIDIA L40S, JAX 0.11.2, blackjax 1.7.1: NUTS **14.5-33.3x** and elliptical slice **23.3-60.1x** against
the documented `jax.vmap`-over-chains pattern, with the same kernel and the same number of inner-loop steps.
`jax.grad` through an elliptical-slice chain is **4.7-8.9x** faster than the padded rewrite.

Where it loses, or the margin shrinks: the margin is **1.05x by dimension 64** and 2.3x at 32, because NUTS's
per-chain state outgrows the register budget; tallymap is for many chains on a small-to-moderate model.  Compile
time is seconds, up to 23 s for the dimension-16 item.  The agreement check is statistical, since the two arms
consume different random streams.

## Run

```
pip install -r requirements.txt   # blackjax; without it only the gradient part runs
python run.py                     # CPU, 128 chains: checks and a timing, prints PASS/FAIL
python run.py --n 16384           # on a GPU
```
