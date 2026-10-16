"""An ensemble: Metropolis chains with adaptive stopping, per-item results, and a varying number of active items.

    python examples/ensemble_toy.py

Each chain samples a 1-D Gaussian of width sigma with random-walk Metropolis and stops after it has accepted
`need` moves (so every chain runs a different number of steps).  Per-item results: the final position and the
number of steps.  The tallies: the total acceptance count and a histogram of the final positions.
`n_active=k` runs only the first k chains of a fixed-capacity batch: changing k never recompiles.
"""
import jax
import jax.numpy as jnp
from jax import lax

import tallymap as tm

NB = 20


def chain(t, x0, key, sigma):
    need = 25

    def body(c):
        x, n, acc, t = c
        k = jax.random.fold_in(key, n)
        u = jax.random.uniform(k, (2,))
        y = x + (u[0] - 0.5) * 2.0
        a = jnp.log(u[1]) < (x * x - y * y) / (2 * sigma * sigma)
        t = t.add("accepted", 0, a.astype(jnp.float32))
        return jnp.where(a, y, x), n + 1, acc + a.astype(jnp.int32), t
    x, n, acc, t = lax.while_loop(lambda c: c[2] < need, body, (x0, 0, 0, t))
    t = t.add("hist", jnp.clip(((x + 5.0) * (NB / 10.0)).astype(jnp.int32), 0, NB - 1), 1.0)
    return t, (x, n)


f = jax.jit(tm.vmap(chain, outputs={"accepted": tm.Sum((1,)), "hist": tm.Sum((NB,))}, in_axes=(0, 0, None),
                    max_steps=10_000))

CAP = 4096                                    # fixed capacity
x0 = jnp.zeros((CAP,), jnp.float32)
keys = jax.random.split(jax.random.PRNGKey(0), CAP)
sigma = jnp.float32(1.5)

r_all = f(x0, keys, sigma, n_active=CAP)
r_half = f(x0, keys, sigma, n_active=CAP // 2)
x, n = r_all.per_item
print("steps per chain: min", int(n.min()), " max", int(n.max()), " mean", float(n.mean()))
print("final-position std", float(jnp.std(x)), "(target sigma", float(sigma), ")")
print("accepted (all / half)", float(r_all.outputs["accepted"][0]), float(r_half.outputs["accepted"][0]))
xh, nh = r_half.per_item
assert int(r_all.stats["n_truncated"]) == 0
assert bool(jnp.all(nh[CAP // 2:] == 0)) and bool(jnp.all(nh[:CAP // 2] == n[:CAP // 2]))  # inactive chains: no-ops
assert float(r_all.outputs["accepted"][0]) == 25 * CAP and float(r_half.outputs["accepted"][0]) == 25 * CAP // 2
print("ENSEMBLE_TOY OK")
