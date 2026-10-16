"""Photon transport into sensor tallies: a calibration-style loss, its gradient (reverse) and Jacobian (forward).

    python examples/photon_toy.py

Photons start inside a unit disk and random-walk (exponential flights of mean length `lam`, isotropic scattering,
absorption exp(-mu * s)) until they leave it.  Where a photon leaves, it deposits its weight times the efficiency of
the sensor at that angle into the tally Q[sensor].  Parameters: theta = (lam, mu) and the per-sensor efficiency
table `eff` (32 entries: reverse mode gives it a sparse, atomically accumulated cotangent).
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm
from tallymap import reference

NS = 32


def photon(t, x0, key, theta, eff):
    lam, mu = theta[0], theta[1]

    def body(c):
        p, d, w, alive, k, t = c
        u = jax.random.uniform(jax.random.fold_in(key, k), (2,))
        s = -jnp.log1p(-u[0]) * lam
        p2 = p + s * d
        out = jnp.sum(p2 * p2) >= 1.0
        i = ((jnp.arctan2(p2[1], p2[0]) + jnp.pi) * (NS / (2 * jnp.pi))).astype(jnp.int32) % NS
        w2 = w * jnp.exp(-mu * s)
        t = t.add("Q", jnp.where(out, i, -1), w2 * eff[i])          # index -1: dropped (no deposit this step)
        phi = 2 * jnp.pi * u[1]
        return p2, jnp.stack([jnp.cos(phi), jnp.sin(phi)]), w2, (~out) & (w2 > 1e-3), k + 1, t
    c = (x0, jnp.array([1.0, 0.0]), jnp.float32(1.0), jnp.bool_(True), 0, t)
    p, d, w, alive, k, t = lax.while_loop(lambda c: c[3], body, c)
    return t, k


opts = dict(outputs={"Q": tm.Sum((NS,))}, in_axes=(0, 0, None, None), max_steps=200)
f = tm.vmap(photon, **opts)

N = 20_000
rng = np.random.default_rng(0)
x0 = jnp.asarray(rng.uniform(-0.5, 0.5, (N, 2)), jnp.float32)
keys = jax.random.split(jax.random.PRNGKey(0), N)
theta = jnp.array([0.3, 0.5], jnp.float32)
eff = jnp.asarray(0.8 + 0.2 * np.cos(np.arange(NS)), jnp.float32)
target = jnp.asarray(rng.uniform(20.0, 40.0, NS), jnp.float32)


def loss(theta, eff, x0, keys):
    q = f(x0, keys, theta, eff).outputs["Q"]
    return jnp.sum((q - target) ** 2) / NS


r = jax.jit(f)(x0, keys, theta, eff)
print("Q[:6]", np.round(np.asarray(r.outputs["Q"][:6]), 2), " mean steps", float(r.per_item.mean()))

# reverse mode: one adjoint kernel for any number of parameters
g_theta, g_eff = jax.jit(jax.grad(loss, argnums=(0, 1)))(theta, eff, x0, keys)
# forward mode: one kernel with P dual lanes
J_theta = jax.jit(jax.jacfwd(loss, argnums=0))(theta, eff, x0, keys)
J_eff = jax.jit(jax.jacfwd(loss, argnums=1))(theta, eff, x0, keys)
e1 = float(jnp.max(jnp.abs(g_theta - J_theta)) / jnp.max(jnp.abs(J_theta)))
e2 = float(jnp.max(jnp.abs(g_eff - J_eff)) / jnp.max(jnp.abs(J_eff)))
print("d loss / d theta: grad", np.asarray(g_theta), " jacfwd", np.asarray(J_theta), f" rel {e1:.1e}")
print(f"d loss / d eff (32 entries): grad vs jacfwd rel {e2:.1e}")

# mechanics check against the dense meaning (jax.vmap + sum) on a small N
n = 512
dense = reference(photon, **opts)
a = f(x0[:n], keys[:n], theta, eff).outputs["Q"]
b = dense(x0[:n], keys[:n], theta, eff).outputs["Q"]
e3 = float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))
print(f"forward vs dense reference rel {e3:.1e}")
assert e1 < 1e-4 and e2 < 1e-4 and e3 < 1e-5, (e1, e2, e3)
print("PHOTON_TOY OK")
