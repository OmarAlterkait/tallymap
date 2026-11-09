"""A shower: items spawn items.  Forward tallies, and the gradient through the spawn tree (the tree adjoint).

    python examples/shower_toy.py

Each particle deposits a fraction of its energy per step along z and may split; a child gets a share of the energy
and runs the same function.  Parameters theta = (deposit fraction, split energy share).  The gradient goes through
every generation (reverse mode: one adjoint launch per generation, deepest first) and is checked against forward
mode (jacfwd: children carry tangents).
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm

NZ, DEPTH = 16, 8


def particle(t, p, theta):
    def body(c):
        s, n, t = c
        k = jax.random.fold_in(s["key"], n)
        u = jax.random.uniform(k, (3,))
        z = s["z"] - jnp.log1p(-u[0]) * 0.5
        dep = s["E"] * theta[0]
        t = t.add("dep", jnp.clip(jnp.floor(z * 2.0).astype(jnp.int32), 0, NZ - 1), dep)
        split = (u[1] < 0.35) & (s["g"] < DEPTH)
        Ec = (s["E"] - dep) * theta[1] * (0.5 + 0.5 * u[2])
        t = t.spawn(dict(E=Ec, z=z, key=jax.random.fold_in(k, 1), g=s["g"] + 1), when=split)
        E = s["E"] - dep - jnp.where(split, Ec, 0.0)
        return dict(E=E, z=z, key=s["key"], g=s["g"]), n + 1, t
    s, n, t = lax.while_loop(lambda c: (c[0]["E"] > 1e-3) & (c[1] < 20), body, (p, jnp.int32(0), t))
    return t


f = tm.vmap(particle, outputs={"dep": tm.Sum((NZ,))}, in_axes=(0, None), spawn=tm.Queue(1 << 15), rev_stats=True)

N = 256
roots = dict(E=jnp.linspace(1.0, 2.0, N, dtype=jnp.float32), z=jnp.zeros((N,), jnp.float32),
             key=jax.random.split(jax.random.PRNGKey(0), N), g=jnp.zeros((N,), jnp.int32))
theta = jnp.array([0.2, 0.6], jnp.float32)
w = jnp.linspace(1.0, 2.0, NZ, dtype=jnp.float32)

r = jax.jit(f)(roots, theta)
print("deposit profile", np.round(np.asarray(r.outputs["dep"]), 1))
print("spawned", int(r.stats["spawned"]), " dropped", int(r.stats["children_dropped"]))
loss = lambda th, roots: jnp.sum(f(roots, th).outputs["dep"] * w)
g = jax.jit(jax.grad(loss))(theta, roots)
J = jax.jit(jax.jacfwd(loss))(theta, roots)
st = tm.diagnostics()["reverse"]["tree"]
err = float(jnp.max(jnp.abs(g - J)) / jnp.max(jnp.abs(J)))
print("grad", np.asarray(g), " jacfwd", np.asarray(J), f" rel {err:.1e}")
print("tree adjoint: generations", int(st["generations"]), " children", int(st["children"]),
      " drop_risk", bool(st["drop_risk"]), " replay mismatches", int(st["replay_mismatch"]))
assert int(r.stats["children_dropped"]) == 0 and not bool(st["drop_risk"]) and int(st["replay_mismatch"]) == 0
assert np.all(np.isfinite(np.asarray(g))) and err < 1e-4, err
print("SHOWER_TOY OK")
