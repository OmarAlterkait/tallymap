"""Walk-on-spheres: a Monte Carlo solver for the Poisson equation, and its gradient in the model parameters.

    python examples/walk_on_spheres.py

Solve  laplacian u = -f(y; theta)  in the unit square with  u = g(x; theta)  on the boundary.  ITEM = one
walker, which jumps to a uniformly random point on the largest circle that fits inside the domain until that
circle's radius drops below eps, adding the standard single-sample ball estimator at each hop.  TALLY =
Sum((M,)), the per-evaluation-point sum of walker estimates.  The number of hops is the divergence: it is
O(log(1/eps)) on average with a geometric tail, so a walker that lands near a corner takes many more hops than
the mean.

The walk itself does not depend on theta, so the pathwise gradient is exact and unbiased.

Measured: about 10x faster than jax.vmap + lax.while_loop forward on an L40S, with a reverse-mode gradient that
plain JAX cannot take at all without first rewriting the loop as a padded masked scan.  The numbers, the arms
and the padding rule are in docs/performance_detail.md.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm

M, EPS = 32, 1e-3


def sdf(x):
    """Distance to the boundary of the unit square."""
    return jnp.minimum(jnp.minimum(x[0], 1.0 - x[0]), jnp.minimum(x[1], 1.0 - x[1]))


def hop(c, key, theta):
    """ONE hop: the ball's source contribution, then the jump to the sphere.  Used by both programs below."""
    x, acc, k = c
    R = sdf(x)
    u = jax.random.uniform(jax.random.fold_in(key, k), (3,))
    r, a = R * jnp.sqrt(u[0]), 2.0 * jnp.pi * u[1]
    y = x + r * jnp.stack([jnp.cos(a), jnp.sin(a)])
    src = theta[0] * jnp.exp(-theta[1] * jnp.sum((y - 0.5) ** 2))
    acc = acc - 0.25 * R * R * jnp.log(jnp.maximum(u[0], 1e-20)) * src
    a2 = 2.0 * jnp.pi * u[2]
    return x + R * jnp.stack([jnp.cos(a2), jnp.sin(a2)]), acc, k + 1


def walk(x0, key, theta):
    """One walker, start to boundary -> (its estimate of u, the number of hops it took)."""
    x, acc, k = lax.while_loop(lambda c: sdf(c[0]) > EPS, lambda c: hop(c, key, theta),
                               (x0, jnp.float32(0.0), jnp.int32(0)))
    return acc + theta[2] + theta[3] * jnp.cos(2.0 * jnp.pi * (x[0] + x[1])), k      # + the boundary value


def walker(t, x0, point, key, theta):
    """ONE item for tallymap.  t is its Tally: thread it through like a PRNG key."""
    acc, k = walk(x0, key, theta)
    return t.add("u", point, acc), k


f = tm.vmap(walker, outputs={"u": tm.Sum((M,))}, in_axes=(0, 0, 0, None), max_steps=4096)

N = 8192
rng = np.random.default_rng(0)
x0 = jnp.asarray(rng.uniform(0.02, 0.98, (N, 2)), jnp.float32)
point = jnp.asarray(np.arange(N) % M, jnp.int32)             # which evaluation point each walker belongs to
keys = jax.random.split(jax.random.PRNGKey(1), N)
theta = jnp.array([1.0, 4.0, 0.3, 0.5], jnp.float32)
target = jnp.asarray(rng.uniform(0.0, 1.0, M), jnp.float32)
W = N / M                                                    # walkers per evaluation point

solve = lambda th: f(x0, point, keys, th).outputs["u"] / W
res = jax.jit(f)(x0, point, keys, theta)
print("u at the first 4 evaluation points", np.round(np.asarray(res.outputs["u"][:4] / W), 4))
print("hops per walker: mean", round(float(res.per_item.mean()), 1), " max", int(res.per_item.max()),
      " truncated", int(res.stats["n_truncated"]))

# The same program in plain JAX: jax.vmap of the SAME walker loop, then one scatter-add.  This is the arm
# docs/performance_detail.md calls `jax.vmap + while_loop`; it has no gradient, because JAX cannot
# reverse-differentiate a while_loop whose trip count depends on the data.
def plain(theta):
    acc, _ = jax.vmap(walk, in_axes=(0, 0, None))(x0, keys, theta)
    return jnp.zeros((M,), jnp.float32).at[point].add(acc) / W


u = jax.jit(solve)(theta)
e_fwd = float(jnp.max(jnp.abs(u - jax.jit(plain)(theta))) / jnp.max(jnp.abs(u)))
print(f"forward vs the plain-JAX arm: rel {e_fwd:.1e}")

# derivatives: reverse mode (one adjoint pass, replayed per item) against forward mode (4 dual lanes, one kernel)
loss = lambda th: jnp.sum((solve(th) - target) ** 2) / M
g = jax.jit(jax.grad(loss))(theta)
J = jax.jit(jax.jacfwd(loss))(theta)
e_grad = float(jnp.max(jnp.abs(g - J)) / jnp.max(jnp.abs(J)))
print("d loss / d theta: grad", np.round(np.asarray(g), 5))
print(f"grad vs jacfwd: rel {e_grad:.1e}")

bad = [n for n, e in (("forward", e_fwd), ("grad-vs-jacfwd", e_grad)) if not (e < 1e-4)]
print("WALK_ON_SPHERES " + ("OK" if not bad else "FAIL " + ",".join(bad)))
assert not bad, bad
