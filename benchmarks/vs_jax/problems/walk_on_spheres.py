"""Walk-on-spheres solver for the Poisson equation in 2-D, measured against plain JAX.

Solve  laplacian u = -f(y; theta)  inside the unit square with  u = g(x; theta)  on its boundary.

ITEM      one walker from one evaluation point.  It jumps repeatedly to a uniformly random point on the largest
          circle that fits inside the domain, radius R = the distance to the boundary, until R <= eps; then it
          takes the boundary value there.  Each ball also contributes the standard single-sample source
          estimator (R^2 / 2) * ln(R/r) * f(y), which is |B| * G_B(x, y) * f(y) / p(y) with the 2-D harmonic
          Green's function.
TALLY     Sum((64,)): the per-evaluation-point sum of walker estimates.  One emit per item, at the end.
DIVERGE   eps.  The hop count is O(log(1/eps)) on average with a geometric tail: a walker that lands near a
          corner takes many more hops than the mean.
THETA     5 parameters (source amplitude and width, boundary offset, amplitude and frequency).  The walk itself
          does not depend on theta, so the pathwise gradient is exact and unbiased.
"""
import arms as A
import harness as H
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm

NAME = "walk_on_spheres"
TITLE = "Walk-on-spheres Poisson solver (2-D)"
GROUP = "natural"
AXIS = "n"
SWEEP = {"full": (1 << 14, 1 << 16, 1 << 19, 1 << 22), "quick": (1 << 10, 1 << 12)}
DIVS = ("mid", "low", "high")
ARMS = ("tm_cuda", "tm_lockstep", "jax_while", "jax_scan", "jax_scan2x", "jax_ckpt", "jax_ckpt2x")
MODES = ("fwd", "jvp", "jacfwd", "grad")
CHECK_VAL = 1 << 12
DIV_VAL = 1 << 19      # the other divergence levels are measured at this one size

M = 64
EPS = {"low": 1e-2, "mid": 1e-3, "high": 1e-5}
TH0 = (1.0, 4.0, 0.3, 0.5, 2.0)


def sdf(x):
    """Distance to the boundary of the unit square."""
    return jnp.minimum(jnp.minimum(x[0], 1.0 - x[0]), jnp.minimum(x[1], 1.0 - x[1]))


def project(x):
    """The closest boundary point of the unit square."""
    d = jnp.stack([x[0], 1.0 - x[0], x[1], 1.0 - x[1]])
    i = jnp.argmin(d)
    return jnp.stack([jnp.where(i == 0, 0.0, jnp.where(i == 1, 1.0, x[0])),
                      jnp.where(i == 2, 0.0, jnp.where(i == 3, 1.0, x[1]))])


def src(y, th):
    r2 = (y[0] - 0.5) ** 2 + (y[1] - 0.5) ** 2
    return th[0] * jnp.exp(-th[1] * r2)


def bnd(x, th):
    return th[2] + th[3] * jnp.cos(th[4] * jnp.pi * (x[0] + x[1]))


def step(c, key, th):
    """ONE hop: the ball's source contribution, then the jump to the sphere.  Shared by every arm."""
    x, acc, k = c
    R = sdf(x)
    u = jax.random.uniform(jax.random.fold_in(key, k), (3,))
    r = R * jnp.sqrt(u[0])
    a = 2.0 * jnp.pi * u[1]
    y = x + r * jnp.stack([jnp.cos(a), jnp.sin(a)])
    g = -0.5 * jnp.log(jnp.maximum(u[0], 1e-20))                  # ln(R/r) with r = R sqrt(u0)
    acc = acc + 0.5 * R * R * g * src(y, th)
    a2 = 2.0 * jnp.pi * u[2]
    return x + R * jnp.stack([jnp.cos(a2), jnp.sin(a2)]), acc, k + 1


def setup(n, div):
    rng = np.random.default_rng(0)
    return dict(n=n, eps=EPS[div], x0=jnp.asarray(rng.uniform(0.02, 0.98, (n, 2)), jnp.float32),
                pid=jnp.asarray(np.arange(n) % M, jnp.int32),
                keys=jax.random.split(jax.random.PRNGKey(1), n), th=jnp.asarray(TH0, jnp.float32),
                wts=jnp.asarray(rng.uniform(0.5, 1.5, M), jnp.float32),
                tgt=jnp.asarray(rng.uniform(0.0, 1.0, M), jnp.float32))


def telemetry(ctx):
    eps = ctx["eps"]

    def one(x0, key, th):
        return lax.while_loop(lambda c: sdf(c[0]) > eps, lambda c: step(c, key, th),
                              (x0, jnp.float32(0.0), jnp.int32(0)))[2]
    k = jax.jit(jax.vmap(one, in_axes=(0, 0, None)))(ctx["x0"], ctx["keys"], ctx["th"])
    return H.trip_stats(k)


def sim_of(ctx, arm, maxs):
    x0, pid, keys, eps, n = ctx["x0"], ctx["pid"], ctx["keys"], ctx["eps"], ctx["n"]
    cond = lambda c: sdf(c[0]) > eps
    w = n / M
    if arm.startswith("tm_"):
        def walker(t, x, p, key, th):
            c = lax.while_loop(cond, lambda c: step(c, key, th), (x, jnp.float32(0.0), jnp.int32(0)))
            acc = c[1] + bnd(project(c[0]), th)
            return t.add("u", p, acc), (acc, c[2])
        f = tm.vmap(walker, outputs={"u": tm.Sum((M,))}, in_axes=(0, 0, 0, None),
                    backend="cuda" if arm == "tm_cuda" else "lockstep")
        return lambda th: f(x0, pid, keys, th).outputs["u"] / w

    def one(x, key, th):
        c0 = (x, jnp.float32(0.0), jnp.int32(0))
        if arm == "jax_while":
            c = lax.while_loop(cond, lambda c: step(c, key, th), c0)
        else:
            c, _ = A.masked_scan(lambda c: step(c, key, th), cond, c0, maxs, remat=A.remat_of(arm))
        return c[1] + bnd(project(c[0]), th)

    def call(th):
        acc = jax.vmap(one, in_axes=(0, 0, None))(x0, keys, th)
        return A.scatter_add(M, pid, acc) / w
    return call


def measured(ctx, arm, mode, maxs):
    sim, loss = sim_of(ctx, arm, maxs), loss_of(ctx, arm, maxs)
    extra = dict(npoints=M, eps=ctx["eps"], padded_steps=A.padded_count(arm, maxs),
                 jacfwd_k=A.jacfwd_k(ctx["th"]))
    return A.mode_fn(mode, sim, loss, ctx["th"]), (ctx["th"],), extra


def loss_of(ctx, arm, maxs):
    sim = sim_of(ctx, arm, maxs)
    return lambda th: jnp.sum(ctx["wts"] * (sim(th) - ctx["tgt"]) ** 2) / M


def check(ctx, maxs):
    return A.cross_arm_check(lambda a: sim_of(ctx, a, maxs), lambda a: loss_of(ctx, a, maxs), ctx["th"])
