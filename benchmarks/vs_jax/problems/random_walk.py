"""Random-walk histogram: the README quickstart item, measured against plain JAX.

ITEM      one walker.  It carries a weight w, starting at 1, multiplied by its own decay factor each step, and
          runs until w drops below 0.01.
TALLY     Sum((64,)): every step adds a value to the bin its random draw selects.  The emit is PER STEP, so the
          tally is written many times per item.
DIVERGE   the spread of the per-item decay factor, which fixes the trip count exactly: a narrow spread is a
          nearly uniform loop, a wide one runs 4 to 200+ steps.
THETA     3 parameters that enter the emitted value only, never the loop condition, so the pathwise gradient is
          the exact gradient of a continuous function.

Because the tally is written every step, the plain-JAX arms cannot put the loop inside the `jax.vmap`: a tally
shared by every item has to be a loop carry.  They therefore `jax.vmap` the SAME scalar `step` function one step
at a time and do one batched `.at[].add` per iteration, with the loop outside.  That is what a user writes by
hand.  The `jax_while` arm's loop condition is `jnp.any(alive)`, so it runs the batch MAXIMUM number of steps.
"""
import arms as A
import harness as H
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm

NAME = "random_walk"
TITLE = "Random-walk histogram (64 bins)"
GROUP = "natural"
AXIS = "n"
SWEEP = {"full": (1 << 14, 1 << 16, 1 << 19, 1 << 22), "quick": (1 << 10, 1 << 12)}
DIVS = ("mid", "low", "high")
ARMS = ("tm_cuda", "tm_xla", "jax_while", "jax_scan", "jax_scan2x", "jax_ckpt", "jax_ckpt2x")
MODES = ("fwd", "jvp", "jacfwd", "grad")
CHECK_VAL = 1 << 12
DIV_VAL = 1 << 19      # the other divergence levels are measured at this one size

NB = 64
WMIN = 0.01
DECAY = {"low": (0.90, 0.92), "mid": (0.60, 0.95), "high": (0.30, 0.98)}
TH0 = (1.0, 0.5, 0.25)


def step(c, key, decay, th):
    """ONE step of ONE walker -> (carry, bin, value).  Every arm calls this function."""
    k, w = c
    u = jax.random.uniform(jax.random.fold_in(key, k))
    b = (u * NB).astype(jnp.int32)
    v = w * th[0] + th[1] * u + th[2] * w * w
    return (k + 1, w * decay), b, v


def setup(n, div):
    lo, hi = DECAY[div]
    rng = np.random.default_rng(0)
    return dict(n=n, keys=jax.random.split(jax.random.PRNGKey(0), n),
                decay=jnp.asarray(rng.uniform(lo, hi, n), jnp.float32),
                th=jnp.asarray(TH0, jnp.float32),
                wq=jnp.asarray(rng.uniform(0.5, 1.5, NB), jnp.float32))


def trips(ctx):
    """The exact trip count of every walker, ceil(log(WMIN) / log(decay)): no device work needed."""
    d = np.asarray(ctx["decay"], np.float64)
    return np.ceil(np.log(WMIN) / np.log(d)).astype(np.int64)


def telemetry(ctx):
    return H.trip_stats(trips(ctx))


def sim_of(ctx, arm, maxs):
    """-> sim(theta) -> the tally, for one arm."""
    keys, decay, n = ctx["keys"], ctx["decay"], ctx["n"]
    if arm.startswith("tm_"):
        def walker(t, key, dec, th):
            def body(c):
                k, w, t = c
                (k2, w2), b, v = step((k, w), key, dec, th)
                return k2, w2, t.add("hist", b, v)
            k, w, t = lax.while_loop(lambda c: c[1] > WMIN, body, (jnp.int32(0), jnp.float32(1.0), t))
            return t, k
        f = tm.vmap(walker, outputs={"hist": tm.Sum((NB,))}, in_axes=(0, 0, None),
                    backend="cuda" if arm == "tm_cuda" else "xla")
        return lambda th: f(keys, decay, th).outputs["hist"]

    vstep = jax.vmap(step, in_axes=((0, 0), 0, 0, None))

    def call(th):
        def body(c):
            k, w, Q = c
            live = w > WMIN
            (k2, w2), b, v = vstep((k, w), keys, decay, th)
            Q = Q.at[b].add(jnp.where(live, v, 0.0), mode="drop")
            return jnp.where(live, k2, k), jnp.where(live, w2, w), Q
        c0 = (jnp.zeros((n,), jnp.int32), jnp.ones((n,), jnp.float32), jnp.zeros((NB,), jnp.float32))
        if arm == "jax_while":
            return lax.while_loop(lambda c: jnp.any(c[1] > WMIN), body, c0)[2]
        return A.padded_loop(body, c0, maxs, remat=A.remat_of(arm))[0][2]
    return call


def measured(ctx, arm, mode, maxs):
    sim = sim_of(ctx, arm, maxs)
    wq = ctx["wq"]
    loss = lambda th: jnp.sum(wq * sim(th) ** 2) / NB
    extra = dict(nbins=NB, padded_steps=A.padded_count(arm, maxs), steps_max=int(trips(ctx).max()),
                 jacfwd_k=A.jacfwd_k(ctx["th"]))
    return A.mode_fn(mode, sim, loss, ctx["th"]), (ctx["th"],), extra


def check(ctx, maxs):
    return A.cross_arm_check(lambda a: sim_of(ctx, a, maxs),
                             lambda a: (lambda t: jnp.sum(ctx["wq"] * sim_of(ctx, a, maxs)(t) ** 2) / NB), ctx["th"])
