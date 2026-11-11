"""The FLOOR: a uniform loop length and a cheap step.  Two shapes of it, because they do not give the same answer.

Every item runs exactly the same number of steps (64) and the step is a handful of flops.  There is no
divergence at all, so tallymap's scheduling advantage is exactly zero and the only lever left is keeping the
loop state in registers instead of round-tripping it to HBM.

This problem was written expecting a loss, on the reasoning that a uniform loop is what `lax.scan` is for.  The
measurement says otherwise: see `docs/performance_detail.md`.  Register residency on its own is worth a few times the
plain-JAX program even with nothing to schedule, and it is where the tally is WRITTEN, not whether the loop is
uniform, that moves the margin.  The real losses are elsewhere: `dense_state` and `trajectory`, both of which
have uniform loops too, and both of which lose because the per-item STATE is too big for registers.

ITEM      one walker, exactly 64 steps.
TALLY     Sum((64,)).
LEVELS    `end_only`: the item emits ONCE, at the end.  The plain-JAX arm then puts the whole item inside the
                      `jax.vmap` and finishes with one scatter-add over the batch, which is about the cheapest
                      program XLA can be given.  This is the FLOOR of what tallymap buys.
          `per_step`: the item emits every step.  A tally shared by every item then has to be a loop carry, so
                      the plain-JAX arm's loop sits outside the `jax.vmap` and does one batched `.at[].add` per
                      iteration, writing N values to HBM 64 times.  tallymap's emit stays in shared memory.
THETA     3 parameters.

The padded-scan arms are not padded at all here: the observed maximum IS the uniform trip count, so `jax_scan`
runs exactly 64 steps.  `jax_scan2x` still pays 128, which measures what over-provisioning costs when there is
nothing to provision for.
"""
import arms as A
import harness as H
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm

NAME = "uniform"
TITLE = "Uniform loop length, cheap step: the floor"
GROUP = "floor"
AXIS = "n"
SWEEP = {"full": (1 << 14, 1 << 16, 1 << 19, 1 << 22), "quick": (1 << 10, 1 << 12)}
DIVS = ("end_only", "per_step")
ARMS = ("tm_cuda", "tm_xla", "jax_while", "jax_scan", "jax_scan2x", "jax_ckpt", "jax_ckpt2x")
MODES = ("fwd", "jvp", "jacfwd", "grad")
CHECK_VAL = 1 << 12
DIV_VAL = 1 << 19      # the other level is measured at this one size

NB, K = 64, 64
TH0 = (1.0, 0.5, 0.25)


def step(c, key, th):
    """ONE step -> (carry, bin, value).  Shared by every arm."""
    k, w = c
    u = jax.random.uniform(jax.random.fold_in(key, k))
    return (k + 1, w * 0.99 + th[2] * u), (u * NB).astype(jnp.int32), w * th[0] + th[1] * u


def setup(n, div):
    rng = np.random.default_rng(0)
    return dict(n=n, per_step=(div == "per_step"), keys=jax.random.split(jax.random.PRNGKey(0), n),
                th=jnp.asarray(TH0, jnp.float32), wq=jnp.asarray(rng.uniform(0.5, 1.5, NB), jnp.float32))


def telemetry(ctx):
    return H.trip_stats(np.full(ctx["n"], K))


def sim_of(ctx, arm, maxs):
    keys, n, per_step = ctx["keys"], ctx["n"], ctx["per_step"]
    if arm.startswith("tm_"):
        def walker(t, key, th):
            def body(c):
                k, w, t = c
                (k2, w2), b, v = step((k, w), key, th)
                return k2, w2, (t.add("hist", b, v) if per_step else t)
            k, w, t = lax.while_loop(lambda c: c[0] < K, body, (jnp.int32(0), jnp.float32(1.0), t))
            if not per_step:
                _, b, v = step((k, w), key, th)
                t = t.add("hist", b, v)
            return t, k
        f = tm.vmap(walker, outputs={"hist": tm.Sum((NB,))}, in_axes=(0, None),
                    backend="auto" if arm == "tm_cuda" else "xla")
        return lambda th: f(keys, th).outputs["hist"]

    if not per_step:
        # the item emits once, so the loop goes INSIDE the jax.vmap and one scatter-add finishes the job
        def one(key, th):
            c0 = (jnp.int32(0), jnp.float32(1.0))
            cond = lambda c: c[0] < K
            body = lambda c: step(c, key, th)[0]
            if arm == "jax_while":
                c = lax.while_loop(cond, body, c0)
            else:
                c, _ = A.masked_scan(body, cond, c0, maxs, remat=A.remat_of(arm))
            _, b, v = step(c, key, th)
            return b, v

        def call_end(th):
            b, v = jax.vmap(one, in_axes=(0, None))(keys, th)
            return A.scatter_add(NB, b, v)
        return call_end

    # the item emits every step, so the shared tally has to be a loop carry and the loop sits outside the vmap
    vstep = jax.vmap(step, in_axes=((0, 0), 0, None))

    def call_steps(th):
        def body(c):
            k, w, Q = c
            (k2, w2), b, v = vstep((k, w), keys, th)
            return k2, w2, Q.at[b].add(v, mode="drop")
        c0 = (jnp.zeros((n,), jnp.int32), jnp.ones((n,), jnp.float32), jnp.zeros((NB,), jnp.float32))
        if arm == "jax_while":
            return lax.while_loop(lambda c: c[0][0] < K, body, c0)[2]
        return A.padded_loop(body, c0, maxs, remat=A.remat_of(arm))[0][2]
    return call_steps


def loss_of(ctx, arm, maxs):
    sim = sim_of(ctx, arm, maxs)
    return lambda th: jnp.sum(ctx["wq"] * sim(th) ** 2) / NB


def measured(ctx, arm, mode, maxs):
    extra = dict(nbins=NB, ksteps=K, emit="per_step" if ctx["per_step"] else "end_only",
                 padded_steps=A.padded_count(arm, maxs), jacfwd_k=A.jacfwd_k(ctx["th"]))
    return A.mode_fn(mode, sim_of(ctx, arm, maxs), loss_of(ctx, arm, maxs), ctx["th"]), (ctx["th"],), extra


def check(ctx, maxs):
    return A.cross_arm_check(lambda a: sim_of(ctx, a, maxs), lambda a: loss_of(ctx, a, maxs), ctx["th"])
