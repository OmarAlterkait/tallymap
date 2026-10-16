"""BAD FIT 4: too few items.  The sweep axis is N, from ONE item upwards.

The same divergent barrier-option path as the `barrier` problem, run at N = 1, 4, 16, ... 2^16.  At N = 1 there
is one GPU thread and no parallelism at all, which is the "one long sequential rollout" case: neither tallymap
nor `jax.vmap` has anything to exploit, and `lax.scan` on its own is the right program.  The sweep then shows
where a per-thread kernel starts to pay for itself.

ITEM      one price path with an early exit on knock-out (identical to the `barrier` problem's item).
TALLY     Sum((1,)): the mid-strike payoff.
AXIS      N, the number of items.
"""
import arms as A
import harness as H
import jax
import jax.numpy as jnp
from jax import lax

import problems.barrier as B
import tallymap as tm

NAME = "small_n"
TITLE = "Very few items, down to N = 1 (a bad fit)"
GROUP = "bad-fit"
AXIS = "n"
SWEEP = {"full": (1, 4, 16, 64, 256, 1 << 10, 1 << 12, 1 << 14, 1 << 16), "quick": (1, 256)}
DIVS = ("mid",)
ARMS = ("tm_cuda", "tm_lockstep", "jax_while", "jax_scan")
MODES = ("fwd",)
CHECK_VAL = 256


def setup(n, div):
    return dict(n=n, bar=B.BAR[div], keys=jax.random.split(jax.random.PRNGKey(3), n),
                th=jnp.asarray(B.TH0, jnp.float32))


def telemetry(ctx):
    bar = ctx["bar"]

    def one(key, th):
        return lax.while_loop(B.cond, lambda c: B.step(c, key, th, bar),
                              (jnp.int32(0), th[2], jnp.bool_(True)))[0]
    return H.trip_stats(jax.jit(jax.vmap(one, in_axes=(0, None)))(ctx["keys"], ctx["th"]))


def sim_of(ctx, arm, maxs):
    keys, bar = ctx["keys"], ctx["bar"]
    if arm.startswith("tm_"):
        def path(t, key, th):
            c = lax.while_loop(B.cond, lambda c: B.step(c, key, th, bar), (jnp.int32(0), th[2], jnp.bool_(True)))
            pay = jnp.maximum(c[1] - th[2], 0.0) * c[2].astype(jnp.float32)
            return t.add("p", 0, pay), c[0]
        f = tm.vmap(path, outputs={"p": tm.Sum((1,))}, in_axes=(0, None),
                    backend="cuda" if arm == "tm_cuda" else "lockstep")
        return lambda th: f(keys, th).outputs["p"]

    def one(key, th):
        c0 = (jnp.int32(0), th[2], jnp.bool_(True))
        if arm == "jax_while":
            c = lax.while_loop(B.cond, lambda c: B.step(c, key, th, bar), c0)
        else:
            c, _ = A.masked_scan(lambda c: B.step(c, key, th, bar), B.cond, c0, maxs, remat=A.remat_of(arm))
        return jnp.maximum(c[1] - th[2], 0.0) * c[2].astype(jnp.float32)

    return lambda th: jnp.sum(jax.vmap(one, in_axes=(0, None))(keys, th), keepdims=True)


def measured(ctx, arm, mode, maxs):
    return (sim_of(ctx, arm, maxs), (ctx["th"],),
            dict(n=ctx["n"], barrier=ctx["bar"], padded_steps=A.padded_count(arm, maxs)))


def check(ctx, maxs):
    return A.cross_arm_check(lambda a: sim_of(ctx, a, maxs),
                             lambda a: (lambda th: jnp.sum(sim_of(ctx, a, maxs)(th))), ctx["th"],
                             arms=("tm_cuda", "jax_while", "jax_scan"))
