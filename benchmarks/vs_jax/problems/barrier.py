"""Up-and-out barrier call option Monte Carlo, measured against plain JAX.

ITEM      one price path, S_{k+1} = S_k exp((r - sig^2/2) dt + sig sqrt(dt) Z_k), 252 daily steps.  If S ever
          exceeds the barrier the option is knocked out and the path may STOP: its payoff is 0 whatever happens
          afterwards.
TALLY     Sum((8,)) a ladder of discounted strike payoffs (one emit per strike), plus the knock-out count.
DIVERGE   a KNOB, not a property of the problem.  With a far barrier no path exits early and every item runs
          exactly 252 steps, which is the regime where `jax.vmap` + `lax.scan` is already the right tool; with a
          near barrier most paths stop in the first few steps.
THETA     (sigma, r, S0): 3 parameters, so `grad` and `jacfwd` are pathwise vega, rho and delta.

Honest caveat on the gradient: a knock-out is a discontinuous functional of the path, so the PATHWISE Greeks of
a barrier option are biased estimators of the true Greeks (the boundary term is missing).  That is a known
problem in finance, handled by smoothing or a likelihood-ratio term.  What is measured here is the cost and the
mechanics: tallymap returns the exact derivative of the program as written, and it agrees with the plain-JAX
arm's derivative of the same program.
"""
import math

import arms as A
import harness as H
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm

NAME = "barrier"
TITLE = "Barrier option Monte Carlo (up-and-out call)"
GROUP = "natural"
AXIS = "n"
SWEEP = {"full": (1 << 14, 1 << 16, 1 << 19, 1 << 22), "quick": (1 << 10, 1 << 12)}
DIVS = ("mid", "low", "high")
ARMS = ("tm_cuda", "tm_xla", "jax_while", "jax_scan", "jax_scan2x", "jax_ckpt", "jax_ckpt2x")
MODES = ("fwd", "jvp", "jacfwd", "grad")
CHECK_VAL = 1 << 12
DIV_VAL = 1 << 19      # the other divergence levels are measured at this one size

NK, KSTEPS, TMAT = 8, 252, 1.0
DT = TMAT / KSTEPS
SQDT = math.sqrt(DT)
BAR = {"low": 3.0, "mid": 1.3, "high": 1.05}            # barrier / S0: 3.0 means a nearly uniform 252-step loop
STRIKES = np.linspace(0.8, 1.2, NK)
TH0 = (0.25, 0.03, 100.0)


def step(c, key, th, bar):
    """ONE daily step of ONE path -> carry (k, S, alive).  Shared by every arm."""
    k, S, alive = c
    z = jax.random.normal(jax.random.fold_in(key, k))
    S = S * jnp.exp((th[1] - 0.5 * th[0] * th[0]) * DT + th[0] * SQDT * z)
    return k + 1, S, alive & (S < bar * th[2])


def cond(c):
    return (c[0] < KSTEPS) & c[2]


def payoffs(S, alive, th, strikes):
    """The discounted payoff of one finished path, per strike (0 if it was knocked out)."""
    return jnp.exp(-th[1] * TMAT) * jnp.maximum(S - strikes * th[2], 0.0) * alive.astype(jnp.float32)


def setup(n, div):
    return dict(n=n, bar=BAR[div], keys=jax.random.split(jax.random.PRNGKey(3), n),
                th=jnp.asarray(TH0, jnp.float32), strikes=jnp.asarray(STRIKES, jnp.float32),
                w=jnp.asarray(np.linspace(1.0, 2.0, NK), jnp.float32))


def telemetry(ctx):
    bar = ctx["bar"]

    def one(key, th):
        return lax.while_loop(cond, lambda c: step(c, key, th, bar),
                              (jnp.int32(0), th[2], jnp.bool_(True)))[0]
    k = jax.jit(jax.vmap(one, in_axes=(0, None)))(ctx["keys"], ctx["th"])
    return H.trip_stats(k)


def sim_of(ctx, arm, maxs):
    keys, bar, strikes, n = ctx["keys"], ctx["bar"], ctx["strikes"], ctx["n"]
    if arm.startswith("tm_"):
        def path(t, key, th, st):
            c = lax.while_loop(cond, lambda c: step(c, key, th, bar), (jnp.int32(0), th[2], jnp.bool_(True)))
            p = payoffs(c[1], c[2], th, st)
            t = t.add("price", jnp.arange(NK, dtype=jnp.int32), p)
            t = t.add("ko", 0, (~c[2]).astype(jnp.float32))
            return t, (p[NK // 2], c[0])
        f = tm.vmap(path, outputs={"price": tm.Sum((NK,)), "ko": tm.Sum((1,))}, in_axes=(0, None, None),
                    backend="cuda" if arm == "tm_cuda" else "xla")
        return lambda th: f(keys, th, strikes).outputs["price"] / n

    def one(key, th):
        c0 = (jnp.int32(0), th[2], jnp.bool_(True))
        if arm == "jax_while":
            c = lax.while_loop(cond, lambda c: step(c, key, th, bar), c0)
        else:
            c, _ = A.masked_scan(lambda c: step(c, key, th, bar), cond, c0, maxs, remat=A.remat_of(arm))
        return payoffs(c[1], c[2], th, strikes)

    return lambda th: jnp.sum(jax.vmap(one, in_axes=(0, None))(keys, th), 0) / n


def loss_of(ctx, arm, maxs):
    sim = sim_of(ctx, arm, maxs)
    return lambda th: jnp.sum(ctx["w"] * sim(th))


def measured(ctx, arm, mode, maxs):
    extra = dict(nstrikes=NK, barrier=ctx["bar"], ksteps=KSTEPS, padded_steps=A.padded_count(arm, maxs),
                 jacfwd_k=A.jacfwd_k(ctx["th"]))
    return A.mode_fn(mode, sim_of(ctx, arm, maxs), loss_of(ctx, arm, maxs), ctx["th"]), (ctx["th"],), extra


def check(ctx, maxs):
    return A.cross_arm_check(lambda a: sim_of(ctx, a, maxs), lambda a: loss_of(ctx, a, maxs), ctx["th"])
