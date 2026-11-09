"""Gillespie SSA (exact stochastic chemical kinetics): the Schloegl network, measured against plain JAX.

Reactions, with A and B buffered:  A + 2X -> 3X,  3X -> A + 2X,  B -> X,  X -> B.
Propensities a = (c0 A X (X-1) / 2, c1 X (X-1) (X-2) / 6, c2 B, c3 X).  One SSA event is
tau = -ln(u)/sum(a), the reaction chosen by a linear search over the cumulative propensities, t += tau,
X += nu_r.  Run until t >= 0.5.

ITEM      one exact realisation.
TALLY     Sum((32,)) a histogram of X(T), plus the sums of X(T) and of the event count.
DIVERGE   the number of reaction events, which the Schloegl model's bistability makes BIMODAL: a realisation
          that settles in the low state fires far fewer reactions than one in the high state.

Gradients: the trajectory is a discrete jump process, so the histogram has NO pathwise derivative with respect
to the rate constants, and there is no `grad` row here.  The `score` mode instead accumulates the path score
d log P(path) / d c2 along the item and tallies score * X(T): a likelihood-ratio (REINFORCE) gradient estimate
obtained in ONE forward pass.  Its row is directly comparable with the `fwd` row, which is the point: the
measurement is what that estimator costs.
"""
import arms as A
import harness as H
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm

NAME = "gillespie"
TITLE = "Gillespie SSA (Schloegl network)"
GROUP = "natural"
AXIS = "n"
SWEEP = {"full": (1 << 14, 1 << 16, 1 << 18, 1 << 20), "quick": (1 << 7, 1 << 9)}
DIVS = ("mid", "low", "high")
ARMS = ("tm_cuda", "tm_xla", "jax_while", "jax_scan", "jax_scan2x")
MODES = ("fwd", "score")
CHECK_VAL = 1 << 7
DIV_VAL = 1 << 18      # the other divergence levels are measured at this one size

NB, T = 32, 0.5
AB, BB = 1.0e5, 2.0e5
C0 = (3.0e-7, 1.0e-4, 1.0e-3, 3.5)
X0 = {"low": (240, 260), "mid": (100, 500), "high": (5, 800)}
XMAX = 1000.0


def props(x, c):
    xf = x.astype(jnp.float32)
    return jnp.stack([c[0] * AB * xf * (xf - 1.0) * 0.5,
                      c[1] * xf * (xf - 1.0) * (xf - 2.0) / 6.0,
                      c[2] * BB,
                      c[3] * xf])


def step(s, key, c, score=False):
    """ONE exact SSA event -> (t, x, n, score).  Shared by every arm.

    With score=True the path score of the B -> X feed rate c2 is accumulated as well.  For one event,
    d/dc2 log[ (a_r / a0) * a0 exp(-a0 tau) ] = d/dc2 [ log a_r - a0 tau ] = [r == 2] / c2 - B tau, because
    a_2 = c2 * B is the only propensity that depends on c2.  That is the only difference between the `fwd` and
    the `score` rows, so their ratio is what the likelihood-ratio gradient estimator costs."""
    t, x, n, sc = s
    a = jnp.maximum(props(x, c), 0.0)
    a0 = jnp.sum(a)
    u = jax.random.uniform(jax.random.fold_in(key, n), (2,))
    tau = -jnp.log(jnp.maximum(u[0], 1e-20)) / jnp.maximum(a0, 1e-20)
    r = jnp.clip(jnp.sum((u[1] * a0 > jnp.cumsum(a)).astype(jnp.int32)), 0, 3)
    dx = jnp.array([1, -1, 1, -1], jnp.int32)[r]
    if score:
        sc = sc + jnp.where(r == 2, 1.0 / jnp.maximum(c[2], 1e-20), 0.0) - BB * tau
    return t + tau, jnp.clip(x + dx, 0, 2000), n + 1, sc


def cond(s):
    return s[0] < T


def hbin(x):
    return jnp.clip((x.astype(jnp.float32) * (NB / XMAX)).astype(jnp.int32), 0, NB - 1)


def c0_of(x0):
    return (jnp.float32(0.0), x0, jnp.int32(0), jnp.float32(0.0))


def setup(n, div):
    rng = np.random.default_rng(0)
    lo, hi = X0[div]
    return dict(n=n, x0_lo=lo, x0_hi=hi, x0=jnp.asarray(rng.integers(lo, hi + 1, n), jnp.int32),
                keys=jax.random.split(jax.random.PRNGKey(2), n), c=jnp.asarray(C0, jnp.float32))


def telemetry(ctx):
    def one(x0, key, c):
        return lax.while_loop(cond, lambda s: step(s, key, c), c0_of(x0))[2]
    k = jax.jit(jax.vmap(one, in_axes=(0, 0, None)))(ctx["x0"], ctx["keys"], ctx["c"])
    return H.trip_stats(k)


def sim_of(ctx, arm, maxs, score=False):
    """-> sim(c) -> the X(T) histogram (and, with score=True, the likelihood-ratio tally as well)."""
    x0, keys, n = ctx["x0"], ctx["keys"], ctx["n"]
    if arm.startswith("tm_"):
        def real(t_, x, key, c):
            s = lax.while_loop(cond, lambda s: step(s, key, c, score), c0_of(x))
            t_ = t_.add("hist", hbin(s[1]), 1.0)
            t_ = t_.add("mean", 0, s[1].astype(jnp.float32))
            if score:
                t_ = t_.add("dmean", 0, s[3] * s[1].astype(jnp.float32))
            return t_, (s[1], s[2])
        outs = {"hist": tm.Sum((NB,)), "mean": tm.Sum((1,))}
        if score:
            outs["dmean"] = tm.Sum((1,))
        f = tm.vmap(real, outputs=outs, in_axes=(0, 0, None),
                    backend="cuda" if arm == "tm_cuda" else "xla")
        def tm_call(c):
            r = f(x0, keys, c)
            return (r.outputs["hist"], r.outputs["dmean"][0] / n) if score else r.outputs["hist"]
        return tm_call

    def one(x, key, c):
        body = lambda s: step(s, key, c, score)
        if arm == "jax_while":
            s = lax.while_loop(cond, body, c0_of(x))
        else:
            s, _ = A.masked_scan(body, cond, c0_of(x), maxs, remat=A.remat_of(arm))
        return s[1], s[3]

    def call(c):
        x, sc = jax.vmap(one, in_axes=(0, 0, None))(x0, keys, c)
        hist = A.scatter_add(NB, hbin(x), jnp.ones((n,), jnp.float32))
        if not score:
            return hist
        return hist, jnp.sum(sc * x.astype(jnp.float32)) / n      # the likelihood-ratio estimate of d E[X(T)]/d c2
    return call


def measured(ctx, arm, mode, maxs):
    sim = sim_of(ctx, arm, maxs, score=(mode == "score"))
    extra = dict(nbins=NB, x0_lo=ctx["x0_lo"], x0_hi=ctx["x0_hi"], padded_steps=A.padded_count(arm, maxs),
                 score=int(mode == "score"))
    return sim, (ctx["c"],), extra


def check(ctx, maxs):
    """No gradient cell: the histogram of a jump process has no pathwise derivative in the rate constants."""
    return A.cross_arm_check(lambda a: sim_of(ctx, a, maxs), None, ctx["c"],
                             arms=("tm_cuda", "jax_while", "jax_scan"), grads=False)
