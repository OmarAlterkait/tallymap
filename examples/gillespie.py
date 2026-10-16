"""Gillespie SSA: exact stochastic chemical kinetics, with a score-function gradient for free.

    python examples/gillespie.py

The Schloegl network (A + 2X -> 3X, 3X -> A + 2X, B -> X, X -> B) with A and B buffered.  ITEM = one exact
realisation: draw a waiting time, pick a reaction, repeat until t >= T.  TALLY = Sum((NB,)), a histogram of
X(T), plus the sums needed below.  The divergence is the number of reaction events, which this model's
bistability makes BIMODAL: a realisation that settles in the low state fires a few hundred reactions, one that
settles high fires thousands.

A jump process has no pathwise derivative of its histogram with respect to the rate constants, so there is no
`jax.grad` here.  The right tool is the SCORE FUNCTION: accumulate d log P(path) / d c2 along the item and
tally score * X(T).  That is a likelihood-ratio (REINFORCE) gradient estimate obtained in ONE forward pass, and
it costs one extra float carry.

Two checks below.  E[score] must be exactly 0, because the score is the derivative of a normalised density;
that is checked against its own standard error.  And the score estimate of d E[X(T)] / d c2 is compared with a
common-random-number finite difference, each with its standard error, since both are Monte Carlo estimates of
the same number at the same sample size.

Measured: 10 to 25 times faster than jax.vmap + lax.while_loop forward on an L40S, because that baseline pays
the batch maximum event count while tallymap pays the mean, and the score carry is free to within the
measurement noise.  See docs/performance.md.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm

NB, T, XMAX = 24, 0.3, 1000.0
A, B = 1.0e5, 2.0e5


def propensities(x, c):
    xf = x.astype(jnp.float32)
    return jnp.stack([c[0] * A * xf * (xf - 1.0) * 0.5,
                      c[1] * xf * (xf - 1.0) * (xf - 2.0) / 6.0,
                      c[2] * B,
                      c[3] * xf])


def event(s, key, c):
    """ONE exact SSA event.  `sc` accumulates d log P(path) / d c2, the score of the B -> X feed rate:
    for one event that is [chosen reaction == 2] / c2 - B * tau, because a_2 = c2 * B is the only propensity
    that depends on c2."""
    t, x, n, sc = s
    a = jnp.maximum(propensities(x, c), 0.0)
    a0 = jnp.sum(a)
    u = jax.random.uniform(jax.random.fold_in(key, n), (2,))
    tau = -jnp.log(jnp.maximum(u[0], 1e-20)) / jnp.maximum(a0, 1e-20)
    r = jnp.clip(jnp.sum((u[1] * a0 > jnp.cumsum(a)).astype(jnp.int32)), 0, 3)
    dx = jnp.array([1, -1, 1, -1], jnp.int32)[r]
    sc = sc + jnp.where(r == 2, 1.0 / c[2], 0.0) - B * tau
    return t + tau, jnp.clip(x + dx, 0, 2000), n + 1, sc


def realisation(t, x0, key, c):
    """ONE item: one exact realisation, tallied into the histogram and into the score estimator."""
    s = lax.while_loop(lambda s: s[0] < T, lambda s: event(s, key, c),
                       (jnp.float32(0.0), x0, jnp.int32(0), jnp.float32(0.0)))
    x, score = s[1].astype(jnp.float32), s[3]
    t = t.add("hist", jnp.clip((x * (NB / XMAX)).astype(jnp.int32), 0, NB - 1), 1.0)
    t = t.add("sums", jnp.array([0, 1, 2, 3], jnp.int32),
              jnp.stack([x, score, score * score, score * x]))        # X, score, score^2, score * X
    return t, (x, s[2])


f = tm.vmap(realisation, outputs={"hist": tm.Sum((NB,)), "sums": tm.Sum((4,))}, in_axes=(0, 0, None),
            max_steps=1 << 16)

N = 4096
rng = np.random.default_rng(0)
x0 = jnp.asarray(rng.integers(100, 501, N), jnp.int32)
keys = jax.random.split(jax.random.PRNGKey(2), N)
c = jnp.array([3.0e-7, 1.0e-4, 1.0e-3, 3.5], jnp.float32)

res = jax.jit(f)(x0, keys, c)
X, events = res.per_item
print("X(T) histogram", np.asarray(res.outputs["hist"], np.int64))
print("events per realisation: mean", round(float(events.mean()), 1), " max", int(events.max()),
      " truncated", int(res.stats["n_truncated"]))
sX, sS, sS2, sSX = (float(v) for v in res.outputs["sums"])
mean_X, mean_score = sX / N, sS / N
se_score = float(np.sqrt(max(sS2 / N - mean_score ** 2, 0.0) / N))
print(f"E[X(T)] = {mean_X:.2f}")

# Check 1: E[score] = 0 exactly, because the score is the derivative of a normalised density.
z = abs(mean_score) / max(se_score, 1e-12)
print(f"E[score] = {mean_score:.4g} +- {se_score:.4g}  ({z:.2f} standard errors from 0)")

# Check 2: the score estimate of d E[X(T)] / d c2, against a common-random-number finite difference.
score_grad = sSX / N
h = 0.05 * float(c[2])
peritem = jax.jit(lambda cc: f(x0, keys, cc).per_item[0])
dX = np.asarray(peritem(c.at[2].add(h)), np.float64) - np.asarray(peritem(c.at[2].add(-h)), np.float64)
fd, se_fd = float(dX.mean()) / (2 * h), float(dX.std(ddof=1)) / (2 * h) / np.sqrt(N)
print(f"d E[X(T)] / d c2:  score function {score_grad:.4g}   finite difference {fd:.4g} +- {se_fd:.2g}")
agree = abs(score_grad - fd) / max(abs(fd), 1e-12)
print(f"the two estimates differ by {agree * 100:.0f}% (both are Monte Carlo estimates at N = {N})")

# The same item in plain JAX: jax.vmap of the SAME event loop, then one scatter-add.  Forward only.
def plain(c):
    def one(x, key):
        s = lax.while_loop(lambda s: s[0] < T, lambda s: event(s, key, c),
                           (jnp.float32(0.0), x, jnp.int32(0), jnp.float32(0.0)))
        return s[1]
    b = jnp.clip((jax.vmap(one)(x0, keys).astype(jnp.float32) * (NB / XMAX)).astype(jnp.int32), 0, NB - 1)
    return jnp.zeros((NB,), jnp.float32).at[b].add(1.0)


e_hist = float(jnp.max(jnp.abs(res.outputs["hist"] - jax.jit(plain)(c))))
print(f"histogram vs the plain-JAX arm: max absolute difference {e_hist:.0f} realisations of {N}")

bad = ([] if z < 4.0 else ["E[score]-is-not-0"]) + ([] if e_hist == 0 else ["histogram"]) \
    + ([] if agree < 1.0 else ["score-vs-finite-difference"])
print("GILLESPIE " + ("OK" if not bad else "FAIL " + ",".join(bad)))
assert not bad, (bad, z, agree, e_hist)
