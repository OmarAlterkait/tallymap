"""INTERACTING SYSTEM 2: a particle filter (sequential Monte Carlo) for a time series.

The second shape of "items that interact every step".  Resampling is an all-to-all over the particles, and it
stays in plain JAX between the calls; what each particle does between two observations is a variable-length
inner loop, and that is the tm.vmap call:

    lax.scan over the T observations                         (plain JAX)
      |-- ONE tm.vmap call: item = one particle.  It integrates its own state from one observation time to the
      |   next with an ADAPTIVE Bogacki-Shampine 3(2) step (the trip count is the particle's own, and it varies
      |   with where the particle sits), adds the diffusion increment, scores itself against the observation and
      |   tallies its weight.
      |-- normalise the weights, accumulate the log-likelihood, systematic resampling   (plain JAX)

ITEM      one particle between two observations.
TALLY     Sum((1,)): the sum of the unnormalised weights of this step, which is the per-step likelihood factor.
          Per-item output: the propagated state and its log weight.
DIVERGE   the diffusion scale, which spreads the particles over the state space and so spreads their adaptive
          step counts.
THETA     (mean-reversion rate, nonlinearity, diffusion scale): 3 parameters.  The drift is strongly
          nonlinear, so a particle far from the origin needs many more adaptive steps than one near it: that is
          where the divergence comes from.
OUTPUT    the log-likelihood estimate, summed over the T observations.

Gradient: `grad` is the standard stop-gradient-through-resampling estimator.  The resampling indices are wrapped
in `lax.stop_gradient`, which makes the gradient of the log-likelihood estimate the usual BIASED estimator used
in practice; it is not the gradient of the true log-likelihood.  That is stated, not fixed: what is measured
here is the cost and the mechanics.
"""
import arms as A
import harness as H
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm

NAME = "particle_filter"
TITLE = "Particle filter with a variable-length inner propagation"
GROUP = "interacting"
AXIS = "n"
SWEEP = {"full": (1 << 12, 1 << 14, 1 << 16, 1 << 18), "quick": (1 << 8, 1 << 10)}
DIVS = ("mid", "low", "high")
ARMS = ("tm_cuda", "tm_lockstep", "jax_while", "jax_scan", "jax_scan2x", "jax_ckpt", "jax_ckpt2x")
MODES = ("fwd", "grad")
CHECK_VAL = 1 << 8
DIV_VAL = 1 << 16      # the other divergence levels are measured at this one size

NOBS = 16                  # observations in the series
DTOBS = 1.0                # time between observations
SIGOBS = 0.3               # observation noise
RTOL, ATOL = 1e-5, 1e-7
# the diffusion scale IS the divergence knob: it sets how far apart the particles get between observations,
# and a particle far from the origin needs many more small adaptive steps than one near it.
SIG = {"low": 0.2, "mid": 1.0, "high": 2.5}
TH0 = (2.0, 2.0)           # (mean-reversion rate, nonlinearity); the diffusion scale is appended per level


def drift(t, x, th):
    return -th[0] * x * (1.0 + th[1] * x * x)


def bs23(t, x, h, th):
    """One Bogacki-Shampine 3(2) trial step of the drift ODE -> (x_new, scaled error norm)."""
    k1 = drift(t, x, th)
    k2 = drift(t + 0.5 * h, x + 0.5 * h * k1, th)
    k3 = drift(t + 0.75 * h, x + 0.75 * h * k2, th)
    xn = x + h * (2.0 / 9.0 * k1 + 1.0 / 3.0 * k2 + 4.0 / 9.0 * k3)
    k4 = drift(t + h, xn, th)
    e = h * ((2.0 / 9.0 - 7.0 / 24.0) * k1 + (1.0 / 3.0 - 0.25) * k2
             + (4.0 / 9.0 - 1.0 / 3.0) * k3 + (-0.125) * k4)
    return xn, jnp.abs(e) / (ATOL + RTOL * jnp.maximum(jnp.abs(x), jnp.abs(xn)))


def step(c, th):
    """ONE adaptive trial step of ONE particle, accepted or rejected.  Shared by every arm."""
    t, x, h, k = c
    h = jnp.minimum(h, DTOBS - t)
    xn, err = bs23(t, x, h, th)
    ok = err <= 1.0
    fac = lax.stop_gradient(jnp.clip(0.9 * jnp.power(jnp.maximum(err, 1e-10), -1.0 / 3.0), 0.2, 5.0))   # (as diffrax)
    return (jnp.where(ok, t + h, t), jnp.where(ok, xn, x), jnp.maximum(h * fac, 1e-7), k + 1)


def cond(c):
    return c[0] < DTOBS


def c0_of(x):
    return (jnp.float32(0.0), x, jnp.float32(1e-2), jnp.int32(0))


def logw_of(x, y):
    return -0.5 * ((y - x) / SIGOBS) ** 2


def setup(n, div):
    rng = np.random.default_rng(0)
    sig = SIG[div]
    y = np.cumsum(rng.normal(0, sig * np.sqrt(DTOBS), NOBS)) * 0.5 + rng.normal(0, SIGOBS, NOBS)
    return dict(n=n, sig=sig, obs=jnp.asarray(y, jnp.float32),
                x0=jnp.asarray(rng.normal(0, 2.0, n), jnp.float32),
                keys=jax.random.split(jax.random.PRNGKey(7), NOBS),
                th=jnp.asarray(TH0 + (sig,), jnp.float32))


def telemetry(ctx):
    """Trip counts of the inner propagation over a REAL forward pass of the filter, with resampling.

    It has to be the real pass: the padded arms are padded to this maximum, and the state distribution after a
    few resampling steps is not the state distribution the initial particles have."""
    prop = propagate_of(ctx, "jax_while", 0)
    x0, keys, obs, n = ctx["x0"], ctx["keys"], ctx["obs"], ctx["n"]

    def run(th):
        def body(x, inp):
            key, y = inp
            kp, kr = jax.random.split(key)
            xn, lw, _, k = prop(x, jax.random.normal(kp, (n,)), th, y)
            return resample(xn, lw, kr), k
        _, ks = lax.scan(body, x0, (keys, obs))
        return ks
    return H.trip_stats(jax.jit(run)(ctx["th"]))


def propagate_of(ctx, arm, maxs):
    """-> f(x, nz, th, y) -> (x_new, logw, weight_sum, trips): one observation step for every particle."""
    if arm.startswith("tm_"):
        def item(t, z, nz, th, y):
            c = lax.while_loop(cond, lambda c: step(c, th), c0_of(z))
            xn = c[1] + th[2] * jnp.sqrt(DTOBS) * nz
            lw = logw_of(xn, y)
            return t.add("w", 0, jnp.exp(lw)), (xn, lw, c[3])
        f = tm.vmap(item, outputs={"w": tm.Sum((1,))}, in_axes=(0, 0, None, None),
                    backend="cuda" if arm == "tm_cuda" else "lockstep")

        def call(x, nz, th, y):
            r = f(x, nz, th, y)
            xn, lw, k = r.per_item
            return xn, lw, r.outputs["w"][0], k
        return call

    def one(z, nz, th, y):
        if arm == "jax_while":
            c = lax.while_loop(cond, lambda c: step(c, th), c0_of(z))
        else:
            c, _ = A.masked_scan(lambda c: step(c, th), cond, c0_of(z), maxs, remat=A.remat_of(arm))
        xn = c[1] + th[2] * jnp.sqrt(DTOBS) * nz
        return xn, logw_of(xn, y), c[3]

    def call(x, nz, th, y):
        xn, lw, k = jax.vmap(one, in_axes=(0, 0, None, None))(x, nz, th, y)
        return xn, lw, jnp.sum(jnp.exp(lw)), k
    return call


def resample(x, lw, key):
    """Systematic resampling, in plain JAX, between the calls.  The indices are stop_gradient-ed: that is what
    makes the gradient below the standard biased SMC estimator."""
    w = jax.nn.softmax(lw)
    cdf = jnp.cumsum(lax.stop_gradient(w))
    n = x.shape[0]
    u = (jnp.arange(n, dtype=jnp.float32) + jax.random.uniform(key)) / n
    idx = jnp.searchsorted(cdf, u)
    return x[jnp.clip(lax.stop_gradient(idx), 0, n - 1)]


def loglik_of(ctx, arm, maxs):
    """The log-likelihood estimate of the whole series: sum_t log( (1/N) sum_i w_it )."""
    prop = propagate_of(ctx, arm, maxs)
    x0, keys, obs, n = ctx["x0"], ctx["keys"], ctx["obs"], ctx["n"]

    def run(th):
        def body(x, inp):
            key, y = inp
            kp, kr = jax.random.split(key)
            nz = jax.random.normal(kp, (n,))
            xn, lw, wsum, _ = prop(x, nz, th, y)
            return resample(xn, lw, kr), jnp.log(wsum / n)
        x, lls = lax.scan(jax.checkpoint(body), x0, (keys, obs))
        return jnp.sum(lls)
    return run


def measured(ctx, arm, mode, maxs):
    ll = loglik_of(ctx, arm, maxs)
    extra = dict(nobs=NOBS, sig=ctx["sig"], padded_steps=A.padded_count(arm, maxs),
                 jacfwd_k=A.jacfwd_k(ctx["th"]))
    return (ll if mode == "fwd" else jax.grad(ll)), (ctx["th"],), extra


def check(ctx, maxs):
    out = A.cross_arm_check(lambda a: (lambda th: jnp.reshape(loglik_of(ctx, a, maxs)(th), (1,))),
                            lambda a: loglik_of(ctx, a, maxs), ctx["th"])
    return out
