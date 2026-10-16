"""A particle filter whose propagation step is a variable-length adaptive solve.

    python examples/particle_filter.py

Resampling is an all-to-all over the particles, which tallymap cannot express inside one call, and does not have
to: it happens BETWEEN the calls.

    lax.scan over the observations                        (plain JAX)
      |-- ONE tm.vmap call.  ITEM = one particle between two observations.  It integrates its own state with an
      |   ADAPTIVE Bogacki-Shampine 3(2) step; the drift is strongly nonlinear, so a particle far from the
      |   origin needs many more small steps than one near it, and the trip count is its own.  Then
      |   it adds the diffusion increment and scores itself against the observation.
      |   TALLY = Sum((1,)), the sum of the unnormalised weights, which is this step's likelihood factor.
      |-- normalise the weights, accumulate the log-likelihood, systematic resampling    (plain JAX)

The output is the log-likelihood estimate of the whole series.

The gradient here is the standard stop-gradient-through-resampling estimator: the resampling indices are wrapped
in lax.stop_gradient, which makes this the usual BIASED estimator of the gradient of the log-likelihood.  That is
what practitioners use; it is a property of the estimator, not of tallymap, which returns the exact derivative of
the program as written.  docs/performance.md has the measured comparison.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm

NOBS, DTOBS, SIGOBS = 12, 1.0, 0.3
RTOL, ATOL = 1e-5, 1e-7


def trial(c, theta):
    """ONE adaptive trial step of the drift ODE, accepted or rejected."""
    t, x, h, k = c
    h = jnp.minimum(h, DTOBS - t)
    d = lambda tt, xx: -theta[0] * xx * (1.0 + theta[1] * xx * xx)
    k1 = d(t, x)
    k2 = d(t + 0.5 * h, x + 0.5 * h * k1)
    k3 = d(t + 0.75 * h, x + 0.75 * h * k2)
    xn = x + h * (2.0 / 9.0 * k1 + 1.0 / 3.0 * k2 + 4.0 / 9.0 * k3)
    k4 = d(t + h, xn)
    e = h * ((2.0 / 9.0 - 7.0 / 24.0) * k1 + (1.0 / 3.0 - 0.25) * k2
             + (4.0 / 9.0 - 1.0 / 3.0) * k3 - 0.125 * k4)
    err = jnp.abs(e) / (ATOL + RTOL * jnp.maximum(jnp.abs(x), jnp.abs(xn)))
    ok = err <= 1.0
    fac = lax.stop_gradient(jnp.clip(0.9 * jnp.power(jnp.maximum(err, 1e-10), -1.0 / 3.0), 0.2, 5.0))   # (as diffrax)
    return jnp.where(ok, t + h, t), jnp.where(ok, xn, x), jnp.maximum(h * fac, 1e-7), k + 1


def propagate(t, x, noise, theta, y):
    """ONE item: integrate this particle to the next observation, then weigh it against the observation."""
    c = lax.while_loop(lambda c: c[0] < DTOBS, lambda c: trial(c, theta),
                       (jnp.float32(0.0), x, jnp.float32(1e-2), jnp.int32(0)))
    xn = c[1] + theta[2] * jnp.sqrt(DTOBS) * noise
    logw = -0.5 * ((y - xn) / SIGOBS) ** 2
    return t.add("w", 0, jnp.exp(logw)), (xn, logw, c[3])


step = tm.vmap(propagate, outputs={"w": tm.Sum((1,))}, in_axes=(0, 0, None, None), max_steps=1 << 14)


def resample(x, logw, key):
    """Systematic resampling, in plain JAX, between the calls.  stop_gradient on the indices is what makes the
    gradient below the standard biased SMC estimator."""
    n = x.shape[0]
    cdf = jnp.cumsum(lax.stop_gradient(jax.nn.softmax(logw)))
    u = (jnp.arange(n, dtype=jnp.float32) + jax.random.uniform(key)) / n
    return x[jnp.clip(lax.stop_gradient(jnp.searchsorted(cdf, u)), 0, n - 1)]


N = 4096
rng = np.random.default_rng(0)
obs = jnp.asarray(np.cumsum(rng.normal(0, 0.3, NOBS)) * 0.5 + rng.normal(0, SIGOBS, NOBS), jnp.float32)
x0 = jnp.asarray(rng.normal(0, 2.0, N), jnp.float32)
keys = jax.random.split(jax.random.PRNGKey(7), NOBS)
theta = jnp.array([2.0, 2.0, 1.0], jnp.float32)   # (mean reversion, nonlinearity, diffusion scale)


def loglik(theta):
    """-> the log-likelihood estimate of the series, sum_t log( (1/N) sum_i w_it )."""
    def body(x, inp):
        key, y = inp
        kp, kr = jax.random.split(key)
        res = step(x, jax.random.normal(kp, (N,)), theta, y)
        xn, logw, trips = res.per_item
        return resample(xn, logw, kr), (jnp.log(res.outputs["w"][0] / N), jnp.mean(trips), jnp.max(trips))
    x, (lls, tmean, tmax) = lax.scan(jax.checkpoint(body), x0, (keys, obs))
    return jnp.sum(lls), (tmean, tmax)


ll, (tmean, tmax) = jax.jit(loglik)(theta)
print(f"log-likelihood estimate {float(ll):.4f}   over {NOBS} observations and {N} particles")
print("adaptive trial steps per particle per observation: mean", round(float(tmean.mean()), 1),
      " largest", int(tmax.max()))

# grad against jacfwd: both differentiate the SAME program, the one with stop_gradient on the resampling, so
# they must agree.  Neither is the gradient of the true log-likelihood.
scalar = lambda th: loglik(th)[0]
g = jax.jit(jax.grad(scalar))(theta)
J = jax.jit(jax.jacfwd(scalar))(theta)
e_grad = float(jnp.max(jnp.abs(g - J)) / jnp.max(jnp.abs(J)))
print("d loglik / d theta (biased, stop-gradient through resampling):", np.round(np.asarray(g), 4))
print(f"grad vs jacfwd: rel {e_grad:.1e}")

# The same item in plain JAX, forward only: jax.vmap of the SAME trial step in a lax.while_loop.
def plain_loglik(theta):
    def one(x, noise, y):
        c = lax.while_loop(lambda c: c[0] < DTOBS, lambda c: trial(c, theta),
                           (jnp.float32(0.0), x, jnp.float32(1e-2), jnp.int32(0)))
        xn = c[1] + theta[2] * jnp.sqrt(DTOBS) * noise
        return xn, -0.5 * ((y - xn) / SIGOBS) ** 2

    def body(x, inp):
        key, y = inp
        kp, kr = jax.random.split(key)
        xn, logw = jax.vmap(one, in_axes=(0, 0, None))(x, jax.random.normal(kp, (N,)), y)
        return resample(xn, logw, kr), jnp.log(jnp.sum(jnp.exp(logw)) / N)
    x, lls = lax.scan(body, x0, (keys, obs))
    return jnp.sum(lls)


ll_plain = float(jax.jit(plain_loglik)(theta))
e_fwd = abs(float(ll) - ll_plain) / max(abs(ll_plain), 1e-12)
print(f"log-likelihood vs the plain-JAX arm: {ll_plain:.4f}, rel {e_fwd:.1e}")

bad = [n for n, e in (("forward", e_fwd), ("grad-vs-jacfwd", e_grad)) if not (e < 1e-3)]
print("PARTICLE_FILTER " + ("OK" if not bad else "FAIL " + ",".join(bad)))
assert not bad, bad
