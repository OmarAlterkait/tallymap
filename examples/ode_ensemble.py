"""An ensemble of adaptive-step ODE solves, with a gradient through the solver.

    python examples/ode_ensemble.py

ITEM = one trajectory of a forced Van der Pol oscillator, integrated with a Bogacki-Shampine 3(2) embedded pair
and the usual error-per-step controller.  One loop iteration is one TRIAL step, accepted or rejected; both cost
the same and both are data dependent.  TALLY = Sum((G,)), group sums of a smooth terminal observable.  The
divergence is the per-item stiffness mu, which sets how many steps each trajectory needs.

The point of this example is the gradient.  A padded masked scan has to store (or recompute) every step of
every trajectory to reverse through it; tallymap replays each item instead, so the memory of the gradient does
not depend on the trip count at all.

Measured on an L40S at N = 1,048,576: the gradient takes 26 ms in 28 MB.  A padded scan given the oracle step
cap takes 187 ms in 26 GB; a checkpointed one at a realistic 2x cap takes 450 ms in 2.5 GB.  See
docs/performance_detail.md, including the comparison against diffrax.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm

G, T = 16, 20.0
RTOL, ATOL = 1e-3, 1e-5


def rhs(t, y, mu, th):
    return jnp.stack([y[1],
                      mu * th[0] * (1.0 - y[0] * y[0]) * y[1] - y[0] - th[1] * y[1] + th[2] * jnp.sin(2.0 * t)])


def trial(c, mu, th):
    """ONE trial step: Bogacki-Shampine 3(2), then the step-size controller.  Accepted or rejected."""
    t, y, h, n = c
    h = jnp.minimum(h, T - t)
    k1 = rhs(t, y, mu, th)
    k2 = rhs(t + 0.5 * h, y + 0.5 * h * k1, mu, th)
    k3 = rhs(t + 0.75 * h, y + 0.75 * h * k2, mu, th)
    yn = y + h * (2.0 / 9.0 * k1 + 1.0 / 3.0 * k2 + 4.0 / 9.0 * k3)
    k4 = rhs(t + h, yn, mu, th)
    e = h * ((2.0 / 9.0 - 7.0 / 24.0) * k1 + (1.0 / 3.0 - 0.25) * k2
             + (4.0 / 9.0 - 1.0 / 3.0) * k3 - 0.125 * k4)
    err = jnp.max(jnp.abs(e) / (ATOL + RTOL * jnp.maximum(jnp.abs(y), jnp.abs(yn))))
    ok = err <= 1.0
    # The controller is NOT differentiated (diffrax stop-gradients the factor the same way).  Its feedback is
    # chaotic for the stiff items: through it, the f32 gradient of this loss has the opposite sign to the f64
    # one and moves 5x under a 1e-7 change of theta.  Detached, the gradient is that of the solution on the
    # chosen steps, and f32 agrees with f64.
    fac = lax.stop_gradient(jnp.clip(0.9 * jnp.power(jnp.maximum(err, 1e-10), -1.0 / 3.0), 0.2, 5.0))
    return jnp.where(ok, t + h, t), jnp.where(ok, yn, y), jnp.maximum(h * fac, 1e-7), n + 1


def observable(y):
    return jnp.tanh(y[0]) + 0.5 * y[1] * jnp.exp(-0.1 * y[1] * y[1])


def trajectory(t, y0, mu, group, theta):
    """ONE item: integrate to T, then add the terminal observable to this item's group bin."""
    c = lax.while_loop(lambda c: c[0] < T, lambda c: trial(c, mu, theta),
                       (jnp.float32(0.0), y0, jnp.float32(1e-2), jnp.int32(0)))
    return t.add("obs", group, observable(c[1])), c[3]


f = tm.vmap(trajectory, outputs={"obs": tm.Sum((G,))}, in_axes=(0, 0, 0, None), max_steps=1 << 16)

N = 4096
rng = np.random.default_rng(0)
y0 = jnp.asarray(np.stack([rng.uniform(-2.0, 2.0, N), rng.uniform(-1.0, 1.0, N)], 1), jnp.float32)
mu = jnp.asarray(np.exp(rng.uniform(np.log(0.5), np.log(6.0), N)), jnp.float32)   # the stiffness spread
group = jnp.asarray(np.arange(N) % G, jnp.int32)
theta = jnp.array([1.0, 0.05, 0.3], jnp.float32)
w = jnp.asarray(rng.uniform(0.5, 1.5, G), jnp.float32)

res = jax.jit(f)(y0, mu, group, theta)
print("group means of the observable", np.round(np.asarray(res.outputs["obs"][:4] / (N / G)), 4))
print("trial steps per trajectory: mean", round(float(res.per_item.mean()), 1),
      " max", int(res.per_item.max()), " truncated", int(res.stats["n_truncated"]))

# The gradient.  Reverse mode replays each item, so its memory does not grow with the step count; forward mode
# runs the 3 parameter tangents as dual lanes of one kernel.  They must agree.
loss = lambda th: jnp.sum(w * (f(y0, mu, group, th).outputs["obs"] / (N / G)) ** 2)
g = jax.jit(jax.grad(loss))(theta)
J = jax.jit(jax.jacfwd(loss))(theta)
e = float(jnp.max(jnp.abs(g - J)) / jnp.max(jnp.abs(J)))
print("d loss / d theta: grad", np.round(np.asarray(g), 5))
print(f"grad vs jacfwd: rel {e:.1e}")

# The same item in plain JAX, forward only: jax.vmap of the SAME trial step in a lax.while_loop.  JAX cannot
# reverse-differentiate that loop, which is why the plain-JAX gradient arm in docs/performance_detail.md is a padded
# masked scan instead.
def plain(theta):
    def one(y0_, mu_):
        c = lax.while_loop(lambda c: c[0] < T, lambda c: trial(c, mu_, theta),
                           (jnp.float32(0.0), y0_, jnp.float32(1e-2), jnp.int32(0)))
        return observable(c[1])
    return jnp.zeros((G,), jnp.float32).at[group].add(jax.vmap(one)(y0, mu)) / (N / G)


o_tm = res.outputs["obs"] / (N / G)
e_fwd = float(jnp.max(jnp.abs(o_tm - jax.jit(plain)(theta))) / jnp.max(jnp.abs(o_tm)))
print(f"forward vs the plain-JAX arm: rel {e_fwd:.1e}")

# grad and jacfwd sum the per-item contributions in a different order (measured 2e-6 to 4e-6, CPU JAX 0.6 to
# 0.11 and L40S).  The forward bound is looser: the kernel backend's arithmetic is not XLA's bit for bit, so a stiff
# trajectory can take a different step sequence and agree only to about the solver tolerance
# (docs/float_semantics.md).
bad = [n for n, v, tol in (("forward", e_fwd, 2e-3), ("grad-vs-jacfwd", e, 1e-4)) if not (v < tol)]
print("ODE_ENSEMBLE " + ("OK" if not bad else "FAIL " + ",".join(bad)))
assert not bad, bad
