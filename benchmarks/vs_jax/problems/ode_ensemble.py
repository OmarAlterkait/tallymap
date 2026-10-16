"""Adaptive Runge-Kutta ODE ensemble, measured against plain JAX and against diffrax itself.

ITEM      one trajectory of a forced Van der Pol oscillator,
            y0' = y1,  y1' = mu * th[0] * (1 - y0^2) * y1 - y0 - th[1] * y1 + th[2] * sin(2 t),
          integrated from t = 0 to t = 20 with a Bogacki-Shampine 3(2) embedded pair and the usual
          error-per-step controller: accept when the scaled error norm is <= 1, then h *= clip(0.9 err^(-1/3),
          0.2, 5).  One loop iteration is one TRIAL step, accepted or rejected; both cost the same and both are
          data dependent.
TALLY     Sum((32,)): group sums of a smooth terminal observable.  One emit per item, at the end.
DIVERGE   the per-item stiffness mu, which sets the number of accepted and of rejected steps.
THETA     3 parameters (stiffness scale, damping, forcing amplitude).

The `diffrax` arm is diffrax's own `diffeqsolve` with `Bosh3` and a `PIDController`, vmapped over the ensemble,
with `RecursiveCheckpointAdjoint` for the gradient.  It is the library a JAX user would actually reach for, and
it needs a `max_steps`, so it is told the same observed maximum the padded-scan arms get.  diffrax is an optional
dependency of this suite: where it is not installed the arm reports status SKIP.
"""
import arms as A
import harness as H
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm

NAME = "ode_ensemble"
TITLE = "Adaptive Runge-Kutta ODE ensemble (Van der Pol)"
GROUP = "natural"
AXIS = "n"
SWEEP = {"full": (1 << 14, 1 << 16, 1 << 18, 1 << 20), "quick": (1 << 8, 1 << 10)}
DIVS = ("mid", "low", "high")
ARMS = ("tm_cuda", "tm_lockstep", "jax_while", "jax_scan", "jax_scan2x", "jax_ckpt", "jax_ckpt2x", "diffrax")
MODES = ("fwd", "jvp", "jacfwd", "grad")
CHECK_VAL = 1 << 8
DIV_VAL = 1 << 18      # the other divergence levels are measured at this one size

G, T = 32, 20.0
RTOL, ATOL = 1e-3, 1e-5
MU = {"low": (1.0, 1.2), "mid": (0.5, 6.0), "high": (0.2, 20.0)}
TH0 = (1.0, 0.05, 0.3)


def rhs(t, y, mu, th):
    return jnp.stack([y[1],
                      mu * th[0] * (1.0 - y[0] * y[0]) * y[1] - y[0] - th[1] * y[1] + th[2] * jnp.sin(2.0 * t)])


def bs23(t, y, h, mu, th):
    """One Bogacki-Shampine 3(2) trial step -> (y_new, scaled error norm)."""
    k1 = rhs(t, y, mu, th)
    k2 = rhs(t + 0.5 * h, y + 0.5 * h * k1, mu, th)
    k3 = rhs(t + 0.75 * h, y + 0.75 * h * k2, mu, th)
    yn = y + h * (2.0 / 9.0 * k1 + 1.0 / 3.0 * k2 + 4.0 / 9.0 * k3)
    k4 = rhs(t + h, yn, mu, th)
    e = h * ((2.0 / 9.0 - 7.0 / 24.0) * k1 + (1.0 / 3.0 - 0.25) * k2
             + (4.0 / 9.0 - 1.0 / 3.0) * k3 + (-0.125) * k4)
    sc = ATOL + RTOL * jnp.maximum(jnp.abs(y), jnp.abs(yn))
    return yn, jnp.max(jnp.abs(e) / sc)


def step(c, mu, th):
    """ONE trial step of ONE trajectory, accepted or rejected.  Shared by every arm."""
    t, y, h, nacc, nrej = c
    h = jnp.minimum(h, T - t)
    yn, err = bs23(t, y, h, mu, th)
    ok = err <= 1.0
    fac = lax.stop_gradient(jnp.clip(0.9 * jnp.power(jnp.maximum(err, 1e-10), -1.0 / 3.0), 0.2, 5.0))   # (as diffrax)
    return (jnp.where(ok, t + h, t), jnp.where(ok, yn, y), jnp.maximum(h * fac, 1e-7),
            nacc + ok.astype(jnp.int32), nrej + (~ok).astype(jnp.int32))


def cond(c):
    return c[0] < T


def obs(y):
    """A smooth terminal observable, so the pathwise gradient is not a hard bin."""
    return jnp.tanh(y[0]) + 0.5 * y[1] * jnp.exp(-0.1 * y[1] * y[1])


def c0_of(y0):
    return (jnp.float32(0.0), y0, jnp.float32(1e-2), jnp.int32(0), jnp.int32(0))


def setup(n, div):
    rng = np.random.default_rng(0)
    lo, hi = MU[div]
    return dict(n=n, mu_lo=lo, mu_hi=hi,
                y0=jnp.asarray(np.stack([rng.uniform(-2.0, 2.0, n), rng.uniform(-1.0, 1.0, n)], 1), jnp.float32),
                mu=jnp.asarray(np.exp(rng.uniform(np.log(lo), np.log(hi), n)), jnp.float32),
                gid=jnp.asarray(np.arange(n) % G, jnp.int32), th=jnp.asarray(TH0, jnp.float32),
                w=jnp.asarray(rng.uniform(0.5, 1.5, G), jnp.float32))


def telemetry(ctx):
    def one(y0, mu, th):
        c = lax.while_loop(cond, lambda c: step(c, mu, th), c0_of(y0))
        return c[3] + c[4]
    k = jax.jit(jax.vmap(one, in_axes=(0, 0, None)))(ctx["y0"], ctx["mu"], ctx["th"])
    return H.trip_stats(k)


def _diffrax_sim(ctx, maxs, mode="fwd"):
    """diffrax's own diffeqsolve, vmapped over the ensemble.

    The adjoint has to be chosen by mode.  `RecursiveCheckpointAdjoint`, diffrax's default and the one to
    compare a reverse-mode gradient against, is a `custom_vjp`: forward-mode autodiff through it raises
    "can't apply forward-mode autodiff (jvp) to a custom_vjp function".  The jvp and jacfwd rows therefore use
    `diffrax.ForwardMode()`, which is a different program, and the rows say so."""
    try:
        import diffrax
    except ImportError as e:
        raise H.Unavailable(f"diffrax is not installed: {e}") from e
    y0, mu, gid, n = ctx["y0"], ctx["mu"], ctx["gid"], ctx["n"]
    term = diffrax.ODETerm(lambda t, y, args: rhs(t, y, args[0], args[1]))
    solver = diffrax.Bosh3()
    ctrl = diffrax.PIDController(rtol=RTOL, atol=ATOL, pcoeff=0.0, icoeff=1.0, dcoeff=0.0)
    adjoint = diffrax.ForwardMode() if mode in ("jvp", "jacfwd") else diffrax.RecursiveCheckpointAdjoint()
    # diffrax's max_steps is a hard bound it raises on, and its controller is not bit-identical to the one the
    # other arms share, so the oracle value is not safe for it: it gets the realistic 2x cap, and says so.
    maxs = 2 * int(maxs)

    def one(y0_, mu_, th):
        sol = diffrax.diffeqsolve(term, solver, t0=0.0, t1=T, dt0=1e-2, y0=y0_, args=(mu_, th),
                                  stepsize_controller=ctrl, saveat=diffrax.SaveAt(t1=True),
                                  adjoint=adjoint, max_steps=int(maxs))
        return obs(sol.ys[0])

    def call(th):
        o = jax.vmap(one, in_axes=(0, 0, None))(y0, mu, th)
        return A.scatter_add(G, gid, o) / (n / G)
    return call


def sim_of(ctx, arm, maxs, mode="fwd"):
    y0, mu, gid, n = ctx["y0"], ctx["mu"], ctx["gid"], ctx["n"]
    w = n / G
    if arm == "diffrax":
        return _diffrax_sim(ctx, maxs, mode)
    if arm.startswith("tm_"):
        def traj(t_, y, m, g, th):
            c = lax.while_loop(cond, lambda c: step(c, m, th), c0_of(y))
            o = obs(c[1])
            return t_.add("obs", g, o), (o, c[3] + c[4])
        f = tm.vmap(traj, outputs={"obs": tm.Sum((G,))}, in_axes=(0, 0, 0, None),
                    backend="cuda" if arm == "tm_cuda" else "lockstep")
        return lambda th: f(y0, mu, gid, th).outputs["obs"] / w

    def one(y, m, th):
        if arm == "jax_while":
            c = lax.while_loop(cond, lambda c: step(c, m, th), c0_of(y))
        else:
            c, _ = A.masked_scan(lambda c: step(c, m, th), cond, c0_of(y), maxs, remat=A.remat_of(arm))
        return obs(c[1])

    def call(th):
        return A.scatter_add(G, gid, jax.vmap(one, in_axes=(0, 0, None))(y0, mu, th)) / w
    return call


def loss_of(ctx, arm, maxs, mode="fwd"):
    sim = sim_of(ctx, arm, maxs, mode)
    return lambda th: jnp.sum(ctx["w"] * sim(th) ** 2)


def measured(ctx, arm, mode, maxs):
    extra = dict(ngroups=G, mu_lo=ctx["mu_lo"], mu_hi=ctx["mu_hi"], jacfwd_k=A.jacfwd_k(ctx["th"]),
                 padded_steps=int(maxs) if arm == "diffrax" else A.padded_count(arm, maxs))
    if arm == "diffrax":
        extra.update(padded_steps=2 * int(maxs), diffrax_max_steps=2 * int(maxs),
                     diffrax_adjoint="ForwardMode" if mode in ("jvp", "jacfwd")
                     else "RecursiveCheckpointAdjoint")
    return (A.mode_fn(mode, sim_of(ctx, arm, maxs, mode), loss_of(ctx, arm, maxs, mode), ctx["th"]),
            (ctx["th"],), extra)


def check(ctx, maxs):
    return A.cross_arm_check(lambda a: sim_of(ctx, a, maxs), lambda a: loss_of(ctx, a, maxs), ctx["th"],
                             arms=("tm_cuda", "jax_while", "jax_scan", "jax_ckpt", "diffrax"))
