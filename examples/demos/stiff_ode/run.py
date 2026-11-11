"""A stiff ODE ensemble: diffrax's Kvaerno3 against the same solver run as a tallymap item.

A forced Van der Pol oscillator with a per-trajectory stiffness mu, augmented with a path functional z' = x^2 so
that the tallied quantity is smooth and under the solver's error control:

    x' = v,   v' = mu (1 - x^2) v - w2 x + A cos(om t),   z' = x^2,   t in [0, 20],   theta = (w2, A, om)

The solver is Kvaerno3 (an implicit ESDIRK) with a chord root finder and diffrax's default PI controller, so the
item holds three nested data-dependent loops.  `kv3.py` is a re-implementation of that, NOT diffrax's code: it
agrees with diffrax to the solver tolerance but takes a different number of steps.

  tm         `kv3.solve` as a tallymap item; tally = group sums of the terminal z.
  jax_while  `jax.vmap` of the identical `kv3.solve`: same code, so the same answer (on the CPU the step counts can
             differ: last-bit differences flip some of the controller's accept/reject decisions).
  diffrax    `diffrax.diffeqsolve(ODETerm, Kvaerno3(), PIDController(rtol=1e-4, atol=1e-6))` under `jax.vmap`,
             when diffrax is installed.  Without it, the gradient is checked against `jacfwd` instead.

  python run.py              # CPU, N = 256
  python run.py --n 262144   # on a GPU
"""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import _demo as D  # noqa: E402
import kv3  # noqa: E402

import tallymap as tm  # noqa: E402

NB, T_END, MAXSTEPS = 32, 20.0, 4096
THETA0 = jnp.asarray([1.0, 0.3, 0.9], jnp.float32)


def item(t, y0, mu, gid, th):
    y, _, ns, _, _ = kv3.solve(y0, mu, th, jnp.float32(T_END))
    return t.add("obs", gid, y[2]), jnp.stack([y[2], ns.astype(jnp.float32)])


f_tm = tm.vmap(item, outputs={"obs": tm.Sum((NB,))}, in_axes=(0, 0, 0, None), max_steps=MAXSTEPS)


def run_tm(y0s, mus, gids, th):
    r = f_tm(y0s, mus, gids, th)
    return r.outputs["obs"], r.per_item


def run_while(y0s, mus, gids, th):
    y, _, ns, _, _ = jax.vmap(lambda a, b: kv3.solve(a, b, th, jnp.float32(T_END)))(y0s, mus)
    return jnp.zeros((NB,), jnp.float32).at[gids].add(y[:, 2]), jnp.stack([y[:, 2], ns.astype(jnp.float32)], 1)


def make_diffrax(dfx):
    term = dfx.ODETerm(lambda t, y, args: kv3.vf(t, y, args[0], args[1]))
    ctrl = dfx.PIDController(rtol=kv3.RTOL_DEF, atol=kv3.ATOL_DEF)

    def run(y0s, mus, gids, th):
        def one(y0, mu):
            s = dfx.diffeqsolve(term, dfx.Kvaerno3(), t0=0.0, t1=T_END, dt0=None, y0=y0, args=(mu, th),
                                stepsize_controller=ctrl, max_steps=MAXSTEPS, saveat=dfx.SaveAt(t1=True), throw=False)
            return s.ys[-1], s.stats["num_steps"]
        y, ns = jax.vmap(one)(y0s, mus)
        return jnp.zeros((NB,), jnp.float32).at[gids].add(y[:, 2]), jnp.stack([y[:, 2], ns.astype(jnp.float32)], 1)
    return run


def loss(f):
    return lambda th, *a: jnp.sum(f(*a, th)[0] ** 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=256)
    n = p.parse_args().n
    r = np.random.default_rng(0)
    mus = jnp.asarray(np.exp(r.uniform(0.0, np.log(30.0), n)), jnp.float32)
    y0s = jnp.asarray(np.tile([2.0, 0.0, 0.0], (n, 1)), jnp.float32)
    gids = jnp.asarray(np.arange(n) % NB, jnp.int32)
    args = (y0s, mus, gids)
    dfx = D.have("diffrax")

    ot, pt = jax.jit(run_tm)(*args, THETA0)
    ow, pw = jax.jit(run_while)(*args, THETA0)
    gt = np.asarray(jax.jit(jax.grad(loss(run_tm)))(THETA0, *args))
    ns = np.asarray(pt[:, 1])
    same = float(np.mean(ns == np.asarray(pw[:, 1])))       # a last-bit difference can move one step-size decision
    # step counts are reported, not judged: on the CPU the kernel (IEEE, libm, no FMA) and XLA:CPU differ in the
    # last bit, which flips some accept/reject decisions of the step controller without changing the answer
    checks = [("tallymap vs the identical while_loop program: the answer",
               D.rel(ot, ow) < 1e-4,
               f"tally rel = {D.rel(ot, ow):.2g}, step counts equal for {100 * same:.0f}% of trajectories"),
              ("the step count varies across trajectories", ns.max() > 1.2 * ns.mean(),
               f"mean {ns.mean():.0f}, max {ns.max():.0f}")]
    arms = {"tallymap": (run_tm, (*args, THETA0)), "vmap + while_loop": (run_while, (*args, THETA0))}
    if dfx is not None:
        rd = make_diffrax(dfx)
        od, pd = jax.jit(rd)(*args, THETA0)
        gd = np.asarray(jax.jit(jax.grad(loss(rd)))(THETA0, *args))
        checks += [("tallymap vs diffrax: tally and terminal z, to the solver tolerance",
                    max(D.rel(ot, od), D.rel(pt[:, 0], pd[:, 0])) < 2e-3,
                    f"tally rel = {D.rel(ot, od):.2g}, per-trajectory z rel = {D.rel(pt[:, 0], pd[:, 0]):.2g}"),
                   ("the gradient vs diffrax's", D.rel(gt, gd) < 5e-2, f"rel = {D.rel(gt, gd):.2g}"),
                   ("kv3's step count is within 2x of diffrax's", 0.5 < ns.sum() / float(pd[:, 1].sum()) < 2.0,
                    f"{ns.sum():,.0f} vs {float(pd[:, 1].sum()):,.0f} attempted steps")]
        arms["diffrax"] = (rd, (*args, THETA0))
    else:
        J = np.asarray(jax.jit(jax.jacfwd(loss(run_tm)))(THETA0, *args))
        checks.append(("the gradient (replay adjoint) vs jacfwd", D.rel(gt, J) < 1e-4, f"rel = {D.rel(gt, J):.2g}"))
    D.timings(n, arms)
    return D.verdict(checks)


if __name__ == "__main__":
    sys.exit(main())
