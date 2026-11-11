"""A branching-process outbreak as a tallymap spawn tree, against the best plain-JAX program for the same job.

There is no incumbent JAX library for individual-level branching processes (the established tools are R packages
that walk one lineage at a time on the CPU), so the baseline is the best plain-JAX program: ONE global generation
frontier pooled over all N outbreaks, each child slot mapped back to its parent by a `searchsorted`.

The model (Lloyd-Smith et al. 2005; Hellewell et al. 2020): an individual infected at t_i in generation g_i draws
nu_i = R0 exp(sigma Z - sigma^2/2), has K_i ~ Poisson(nu_i) offspring infected at t_i + Weibull(alpha, beta), and is
detected (so K_i = 0) with probability p_det.  A lineage stops at t > 40 days or generation 10.

  tm         item = one infected individual; its offspring loop `spawn`s each child, which runs the same function.
  jax_front  a `lax.scan` over generations of one global frontier of `front` slots.  `front` must be chosen in
             advance; here it is sized from what the spawn tree measures, and its dropped count must be zero.

theta = (log R0, Weibull shape, Weibull scale, logit p_det).  Only the two generation-interval parameters have a
pathwise derivative; the other two enter through discrete draws, so theirs is exactly zero in both arms.

  python run.py             # CPU, N = 1024
  python run.py --n 262144  # on a GPU
"""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _demo as D  # noqa: E402

import tallymap as tm  # noqa: E402

NT, T_OBS, GMAX = 32, 40.0, 10       # incidence bins, observation horizon (days), generation cap
NU_MAX, SIGMA, MAXOFF = 400.0, 1.0, 512
THETA0 = np.array([np.log(1.55), 2.5, 6.0, -1.2], np.float32)


def draw_nu(key, logR0):
    return jnp.minimum(jnp.exp(logR0 + SIGMA * jax.random.normal(key) - 0.5 * SIGMA * SIGMA), NU_MAX)


def poisson_count(key, lam):
    """K ~ Poisson(lam) by exponential interarrivals."""
    def body(c):
        n, s, i = c
        return n + 1, s - jnp.log(jax.random.uniform(jax.random.fold_in(key, i), minval=1e-20)), i + 1
    n, _, _ = lax.while_loop(lambda c: (c[1] <= lam) & (c[0] < MAXOFF), body,
                             (jnp.int32(0), jnp.float32(0.0), jnp.int32(0)))
    return jnp.maximum(n - 1, 0)


def weibull(key, alpha, beta):
    return beta * jnp.power(-jnp.log(jax.random.uniform(key, minval=1e-20)), 1.0 / alpha)


def smooth_w(t):
    return jnp.exp(-0.5 * ((t - 14.0) / 7.0) ** 2)


def tbin(t):
    return jnp.clip((t * (NT / T_OBS)).astype(jnp.int32), 0, NT - 1)


def individual(t, p, theta):
    kd, kn, ko, kg = jax.random.split(p["key"], 4)
    detected = jax.random.uniform(kd) < jax.nn.sigmoid(theta[3])
    n = jnp.where(detected, 0, poisson_count(ko, draw_nu(kn, theta[0])))
    t = t.add("inc", tbin(p["t"]), 1.0)
    t = t.add("gen", jnp.clip(p["g"], 0, GMAX), 1.0)
    t = t.add("size", 0, 1.0)
    t = t.add("smooth", 0, smooth_w(p["t"]))
    t = t.add("drawn", jnp.clip(p["g"], 0, GMAX), n.astype(jnp.float32))     # what the frontier must hold

    def body(c):
        i, t = c
        ki = jax.random.fold_in(kg, i)
        tc = p["t"] + weibull(ki, theta[1], theta[2])
        ok = (tc < T_OBS) & (p["g"] + 1 <= GMAX)
        return i + 1, t.spawn(dict(t=tc, g=p["g"] + 1, key=jax.random.fold_in(ki, 7)), when=ok)
    _, t = lax.while_loop(lambda c: c[0] < n, body, (jnp.int32(0), t))
    return t


f_tm = tm.vmap(individual, outputs={"inc": tm.Sum((NT,)), "gen": tm.Sum((GMAX + 1,)), "size": tm.Sum((1,)),
                                    "smooth": tm.Sum((1,)), "drawn": tm.Sum((GMAX + 1,))},
               in_axes=(0, None), spawn=tm.Queue(1 << 22), max_steps=MAXOFF)


def make_front(front):
    """One global frontier of `front` slots; children are packed densely by a searchsorted on the cumsum."""
    def call(roots, theta):
        n0 = roots["t"].shape[0]
        slots = jnp.arange(front)
        pad = lambda a, v: jnp.concatenate([a, jnp.full((front - n0,) + a.shape[1:], v, a.dtype)])  # noqa: E731

        def gen_step(c, _):
            tt, gg, kk, al, drop = c
            ks = jax.vmap(lambda x: jax.random.split(x, 4))(kk)
            det = jax.vmap(jax.random.uniform)(ks[:, 0]) < jax.nn.sigmoid(theta[3])
            nu = jax.vmap(lambda x: draw_nu(x, theta[0]))(ks[:, 1])
            n = jnp.where(al & ~det, jax.vmap(poisson_count)(ks[:, 2], nu), 0)
            m = jnp.where(al, 1.0, 0.0)
            out = (jnp.zeros((NT,), jnp.float32).at[tbin(tt)].add(m),
                   jnp.zeros((GMAX + 1,), jnp.float32).at[jnp.clip(gg, 0, GMAX)].add(m),
                   jnp.sum(m), jnp.sum(jnp.where(al, smooth_w(tt), 0.0)))
            cs = jnp.cumsum(n)
            par = jnp.minimum(jnp.searchsorted(cs, slots, side="right"), front - 1)   # child slot -> parent
            ki = jax.vmap(jax.random.fold_in)(ks[:, 3][par], slots - (cs - n)[par])
            tc = tt[par] + jax.vmap(lambda k_: weibull(k_, theta[1], theta[2]))(ki)
            gc = gg[par] + 1
            ok = (slots < cs[-1]) & (tc < T_OBS) & (gc <= GMAX)
            kc = jax.vmap(lambda k_: jax.random.fold_in(k_, 7))(ki)
            return (jnp.where(ok, tc, 0.0), jnp.where(ok, gc, 0), kc, ok, drop + jnp.maximum(cs[-1] - front, 0)), out

        c0 = (pad(roots["t"], 0.0), pad(roots["g"], 0), pad(roots["key"], 0),
              jnp.arange(front) < n0, jnp.int32(0))
        c, outs = lax.scan(gen_step, c0, None, length=GMAX + 1)
        return tuple(jnp.sum(o, axis=0) for o in outs) + (c[-1],)
    return call


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1024)
    n = p.parse_args().n
    rng = np.random.default_rng(0)
    roots = dict(t=jnp.asarray(rng.uniform(0.0, 4.0, n), jnp.float32), g=jnp.zeros((n,), jnp.int32),
                 key=jax.random.split(jax.random.PRNGKey(11), n))
    theta = jnp.asarray(THETA0)

    rt = jax.jit(f_tm)(roots, theta)
    front = max(n, int(1.05 * float(np.max(np.asarray(rt.outputs["drawn"])))) + 1)      # measured, not guessed
    fj = make_front(front)
    o = jax.jit(fj)(roots, theta)
    loss_tm = lambda th: f_tm(roots, th).outputs["smooth"][0] / n                          # noqa: E731
    loss_jx = lambda th: fj(roots, th)[3] / n                                              # noqa: E731
    g = np.asarray(jax.jit(jax.grad(loss_tm))(theta))
    J = np.asarray(jax.jit(jax.jacfwd(loss_tm))(theta))
    gx = np.asarray(jax.jit(jax.grad(loss_jx))(theta))
    st, sj = float(rt.outputs["size"][0]) / n, float(o[2]) / n
    checks = [
        (f"no children dropped (spawn Queue; frontier sized {front:,})",
         int(rt.stats["children_dropped"]) == 0 and int(o[-1]) == 0,
         f"{int(rt.stats['spawned']):,} children spawned; dropped {int(rt.stats['children_dropped'])} / {int(o[-1])}"),
        ("incidence and infections by generation vs the frontier", max(D.rel(rt.outputs["inc"], o[0]),
                                                                       D.rel(rt.outputs["gen"], o[1])) < 1e-5,
         f"rel = {D.rel(rt.outputs['inc'], o[0]):.2g} / {D.rel(rt.outputs['gen'], o[1]):.2g}"),
        ("outbreak size per index case", abs(st - sj) / sj < 1e-4, f"{st:.4f} vs {sj:.4f}"),
        ("tree adjoint grad vs jacfwd", D.rel(g, J) < 1e-5, f"grad = {np.array2string(g, precision=5)}"),
        ("jacfwd vs the frontier's grad", D.rel(J, gx) < 1e-5, f"rel = {D.rel(J, gx):.2g}"),
        ("log R0 and logit p_det have no pathwise derivative", g[0] == 0 and g[3] == 0 and abs(g[1]) + abs(g[2]) > 0,
         f"d/d log R0 = {g[0]:.3g}, d/d logit p_det = {g[3]:.3g}")]
    D.timings(n, {"tallymap spawn tree": (lambda th: f_tm(roots, th).outputs["inc"], (theta,)),
                  "global generation frontier": (lambda th: fj(roots, th)[0], (theta,))})
    return D.verdict(checks)


if __name__ == "__main__":
    sys.exit(main())
