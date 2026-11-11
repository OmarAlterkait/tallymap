"""Adaptive quadrature, one integral per item, against quadax.quadgk.

N integrals of `w / (1 + a (x - c)^2)` over [0, 1] to 1e-6, with the peak sharpness `a` drawn log-uniformly
per item, so the number of subdivisions varies from item to item.

  tm         one integral = one tallymap item: LOCALLY adaptive Simpson with an explicit stack of depth 24.
  jax_while  the identical Simpson function under `jax.vmap` + `lax.while_loop`: same code, other execution model.
  quadax     `quadax.quadgk` (quadax 0.3.0) under `jax.vmap` at its default `max_ninter=50`: GLOBALLY adaptive
             Gauss-Kronrod, a different algorithm, so it is compared at the same tolerance, not bitwise.
             Skipped when quadax is not installed.

The closed form `w / sqrt(a) (atan(sqrt(a) (1 - c)) + atan(sqrt(a) c))` checks all of them.

  python run.py            # CPU, N = 4096
  python run.py --n 262144 # the size the README's numbers were measured at, on a GPU
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

NB = 32              # tally bins (groups of items)
TOL = 1e-6
DMAX = 24            # the item's stack depth
MAXIT = 4096


def integrand(x, a, c, w):
    return w / (1.0 + a * (x - c) ** 2)


def simpson(a, c, w):
    """Local adaptive Simpson with an explicit stack of (lo, hi) pairs: the recursive rule made iterative."""
    def simp(lo, hi):
        m = 0.5 * (lo + hi)
        return (hi - lo) / 6.0 * (integrand(lo, a, c, w) + 4.0 * integrand(m, a, c, w) + integrand(hi, a, c, w)), m

    def body(s):
        sp, lo_s, hi_s, acc, k = s
        sp = sp - 1
        lo, hi = lo_s[sp], hi_s[sp]
        whole, m = simp(lo, hi)
        left, _ = simp(lo, m)
        right, _ = simp(m, hi)
        done = (jnp.abs(left + right - whole) < TOL * (hi - lo + 1e-6)) | (sp >= DMAX - 2)
        acc = acc + jnp.where(done, left + right + (left + right - whole) / 15.0, 0.0)
        lo_s = jnp.where(done, lo_s, lo_s.at[sp].set(lo).at[sp + 1].set(m))
        hi_s = jnp.where(done, hi_s, hi_s.at[sp].set(m).at[sp + 1].set(hi))
        return sp + jnp.where(done, 0, 2), lo_s, hi_s, acc, k + 1

    lo0 = jnp.zeros((DMAX,), jnp.float32)
    hi0 = jnp.zeros((DMAX,), jnp.float32).at[0].set(1.0)
    _, _, _, acc, k = lax.while_loop(lambda s: (s[0] > 0) & (s[4] < MAXIT), body,
                                     (jnp.int32(1), lo0, hi0, jnp.float32(0.0), jnp.int32(0)))
    return acc, k


def item(t, a, c, gid, w):
    v, k = simpson(a, c, w)
    return t.add("I", gid, v), jnp.stack([v, k.astype(jnp.float32)])


f_tm = tm.vmap(item, outputs={"I": tm.Sum((NB,))}, in_axes=(0, 0, 0, None), max_steps=MAXIT + 2)


def run_tm(a, c, gid, w):
    r = f_tm(a, c, gid, w)
    return r.outputs["I"], r.per_item[:, 0], r.per_item[:, 1]


def run_while(a, c, gid, w):
    v, k = jax.vmap(lambda x, y: simpson(x, y, w))(a, c)
    return jnp.zeros((NB,), jnp.float32).at[gid].add(v), v


def make_quadax(quadax):
    def run(a, c, gid, w):
        v = jax.vmap(lambda aa, cc: quadax.quadgk(integrand, [0.0, 1.0], args=(aa, cc, w), epsabs=TOL,
                                                  epsrel=TOL, max_ninter=50)[0])(a, c)
        return jnp.zeros((NB,), jnp.float32).at[gid].add(v), v
    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4096)
    n = p.parse_args().n
    r = np.random.default_rng(0)
    a = np.exp(r.uniform(0.0, np.log(400.0), n)).astype(np.float32)
    c = r.uniform(0.2, 0.8, n).astype(np.float32)
    exact = (np.arctan(np.sqrt(a) * (1 - c)) + np.arctan(np.sqrt(a) * c)) / np.sqrt(a)       # w = 1, in f64
    a, c, gid, w = jnp.asarray(a), jnp.asarray(c), jnp.asarray(np.arange(n) % NB, jnp.int32), jnp.float32(1.0)
    quadax = D.have("quadax")

    It, vt, kt = jax.jit(run_tm)(a, c, gid, w)
    Iw, vw = jax.jit(run_while)(a, c, gid, w)
    gt = float(jax.jit(jax.grad(lambda w_: jnp.sum(run_tm(a, c, gid, w_)[0] ** 2)))(w))
    ge = 2.0 * float(np.sum(np.bincount(np.arange(n) % NB, exact) ** 2))                      # d/dw sum(I^2)
    checks = [
        ("tallymap vs the identical while_loop program", D.rel(It, Iw) < 1e-5 and D.rel(vt, vw) < 1e-5,
         f"tally rel = {D.rel(It, Iw):.2g}, per-item rel = {D.rel(vt, vw):.2g}"),
        ("tallymap vs the closed form", D.rel(vt, exact) < 1e-4, f"per-item rel = {D.rel(vt, exact):.2g}"),
        ("the gradient w.r.t. the amplitude vs the closed form", abs(gt - ge) / ge < 1e-3, f"{gt:.7g} vs {ge:.7g}"),
        ("the subdivision count varies across items", float(kt.max()) > 1.5 * float(kt.mean()),
         f"mean {float(kt.mean()):.1f}, max {float(kt.max()):.0f}")]
    arms = {"tallymap": (run_tm, (a, c, gid, w)), "vmap + while_loop": (run_while, (a, c, gid, w))}
    if quadax is not None:
        rq = make_quadax(quadax)
        Iq, vq = jax.jit(rq)(a, c, gid, w)
        gq = float(jax.jit(jax.grad(lambda w_: jnp.sum(rq(a, c, gid, w_)[0] ** 2)))(w))
        checks += [("tallymap vs quadax.quadgk, to the tolerance", D.rel(vt, vq) < 1e-3 and D.rel(It, Iq) < 1e-4,
                    f"per-item rel = {D.rel(vt, vq):.2g}, tally rel = {D.rel(It, Iq):.2g}"),
                   ("the gradient vs quadax's", abs(gt - gq) / abs(gq) < 1e-3, f"{gt:.7g} vs {gq:.7g}")]
        arms["quadax.quadgk"] = (rq, (a, c, gid, w))
    D.timings(n, arms)
    return D.verdict(checks)


if __name__ == "__main__":
    sys.exit(main())
