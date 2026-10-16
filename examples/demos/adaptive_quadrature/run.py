"""Adaptive quadrature per item: quadax's globally adaptive Gauss-Kronrod against a local adaptive item.

The integral is `w / (1 + a (x - c)^2)` over [0, 1] to 1e-6, with a per-item peak sharpness `a` drawn
log-uniformly.  A sharper peak means more subdivisions, so `--div` is a direct knob on how much the per-item
work varies.

  quadax      the incumbent, `quadax.quadgk` (quadax 0.3.0) run as it ships under `jax.vmap`, at its default
              `max_ninter=50`.  It is GLOBALLY adaptive, so it carries four length-50 interval arrays per item
              (`quadax/adaptive.py:217-224`), about 200 floats of per-item state.
  tm          one integral = one tallymap item: LOCAL adaptive Simpson with an explicit stack of depth 24,
              48 floats of per-item state.
  jax_while   the identical local adaptive Simpson under `jax.vmap` + `lax.while_loop`.  This is the fairness
              control: it runs the same Python function as `tm`, so the only difference is the execution model.

These are two different algorithms, and the README says so.  quadax is globally adaptive and more robust on a
nasty integrand; the item is the locally adaptive algorithm that fits in registers.  They are compared at the
same requested tolerance, and `--quick` measures how far apart their answers land.

  python run.py --quick
  python run.py --full
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _demo as D  # noqa: E402

import tallymap as tm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LIBS = ("quadax", "jax")
NB = 32                                                    # tally bins (groups of items)
DIVS = {"low": 40.0, "mid": 400.0, "high": 4000.0}         # per item the sharpness is drawn from [1, this]
TOL = 1e-6
DMAX = 24                                                  # the item's stack depth
ARMS = ("tm", "jax_while", "quadax")
LABEL = {"tm": "tallymap", "jax_while": "vmap + while_loop", "quadax": "quadax.quadgk"}


def inputs(n, div, seed=0):
    r = np.random.default_rng(seed)
    a = np.exp(r.uniform(0.0, np.log(DIVS[div]), n)).astype(np.float32)
    c = r.uniform(0.2, 0.8, n).astype(np.float32)
    return jnp.asarray(a), jnp.asarray(c), jnp.asarray((np.arange(n) % NB).astype(np.int32))


def integrand(x, a, c, w):
    """A Lorentzian peak at c of width 1/sqrt(a), scaled by w (the differentiated parameter)."""
    return w / (1.0 + a * (x - c) ** 2)


def simpson(a, c, w, maxit):
    """Local adaptive Simpson with an explicit stack: the textbook recursive rule, made iterative.
    Per-item state: a stack of (lo, hi) pairs of depth DMAX, so 2*DMAX floats."""
    def simp(lo, hi):
        m = 0.5 * (lo + hi)
        h = hi - lo
        f0, f1, f2 = integrand(lo, a, c, w), integrand(m, a, c, w), integrand(hi, a, c, w)
        return h / 6.0 * (f0 + 4.0 * f1 + f2), m

    def body(s):
        sp, lo_s, hi_s, acc, k = s
        sp = sp - 1
        lo, hi = lo_s[sp], hi_s[sp]
        whole, m = simp(lo, hi)
        left, _ = simp(lo, m)
        right, _ = simp(m, hi)
        err = jnp.abs(left + right - whole)
        done = (err < TOL * (hi - lo + 1e-6)) | (sp >= DMAX - 2)
        acc = acc + jnp.where(done, left + right + (left + right - whole) / 15.0, 0.0)
        lo_s = jnp.where(done, lo_s, lo_s.at[sp].set(lo).at[sp + 1].set(m))
        hi_s = jnp.where(done, hi_s, hi_s.at[sp].set(m).at[sp + 1].set(hi))
        sp = sp + jnp.where(done, 0, 2)
        return sp, lo_s, hi_s, acc, k + 1

    lo0 = jnp.zeros((DMAX,), jnp.float32).at[0].set(0.0)
    hi0 = jnp.zeros((DMAX,), jnp.float32).at[0].set(1.0)
    sp, lo_s, hi_s, acc, k = lax.while_loop(lambda s: (s[0] > 0) & (s[4] < maxit), body,
                                            (jnp.int32(1), lo0, hi0, jnp.float32(0.0), jnp.int32(0)))
    return acc, k


def make_tm(maxit, **opts):
    def item(t, a, c, gid, w):
        v, k = simpson(a, c, w, maxit)
        t = t.add("I", gid, v)
        return t, jnp.stack([v, k.astype(jnp.float32)])
    return tm.vmap(item, outputs={"I": tm.Sum((NB,))}, in_axes=(0, 0, 0, None),
                   max_steps=maxit + 2, **opts)


def make_jax_while(maxit):
    def f(a, c, gid, w):
        v, k = jax.vmap(lambda x, y: simpson(x, y, w, maxit))(a, c)
        return {"I": jnp.zeros((NB,), jnp.float32).at[gid].add(v)}, jnp.stack([v, k.astype(jnp.float32)], 1)
    return f


def make_quadax(ninter):
    import quadax

    def one(a, c, w):
        v, info = quadax.quadgk(lambda x, aa, cc, ww: integrand(x, aa, cc, ww), [0.0, 1.0], args=(a, c, w),
                                epsabs=TOL, epsrel=TOL, max_ninter=ninter)
        return v, info.neval

    def f(a, c, gid, w):
        v, k = jax.vmap(one, in_axes=(0, 0, None))(a, c, w)
        return ({"I": jnp.zeros((NB,), jnp.float32).at[gid].add(v)},
                jnp.stack([v, k.astype(jnp.float32)], 1))
    return f


def wrap(g):
    def f(*args):
        r = g(*args)
        return r.outputs, r.per_item
    return f


def build(arm, maxit, ninter):
    if arm == "tm":
        return wrap(make_tm(maxit))
    if arm == "jax_while":
        return make_jax_while(maxit)
    if arm == "quadax":
        return make_quadax(ninter)
    raise SystemExit(f"unknown arm {arm}")


# ------------------------------------------------------------------------------------------------ one measurement
def one(a):
    D.banner(f"quad arm={a.arm} mode={a.mode} n={a.n} div={a.div} maxit={a.maxit} ninter={a.ninter}", LIBS)
    aa, cc, gid = inputs(a.n, a.div)
    w = jnp.float32(1.0)
    base = dict(demo="adaptive_quadrature", arm=a.arm, mode=a.mode, n=a.n, div=a.div, maxit=a.maxit,
                ninter=a.ninter, tol=TOL, dmax=DMAX)
    f = build(a.arm, a.maxit, a.ninter)
    call = lambda w_: f(aa, cc, gid, w_)                                                 # noqa: E731

    def go():
        if a.mode == "fwd":
            o, r = D.measure(call, (w,), reps=a.reps)
            od, pi = o
            D.row(**base, status="ok", **r, I0=float(od["I"][0]),
                  it_mean=float(np.asarray(pi[:, 1]).mean()), it_max=float(np.asarray(pi[:, 1]).max()))
        elif a.mode == "grad":
            o, r = D.measure(jax.grad(lambda w_: jnp.sum(call(w_)[0]["I"] ** 2)), (w,), reps=a.reps)
            D.row(**base, status="ok", **r, grad=float(np.asarray(o)))
        else:
            raise SystemExit(a.mode)
    D.guard(go, base)


# -------------------------------------------------------------------------------------------------------- --quick
def quick(a):
    n = min(a.n, 4096)
    D.banner(f"quick n={n}", LIBS)
    aa, cc, gid = inputs(n, "mid")
    w = jnp.float32(1.0)
    checks = []
    o = {arm: jax.jit(lambda w_, arm=arm: build(arm, 4096, 50)(aa, cc, gid, w_))(w) for arm in ARMS}
    g = {arm: float(np.asarray(jax.jit(jax.grad(
        lambda w_, arm=arm: jnp.sum(build(arm, 4096, 50)(aa, cc, gid, w_)[0]["I"] ** 2)))(w)))
        for arm in ("tm", "quadax")}
    ot, pt = o["tm"]
    ow, pw = o["jax_while"]
    oq, pq = o["quadax"]
    checks.append(("tallymap vs the identical while_loop program, bitwise",
                   D.rel(ot["I"], ow["I"]) == 0.0 and D.rel(pt[:, 0], pw[:, 0]) == 0.0,
                   f"tally rel = {D.rel(ot['I'], ow['I']):.2g}, "
                   f"per-item integral rel = {D.rel(pt[:, 0], pw[:, 0]):.2g}"))
    rq, rqi = D.rel(ot["I"], oq["I"]), D.rel(pt[:, 0], pq[:, 0])
    # two different algorithms at the same tolerance, so the contract is the tolerance, not bitwise
    checks.append(("the item's answer matches quadax.quadgk to the requested tolerance",
                   rqi < 1e-3 and rq < 1e-4,
                   f"per-item rel = {rqi:.2g}, tally rel = {rq:.2g} (two different algorithms at "
                   f"epsabs=epsrel={TOL:g})"))
    checks.append(("the gradient w.r.t. the integrand amplitude matches quadax's",
                   abs(g["tm"] - g["quadax"]) / abs(g["quadax"]) < 1e-3,
                   f"{g['tm']:.8g} vs {g['quadax']:.8g}, rel = "
                   f"{abs(g['tm'] - g['quadax']) / abs(g['quadax']):.2g}"))
    it = np.asarray(pt[:, 1])
    checks.append(("the subdivision count varies across items", it.max() > 1.5 * it.mean(),
                   f"mean {it.mean():.1f}, max {it.max():.0f} subdivisions"))
    return D.verdict(checks)


# --------------------------------------------------------------------------------------------------------- --full
NS = (16384, 65536, 262144, 1048576)


def plots(rows, meta):
    """Redraw plots/ from `rows`.  Called by --full and by --plots."""
    P = os.path.join(HERE, "plots")
    fwd = [r for r in rows if r.get("mode") == "fwd" and r.get("div") == "mid"]
    gr = [r for r in rows if r.get("mode") == "grad" and r.get("div") == "mid"]
    D.panel(os.path.join(P, "fwd_time.png"), fwd, ARMS, LABEL,
            "Adaptive quadrature to 1e-6, forward: time per call", "ms per call")
    D.panel(os.path.join(P, "grad_time.png"), gr, ("tm", "quadax"), LABEL,
            "Gradient w.r.t. the integrand amplitude: time", "ms per call")
    D.panel(os.path.join(P, "grad_mem.png"), gr, ("tm", "quadax"), LABEL,
            "Gradient w.r.t. the integrand amplitude: peak device memory", "peak device MB", ykey="peak_mb")

def full(a):
    script = os.path.join(HERE, "run.py")
    out = os.path.join(HERE, "results", "adaptive_quadrature.json")
    jobs = [dict(arm=arm, mode="fwd", n=n, div="mid") for n in NS for arm in ARMS]
    jobs += [dict(arm=arm, mode="grad", n=n, div="mid") for n in (65536, 262144) for arm in ("tm", "quadax")]
    jobs += [dict(arm=arm, mode="grad", n=262144, div=d) for d in ("low", "high") for arm in ("tm", "quadax")]
    jobs += [dict(arm=arm, mode="fwd", n=262144, div=d) for d in ("low", "high") for arm in ARMS]
    rows, meta = D.sweep(script, jobs, out, libs=LIBS, extra_meta=dict(ns=list(NS)))
    plots(rows, meta)

    print(f"\n{len(rows)} rows -> {out}")
    return 0


def main():
    p = D.parser(maxit=4096, ninter=50)
    a = p.parse_args()
    if a.one:
        one(a)
        return 0
    if a.plots:
        rows, meta = D.load(os.path.join(HERE, "results", "adaptive_quadrature.json"))
        plots(rows, meta)
        return 0
    return quick(a) if a.quick else full(a)


if __name__ == "__main__":
    sys.exit(main())
