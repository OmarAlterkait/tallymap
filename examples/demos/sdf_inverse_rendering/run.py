"""Sphere tracing a signed distance field, with an inverse-rendering gradient.

There is no incumbent JAX library here, and the README says so.  Differentiable sphere tracers in JAX are
written by hand, and what they are written as is a `lax.scan` padded to a fixed step budget, because
`lax.while_loop` has no reverse-mode rule.  So the baselines are the two programs a practitioner actually
writes, and the padded ones are padded to the OBSERVED maximum march length, never to an inflated cap.

The scene is a CSG ring of eight spheres combined with the standard polynomial smooth-min union, plus a ground
plane.  `theta = (k_smoothmin, radius_scale, dx, dy, dz)` are five differentiated scene parameters.  A pinhole
camera casts N rays; each marches `t <- t + max(d(p), 1e-4)` until `|d| < eps * max(t, 1)` (a hit), `t > 30`
(a miss) or the step cap.  Shading is half-Lambert with the normal from central differences of the SDF.  The
tally is an inverse-rendering residual, `Sum((32,))` per-tile sums of `(shade - target)^2` against a target
image rendered once at a different `theta`, so `jax.grad` of its total is the gradient a practitioner wants.

  tm          item = one ray, a `lax.while_loop` that stops at the hit or the miss.
  jax_while   `jax.vmap` of the identical march.  Forward only: JAX cannot reverse-differentiate it.
  jax_scan    `jax.vmap` of the same march rewritten as a masked `lax.scan` of the observed maximum length.
  jax_ckpt    `jax_scan` with a nested, `jax.checkpoint`-ed inner scan (recursive checkpointing).

Divergence knob: `--div` sets (camera half-angle, hit tolerance).  A wider cone means more rays that graze a
silhouette or miss entirely; a tighter tolerance means more small steps at the end of a grazing march.

Estimator caveat, stated and not fixed: the hit point of a sphere-traced ray is a discontinuous function of the
scene parameters at a silhouette, so these pathwise gradients are biased estimators of the true shape gradient.
tallymap returns the exact derivative of the program as written, and `--quick` checks that all three
differentiable arms agree to the last bit.  It is the caveat every sphere tracer carries; the standard fixes
(edge sampling, reparameterisation) sit in the user's renderer.

  python run.py --quick
  python run.py --full
"""
import math
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
LIBS = ("jax",)
NB = 32                      # image tiles (the tally)
K = 8                        # SDF primitives
TMAX = 30.0
DIV = {"low": (0.18, 1e-3), "mid": (0.45, 1e-4), "high": (0.90, 1e-5)}
ARMS = ("tm", "jax_while", "jax_scan", "jax_ckpt")
LABEL = {"tm": "tallymap", "jax_while": "vmap + while_loop", "jax_scan": "vmap + padded scan",
         "jax_ckpt": "padded scan + remat"}

CENTRES = np.asarray([[math.cos(2 * math.pi * i / K) * 1.6, math.sin(2 * math.pi * i / K) * 1.6,
                       0.4 * math.sin(3.0 * i)] for i in range(K)], np.float32)
RADII = np.asarray([0.45 + 0.12 * math.cos(1.7 * i) for i in range(K)], np.float32)
THETA0 = np.asarray([0.35, 1.0, 0.0, 0.0, 0.0], np.float32)     # (smooth-min k, radius scale, 3 offsets)
THETA_T = np.asarray([0.30, 1.06, 0.05, -0.04, 0.03], np.float32)    # the target image's theta
LIGHT = jnp.asarray([-0.267, -0.802, 0.535], jnp.float32)


def sdf(p, th, centres, radii):
    """Smooth union of K spheres and a ground plane.  theta: (k, rscale, dx, dy, dz)."""
    k, rs = th[0], th[1]
    off = jnp.stack([th[2], th[3], th[4]])
    d = jnp.float32(1e9)
    for i in range(K):
        di = jnp.linalg.norm(p - (centres[i] + off)) - radii[i] * rs
        h = jnp.clip(0.5 + 0.5 * (d - di) / k, 0.0, 1.0)             # polynomial smooth min
        d = d * (1.0 - h) + di * h - k * h * (1.0 - h)
    return jnp.minimum(d, p[2] + 1.2)


def normal(p, th, centres, radii):
    e = jnp.float32(1e-3)
    g = jnp.stack([sdf(p + jnp.asarray([e, 0., 0.], jnp.float32), th, centres, radii)
                   - sdf(p - jnp.asarray([e, 0., 0.], jnp.float32), th, centres, radii),
                   sdf(p + jnp.asarray([0., e, 0.], jnp.float32), th, centres, radii)
                   - sdf(p - jnp.asarray([0., e, 0.], jnp.float32), th, centres, radii),
                   sdf(p + jnp.asarray([0., 0., e], jnp.float32), th, centres, radii)
                   - sdf(p - jnp.asarray([0., 0., e], jnp.float32), th, centres, radii)])
    return g / (jnp.linalg.norm(g) + 1e-12)


def shade(hit, p, th, centres, radii):
    """Half-Lambert shading: smooth and never saturated, so the inverse-rendering gradient does not vanish."""
    return jnp.where(hit, 0.5 + 0.5 * jnp.dot(normal(p, th, centres, radii), LIGHT), jnp.float32(0.0))


def march_while(o, dr, th, centres, radii, maxs, eps):
    """Data-dependent sphere tracing (`lax.while_loop`) -> (hit, t, steps)."""
    def body(c):
        i, t, _ = c
        d = sdf(o + t * dr, th, centres, radii)
        return i + 1, t + jnp.maximum(d, 1e-4), d

    def cond(c):
        i, t, d = c
        return (i < maxs) & (t < TMAX) & (d > eps * jnp.maximum(t, 1.0))
    i, t, d = lax.while_loop(cond, body, (jnp.int32(0), jnp.float32(1e-3), jnp.float32(1e9)))
    return (d <= eps * jnp.maximum(t, 1.0)) & (t < TMAX), t, i


def march_scan(o, dr, th, centres, radii, maxs, eps, remat=False):
    """The differentiable plain-JAX way: a masked scan of `maxs` steps."""
    def step(c, _):
        i, t, d, live = c
        dd = sdf(o + t * dr, th, centres, radii)
        d = jnp.where(live, dd, d)
        t = jnp.where(live, t + jnp.maximum(d, 1e-4), t)
        i = i + live.astype(jnp.int32)
        live = live & (t < TMAX) & (d > eps * jnp.maximum(t, 1.0))
        return (i, t, d, live), None
    init = (jnp.int32(0), jnp.float32(1e-3), jnp.float32(1e9), jnp.bool_(True))
    if remat:
        m = int(round(math.sqrt(maxs)))
        outer = (maxs + m - 1) // m
        inner = jax.checkpoint(lambda c, _: lax.scan(step, c, None, length=m))
        c, _ = lax.scan(inner, init, None, length=outer)
    else:
        c, _ = lax.scan(step, init, None, length=maxs)
    i, t, d, live = c
    return (d <= eps * jnp.maximum(t, 1.0)) & (t < TMAX), t, i


def rays(n, div, seed=0):
    r = np.random.default_rng(seed)
    half = DIV[div][0]
    u = r.uniform(-half, half, n).astype(np.float32)
    v = r.uniform(-half, half, n).astype(np.float32)
    dirs = np.stack([u, np.ones_like(u), v], 1)                    # pinhole camera at y = -6 looking along +y
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    orig = np.tile(np.asarray([0.0, -6.0, 0.6], np.float32), (n, 1))
    tile = (np.floor((u / half * 0.5 + 0.5) * 8).clip(0, 7) * 4
            + np.floor((v / half * 0.5 + 0.5) * 4).clip(0, 3)).astype(np.int32)
    return jnp.asarray(orig), jnp.asarray(dirs), jnp.asarray(tile)


def make_tm(maxs, eps, **opts):
    def item(t, o, dr, tile, tgt, th, centres, radii):
        hit, tt, i = march_while(o, dr, th, centres, radii, maxs, eps)
        s = shade(hit, o + tt * dr, th, centres, radii)
        t = t.add("res", tile, (s - tgt) ** 2)
        t = t.add("cnt", jnp.asarray([0, 1]), jnp.stack([i.astype(jnp.float32), hit.astype(jnp.float32)]))
        return t, jnp.stack([s, tt, i.astype(jnp.float32), hit.astype(jnp.float32)])
    return tm.vmap(item, outputs={"res": tm.Sum((NB,)), "cnt": tm.Sum((2,))},
                   in_axes=(0, 0, 0, 0, None, None, None), max_steps=maxs + 2, **opts)


def make_jax(maxs, eps, kind):
    def one(o, dr, tgt, th, c, rr):
        if kind == "while":
            hit, tt, i = march_while(o, dr, th, c, rr, maxs, eps)
        else:
            hit, tt, i = march_scan(o, dr, th, c, rr, maxs, eps, remat=(kind == "ckpt"))
        return shade(hit, o + tt * dr, th, c, rr), tt, i, hit

    def f(orig, dirs, tiles, tgt, th, c, rr):
        s, tt, i, hit = jax.vmap(one, in_axes=(0, 0, 0, None, None, None))(orig, dirs, tgt, th, c, rr)
        res = jnp.zeros((NB,), jnp.float32).at[tiles].add((s - tgt) ** 2)
        cnt = jnp.stack([i.sum().astype(jnp.float32), hit.sum().astype(jnp.float32)])
        return {"res": res, "cnt": cnt}, jnp.stack([s, tt, i.astype(jnp.float32), hit.astype(jnp.float32)], 1)
    return f


def wrap(g):
    def f(*args):
        r = g(*args)
        return r.outputs, r.per_item
    return f


def build(arm, maxs, eps):
    if arm == "tm":
        return wrap(make_tm(maxs, eps))
    return make_jax(maxs, eps, {"jax_while": "while", "jax_scan": "scan", "jax_ckpt": "ckpt"}[arm])


def setup(n, div):
    half, eps = DIV[div]
    orig, dirs, tiles = rays(n, div)
    c, rr = jnp.asarray(CENTRES), jnp.asarray(RADII)

    def tgt_fn(o, d):
        hit, tt, _ = march_scan(o, d, jnp.asarray(THETA_T), c, rr, 1024, eps)
        return shade(hit, o + tt * d, jnp.asarray(THETA_T), c, rr)
    tgt = jax.block_until_ready(jax.jit(jax.vmap(tgt_fn))(orig, dirs))
    return orig, dirs, tiles, tgt, c, rr, eps, half


# ------------------------------------------------------------------------------------------------ one measurement
def one(a):
    orig, dirs, tiles, tgt, c, rr, eps, half = setup(a.n, a.div)
    D.banner(f"sdf arm={a.arm} mode={a.mode} n={a.n} div={a.div} half={half} eps={eps} maxs={a.maxs}", LIBS)
    th = jnp.asarray(THETA0)
    base = dict(demo="sdf_inverse_rendering", arm=a.arm, mode=a.mode, n=a.n, div=a.div, maxs=a.maxs,
                eps=eps, half=half)

    if a.mode == "calib":                      # the observed maximum march length, which sizes the scan arms
        f = build("jax_while", a.maxs, eps)
        _, pi = jax.block_until_ready(jax.jit(lambda t_: f(orig, dirs, tiles, tgt, t_, c, rr))(th))
        st = np.asarray(pi[:, 2])
        D.row(**base, status="ok", steps_mean=float(st.mean()), steps_max=float(st.max()),
              hit_frac=float(np.asarray(pi[:, 3]).mean()))
        return

    f = build(a.arm, a.maxs, eps)
    loss = lambda th_: jnp.sum(f(orig, dirs, tiles, tgt, th_, c, rr)[0]["res"])           # noqa: E731

    def go():
        if a.mode == "fwd":
            o, r = D.measure(lambda t_: f(orig, dirs, tiles, tgt, t_, c, rr), (th,), reps=a.reps)
            od, pi = o
            D.row(**base, status="ok", **r, res0=float(od["res"][0]),
                  steps_mean=float(np.asarray(pi[:, 2]).mean()),
                  steps_max=float(np.asarray(pi[:, 2]).max()),
                  hit_frac=float(np.asarray(pi[:, 3]).mean()))
        elif a.mode == "grad":
            o, r = D.measure(jax.grad(loss), (th,), reps=a.reps)
            D.row(**base, status="ok", **r, grad=np.asarray(o))
        elif a.mode == "jacfwd":
            o, r = D.measure(jax.jacfwd(loss), (th,), reps=a.reps)
            D.row(**base, status="ok", **r, jac=np.asarray(o))
        else:
            raise SystemExit(a.mode)
    D.guard(go, base)


# -------------------------------------------------------------------------------------------------------- --quick
def quick(a):
    n = min(a.n, 4096)
    D.banner(f"quick n={n}", LIBS)
    orig, dirs, tiles, tgt, c, rr, eps, half = setup(n, "mid")
    th = jnp.asarray(THETA0)
    # size the padded arms from what the while_loop arm actually does, not from a guess
    fw = build("jax_while", 4096, eps)
    _, pw = jax.jit(lambda t_: fw(orig, dirs, tiles, tgt, t_, c, rr))(th)
    maxs = int(np.asarray(pw[:, 2]).max())
    out = {}
    for arm in ARMS:
        f = build(arm, maxs, eps)
        o, pi = jax.block_until_ready(jax.jit(lambda t_: f(orig, dirs, tiles, tgt, t_, c, rr))(th))
        g = None
        if arm != "jax_while":
            g = np.asarray(jax.block_until_ready(jax.jit(jax.grad(
                lambda t_: jnp.sum(f(orig, dirs, tiles, tgt, t_, c, rr)[0]["res"])))(th)))
        out[arm] = (np.asarray(o["res"]), np.asarray(pi), g)
    ro, rp, rg = out["jax_scan"]
    checks = [(f"the padded arms are padded to the observed maximum ({maxs} steps)", True,
               f"mean {np.asarray(pw[:, 2]).mean():.1f}, max {maxs} march steps")]
    for arm in ARMS:
        res, pi, g = out[arm]
        ok = (D.rel(res, ro) == 0.0 and float(np.mean(pi[:, 3] == rp[:, 3])) == 1.0
              and float(np.mean(pi[:, 2] == rp[:, 2])) == 1.0)
        checks.append((f"{LABEL[arm]} vs the padded scan, bitwise", ok,
                       f"residual rel = {D.rel(res, ro):.2g}, hit flag and step count equal for "
                       f"{100 * float(np.mean(pi[:, 3] == rp[:, 3])):.0f}% / "
                       f"{100 * float(np.mean(pi[:, 2] == rp[:, 2])):.0f}% of rays"))
        if g is not None and arm != "jax_scan":
            # the forward pass is bitwise identical across arms; the gradient is not, because each arm
            # accumulates the five parameter cotangents in a different order
            checks.append((f"{LABEL[arm]}: the inverse-rendering gradient vs the padded scan's",
                           D.rel(g, rg) < 1e-4, f"rel = {D.rel(g, rg):.2g}"))
    jt = np.asarray(jax.jit(jax.jacfwd(lambda t_: jnp.sum(
        build("tm", maxs, eps)(orig, dirs, tiles, tgt, t_, c, rr)[0]["res"])))(th))
    _, _, gt = out["tm"]
    checks.append(("jacfwd agrees with grad inside tallymap", D.rel(jt, gt) < 1e-5,
                   f"rel = {D.rel(jt, gt):.2g}"))
    st = np.asarray(out["tm"][1][:, 2])
    checks.append(("the march length varies strongly across rays", st.max() > 4 * st.mean(),
                   f"mean {st.mean():.1f}, max {st.max():.0f} (ratio {st.max() / st.mean():.1f})"))
    return D.verdict(checks)


# --------------------------------------------------------------------------------------------------------- --full
NS = (16384, 65536, 262144, 1048576)
MAXS = {"low": 118, "mid": 242, "high": 467}        # fallbacks only; `--full` measures these first


def plots(rows, meta):
    """Redraw plots/ from `rows`.  Called by --full and by --plots."""
    P = os.path.join(HERE, "plots")
    fwd = [r for r in rows if r.get("mode") == "fwd" and r.get("div") == "mid"]
    gr = [r for r in rows if r.get("mode") == "grad" and r.get("div") == "mid"]
    D.panel(os.path.join(P, "fwd_time.png"), fwd, ARMS, LABEL,
            f"Sphere tracing, forward: time per call (scan length {MAXS['mid']} = the observed maximum)",
            "ms per call", xlabel="rays")
    D.panel(os.path.join(P, "grad_time.png"), gr, ("tm", "jax_scan", "jax_ckpt"), LABEL,
            "Inverse-rendering gradient w.r.t. five scene parameters: time", "ms per call", xlabel="rays",
            note="`grad` of the while_loop arm raises in JAX, so it has no line here.")
    D.panel(os.path.join(P, "grad_mem.png"), gr, ("tm", "jax_scan", "jax_ckpt"), LABEL,
            "Inverse-rendering gradient: peak device memory", "peak device MB", xlabel="rays", ykey="peak_mb")

def full(a):
    script = os.path.join(HERE, "run.py")
    out = os.path.join(HERE, "results", "sdf_inverse_rendering.json")
    # stage 1: measure the longest march at the largest N, which is what the padded arms are sized for
    rows = D.run_jobs(script, [dict(arm="jax_while", mode="calib", n=max(NS), div=d, maxs=4096) for d in DIV])
    ms = {r["div"]: int(r["steps_max"]) for r in rows if r.get("status") == "ok"}
    ms = {d: ms.get(d, MAXS[d]) for d in DIV}
    print("\npadded-scan lengths (the observed maximum march at the largest N): " + str(ms), flush=True)

    jobs = [dict(arm=arm, mode="fwd", n=n, div="mid", maxs=ms["mid"]) for n in NS for arm in ARMS]
    jobs += [dict(arm=arm, mode="grad", n=n, div="mid", maxs=ms["mid"]) for n in NS
             for arm in ("tm", "jax_scan", "jax_ckpt")]
    for d in ("low", "high"):
        jobs += [dict(arm=arm, mode="fwd", n=262144, div=d, maxs=ms[d]) for arm in ARMS]
        jobs += [dict(arm=arm, mode="grad", n=262144, div=d, maxs=ms[d]) for arm in ("tm", "jax_ckpt")]
    rows += D.run_jobs(script, jobs)
    rows, meta = D.write_results(out, rows, libs=LIBS, extra_meta=dict(ns=list(NS), scan_lengths=ms))
    plots(rows, meta)

    print(f"\n{len(rows)} rows -> {out}")
    return 0


def main():
    p = D.parser(maxs=242)
    a = p.parse_args()
    if a.one:
        one(a)
        return 0
    if a.plots:
        rows, meta = D.load(os.path.join(HERE, "results", "sdf_inverse_rendering.json"))
        plots(rows, meta)
        return 0
    return quick(a) if a.quick else full(a)


if __name__ == "__main__":
    sys.exit(main())
