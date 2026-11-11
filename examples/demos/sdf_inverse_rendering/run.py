"""Sphere tracing a signed distance field, with an inverse-rendering gradient.

There is no incumbent JAX library here: differentiable sphere tracers in JAX are written by hand, as a `lax.scan`
padded to a fixed step budget, because `lax.while_loop` has no reverse-mode rule.  So the baseline is that program,
padded to the OBSERVED maximum march length, never to an inflated cap.

The scene is a smooth-min union of eight spheres plus a ground plane; theta = (smooth-min k, radius scale, dx, dy,
dz).  A pinhole camera casts N rays; each marches t <- t + max(d(p), 1e-4) until a hit, a miss (t > 30) or the step
cap, and is shaded half-Lambert.  The tally is Sum((32,)), per-tile sums of (shade - target)^2 against a target
image rendered at another theta, so `jax.grad` of its total is the inverse-rendering gradient.

  tm         item = one ray, a `lax.while_loop` that stops at the hit or the miss.
  jax_scan   `jax.vmap` of the same march as a masked `lax.scan` of the observed maximum length.

Both arms run the same march, so forward and gradient agree to f32 rounding (on the CPU the kernel's libm and
XLA's math functions differ in the last bit, and the arms sum the cotangents in different orders).

  python run.py              # CPU, N = 4096 rays
  python run.py --n 1048576  # on a GPU
"""
import argparse
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

NB, K, TMAX, HALF, EPS = 32, 8, 30.0, 0.45, 1e-4
CENTRES = jnp.asarray([[math.cos(2 * math.pi * i / K) * 1.6, math.sin(2 * math.pi * i / K) * 1.6,
                        0.4 * math.sin(3.0 * i)] for i in range(K)], jnp.float32)
RADII = jnp.asarray([0.45 + 0.12 * math.cos(1.7 * i) for i in range(K)], jnp.float32)
THETA0 = jnp.asarray([0.35, 1.0, 0.0, 0.0, 0.0], jnp.float32)
THETA_T = jnp.asarray([0.30, 1.06, 0.05, -0.04, 0.03], jnp.float32)        # the target image's theta
LIGHT = jnp.asarray([-0.267, -0.802, 0.535], jnp.float32)


def sdf(p, th):
    off = jnp.stack([th[2], th[3], th[4]])
    d = jnp.float32(1e9)
    for i in range(K):
        di = jnp.linalg.norm(p - (CENTRES[i] + off)) - RADII[i] * th[1]
        h = jnp.clip(0.5 + 0.5 * (d - di) / th[0], 0.0, 1.0)              # polynomial smooth min
        d = d * (1.0 - h) + di * h - th[0] * h * (1.0 - h)
    return jnp.minimum(d, p[2] + 1.2)


def shade(hit, p, th):
    e = jnp.eye(3, dtype=jnp.float32) * 1e-3
    g = jnp.stack([sdf(p + e[i], th) - sdf(p - e[i], th) for i in range(3)])
    return jnp.where(hit, 0.5 + 0.5 * jnp.dot(g / (jnp.linalg.norm(g) + 1e-12), LIGHT), jnp.float32(0.0))


def is_hit(t, d):
    return (d <= EPS * jnp.maximum(t, 1.0)) & (t < TMAX)


def march_while(o, dr, th, maxs):
    def body(c):
        i, t, _ = c
        d = sdf(o + t * dr, th)
        return i + 1, t + jnp.maximum(d, 1e-4), d
    i, t, d = lax.while_loop(lambda c: (c[0] < maxs) & (c[1] < TMAX) & (c[2] > EPS * jnp.maximum(c[1], 1.0)), body,
                             (jnp.int32(0), jnp.float32(1e-3), jnp.float32(1e9)))
    return is_hit(t, d), t, i


def march_scan(o, dr, th, maxs):
    def step(c, _):
        i, t, d, live = c
        d = jnp.where(live, sdf(o + t * dr, th), d)
        t = jnp.where(live, t + jnp.maximum(d, 1e-4), t)
        i = i + live.astype(jnp.int32)
        return (i, t, d, live & (t < TMAX) & (d > EPS * jnp.maximum(t, 1.0))), None
    (i, t, d, _), _ = lax.scan(step, (jnp.int32(0), jnp.float32(1e-3), jnp.float32(1e9), jnp.bool_(True)), None,
                               length=maxs)
    return is_hit(t, d), t, i


def make_tm(maxs):
    def item(t, o, dr, tile, tgt, th):
        hit, tt, i = march_while(o, dr, th, maxs)
        s = shade(hit, o + tt * dr, th)
        return t.add("res", tile, (s - tgt) ** 2), jnp.stack([s, i.astype(jnp.float32)])
    f = tm.vmap(item, outputs={"res": tm.Sum((NB,))}, in_axes=(0, 0, 0, 0, None), max_steps=maxs + 2)

    def run(orig, dirs, tiles, tgt, th):
        r = f(orig, dirs, tiles, tgt, th)
        return r.outputs["res"], r.per_item
    return run


def make_scan(maxs):
    def run(orig, dirs, tiles, tgt, th):
        def one(o, dr, tg):
            hit, tt, i = march_scan(o, dr, th, maxs)
            return shade(hit, o + tt * dr, th), i
        s, i = jax.vmap(one)(orig, dirs, tgt)
        return jnp.zeros((NB,), jnp.float32).at[tiles].add((s - tgt) ** 2), jnp.stack([s, i.astype(jnp.float32)], 1)
    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4096)
    n = p.parse_args().n
    r = np.random.default_rng(0)
    u, v = r.uniform(-HALF, HALF, n).astype(np.float32), r.uniform(-HALF, HALF, n).astype(np.float32)
    dirs = np.stack([u, np.ones_like(u), v], 1)
    dirs = jnp.asarray(dirs / np.linalg.norm(dirs, axis=1, keepdims=True))
    orig = jnp.tile(jnp.asarray([0.0, -6.0, 0.6], jnp.float32), (n, 1))
    tiles = jnp.asarray(np.floor((u / HALF * 0.5 + 0.5) * 8).clip(0, 7) * 4
                        + np.floor((v / HALF * 0.5 + 0.5) * 4).clip(0, 3), jnp.int32)

    def target(o, d):
        hit, tt, _ = march_scan(o, d, THETA_T, 1024)
        return shade(hit, o + tt * d, THETA_T)
    tgt = jax.jit(jax.vmap(target))(orig, dirs)

    _, probe = jax.jit(make_tm(4096))(orig, dirs, tiles, tgt, THETA0)
    maxs = int(np.asarray(probe[:, 1]).max())                                  # the observed maximum
    ft, fs = make_tm(maxs), make_scan(maxs)
    rt, pt = jax.jit(ft)(orig, dirs, tiles, tgt, THETA0)
    rs, ps = jax.jit(fs)(orig, dirs, tiles, tgt, THETA0)
    gt = jax.jit(jax.grad(lambda th: jnp.sum(ft(orig, dirs, tiles, tgt, th)[0])))(THETA0)
    gs = jax.jit(jax.grad(lambda th: jnp.sum(fs(orig, dirs, tiles, tgt, th)[0])))(THETA0)
    jt = jax.jit(jax.jacfwd(lambda th: jnp.sum(ft(orig, dirs, tiles, tgt, th)[0])))(THETA0)
    st = np.asarray(pt[:, 1])
    checks = [("forward vs the padded scan", D.rel(rt, rs) < 1e-5 and D.rel(pt, ps) < 1e-5,
               f"residual rel = {D.rel(rt, rs):.2g}, per-ray shade and step count rel = {D.rel(pt, ps):.2g}"),
              ("inverse-rendering gradient vs the padded scan's", D.rel(gt, gs) < 1e-4, f"rel = {D.rel(gt, gs):.2g}"),
              ("jacfwd vs grad inside tallymap", D.rel(jt, gt) < 1e-5, f"rel = {D.rel(jt, gt):.2g}"),
              ("the march length varies across rays", st.max() > 4 * st.mean(),
               f"mean {st.mean():.1f}, max {st.max():.0f} steps (the scan is padded to the max)")]
    D.timings(n, {"tallymap grad": (jax.grad(lambda th: jnp.sum(ft(orig, dirs, tiles, tgt, th)[0])), (THETA0,)),
                  "padded scan grad": (jax.grad(lambda th: jnp.sum(fs(orig, dirs, tiles, tgt, th)[0])), (THETA0,))})
    return D.verdict(checks)


if __name__ == "__main__":
    sys.exit(main())
