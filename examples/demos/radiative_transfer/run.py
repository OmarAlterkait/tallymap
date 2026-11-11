"""Radiative transfer on Ray-trax's own steady-state problem: Ray-trax against one ray per tallymap item.

Ray-trax (github.com/lorenzobranca/Ray-trax) is a differentiable 3-D ray tracer in JAX for the emission-absorption
transfer equation on a regular grid.  Its per-ray kernel is a `lax.fori_loop` of a fixed number of steps that
interpolates the emissivity trilinearly, looks up the opacity at the nearest voxel, updates (I, tau) and deposits
I trilinearly onto a FULL GRID ALLOCATED PER RAY; rays are batched with `jax.vmap` and summed.

  tm        item = one ray from the source, a `while_loop` that stops when it leaves the box; tally Sum((G^3,)).
  jax_scan  the best plain-JAX rewrite: ONE shared grid in a `lax.scan` carry with the rays vectorised inside the
            step, padded to the longest exit path.
  raytrax   `ray_trax.ray_trax_3D.compute_radiation_field_from_source` as it ships, when Ray-trax is importable
            (a source repository: clone it and set RAYTRAX_PATH, see the README).

The step function below is Ray-trax's `trace_ray` body; tm and jax_scan both call it.  The checks: forward field
and the inverse-problem gradient w.r.t. both fields, tm against jax_scan (and against Ray-trax when present).

  python run.py              # CPU, 2048 rays on a 16^3 grid
  python run.py --n 65536    # on a GPU
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

G, DS = 16, 0.5                                   # grid size, ray-march step in grid units (Ray-trax's default)
MAXS = int(np.ceil(np.sqrt(3) * (G - 1) / 2 / DS)) + 1     # the longest exit path from the centre, in steps


def smooth_field(seed, lo, hi):
    f = np.fft.fftfreq(G)
    k = f[:, None, None] ** 2 + f[None, :, None] ** 2 + f[None, None, :] ** 2
    a = np.real(np.fft.ifftn(np.fft.fftn(np.random.default_rng(seed).normal(size=(G, G, G))) * np.exp(-k * G * G / 8)))
    return np.asarray(lo + (hi - lo) * (a - a.min()) / (a.max() - a.min()), np.float32).reshape(-1)


def corners(x, y, z):
    """Ray-trax's `_corners`: 8 clipped corner indices and their trilinear weights."""
    fx, fy, fz = jnp.floor(x), jnp.floor(y), jnp.floor(z)
    x0, y0, z0 = (jnp.clip(v.astype(jnp.int32), 0, G - 1) for v in (fx, fy, fz))
    x1, y1, z1 = (jnp.clip(v + 1, 0, G - 1) for v in (x0, y0, z0))
    dx, dy, dz = x - fx, y - fy, z - fz
    w, idx = [], []
    for xi, wx in ((x0, 1 - dx), (x1, dx)):
        for yi, wy in ((y0, 1 - dy), (y1, dy)):
            for zi, wz in ((z0, 1 - dz), (z1, dz)):
                w.append(wx * wy * wz)
                idx.append((xi * G + yi) * G + zi)
    return w, idx


def inside(x, y, z):
    return (x >= 0) & (x <= G - 1) & (y >= 0) & (y <= G - 1) & (z >= 0) & (z <= G - 1)


def march_step(jf, kf, x, y, z, I, tau):
    """One step of Ray-trax's `trace_ray` -> the new (I, tau) and the 8 (index, value) deposits."""
    w, idx = corners(x, y, z)
    j_val = sum(wi * jf[ii] for wi, ii in zip(w, idx))
    ix, iy, iz = (jnp.clip(jnp.floor(v).astype(jnp.int32), 0, G - 1) for v in (x, y, z))
    dtau = kf[(ix * G + iy) * G + iz] * DS
    I2 = I * jnp.exp(-dtau) + j_val * jnp.exp(-tau) * DS
    dep = I2 * inside(x, y, z).astype(I2.dtype)
    return I2, tau + dtau, [(ii, wi * dep) for wi, ii in zip(w, idx)]


def ray(t, d, jf, kf, src):
    def body(c):
        x, y, z, I, tau, t = c
        I2, tau2, dep = march_step(jf, kf, x, y, z, I, tau)
        for ii, vv in dep:
            t = t.add("J", ii, vv)
        return x + d[0] * DS, y + d[1] * DS, z + d[2] * DS, I2, tau2, t
    c = lax.while_loop(lambda c: inside(c[0], c[1], c[2]), body,
                       (src[0], src[1], src[2], jnp.float32(0.0), jnp.float32(0.0), t))
    return c[-1]


f_tm = tm.vmap(ray, outputs={"J": tm.Sum((G ** 3,))}, in_axes=(0, None, None, None), max_steps=MAXS + 2)


def run_scan(dirs, jf, kf, src):
    n = dirs.shape[0]

    def step(c, _):
        x, y, z, I, tau, J = c
        I2, tau2, dep = jax.vmap(lambda *a: march_step(jf, kf, *a))(x, y, z, I, tau)
        for ii, vv in dep:
            J = J.at[ii].add(vv)
        return (x + dirs[:, 0] * DS, y + dirs[:, 1] * DS, z + dirs[:, 2] * DS, I2, tau2, J), None
    c0 = tuple(jnp.full((n,), v, jnp.float32) for v in (src[0], src[1], src[2], 0.0, 0.0)) + (jnp.zeros((G ** 3,)),)
    return lax.scan(step, c0, None, length=MAXS)[0][-1]


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r, ph = np.sqrt(1 - z * z), np.pi * (1 + 5 ** 0.5) * i
    return np.stack([r * np.cos(ph), r * np.sin(ph), z], 1).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2048)
    n = p.parse_args().n
    if os.environ.get("RAYTRAX_PATH"):
        sys.path.append(os.environ["RAYTRAX_PATH"])
    rt = D.have("ray_trax")
    if rt is not None:
        from ray_trax.core import sample_sphere
        from ray_trax.ray_trax_3D import compute_radiation_field_from_source
        dirs = jnp.asarray(sample_sphere(n), jnp.float32)          # the library's own direction set
    else:
        dirs = jnp.asarray(fibonacci_sphere(n))
    jf, kf = jnp.asarray(smooth_field(1, 0.0, 1.0)), jnp.asarray(smooth_field(2, 0.01, 0.4))
    src = jnp.full((3,), (G - 1) / 2.0, jnp.float32)
    rng = np.random.default_rng(0)
    wts = jnp.asarray(rng.uniform(0.5, 1.5, G ** 3), jnp.float32)
    tgt = jnp.asarray(rng.uniform(0, 2, G ** 3), jnp.float32)
    sims = {"tallymap": lambda j, k: f_tm(dirs, j, k, src).outputs["J"],
            "plain-JAX shared scan": lambda j, k: run_scan(dirs, j, k, src)}
    if rt is not None:
        sims["Ray-trax"] = lambda j, k: compute_radiation_field_from_source(
            j.reshape(G, G, G), k.reshape(G, G, G), src, num_rays=n, step_size=DS, max_steps=MAXS,
            ray_batch_size=None).reshape(-1) / (4 * jnp.pi / n)

    J = {k: jax.jit(f)(jf, kf) for k, f in sims.items()}
    grad = {k: jax.jit(jax.grad(lambda j, q, f=f: jnp.sum(wts * (f(j, q) - tgt) ** 2) / G ** 3, argnums=(0, 1)))(jf, kf)
            for k, f in sims.items()}
    checks = []
    for k in list(sims)[1:]:
        rg = max(D.rel(grad["tallymap"][i], grad[k][i]) for i in range(2))
        checks += [(f"field vs {k}", D.rel(J["tallymap"], J[k]) < 2e-4, f"rel = {D.rel(J['tallymap'], J[k]):.2g}"),
                   (f"gradient w.r.t. emissivity and opacity vs {k}", rg < 1e-3, f"rel = {rg:.2g}")]
    D.timings(n, {k: (f, (jf, kf)) for k, f in sims.items()})
    return D.verdict(checks)


if __name__ == "__main__":
    sys.exit(main())
