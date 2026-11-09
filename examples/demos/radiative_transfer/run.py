"""Radiative transfer two ways: Ray-trax on its own steady-state problem, and a volumetric Monte-Carlo walk.

`--prog ray` is the head-to-head.  Ray-trax (github.com/lorenzobranca/Ray-trax, arXiv:2511.09389) is a
GPU-oriented differentiable 3-D ray tracer in JAX for the emission-absorption transfer equation on regular
grids.  Its per-ray kernel (`ray_trax/core.py:126-195`) is a `lax.fori_loop` of a STATIC number of steps; per
step it interpolates the emissivity trilinearly, looks up the opacity at the nearest voxel, updates (I, tau)
semi-analytically and deposits I trilinearly onto a grid.  The deposit target is a FULL GRID ALLOCATED PER RAY
(`core.py:191`); rays are batched with `jax.vmap` and summed over the ray axis (`core.py:202-240`), with an
optional `ray_batch_size` micro-batching knob whose stated purpose is to cap that peak memory.

Ray-trax is a source repository with no packaging, so it cannot go in `requirements.txt`.  Clone it and put it
on `PYTHONPATH`, or point `--raytrax` at the clone; see this demo's README.

  raytrax           the library's public `compute_radiation_field_from_source`, `ray_batch_size=None`
                    (its default: one `vmap` over all rays).
  raytrax_b         the same call with the library's own `ray_batch_size` set so that one micro-batch of
                    per-ray grids stays under 1 GB, which is the setting its docstring recommends when memory
                    is tight.
  jax_scan_shared   the best plain-JAX rewrite: ONE shared grid in a `lax.scan` carry with the rays vectorised
                    inside the step, so O(grid) memory instead of O(rays x grid), still padded to the horizon.
  jax_ckpt          `jax_scan_shared` with a recursively checkpointed (nested, remat'd) scan.
  tm                tallymap: item = one ray, tally = `Sum((G^3,))`, and the ray's own `while_loop` stops when
                    it leaves the box.
  tm_exact          the same with `Sum(..., fixed_point=True)`, the int64 fixed-point accumulator.

The step function is written once in this file and every arm calls it, except the two library arms, which call
the library.  `--quick` checks all of them against each other and against an f64 run of the same program.

`--prog mc` is a volumetric Monte-Carlo transport walk, the shape of every Monte-Carlo transport code (MCX/MMC,
MYSTIC, SMART-G, gCMCRT, Mitsuba's volpath, gPMC).  None of those is a JAX library, so the baselines there are
a faithful plain-JAX port of the same item function, not those codes, and the README says so.  Free flights use
a fixed majorant (Woodcock / delta tracking), the scattering coefficient enters as a ratio-tracking weight
rather than a real/null decision, absorption is implicit capture, and termination is escape or Russian roulette
at a fixed survival probability.  Every field therefore enters only through multiplicative weights, so the path
does not depend on the parameters and the pathwise derivative is both the exact derivative of the program and
an unbiased estimator of the derivative.

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
LIBS = ("jax",)
DS = 0.5                        # ray-march step in grid units (Ray-trax's default)
# Horizon in steps, as a multiple of the longest exit path from the source (the box half-diagonal):
# low = the observed maximum, which is the fairest possible pad; mid = 2x; high = 4x.  All three are below the
# library's own shipped default of max_steps=500.
HORIZON = {"low": 1.0, "mid": 2.0, "high": 4.0}
BATCH_BYTES = 1 << 30           # cap for ONE micro-batch of per-ray grids in the raytrax_b arm
RAY_ARMS = ("tm", "tm_exact", "raytrax", "raytrax_b", "jax_scan_shared", "jax_ckpt")
RAY_LABEL = {"tm": "tallymap", "tm_exact": "tallymap, fixed_point=True", "raytrax": "Ray-trax (shipped)",
             "raytrax_b": "Ray-trax (micro-batched)", "jax_scan_shared": "plain-JAX shared scan",
             "jax_ckpt": "shared scan + remat"}


def smooth_field(G, seed, lo, hi):
    """A smooth turbulent-looking field: a random field damped in k space, then rescaled to [lo, hi].
    Ray-trax's own test fields are built the same way."""
    f = np.fft.fftfreq(G)
    k = f[:, None, None] ** 2 + f[None, :, None] ** 2 + f[None, None, :] ** 2
    a = np.fft.fftn(np.random.default_rng(seed).normal(size=(G, G, G))) * np.exp(-k * G * G / 8.0)
    a = np.real(np.fft.ifftn(a))
    a = (a - a.min()) / (a.max() - a.min())
    return np.asarray(lo + (hi - lo) * a, np.float32)


def raytrax_path(given=""):
    """Where the Ray-trax clone is: --raytrax, then $RAYTRAX_PATH, then a `Ray-trax` directory next to this
    file, then wherever `ray_trax` is already importable from (PYTHONPATH)."""
    for p in (given, os.environ.get("RAYTRAX_PATH", ""), os.path.join(HERE, "Ray-trax")):
        if p and os.path.isdir(os.path.join(p, "ray_trax")):
            return p
    try:
        import importlib.util
        sp = importlib.util.find_spec("ray_trax")
        if sp is not None and sp.submodule_search_locations:
            return os.path.dirname(list(sp.submodule_search_locations)[0])
    except Exception:
        pass
    return ""


def add_raytrax(given=""):
    p = raytrax_path(given)
    if not p:
        raise SystemExit("Ray-trax not found: clone it and pass --raytrax <dir> or set RAYTRAX_PATH "
                         "(see this demo's README)")
    if p not in sys.path:
        sys.path.append(p)                      # append, not insert: site-packages keeps priority
    return p


# ============================================================================= prog=ray: the ray-march step
def corners(G, x, y, z):
    """Ray-trax's `_corners` (core.py:52-72): 8 clipped corner indices and their trilinear weights."""
    fx, fy, fz = jnp.floor(x), jnp.floor(y), jnp.floor(z)
    x0 = jnp.clip(fx.astype(jnp.int32), 0, G - 1)
    y0 = jnp.clip(fy.astype(jnp.int32), 0, G - 1)
    z0 = jnp.clip(fz.astype(jnp.int32), 0, G - 1)
    x1, y1, z1 = jnp.clip(x0 + 1, 0, G - 1), jnp.clip(y0 + 1, 0, G - 1), jnp.clip(z0 + 1, 0, G - 1)
    dx, dy, dz = x - fx, y - fy, z - fz
    w = ((1 - dx) * (1 - dy) * (1 - dz), (1 - dx) * (1 - dy) * dz,
         (1 - dx) * dy * (1 - dz), (1 - dx) * dy * dz,
         dx * (1 - dy) * (1 - dz), dx * (1 - dy) * dz,
         dx * dy * (1 - dz), dx * dy * dz)
    idx = (((x0 * G + y0) * G + z0), ((x0 * G + y0) * G + z1),
           ((x0 * G + y1) * G + z0), ((x0 * G + y1) * G + z1),
           ((x1 * G + y0) * G + z0), ((x1 * G + y0) * G + z1),
           ((x1 * G + y1) * G + z0), ((x1 * G + y1) * G + z1))
    return w, idx


def inside_box(G, x, y, z):
    return (x >= 0) & (x <= G - 1) & (y >= 0) & (y <= G - 1) & (z >= 0) & (z <= G - 1)


def march_step(G, jf, kf, x, y, z, I, tau):
    """ONE ray-march step: Ray-trax's `trace_ray` body (core.py:165-183) on flattened fields.
    -> the new (I, tau) and the 8 (index, value) deposits of this step."""
    w, idx = corners(G, x, y, z)
    j_val = sum(wi * jf[ii] for wi, ii in zip(w, idx))                  # trilinear interpolation
    ix = jnp.clip(jnp.floor(x).astype(jnp.int32), 0, G - 1)
    iy = jnp.clip(jnp.floor(y).astype(jnp.int32), 0, G - 1)
    iz = jnp.clip(jnp.floor(z).astype(jnp.int32), 0, G - 1)
    kap = kf[(ix * G + iy) * G + iz]                                    # nearest-voxel opacity
    dtau = kap * DS
    I2 = I * jnp.exp(-dtau) + j_val * jnp.exp(-tau) * DS
    dep = I2 * inside_box(G, x, y, z).astype(I2.dtype)
    return I2, tau + dtau, [(ii, wi * dep) for wi, ii in zip(w, idx)]


def ray_tm(G, maxs, fixed_point=False, **o):
    def ray(t, d, jf, kf, src):
        def body(c):
            x, y, z, I, tau, k, t = c
            I2, tau2, dep = march_step(G, jf, kf, x, y, z, I, tau)
            for ii, vv in dep:
                t = t.add("J", ii, vv)
            return (x + d[0] * DS, y + d[1] * DS, z + d[2] * DS, I2, tau2, k + 1, t)
        c = (src[0], src[1], src[2], jnp.float32(0.0), jnp.float32(0.0), jnp.int32(0), t)
        x, y, z, I, tau, k, t = lax.while_loop(lambda c: inside_box(G, c[0], c[1], c[2]), body, c)
        return t, k
    return tm.vmap(ray, outputs={"J": tm.Sum((G ** 3,), fixed_point=fixed_point)}, in_axes=(0, None, None, None),
                   max_steps=maxs, **o)


def ray_scan(G, maxs, remat=False, seg=0):
    """ONE shared grid in the scan carry, rays vectorised inside the step: O(grid) memory, still padded."""
    nv = G ** 3

    def call(dirs, jf, kf, src):
        n = dirs.shape[0]

        def step(c, _):
            x, y, z, I, tau, J = c
            I2, tau2, dep = jax.vmap(lambda *a: march_step(G, jf, kf, *a))(x, y, z, I, tau)
            for ii, vv in dep:
                J = J.at[ii].add(vv)
            return (x + dirs[:, 0] * DS, y + dirs[:, 1] * DS, z + dirs[:, 2] * DS, I2, tau2, J), None
        dt = jf.dtype
        c0 = (jnp.full((n,), src[0], dt), jnp.full((n,), src[1], dt), jnp.full((n,), src[2], dt),
              jnp.zeros((n,), dt), jnp.zeros((n,), dt), jnp.zeros((nv,), dt))
        if remat:
            inner = jax.checkpoint(lambda c, _: lax.scan(step, c, None, length=maxs // seg))
            c, _ = lax.scan(inner, c0, None, length=seg)
        else:
            c, _ = lax.scan(step, c0, None, length=maxs)
        return c[-1]
    return call


def ray_library(G, maxs, batch, n, rtpath):
    add_raytrax(rtpath)
    from ray_trax.ray_trax_3D import compute_radiation_field_from_source

    def call(jf, kf, src):
        # the library returns J normalised by 4 pi / N; divide that out so every arm returns the same quantity
        return compute_radiation_field_from_source(
            jf.reshape(G, G, G), kf.reshape(G, G, G), src, num_rays=n, step_size=DS, max_steps=maxs,
            ray_batch_size=batch).reshape(-1) / (4 * jnp.pi / n)
    return call


def ray_setup(a):
    G = a.grid
    maxs = a.maxs or int(np.ceil(HORIZON[a.div] * np.sqrt(3) * (G - 1) / 2 / DS)) + 1
    jf = jnp.asarray(smooth_field(G, 1, 0.0, 1.0)).reshape(-1)
    kf = jnp.asarray(smooth_field(G, 2, 0.01, 0.4)).reshape(-1)
    src = jnp.asarray([(G - 1) / 2.0] * 3, jnp.float32)
    add_raytrax(a.raytrax)
    from ray_trax.core import sample_sphere  # the library's own direction set
    dirs = jnp.asarray(sample_sphere(a.n), jnp.float32)
    rng = np.random.default_rng(0)
    wts = jnp.asarray(rng.uniform(0.5, 1.5, G ** 3), jnp.float32)
    tgt = jnp.asarray(rng.uniform(0.0, 2.0, G ** 3), jnp.float32)
    batch = a.batch or max(1, min(a.n, BATCH_BYTES // (G ** 3 * 4)))
    return G, maxs, jf, kf, src, dirs, wts, tgt, batch


def ray_sims(arm, a, G, maxs, dirs, src, batch, seg):
    """-> (sim(jf, kf) -> flat J, steps_fn or None)."""
    if arm in ("tm", "tm_exact"):
        f = ray_tm(G, maxs, fixed_point=(arm == "tm_exact"))
        return (lambda jf_, kf_: f(dirs, jf_, kf_, src).outputs["J"],
                lambda jf_, kf_: np.asarray(jax.jit(lambda j, k: f(dirs, j, k, src).per_item)(jf_, kf_)))
    if arm in ("jax_scan_shared", "jax_ckpt"):
        g = ray_scan(G, maxs, remat=(arm == "jax_ckpt"), seg=seg)
        return (lambda jf_, kf_: g(dirs, jf_, kf_, src), None)
    if arm in ("raytrax", "raytrax_b"):
        g = ray_library(G, maxs, None if arm == "raytrax" else batch, a.n, a.raytrax)
        return (lambda jf_, kf_: g(jf_, kf_, src), None)
    raise SystemExit(f"unknown arm {arm}")


# ====================================================================== prog=mc: volumetric Monte-Carlo walk
NSENS = 32                                                        # escaping-flux direction bins
MC_DIV = {"low": (0.25, 0.5), "mid": (1.0, 0.9), "high": (3.0, 0.98)}     # (majorant, roulette survival)
MC_MAXS = {"low": 128, "mid": 384, "high": 1536}                  # above the observed maximum; --quick checks it
MC_ARMS = ("tm", "jax_while", "jax_scan", "jax_ckpt")
MC_LABEL = {"tm": "tallymap", "jax_while": "vmap + while_loop", "jax_scan": "padded scan",
            "jax_ckpt": "padded scan + remat"}


def sensor_bin(d):
    phi = (jnp.arctan2(d[1], d[0]) + jnp.pi) * (8 / (2 * jnp.pi))
    mu = (d[2] + 1.0) * 2.0
    return jnp.clip(phi.astype(jnp.int32), 0, 7) * 4 + jnp.clip(mu.astype(jnp.int32), 0, 3)


def flight(G, ss, sa, p, d, w, key, k, smaj, psurv, th):
    """ONE collision step.  -> the new (p, d, w, alive) and the (voxel, absorbed) and (sensor, escaped)
    deposits.  `th` = (absorption scale, scattering scale, source power), the three global parameters."""
    u = jax.random.uniform(jax.random.fold_in(key, k), (4,))
    s = -jnp.log1p(-u[0]) / smaj
    p2 = p + s * d
    out = ~jnp.all((p2 >= 0.0) & (p2 <= float(G)))
    i = jnp.clip(jnp.floor(p2).astype(jnp.int32), 0, G - 1)
    v = (i[0] * G + i[1]) * G + i[2]
    w2 = w * jnp.where(out, 1.0, th[1] * ss[v] / smaj)                 # ratio-tracking scatter weight
    att = jnp.exp(-th[0] * sa[v] * s)
    dep = jnp.where(out, 0.0, w2 * (1.0 - att))                        # absorbed in this voxel
    w3 = jnp.where(out, w2, w2 * att)
    ct = 1.0 - 2.0 * u[1]
    st = jnp.sqrt(jnp.maximum(1.0 - ct * ct, 0.0))
    ph = 2.0 * jnp.pi * u[2]
    d2 = jnp.stack([st * jnp.cos(ph), st * jnp.sin(ph), ct])
    live = u[3] < psurv                                                # roulette: a field-independent decision
    return (p2, jnp.where(out, d, d2), jnp.where(live, w3 / psurv, w3), (~out) & live,
            (v, dep), (sensor_bin(d), jnp.where(out, w2, 0.0)))


def mc_tm(G, maxs, smaj, psurv, **o):
    def photon(t, p0, d0, key, ss, sa, th):
        def body(c):
            p, d, w, alive, k, t = c
            p2, d2, w2, live, (v, dep), (sb, esc) = flight(G, ss, sa, p, d, w, key, k, smaj, psurv, th)
            t = t.add("dep", v, dep)
            t = t.add("esc", sb, esc)
            return (p2, d2, w2, live, k + 1, t)
        c = (p0, d0, th[2], jnp.bool_(True), jnp.int32(0), t)
        p, d, w, alive, k, t = lax.while_loop(lambda c: c[3], body, c)
        return t, k
    return tm.vmap(photon, outputs={"dep": tm.Sum((G ** 3,)), "esc": tm.Sum((NSENS,))},
                   in_axes=(0, 0, 0, None, None, None), max_steps=maxs, **o)


# The fair plain-JAX shape: ONE shared tally in the loop carry with all N photons vectorised inside the step.
# `jax.vmap` of the item with its own dense (Nvox,) tally per photon is the other option and is hopeless: it
# needs N x Nvox floats, 16 GB at N = 1M and Nvox = 4096.
def mc_vec_step(G, smaj, psurv, c, ss, sa, th, keys):
    p, d, w, alive, k, dp, es = c
    f = lambda p_, d_, w_, k_, key_: flight(G, ss, sa, p_, d_, w_, key_, k_, smaj, psurv, th)   # noqa: E731
    p2, d2, w2, live, (v, dep), (sb, esc) = jax.vmap(f)(p, d, w, k, keys)
    m = alive.astype(jnp.float32)
    return (jnp.where(alive[:, None], p2, p), jnp.where(alive[:, None], d2, d), jnp.where(alive, w2, w),
            alive & live, k + alive.astype(jnp.int32), dp.at[v].add(dep * m), es.at[sb].add(esc * m))


def mc_jax(arm, G, maxs, smaj, psurv, seg):
    nv = G ** 3

    def call(p0, d0, keys, ss, sa, th):
        n = p0.shape[0]
        c0 = (p0, d0, jnp.full((n,), th[2]), jnp.ones((n,), jnp.bool_), jnp.zeros((n,), jnp.int32),
              jnp.zeros((nv,), jnp.float32), jnp.zeros((NSENS,), jnp.float32))
        step = lambda c: mc_vec_step(G, smaj, psurv, c, ss, sa, th, keys)                # noqa: E731
        if arm == "jax_while":                     # pays the batch MAXIMUM trip count
            c = lax.while_loop(lambda c: jnp.any(c[3]) & (jnp.max(c[4]) < maxs), step, c0)
        elif arm == "jax_scan":                    # padded to `maxs`
            c, _ = lax.scan(lambda c, _: (step(c), None), c0, None, length=maxs)
        elif arm == "jax_ckpt":                    # + recursive checkpointing
            inner = jax.checkpoint(lambda c, _: lax.scan(lambda c, _: (step(c), None), c, None,
                                                         length=maxs // seg))
            c, _ = lax.scan(inner, c0, None, length=seg)
        else:
            raise SystemExit(f"unknown arm {arm}")
        return c[5], c[6], c[4]
    return call


def mc_setup(a):
    G = a.grid
    smaj, psurv = MC_DIV[a.div]
    maxs = a.maxs or MC_MAXS[a.div]
    ss = jnp.asarray(smooth_field(G, 1, 0.05 * smaj, smaj)).reshape(-1)   # scattering, below the majorant
    sa = jnp.asarray(smooth_field(G, 2, 0.0, 0.3 * smaj)).reshape(-1)     # absorption
    th = jnp.array([1.0, 1.0, 1.0], jnp.float32)
    rng = np.random.default_rng(0)
    c = (G - 1) / 2.0
    p0 = jnp.asarray(np.full((a.n, 3), c), jnp.float32)                   # isotropic source at the box centre
    u = rng.uniform(0.0, 1.0, (a.n, 2))
    ct = 1 - 2 * u[:, 0]
    st = np.sqrt(np.maximum(1 - ct * ct, 0))
    d0 = jnp.asarray(np.stack([st * np.cos(2 * np.pi * u[:, 1]), st * np.sin(2 * np.pi * u[:, 1]), ct], 1),
                     jnp.float32)
    keys = jax.random.split(jax.random.PRNGKey(7), a.n)
    wts = jnp.asarray(rng.uniform(0.5, 1.5, G ** 3), jnp.float32)
    tgt = jnp.asarray(rng.uniform(0.0, 1.0, G ** 3), jnp.float32) * (a.n / G ** 3)
    tge = jnp.asarray(rng.uniform(0.0, 1.0, NSENS), jnp.float32) * (a.n / NSENS)
    return G, smaj, psurv, maxs, ss, sa, th, p0, d0, keys, wts, tgt, tge


def mc_sims(arm, G, maxs, smaj, psurv, seg, p0, d0, keys):
    if arm == "tm":
        f = mc_tm(G, maxs, smaj, psurv)

        def sim(ss_, sa_, th_):
            r = f(p0, d0, keys, ss_, sa_, th_)
            return r.outputs["dep"], r.outputs["esc"]
        steps = lambda s_, a_, t_: (                                                     # noqa: E731
            np.asarray(jax.jit(lambda x, y, z: f(p0, d0, keys, x, y, z).per_item)(s_, a_, t_)),
            int(jax.jit(lambda x, y, z: f(p0, d0, keys, x, y, z).stats["n_truncated"])(s_, a_, t_)))
        return sim, steps
    g = mc_jax(arm, G, maxs, smaj, psurv, seg)
    return (lambda ss_, sa_, th_: g(p0, d0, keys, ss_, sa_, th_)[:2],
            lambda s_, a_, t_: (np.asarray(jax.jit(lambda x, y, z: g(p0, d0, keys, x, y, z)[2])(s_, a_, t_)), -1))


def sqrt_seg(maxs):
    """The DIVISOR of `maxs` nearest sqrt(maxs).  It must divide exactly, or the nested scan runs fewer than
    `maxs` steps and silently truncates the longest items."""
    import math
    best = 1
    for d in range(1, maxs + 1):
        if maxs % d == 0 and abs(d - math.sqrt(maxs)) < abs(best - math.sqrt(maxs)):
            best = d
    return best


# ------------------------------------------------------------------------------------------------ one measurement
def one(a):
    seg = a.seg or sqrt_seg(a.maxs or 1)
    if a.prog == "ray" and a.mode == "check64":
        # The arbiter of the f32 disagreements: an f64 run of the shared-scan program, in its own process
        # because enabling x64 is a global switch.  The arms stay f32: their fields are explicitly float32.
        jax.config.update("jax_enable_x64", True)
        G, maxs, jf, kf, src, dirs, wts, tgt, batch = ray_setup(a)
        seg = a.seg or sqrt_seg(maxs)
        D.banner(f"ray check64 rays={a.n} G={G} maxs={maxs}", LIBS)
        out = {}
        for arm in RAY_ARMS:
            fo, _ = ray_sims(arm, a, G, maxs, dirs, src, batch, seg)
            out[arm] = jax.jit(fo)(jf, kf)
        ref = jax.jit(ray_scan(G, maxs))(dirs.astype(jnp.float64), jf.astype(jnp.float64),
                                         kf.astype(jnp.float64), src.astype(jnp.float64))
        ft, st = ray_sims("tm", a, G, maxs, dirs, src, batch, seg)
        k = st(jf, kf)
        nv = G ** 3
        lt = lambda jf_, kf_: jnp.sum(wts * (ft(jf_, kf_) - tgt) ** 2) / nv            # noqa: E731
        fr, _ = ray_sims("raytrax", a, G, maxs, dirs, src, batch, seg)
        lr = lambda jf_, kf_: jnp.sum(wts * (fr(jf_, kf_) - tgt) ** 2) / nv            # noqa: E731
        gt = jax.jit(jax.grad(lt, argnums=(0, 1)))(jf, kf)
        gr = jax.jit(jax.grad(lr, argnums=(0, 1)))(jf, kf)
        D.row(demo="radiative_transfer", prog="ray", arm="check64", mode="check64", n=a.n, div=a.div,
              grid=G, maxs=maxs, status="ok", steps_mean=float(k.mean()), steps_min=int(k.min()),
              steps_max=int(k.max()), gradj_rel_raytrax=D.rel(gt[0], gr[0]),
              gradk_rel_raytrax=D.rel(gt[1], gr[1]),
              **{f"rel64_{arm}": D.rel(v, ref) for arm, v in out.items()})
        return
    if a.prog == "ray":
        G, maxs, jf, kf, src, dirs, wts, tgt, batch = ray_setup(a)
        seg = a.seg or sqrt_seg(maxs)
        D.banner(f"ray arm={a.arm} mode={a.mode} rays={a.n} div={a.div} G={G} maxs={maxs}", LIBS)
        nv = G ** 3
        base = dict(demo="radiative_transfer", prog="ray", arm=a.arm, mode=a.mode, n=a.n, div=a.div,
                    grid=G, nvox=nv, maxs=maxs, seg=seg, batch=batch,
                    raytrax_commit=os.popen(f"git -C {raytrax_path(a.raytrax)} rev-parse --short HEAD"
                                            ).read().strip() or "unknown")
        sim, steps_fn = ray_sims(a.arm, a, G, maxs, dirs, src, batch, seg)
        loss = lambda jf_, kf_: jnp.sum(wts * (sim(jf_, kf_) - tgt) ** 2) / nv            # noqa: E731
        fns = {"fwd": sim,
               "jvp": lambda jf_, kf_: jax.jvp(sim, (jf_, kf_), (jnp.ones_like(jf_), jnp.ones_like(kf_))),
               "grad": jax.grad(loss, argnums=(0, 1))}

        def go():
            out, r = D.measure(fns[a.mode], (jf, kf), reps=a.reps)
            leaf = jax.tree_util.tree_leaves(out)[0]
            r.update(base, status="ok", out_norm=float(jnp.linalg.norm(jnp.asarray(leaf).ravel())))
            if a.mode == "fwd" and steps_fn is not None:
                k = steps_fn(jf, kf)
                r.update(steps_mean=float(k.mean()), steps_max=int(k.max()), steps_min=int(k.min()))
            D.row(**r)
        return D.guard(go, base)

    G, smaj, psurv, maxs, ss, sa, th, p0, d0, keys, wts, tgt, tge = mc_setup(a)
    seg = a.seg or sqrt_seg(maxs)
    D.banner(f"mc arm={a.arm} mode={a.mode} n={a.n} div={a.div} G={G} smaj={smaj} p={psurv} maxs={maxs}", LIBS)
    nv = G ** 3
    base = dict(demo="radiative_transfer", prog="mc", arm=a.arm, mode=a.mode, n=a.n, div=a.div, grid=G,
                nvox=nv, maxs=maxs, seg=seg, smaj=smaj, p_surv=psurv)
    sim, steps_fn = mc_sims(a.arm, G, maxs, smaj, psurv, seg, p0, d0, keys)

    def loss(ss_, sa_, th_):
        dp, es = sim(ss_, sa_, th_)
        return (jnp.sum(wts * (dp - tgt) ** 2) / nv + jnp.sum((es - tge) ** 2) / NSENS) / a.n
    fns = {"fwd": sim,
           "jacfwd": jax.jacfwd(loss, argnums=2),
           "grad": jax.grad(loss, argnums=(0, 1, 2))}

    def go():
        out, r = D.measure(fns[a.mode], (ss, sa, th), reps=a.reps)
        leaf = jax.tree_util.tree_leaves(out)[0]
        r.update(base, status="ok", out_norm=float(jnp.linalg.norm(jnp.asarray(leaf).ravel())))
        if a.mode == "fwd":
            k, ntr = steps_fn(ss, sa, th)
            r.update(steps_mean=float(k.mean()), steps_max=int(k.max()),
                     steps_p99=float(np.percentile(k, 99)), n_truncated=ntr)
        D.row(**r)
    D.guard(go, base)


# -------------------------------------------------------------------------------------------------------- --quick
class _A:
    pass


def quick(a):
    checks = []

    # ---- prog=ray: every arm against every other, and all of them against an f64 run of the same program
    q = _A()
    q.n, q.div, q.grid, q.maxs, q.batch, q.seg, q.raytrax = min(a.n, 2048), "low", 16, 0, 0, 0, a.raytrax
    G, maxs, jf, kf, src, dirs, wts, tgt, batch = ray_setup(q)
    nv = G ** 3
    D.banner(f"quick ray rays={q.n} G={G} maxs={maxs}", LIBS)
    # Enabling f64 is a global switch, so the arbiter runs in its own process.
    cmd = [sys.executable, os.path.join(HERE, "run.py"), "--one", "--prog", "ray", "--mode", "check64",
           "--n", str(q.n), "--div", q.div, "--grid", str(q.grid)] + (["--raytrax", a.raytrax]
                                                                      if a.raytrax else [])
    import json
    import subprocess
    pr = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    rr = [json.loads(ln[4:]) for ln in pr.stdout.splitlines() if ln.startswith("ROW ")]
    if not rr:
        checks.append(("ray: every arm vs an f64 run of the same program", False,
                       f"the f64 arbiter did not run: {(pr.stdout + pr.stderr)[-300:]}"))
    else:
        c = rr[0]
        for arm in RAY_ARMS:
            v = c.get(f"rel64_{arm}")
            checks.append((f"ray: {RAY_LABEL[arm]} vs an f64 run of the same program",
                           v is not None and v < 1e-4, f"rel = {v:.2g}" if v is not None else "missing"))
        gj, gk = c["gradj_rel_raytrax"], c["gradk_rel_raytrax"]
        checks.append(("ray: the inverse-problem gradient agrees with Ray-trax's", max(gj, gk) < 1e-3,
                       f"emissivity rel = {gj:.2g}, opacity rel = {gk:.2g}"))
        checks.append(("ray: the march length varies across rays", c["steps_max"] > c["steps_min"],
                       f"min {c['steps_min']}, mean {c['steps_mean']:.1f}, max {c['steps_max']} "
                       f"of maxs={c['maxs']}"))

    # ---- prog=mc: all four arms, the per-item collision count, and the gradient
    q2 = _A()
    q2.n, q2.div, q2.grid, q2.maxs, q2.seg = min(a.n, 2048), "mid", 8, 0, 0
    G, smaj, psurv, maxs, ss, sa, th, p0, d0, keys, wts, tgt, tge = mc_setup(q2)
    nv = G ** 3
    seg = sqrt_seg(maxs)
    ftm, stm = mc_sims("tm", G, maxs, smaj, psurv, seg, p0, d0, keys)
    dt, et = jax.jit(ftm)(ss, sa, th)
    kk, ntr = stm(ss, sa, th)
    for arm in ("jax_while", "jax_scan", "jax_ckpt"):
        fo, so = mc_sims(arm, G, maxs, smaj, psurv, seg, p0, d0, keys)
        do, eo = jax.jit(fo)(ss, sa, th)
        ko, _ = so(ss, sa, th)
        rd, re = D.rel(dt, do), D.rel(et, eo)
        eq = float(np.mean(np.asarray(kk) == ko))
        checks.append((f"mc: deposition and escaping flux vs {MC_LABEL[arm]}",
                       max(rd, re) < 1e-5 and eq == 1.0,
                       f"rel = {rd:.2g} / {re:.2g}, collision count equal for {100 * eq:.0f}% of photons"))

    def mkloss(f):
        return lambda s_, a_, t_: (jnp.sum(wts * (f(s_, a_, t_)[0] - tgt) ** 2) / nv
                                   + jnp.sum((f(s_, a_, t_)[1] - tge) ** 2) / NSENS) / q2.n
    fs, _ = mc_sims("jax_scan", G, maxs, smaj, psurv, seg, p0, d0, keys)
    gt = jax.jit(jax.grad(mkloss(ftm), argnums=(0, 1, 2)))(ss, sa, th)
    gs = jax.jit(jax.grad(mkloss(fs), argnums=(0, 1, 2)))(ss, sa, th)
    jt = jax.jit(jax.jacfwd(mkloss(ftm), argnums=2))(ss, sa, th)
    rs = [D.rel(gt[i], gs[i]) for i in range(3)]
    checks.append(("mc: the gradient w.r.t. both fields and the three globals vs the padded scan",
                   max(rs) < 1e-4, "rel = " + " / ".join(f"{x:.2g}" for x in rs)))
    checks.append(("mc: jacfwd agrees with grad inside tallymap", D.rel(gt[2], jt) < 1e-4,
                   f"rel = {D.rel(gt[2], jt):.2g}"))
    checks.append((f"mc: no item hit max_steps={maxs}", ntr == 0,
                   f"n_truncated = {ntr}, collisions mean {kk.mean():.1f} max {kk.max()}"))
    return D.verdict(checks)


# --------------------------------------------------------------------------------------------------------- --full
def plots(rows, meta):
    """Redraw plots/ from `rows`.  Called by --full and by --plots."""
    P = os.path.join(HERE, "plots")
    for mode, t in (("fwd", "forward"), ("grad", "reverse-mode gradient")):
        rr = [r for r in rows if r.get("prog") == "ray" and r.get("mode") == mode
              and r.get("div") == "low" and r.get("grid") == 32]
        D.panel(os.path.join(P, f"ray_{mode}_time.png"), rr, RAY_ARMS, RAY_LABEL,
                f"Ray-trax's steady-state problem, 32^3 grid, {t}: time", "ms per call", xlabel="rays")
        D.panel(os.path.join(P, f"ray_{mode}_mem.png"), rr, RAY_ARMS, RAY_LABEL,
                f"Ray-trax's steady-state problem, 32^3 grid, {t}: peak device memory",
                "peak device MB", xlabel="rays", ykey="peak_mb")
        mm = [r for r in rows if r.get("prog") == "mc" and r.get("mode") == mode and r.get("div") == "mid"]
        D.panel(os.path.join(P, f"mc_{mode}_time.png"), mm, MC_ARMS, MC_LABEL,
                f"Volumetric Monte-Carlo transport, 16^3 grid, {t}: time", "ms per call", xlabel="photons")
        D.panel(os.path.join(P, f"mc_{mode}_mem.png"), mm, MC_ARMS, MC_LABEL,
                f"Volumetric Monte-Carlo transport, 16^3 grid, {t}: peak device memory",
                "peak device MB", xlabel="photons", ykey="peak_mb")
    gg = [r for r in rows if r.get("prog") == "ray" and r.get("mode") == "fwd" and r.get("div") == "mid"]
    for r in gg:
        r["nvoxx"] = r.get("nvox")
    D.panel(os.path.join(P, "ray_grid_fwd.png"), gg, ("tm", "raytrax", "raytrax_b", "jax_scan_shared"),
            RAY_LABEL, "Ray-trax's problem at 16,384 rays: forward time vs voxel count",
            "ms per call", xlabel="voxels", xkey="nvoxx")

def full(a):
    script = os.path.join(HERE, "run.py")
    out = os.path.join(HERE, "results", "radiative_transfer.json")
    jobs = []
    for n in (1024, 4096, 16384, 65536):
        for arm in RAY_ARMS:
            for mode in ("fwd", "grad"):
                jobs.append(dict(prog="ray", arm=arm, mode=mode, n=n, div="low", grid=32))
    for g in (16, 32, 64):
        for arm in ("tm", "raytrax", "raytrax_b", "jax_scan_shared"):
            for mode in ("fwd", "grad"):
                jobs.append(dict(prog="ray", arm=arm, mode=mode, n=16384, div="mid", grid=g))
    for d in ("low", "mid", "high"):
        for arm in ("tm", "raytrax", "jax_scan_shared"):
            jobs.append(dict(prog="ray", arm=arm, mode="fwd", n=16384, div=d, grid=32))
    for n in (65536, 262144, 1048576, 4194304):
        for arm in MC_ARMS:
            jobs.append(dict(prog="mc", arm=arm, mode="fwd", n=n, div="mid", grid=16))
    for n in (65536, 262144, 1048576):
        for arm in ("tm", "jax_scan", "jax_ckpt"):
            jobs.append(dict(prog="mc", arm=arm, mode="grad", n=n, div="mid", grid=16))
    rows, meta = D.sweep(script, jobs, out, libs=LIBS,
                         extra_meta=dict(raytrax_repo="github.com/lorenzobranca/Ray-trax",
                                         raytrax_commit="89347946fba4eb47817ea6e7ab76ec8ff9117614"))
    plots(rows, meta)

    print(f"\n{len(rows)} rows -> {out}")
    return 0


def main():
    p = D.parser(grid=32, maxs=0, seg=0, batch=0, raytrax="")
    p.add_argument("--prog", default="ray", choices=("ray", "mc"))
    a = p.parse_args()
    if a.one:
        one(a)
        return 0
    if a.plots:
        rows, meta = D.load(os.path.join(HERE, "results", "radiative_transfer.json"))
        plots(rows, meta)
        return 0
    return quick(a) if a.quick else full(a)


if __name__ == "__main__":
    sys.exit(main())
