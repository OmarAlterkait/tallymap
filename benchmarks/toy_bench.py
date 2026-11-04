"""Cost measurements for the toy walk program (GPU).  env:
BK=cuda|lockstep, N (items), MODES=fwd,jvp,jacfwd,grad,vmap,host,second,second2  LANES=off|auto|force
Prints one line per measurement: name, ms (median of 7 after warmup), plus the compiled-HLO hash so A/B runs can
assert they compiled different programs."""
import hashlib
import os
import statistics
import time

import jax
import jax.numpy as jnp
from jax import lax

import tallymap as tm

BK = os.environ.get("BK", "cuda")
N = int(os.environ.get("N", 1 << 20))
MODES = os.environ.get("MODES", "fwd,jvp,jacfwd,grad,vmap,host").split(",")
NB = int(os.environ.get("NBINS", 256))
BV = int(os.environ.get("BV", 8))
print("NBINS", NB, "BV", BV, "jax", jax.__version__, jax.__file__, jax.devices(), "tallymap", tm.__file__, "bk", BK,
      "N", N, flush=True)


def bits(key, k):
    return jax.random.bits(jax.random.fold_in(key, k))


def walk(t, x, key, tab, th):
    def body(c):
        k, w, t = c
        b = bits(key, k)
        i = (b % NB).astype(jnp.int32)
        t = t.add("Q", i, w * tab[i] * th[0] + th[1] * th[1])
        return k + 1, w * (0.7 + 0.2 * th[0]) + 0.01 * jnp.sin(th[1] * w) / (1.0 + w * w), t
    k, w, t = lax.while_loop(lambda c: (bits(key, c[0] + 1000) % 16 != 0) | (c[0] == 0), body, (0, x, t))
    return t, w


xs = jnp.linspace(0.5, 1.5, N, dtype=jnp.float32)
keys = jax.random.split(jax.random.PRNGKey(0), N)
tab = jnp.linspace(1.0, 2.0, NB, dtype=jnp.float32)
TH = jnp.array([0.9, 0.3], jnp.float32)
WQ = jnp.linspace(0.5, 2.0, NB, dtype=jnp.float32)


LANES = os.environ.get("LANES")        # private _lanes for every call: off | auto | force | <int chunk>
OPTS = {k: (int(v) if v.lstrip("-").isdigit() else {"False": False, "True": True}.get(v, v)) for k, v in
        (kv.split("=") for kv in os.environ.get("OPTS", "").split(",") if kv)}   # extra tm.vmap options: k=v,k=v


def mk(**o):
    if LANES:
        o = dict(o, _lanes=int(LANES) if LANES.isdigit() else LANES)
    o = dict(OPTS, **o)
    return tm.vmap(walk, outputs={"Q": tm.Sum((NB,))}, in_axes=(0, 0, None, None), backend=BK,
                   cuda_math=os.environ.get("MATH", "approx"), **o)


fF, fR = mk(), mk()          # (two separately traced functions: forward-mode rows / reverse-mode rows)


def loss(f, th):
    return jnp.sum(f(xs, keys, tab, th).outputs["Q"] * WQ)


def bench(name, fn, *args):
    j = jax.jit(fn)
    hlo = j.lower(*args).compile().as_text()
    h = hashlib.sha1(hlo.encode()).hexdigest()[:10]
    ncc = hlo.count("custom_call_target=\"tallymap_cuda_launch\"")
    jax.block_until_ready(j(*args))
    ts = []
    for _ in range(7):
        t0 = time.perf_counter()
        jax.block_until_ready(j(*args))
        ts.append((time.perf_counter() - t0) * 1e3)
        lanes = "+dual_lanes" in str(jax.make_jaxpr(fn)(*args))
    print(f"BENCH tm   {name:28s} {statistics.median(ts):9.3f} ms  min {min(ts):8.3f}  hlo {h} tallymap_launches {ncc}"
          f" lanes_used {lanes}", flush=True)


if "fwd" in MODES:
    bench("fwd", lambda th: fF(xs, keys, tab, th).outputs["Q"], TH)
if "jvp" in MODES:
    Qof = lambda s: fF(xs, keys, tab, s).outputs["Q"]
    bench("jvp (primal+tangent)", lambda th, v: jax.jvp(Qof, (th,), (v,)), TH, TH)
    bench("jvp tangent only", lambda th, v: jax.jvp(Qof, (th,), (v,))[1], TH, TH)
if "cse" in MODES:        # 9 jax.jvp calls at the same primal point (a reconstruction fit): primal kernels CSE'd?
    def nine(th):
        outs = [jax.jvp(lambda s: fF(xs, keys, tab, s).outputs["Q"], (th,), (jnp.array([1.0, 0.1 * i]),))
                for i in range(9)]
        return outs[0][0], [o[1] for o in outs]
    bench("9 jvps + primal", nine, TH)
if "jacfwd" in MODES:
    P = jnp.concatenate([tab[:6], TH])
    g = lambda p: fF(xs, keys, tab.at[:6].set(p[:6]), p[6:]).outputs["Q"]
    bench("jacfwd T=8", jax.jacfwd(g), P)
    bench("value+jacfwd T=8", lambda p: (g(p), jax.jacfwd(g)(p)), P)
if "second" in MODES:     # second order: cost relative to grad
    gl = jax.grad(lambda th: loss(fR, th))
    bench("hvp (jvp of grad)", lambda th, v: jax.jvp(gl, (th,), (v,))[1], TH, TH)
    bench("grad-of-jvp", jax.grad(lambda th, v: jax.jvp(lambda s: loss(fF, s), (th,), (v,))[1]), TH, TH)
    bench("hessian 2x2 (jacfwd(jacrev))", jax.hessian(lambda th: loss(fR, th)), TH)
    bench("grad-of-grad (rev-over-rev)", jax.grad(lambda th, v: jnp.vdot(gl(th), v)), TH, TH)
if "second2" in MODES:    # second order w.r.t. table entries (sparse table cotangents) and a 256-row jacrev
    gl = jax.grad(lambda th: loss(fR, th))
    P8 = jnp.concatenate([tab[:6], TH])
    L8 = lambda p: jnp.sum(fR(xs, keys, tab.at[:6].set(p[:6]), p[6:]).outputs["Q"] * WQ)
    bench("hessian 8x8 (table entries)", jax.hessian(L8), P8)
    bench("jacrev Q[256] wrt th", jax.jacrev(lambda th: fR(xs, keys, tab, th).outputs["Q"]), TH)
if "grad" in MODES:
    bench("grad", jax.grad(lambda th: loss(fR, th)), TH)
    bench("value_and_grad", jax.value_and_grad(lambda th: loss(fR, th)), TH)
if "vmap" in MODES:
    ths = jnp.stack([TH * (1 + 0.05 * i) for i in range(BV)])
    bench(f"vmap over theta B={BV}", jax.vmap(lambda th: fF(xs, keys, tab, th).outputs["Q"]), ths)
    xb = jnp.stack([xs * (1 + 0.05 * i) for i in range(BV)])
    bench(f"vmap over events B={BV}", jax.vmap(lambda x: fF(x, keys, tab, TH).outputs["Q"]), xb)
    fM = mk(_fold=False)
    bench(f"vmap theta B={BV} (lax.map)", jax.vmap(lambda th: fM(xs, keys, tab, th).outputs["Q"]), ths)
    bench(f"vmap events B={BV} (lax.map)", jax.vmap(lambda x: fM(x, keys, tab, TH).outputs["Q"]), xb)
if "host" in MODES:      # host overhead of one eager call / one retrace (small N so the kernel is negligible)
    n = 1024
    f = fF
    f(xs[:n], keys[:n], tab, TH).outputs["Q"].block_until_ready()
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        f(xs[:n], keys[:n], tab, TH).outputs["Q"].block_until_ready()
        ts.append((time.perf_counter() - t0) * 1e3)
    print(f"BENCH tm   {'eager call N=1024':28s} {statistics.median(ts):9.3f} ms", flush=True)
    jf = jax.jit(lambda th: f(xs[:n], keys[:n], tab, th).outputs["Q"])
    jf(TH).block_until_ready()
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        jf(TH).block_until_ready()
        ts.append((time.perf_counter() - t0) * 1e3)
    print(f"BENCH tm   {'jitted call N=1024':28s} {statistics.median(ts):9.3f} ms", flush=True)
    t0 = time.perf_counter()
    for _ in range(5):
        jax.make_jaxpr(lambda th: f(xs[:n], keys[:n], tab, th).outputs["Q"])(TH)
    print(f"BENCH tm   {'trace (make_jaxpr)':28s} {(time.perf_counter() - t0) / 5 * 1e3:9.3f} ms", flush=True)
