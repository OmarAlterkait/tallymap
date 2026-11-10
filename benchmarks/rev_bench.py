"""Reverse-mode cost: time and memory of primal / jvp / jacfwd(P) / grad (checkpointed) vs path length K,
parameter count P and backend.  python benchmarks/rev_bench.py
env: RB_N (65536)  RB_KS (10,30,100,300,1000)  RB_PS (1,10,tab)  RB_BACKENDS (cuda,xla)  RB_MATH (ieee,approx)
     RB_MODES (fwd,jvp,jacfwd,ckpt,dense)  RB_REPS (5)  RB_TAB (11096)  RB_SLOTS (16: rev_checkpoints, comma list)
Rows: 'ROW {json}'.  Instrument: every arm records the kernel hash(es) it built (cuda) and the vjp program summary
(tallymap._reverse.LAST); arms that must differ are asserted to have different kernels.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm
from tallymap import _reverse

jax.config.update("jax_default_matmul_precision", "highest")
GPU = jax.default_backend() == "gpu"
N = int(os.environ.get("RB_N", 65536))
KS = [int(k) for k in os.environ.get("RB_KS", "10,30,100,300,1000").split(",")]
PS = os.environ.get("RB_PS", "1,10,tab").split(",")
BACKENDS = os.environ.get("RB_BACKENDS", "cuda,xla" if GPU else "xla").split(",")
MATHS = os.environ.get("RB_MATH", "ieee,approx").split(",")
SPARSE_MIN0 = _reverse.SPARSE_MIN
MODES = os.environ.get("RB_MODES", "fwd,jvp,jacfwd,ckpt,dense").split(",")
REPS = int(os.environ.get("RB_REPS", 5))
SLOTS = [int(c) for c in os.environ.get("RB_SLOTS", "16").split(",")]
NTAB = int(os.environ.get("RB_TAB", 11096))
NB = 256                      # output bins (sensor-like)
DEV = jax.devices()[0].device_kind
print("jax", jax.__version__, DEV, "tallymap", tm.__file__, "N", N, flush=True)

# --- instrument: record every cuda kernel built (hash + source) --------------------------------------------
BUILT = []
if GPU:
    import tallymap.backends.kernel as SC
    _K0 = SC.Kernel.__init__

    def _kinit(self, p, **kw):
        _K0(self, p, **kw)
        BUILT.append((self.info.get("hash"), self.src))
    SC.Kernel.__init__ = _kinit


def ptxas(src):
    """(registers, stack bytes, local bytes) of a generated kernel (sm of this GPU)."""
    try:
        from tallymap.backends.kernel.compile import device_arch
        d = tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "/tmp"))
        cu = os.path.join(d, "k.cu")
        open(cu, "w").write(src)
        r = subprocess.run(["/usr/local/cuda/bin/nvcc", "-cubin", f"-arch=sm_{device_arch()}", "-fmad=false",
                            "-std=c++17", "-Xptxas", "-v", cu, "-o", os.path.join(d, "k.cubin")],
                           capture_output=True, text=True)
        txt = r.stderr
        import re
        regs = max([int(x) for x in re.findall(r"Used (\d+) registers", txt)] or [-1])
        stack = max([int(x) for x in re.findall(r"(\d+) bytes stack frame", txt)] or [-1])
        lmem = max([int(x) for x in re.findall(r"(\d+) bytes lmem", txt)] or [0])
        return regs, stack, lmem
    except Exception as e:  # pragma: no cover
        return -1, -1, str(e)[:80]


def kb(key, k):
    return jax.random.bits(jax.random.fold_in(key, k))


def make_item(K, P):
    """walk: per-item length in [K/2, 3K/2]; nonlinear geometry-like state (y, d) + weight w.
    P='1': th[0] scales the weight; P='10': 10 params enter the weight update; P='tab': an NTAB table (QE-like)
    multiplies every emit, read at the emitted bin (sparse cotangent)."""
    def item(t, x, key, th, tab):
        L = (K // 2 + kb(key, 999_999) % jnp.uint32(K + 1)).astype(jnp.int32)

        def body(c):
            (k, w, y, d), t = c
            b = kb(key, k)
            i = (b % jnp.uint32(NB)).astype(jnp.int32)
            d2 = d + 0.1 * jnp.sin(3.0 * y + d)
            y2 = y + 0.05 * jnp.cos(d2)
            if P == "10":
                w2 = w * (0.999 + 0.0001 * jnp.sum(th * jnp.cos(jnp.arange(10, dtype=jnp.float32) * y)))
            else:
                w2 = w * (0.998 + 0.001 * th[0] * jnp.cos(y))
            v = w * (1 + 0.1 * y * y)
            if P == "tab":
                j = ((b >> 8) % jnp.uint32(NTAB)).astype(jnp.int32)
                v = v * tab[j]
            return (k + 1, w2, y2, d2), t.add("Q", i, v)
        (k, w, y, d), t = lax.while_loop(lambda c: c[0][0] < L, body, ((0, x, x, 0.3 * x), t))
        return t, w
    return item


def timed(fn, *args):
    lo = jax.jit(fn).lower(*args)
    t0 = time.perf_counter()
    c = lo.compile()
    comp = time.perf_counter() - t0
    try:
        ma = c.memory_analysis()
        mem = int(ma.temp_size_in_bytes + ma.output_size_in_bytes)
    except Exception:
        mem = -1
    o = jax.block_until_ready(c(*args))
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        o = jax.block_until_ready(c(*args))
        ts.append(time.perf_counter() - t0)
    return o, 1e3 * float(np.median(ts)), comp, mem


def run_arm(backend, math, K, P, mode):
    item = make_item(K, P)
    opts = dict(backend=backend)
    if backend == "auto":
        opts["cuda_math"] = math
    _reverse.SPARSE_MIN = 10 ** 9 if mode == "dense" else SPARSE_MIN0     # 'dense': every table on the dense path
    if mode.startswith("ckpt"):
        if mode.startswith("ckpt_s"):
            opts["rev_checkpoints"] = int(mode[6:])
    f = tm.vmap(item, outputs={"Q": tm.Sum((NB,))}, in_axes=(0, 0, None, None), **opts)
    xs = jnp.linspace(0.5, 1.5, N).astype(jnp.float32)
    keys = jax.random.split(jax.random.PRNGKey(0), N)
    th = (jnp.ones(10, jnp.float32) * 0.9) if P == "10" else jnp.array([0.9], jnp.float32)
    tab = (1.0 + 0.1 * jnp.sin(jnp.arange(NTAB))).astype(jnp.float32)
    wq = jnp.cos(jnp.arange(NB)).astype(jnp.float32)

    def L(th, tab, xs, keys):
        r = f(xs, keys, th, tab)
        return jnp.sum(r.outputs["Q"] * wq) + jnp.sum(r.per_item ** 2)
    wrt = 1 if P == "tab" else 0
    nb0 = len(BUILT)
    if mode == "fwd":
        fn = L
    elif mode == "jvp":
        def fn(th, tab, xs, keys):
            if wrt:
                return jax.jvp(lambda q: L(th, q, xs, keys), (tab,), (jnp.ones_like(tab),))
            return jax.jvp(lambda q: L(q, tab, xs, keys), (th,), (jnp.ones_like(th),))
    elif mode == "jacfwd":
        if P == "tab":
            return None
        fn = jax.jacfwd(L, argnums=0)
    else:
        fn = jax.grad(L, argnums=wrt)
    o, ms, comp, mem = timed(fn, th, tab, xs, keys)
    built = BUILT[nb0:]
    row = dict(dev=DEV, backend=backend, math=math if backend == "auto" else "-", K=K, P=P, mode=mode, N=N,
               ms=ms, compile_s=comp, xla_mem=mem, kernels=[h for h, _ in built])
    if built:
        rs = [ptxas(s) for _, s in built]
        row["regs"] = [r[0] for r in rs]
        row["stack"] = [r[1] for r in rs]
    if mode.startswith(("ckpt", "dense")):
        row["vjp"] = dict(sparse=_reverse.LAST.get("sparse"), dense=_reverse.LAST.get("dense"),
                          loops=_reverse.LAST.get("loops"))
        row["g"] = np.asarray(o, np.float64).ravel()[:4].tolist()
        row["gnorm"] = float(np.linalg.norm(np.asarray(o, np.float64)))
        row["_g"] = np.asarray(o, np.float64)
    if mode == "jacfwd":
        row["_g"] = np.asarray(o, np.float64)
        row["gnorm"] = float(np.linalg.norm(row["_g"]))
    return row


ROWS = []
for backend in BACKENDS:
    for math in (MATHS if backend == "auto" else ["-"]):
        for P in PS:
            for K in KS:
                arms = {}
                modes = [m for x in MODES
                         for m in ([f"ckpt_s{c}" for c in SLOTS] if x == "ckpt" and len(SLOTS) > 1 else [x])]
                for mode in modes:
                    if mode == "dense" and P != "tab":
                        continue
                    try:
                        r = run_arm(backend, math, K, P, mode)
                    except Exception as e:
                        r = dict(backend=backend, math=math, K=K, P=P, mode=mode,
                                 error=f"{type(e).__name__}: {str(e)[:300]}")
                    if r is None:
                        continue
                    arms[mode] = r
                    g = r.pop("_g", None)
                    if g is not None:
                        r["_gv"] = g
                    pr = {k: v for k, v in r.items() if k != "_gv"}
                    print("ROW", json.dumps(pr), flush=True)
                    ROWS.append(pr)
                # instrument + consistency across arms of this (backend, math, P, K)
                ref = arms.get("jacfwd") or arms.get("ckpt") or arms.get(f"ckpt_s{SLOTS[-1]}")
                for m in [x for x in arms if x.startswith(("ckpt", "dense"))]:
                    if m in arms and ref is not None and "_gv" in arms[m] and "_gv" in ref and arms[m] is not ref:
                        a, b = arms[m]["_gv"], ref["_gv"]
                        e = float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30))
                        cmp = dict(backend=backend, math=math, P=P, K=K, arm=m,
                                   ref="jacfwd" if "jacfwd" in arms else "ckpt", relerr=e)
                        print("CMP", json.dumps(cmp), flush=True)
                if backend == "auto":
                    for a, b in (("ckpt", "dense"),):
                        if a in arms and b in arms and "kernels" in arms[a] and "kernels" in arms[b]:
                            same = set(arms[a]["kernels"]) & set(arms[b]["kernels"]) - {None}
                            vk = [h for h in arms[a]["kernels"] if h not in arms.get("fwd", {}).get("kernels", [])]
                            print("ABCHECK", a, b, "K", K, "P", P, "shared kernels", len(same), "of",
                                  len(arms[a]["kernels"]), flush=True)
print("DONE", len(ROWS), flush=True)
