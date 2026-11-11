"""How NVRTC's compile time scales with the SIZE of one item, on a reproducible program.

The cuda emitter inlines the whole item into one kernel, so for a large item the compile time IS the size of
that one function.  `docs/design_notes.md` section 9 has the numbers for the photon-transport engine's 4 MB kernel,
which is not public; this is the same curve on a program this repository can run: an item whose loop body is M
stages of 8-element array arithmetic, so every stage becomes straight-line code.

    CS_MS (stage counts, default 25,50,100,150,200)   CS_N (items, default 65536)   CS_LANES (jacfwd columns, 0 = off)

Prints one `CS {json}` line per program: generated source bytes, NVRTC seconds, registers and local bytes.
Run it with a FRESH TALLYMAP_CACHE, or the cached cubins make every compile time 0.
"""
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm

MS = [int(x) for x in os.environ.get("CS_MS", "25,50,100,150,200").split(",")]
N = int(os.environ.get("CS_N", 65536))
T = int(os.environ.get("CS_LANES", 0))
NB = 64
print(f"COMPILE SCALING N {N} stages {MS} jacfwd {T} {jax.devices()} jax {jax.__version__}", flush=True)


def make(M):
    C = jnp.asarray(np.linspace(0.9, 1.1, 8 * M, dtype=np.float32)).reshape(M, 8)

    def item(t, x, key, th):
        def body(c):
            k, v, t = c
            for i in range(M):                        # M stages on an 8-vector: straight-line code
                v = jnp.sin(v * C[i] + th[0]) * 0.99 + jnp.cos(v[::-1] * th[1]) * 0.01
            b = jax.random.bits(jax.random.fold_in(key, k))
            t = t.add("Q", (b % NB).astype(jnp.int32), jnp.sum(v))
            return k + 1, v, t
        k, v, t = lax.while_loop(lambda c: c[0] < 2, body, (0, jnp.full((8,), x), t))
        return t, jnp.sum(v)
    return item


xs = jnp.linspace(0.5, 1.5, N, dtype=jnp.float32)
keys = jax.random.split(jax.random.PRNGKey(0), N)
TH = jnp.full((max(T, 2),), 0.3, jnp.float32)

for M in MS:
    f = tm.vmap(make(M), outputs={"Q": tm.Sum((NB,))}, in_axes=(0, 0, None), backend="auto")
    arms = [("fwd", lambda th: f(xs, keys, th).outputs["Q"])]
    if T:
        arms.append(("jacfwd", jax.jacfwd(lambda th: f(xs, keys, th).outputs["Q"])))
    for which, fn in arms:
        tm.diagnostics(reset=True)
        t0 = time.perf_counter()
        jax.block_until_ready(jax.jit(fn)(TH))
        wall = time.perf_counter() - t0
        for k in tm.diagnostics()["kernels"]:
            r = k.get("resources") or {}
            print("CS " + json.dumps(dict(stages=M, which=which, wall_s=round(wall, 1), schedule=k["schedule"],
                                          src_bytes=os.path.getsize(k["source"]), compile_s=k["compile_s"],
                                          regs=r.get("regs"), local_bytes=r.get("local_bytes"))), flush=True)
