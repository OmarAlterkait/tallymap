"""Event scheduling (`_schedule='events'`) against the default step refill, on the one item shape it is for:
a divergent while loop whose body is a LOPSIDED switch -- one branch far more expensive than the others, and
taken rarely.  None of the `benchmarks/vs_jax` problems has a top-level `cond` with a dynamic index in its step
body, so on every one of them the events arm compiles to the identical kernel; this is the measurement that
says what the mechanism buys (docs/design.md, section 1b).

    EV_N (items, default 2^22)   EV_P (probability of the costly branch)   EV_W (its inner trip count)
    EV_MODES (fwd,grad)          EV_REPS (9)

Prints one BENCH line per schedule: median and minimum jax.jit wall ms per call, the compiled-HLO hash (so the
arms can be seen to differ), and each kernel's (mode, registers, local bytes, cubin, cut branches).
"""
import hashlib
import os
import statistics
import time

import jax
import jax.numpy as jnp
from jax import lax

import tallymap as tm

N = int(os.environ.get("EV_N", 1 << 22))
P = float(os.environ.get("EV_P", 0.05))
W = int(os.environ.get("EV_W", 256))
MODES = os.environ.get("EV_MODES", "fwd,grad").split(",")
REPS = int(os.environ.get("EV_REPS", 9))
NB = 256
F = jnp.float32
print(f"EVENTS N {N} P {P} W {W} {jax.devices()} jax {jax.__version__}", flush=True)

xs = jnp.linspace(0.01, 0.99, N, dtype=F)
keys = jax.random.split(jax.random.PRNGKey(0), N)
TH = F(0.7)


def item(t, x0, key, th):
    """One item: a data-dependent number of steps, each choosing one of three branches; branch 2 costs W times
    the others and is taken with probability P."""
    def body(c):
        k, x, t = c
        u = jax.random.uniform(jax.random.fold_in(key, k))
        idx = jnp.where(u < P, 2, (u * 10).astype(jnp.int32) % 2)

        def cheap(j):
            return lambda x, t: (x * 0.9 + th * u + j, t.add("Q", (k + j) % NB, x * th))

        def costly(x, t):
            y = lax.fori_loop(0, W, lambda i, y: jnp.sin(y) * th + 0.01 * i, x)
            return y, t.add("Q", (k * 3) % NB, y)
        x, t = lax.switch(idx, [cheap(0), cheap(1), costly], x, t)
        return k + 1, x, t
    k, x, t = lax.while_loop(lambda c: c[0] < 4 + (x0 * 30).astype(jnp.int32), body, (0, x0, t))
    return t, x * th


def bench(name, fn, *args):
    j = jax.jit(fn)
    h = hashlib.sha1(j.lower(*args).compile().as_text().encode()).hexdigest()[:10]
    jax.block_until_ready(j(*args))
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        jax.block_until_ready(j(*args))
        ts.append((time.perf_counter() - t0) * 1e3)
    ks = [(k["mode"], (k.get("resources") or {}).get("regs"), (k.get("resources") or {}).get("local_bytes"),
           k["hash"][:8], (k.get("cut") or {}).get("branches")) for k in tm.diagnostics()["kernels"][-2:]]
    print(f"BENCH {name:16s} {statistics.median(ts):9.3f} ms  min {min(ts):9.3f}  hlo {h}  {ks}", flush=True)


for mode in MODES:
    for sched in ("step", "events", "item"):
        f = tm.vmap(item, outputs={"Q": tm.Sum((NB,), F)}, in_axes=(0, 0, None), backend="cuda", _schedule=sched)
        if mode == "fwd":
            bench(f"fwd {sched}", lambda th, f=f: f(xs, keys, th).outputs["Q"], TH)
        else:
            bench(f"grad {sched}", jax.grad(lambda th, f=f: jnp.sum(f(xs, keys, th).outputs["Q"] ** 2)), TH)
