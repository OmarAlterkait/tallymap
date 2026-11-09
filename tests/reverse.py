"""Reverse-mode matrix.  python tests/reverse.py
env: BACKENDS=lockstep,cuda  TEST_X64=1  ONLY=substr  KS=1,10,100,1000

Every gradient is compared, as full VECTORS, against jax.grad of a PADDED-SCAN version of the same item (dense
per-item tallies, jax.vmap + sum: tallymap.reference), and where cheap against jacfwd / jvp of
tallymap's forward mode.  The evaluation point is always a jit ARGUMENT (F3 rule).  Instrument checks: the
sparse/dense table path and the scheme actually built are read back from tallymap._reverse.LAST and asserted before a
result is accepted.
"""
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.environ.get("TEST_X64"):
    import jax
    jax.config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm
from tallymap import _reverse

if os.environ.get("TX_UNFUSED") == "1":                 # run_all --unfused: the two-node jvp design
    from tallymap import _primitive as _P0
    _P0.FUSE_JVP = False
if os.environ.get("TX_T3") == "1":                      # run_all --t3: the single-loop adjoint wherever it applies
    _reverse.SINGLE = "force"
from tallymap import reference

jax.config.update("jax_default_matmul_precision", "highest")
X64 = bool(jax.config.jax_enable_x64)
F = jnp.float64 if X64 else jnp.float32
GPU = jax.default_backend() == "gpu"
BACKENDS = os.environ.get("BACKENDS", "lockstep,cuda" if GPU else "lockstep").split(",")
ONLY = os.environ.get("ONLY", "")
KS = [int(k) for k in os.environ.get("KS", "1,10,100,1000").split(",")]
N = int(os.environ.get("RV_N", 48))
TOL = 1e-9 if X64 else 3e-4
print("jax", jax.__version__, jax.__file__, jax.devices(), "x64", X64, "tallymap", tm.__file__, flush=True)
RES = []


class Fail(Exception):
    pass


def relerr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        raise Fail(f"shape {a.shape} vs {b.shape}")
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30)) if b.size else 0.0


def close(a, b, what, tol=None):
    tol = TOL if tol is None else tol
    msgs = []
    for k in a:
        e = relerr(a[k], b[k])
        ref = float(np.max(np.abs(np.asarray(b[k]))))
        nz = int(np.count_nonzero(np.asarray(a[k])))
        msgs.append(f"{k}:{e:.1e}" + (f"(|ref| {ref:.2g}, nz {nz})" if e == 0 else ""))
        if not e <= tol:
            raise Fail(f"{what} {k} rel err {e:.3g} > {tol:.1g}  ({' '.join(msgs)})")
    return " ".join(msgs)


def case(name, backends=None):
    def deco(fn):
        for b in (backends or BACKENDS):
            if b not in BACKENDS or (ONLY and ONLY not in name):
                continue
            t0 = time.perf_counter()
            try:
                r = fn(b)
                msg = "PASS" + (f" ({r})" if isinstance(r, str) else "")
            except Fail as e:
                msg = f"FAIL {e}"
            except Exception as e:
                msg = f"RAISE {type(e).__name__}: {str(e)[:300]}"
                traceback.print_exc()
            RES.append((name, b, msg))
            print(f"{name:58s} {b:8s} {msg}  [{time.perf_counter() - t0:.1f}s]", flush=True)
        return fn
    return deco


def kb(key, k):
    return jax.random.bits(jax.random.fold_in(key, k))


def mask_tree(act, new, old):
    return jax.tree_util.tree_map(lambda a, b: jnp.where(act, a, b), new, old)


# ------------------------------------------------------------------------------------------- toy 1: long walk
def make_walk(NB, K):
    """per-item path length in [K//2, K//2 + K] (integer exact from the key); nonlinear state (w, y); a table
    read per step; th = 3 broadcast params.  Returns (item, padded item, KMAX)."""
    KMAX = K // 2 + K + 1

    def length(key):
        return (K // 2 + kb(key, 999_999) % jnp.uint32(K + 1)).astype(jnp.int32)

    def step(c, key, tab, th):
        k, w, y = c
        i = (kb(key, k) % jnp.uint32(NB)).astype(jnp.int32)
        v = w * tab[i] * (1 + 0.1 * y * y) * th[0]
        y2 = y + 0.05 * jnp.sin(3 * y + th[1])
        w2 = w * (0.999 + 0.001 * jnp.cos(y * th[2]))
        return (k + 1, w2, y2), i, v

    def item(t, x, key, tab, th):
        L = length(key)

        def body(c):
            s, t = c
            s2, i, v = step(s, key, tab, th)
            return s2, t.add("Q", i, v)
        (k, w, y), t = lax.while_loop(lambda c: c[0][0] < L, body, ((0, x, 0.5 * x), t))
        return t, (w, y)

    def padded(t, x, key, tab, th):
        L = length(key)

        def body(c, _):
            s, t = c
            act = s[0] < L
            s2, i, v = step(s, key, tab, th)
            return (mask_tree(act, s2, s), t.add("Q", jnp.where(act, i, -1), v)), None
        ((k, w, y), t), _ = lax.scan(body, ((0, x, 0.5 * x), t), None, length=KMAX)
        return t, (w, y)
    return item, padded


def inputs(NB, seed=0, n=N):
    xs = jnp.linspace(0.5, 1.5, n).astype(F)
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    tab = (1.0 + 0.5 * jnp.sin(jnp.arange(NB))).astype(F)
    th = jnp.array([0.9, 0.3, 1.7], F)
    wq = (1.0 + jnp.cos(jnp.arange(NB))).astype(F)
    return xs, keys, tab, th, wq


def loss_of(f, wq):
    def L(x, keys, tab, th):
        r = f(x, keys, tab, th)
        w, y = r.per_item
        return jnp.sum(r.outputs["Q"] * wq) + jnp.sum(w * w) + jnp.sum(jnp.sin(y))
    return L


def grads(L, args):
    g = jax.jit(jax.grad(L, argnums=(0, 2, 3)))(*args)
    return dict(x=g[0], tab=g[1], th=g[2])


def walk_case(b, NB, K, **ro):
    item, padded = make_walk(NB, K)
    xs, keys, tab, th, wq = inputs(NB)
    AX = (0, 0, None, None)
    f = tm.vmap(item, outputs={"Q": tm.Sum((NB,), F)}, in_axes=AX, backend=b, **ro)
    r = reference(padded, outputs={"Q": tm.Sum((NB,), F)}, in_axes=AX)
    g = grads(loss_of(f, wq), (xs, keys, tab, th))
    info = dict(_reverse.LAST)
    gr = grads(loss_of(r, wq), (xs, keys, tab, th))
    return g, gr, info, (f, xs, keys, tab, th, wq, AX, item)


for K in KS:
    @case(f"walk K~{K} NB=8 (dense table): rev(ckpt) == grad(padded)")
    def _(b, K=K):
        g, gr, info, _ = walk_case(b, 8, K)
        if info["sparse"]:
            raise Fail(f"instrument: expected a dense table, got {info}")
        act = [lp["n_active"] for lp in info["loops"]], [lp["n_float"] for lp in info["loops"]]
        return close(g, gr, "ckpt") + f" act {act[0]}/{act[1]}"

    @case(f"walk K~{K} NB=4096 (sparse table): rev(ckpt) == grad(padded)")
    def _(b, K=K):
        g, gr, info, _ = walk_case(b, 4096, K)
        if not info["sparse"] or any(i for i, s in info["dense"] if s >= 32):
            raise Fail(f"instrument: expected the 4096 table on the sparse path, got {info}")
        return close(g, gr, "ckpt sparse")

    if K <= 100:
        @case(f"walk K~{K} NB=4096: rev(dense path forced for the big table) == grad(padded)")
        def _(b, K=K):
            g, gr, info, _ = walk_case(b, 4096, K, _rev_sparse_min=10 ** 9)
            if info["sparse"]:
                raise Fail(f"instrument: expected the dense path, got {info}")
            return close(g, gr, "dense")


@case("walk K~100: slots 16 vs slots 3 per-item cotangents (two checkpoint layouts; bitwise on cuda)")
def _(b):
    g1, gr, _, _ = walk_case(b, 64, 100)
    g2, _, info, _ = walk_case(b, 64, 100, rev_checkpoints=3)
    if info["loops"][0]["checkpoints"] != 3:
        raise Fail(f"instrument: slots {info['loops']}")
    dx = np.asarray(g1["x"]) - np.asarray(g2["x"])
    nd = int((dx != 0).sum())
    if b == "cuda" and nd:
        raise Fail(f"cuda: {nd}/{dx.size} per-item cotangents differ between slots 16 and 3 "
                   f"(max {np.abs(dx).max():.3g})")
    return f"per-item x differs in {nd}/{dx.size} (max {np.abs(dx).max():.2g}); " + close(g1, g2, "slots 16 vs 3")


@case("walk K~40: per-item cotangents are bitwise equal for slots 4 / 16 / 64 (C changes work, not values)")
def _(b):
    # The checkpoint stride is C (not ceil(K/C)), so C decides which states are stored and which are refilled
    # from a checkpoint.  A refilled state runs the identical body on the identical input, so it is bitwise the
    # stored one: changing C must not move a single per-item cotangent.  K ~ 40 spans the three regimes here
    # (K > C^2 at C=4, K > C at C=16, K < C at C=64).
    ref, out = None, []
    for C in (4, 16, 64):
        g, _, info, _ = walk_case(b, 64, 40, rev_checkpoints=C)
        if info["loops"][0]["checkpoints"] != C:
            raise Fail(f"instrument: slots {info['loops']}")
        x = np.asarray(g["x"])
        if ref is None:
            ref = x
            continue
        nd = int((x != ref).sum())
        if b == "cuda" and nd:
            raise Fail(f"cuda: slots {C} moved {nd}/{x.size} per-item cotangents against slots 4 "
                       f"(max {np.abs(x - ref).max():.3g})")
        out.append(f"C={C}: {nd}/{x.size} differ")
    return "; ".join(out) or "bitwise equal"


for C in (1, 3, 7):
    @case(f"walk K~200 slots={C} (K > C^2: sub-segment replay path)")
    def _(b, C=C):
        g, gr, info, _ = walk_case(b, 64, 200, rev_checkpoints=C)
        if info["loops"][0]["checkpoints"] != C:
            raise Fail(f"instrument: slots {info['loops']}")
        return close(g, gr, f"slots{C}")


@case("walk K exactly 16/256/257 lengths, slots=16 (boundaries)")
def _(b):
    out = []
    for K in (16, 256, 257):
        NB = 32

        def item(t, x, key, tab, th, K=K):
            def body(c):
                (k, w, y), t = c
                i = (kb(key, k) % jnp.uint32(NB)).astype(jnp.int32)
                return (k + 1, w * (0.999 + 0.001 * jnp.cos(y * th[2])), y + 0.05 * jnp.sin(3 * y + th[1])), \
                    t.add("Q", i, w * tab[i] * th[0])
            (k, w, y), t = lax.while_loop(lambda c: c[0][0] < K, body, ((0, x, 0.5 * x), t))
            return t, (w, y)

        def padded(t, x, key, tab, th, K=K):
            def body(c, _):
                (k, w, y), t = c
                i = (kb(key, k) % jnp.uint32(NB)).astype(jnp.int32)
                return ((k + 1, w * (0.999 + 0.001 * jnp.cos(y * th[2])), y + 0.05 * jnp.sin(3 * y + th[1])),
                        t.add("Q", i, w * tab[i] * th[0])), None
            ((k, w, y), t), _ = lax.scan(body, ((0, x, 0.5 * x), t), None, length=K)
            return t, (w, y)
        xs, keys, tab, th, wq = inputs(NB)
        AX = (0, 0, None, None)
        f = tm.vmap(item, outputs={"Q": tm.Sum((NB,), F)}, in_axes=AX, backend=b, rev_checkpoints=16)
        r = reference(padded, outputs={"Q": tm.Sum((NB,), F)}, in_axes=AX)
        out.append(f"K={K} " + close(grads(loss_of(f, wq), (xs, keys, tab, th)),
                                     grads(loss_of(r, wq), (xs, keys, tab, th)), f"K={K}"))
    return "; ".join(out)


@case("walk K~30: rev == jacfwd(forward) on th (3 params) and jvp on tab/x directions")
def _(b):
    g, gr, info, (f, xs, keys, tab, th, wq, AX, item) = walk_case(b, 4096, 30)
    ff = tm.vmap(item, outputs={"Q": tm.Sum((4096,), F)}, in_axes=AX, backend=b)
    Lf = loss_of(ff, wq)
    jth = jax.jit(jax.jacfwd(Lf, argnums=3))(xs, keys, tab, th)
    dx = jnp.cos(jnp.arange(N)).astype(F)
    dt = jnp.sin(jnp.arange(4096) * 0.37).astype(F)
    jx = jax.jit(lambda x, k, tb, t: jax.jvp(lambda x_: Lf(x_, k, tb, t), (x,), (dx,))[1])(xs, keys, tab, th)
    jt = jax.jit(lambda x, k, tb, t: jax.jvp(lambda tb_: Lf(x, k, tb_, t), (tb,), (dt,))[1])(xs, keys, tab, th)
    e = [relerr(g["th"], jth), relerr(jnp.vdot(g["x"], dx), jx), relerr(jnp.vdot(g["tab"], dt), jt)]
    if max(e) > TOL:
        raise Fail(f"rev vs forward: th {e[0]:.2g} x-dir {e[1]:.2g} tab-dir {e[2]:.2g}")
    return f"th {e[0]:.1e} x-dir {e[1]:.1e} tab-dir {e[2]:.1e}"


# ------------------------------------------------------------------------------------------- toy 2: nested loops
def make_nested(NB):
    K1, K2 = 6, 6

    def inner_step(d, k, key, tab, th):
        j, z = d
        i = (kb(key, 1000 * k + j) % jnp.uint32(NB)).astype(jnp.int32)
        return (j + 1, z * (0.9 + 0.1 * jnp.cos(z * th[1])) + 0.01 * tab[i]), i, z * tab[i] * th[0]

    def L1(key):
        return (2 + kb(key, 7) % jnp.uint32(K1 - 1)).astype(jnp.int32)

    def L2(key, k):
        return (1 + kb(key, 100 + k) % jnp.uint32(K2)).astype(jnp.int32)

    def item(t, x, key, tab, th):
        def outer(c):
            k, y, t = c
            n2 = L2(key, k)

            def inner(d):
                s, t = d
                s2, i, v = inner_step(s, k, key, tab, th)
                return s2, t.add("Q", i, v)
            (_, y), t = lax.while_loop(lambda d: d[0][0] < n2, inner, ((0, y), t))
            return k + 1, y + 0.1 * jnp.sin(y * th[2]), t
        k, y, t = lax.while_loop(lambda c: c[0] < L1(key), outer, (0, x, t))
        return t, (y, y * y)

    def padded(t, x, key, tab, th):
        n1 = L1(key)

        def outer(c, _):
            k, y, t = c
            a1 = k < n1
            n2 = L2(key, k)

            def inner(d, _):
                s, t = d
                act = a1 & (s[0] < n2)
                s2, i, v = inner_step(s, k, key, tab, th)
                return (mask_tree(act, s2, s), t.add("Q", jnp.where(act, i, -1), v)), None
            ((_, y2), t), _ = lax.scan(inner, ((0, y), t), None, length=K2)
            return (mask_tree(a1, (k + 1, y2 + 0.1 * jnp.sin(y2 * th[2])), (k, y)) + (t,)), None
        (k, y, t), _ = lax.scan(outer, (0, x, t), None, length=K1)
        return t, (y, y * y)
    return item, padded


for NB in (8, 512):
    @case(f"nested while (emits + table reads in the inner loop) NB={NB}")
    def _(b, NB=NB):
        item, padded = make_nested(NB)
        xs, keys, tab, th, wq = inputs(NB)
        AX = (0, 0, None, None)
        f = tm.vmap(item, outputs={"Q": tm.Sum((NB,), F)}, in_axes=AX, backend=b)
        r = reference(padded, outputs={"Q": tm.Sum((NB,), F)}, in_axes=AX)
        g = grads(loss_of(f, wq), (xs, keys, tab, th))
        info = dict(_reverse.LAST)
        if (NB >= 32) != bool(info["sparse"]):
            raise Fail(f"instrument: {info}")
        return close(g, grads(loss_of(r, wq), (xs, keys, tab, th)), "nested") + f" loops {len(info['loops'])}"


# ------------------------------------------------------------------------------------------- toy 3: cond / scan / mixed
def make_branchy(NB, mixed=False, use_scan=False):
    K = 20

    def step(c, key, tab, th):
        k, w, y = c
        bits = kb(key, k)
        i = (bits % jnp.uint32(NB)).astype(jnp.int32)
        br = ((bits >> 8) % jnp.uint32(3)).astype(jnp.int32)
        v, w2 = lax.switch(br, [lambda: (w * tab[i], w * 0.97), lambda: (w * th[0], w * (1.0 + 0.01 * y)),
                                lambda: (jnp.zeros_like(w), w)])
        if use_scan:
            def sb(a, s):
                return a + tab[(i + s) % NB] * 0.1, None
            acc, _ = lax.scan(sb, jnp.zeros_like(w), jnp.arange(3))
            v = v + acc * w
        if mixed:
            v = v + 1e-2 * w * jnp.sum(tab * tab)          # a DENSE use of the table next to the reads
        return (k + 1, w2, y + 0.05 * jnp.sin(3 * y + th[1])), i, v

    def item(t, x, key, tab, th):
        def body(c):
            s, t = c
            s2, i, v = step(s, key, tab, th)
            return s2, t.add("Q", i, v)
        (k, w, y), t = lax.while_loop(lambda c: c[0][0] < K, body, ((0, x, 0.5 * x), t))
        return t, (w, y)

    def padded(t, x, key, tab, th):
        def body(c, _):
            s, t = c
            s2, i, v = step(s, key, tab, th)
            return (s2, t.add("Q", i, v)), None
        ((k, w, y), t), _ = lax.scan(body, ((0, x, 0.5 * x), t), None, length=K)
        return t, (w, y)
    return item, padded


def branchy_case(b, NB, mixed=False, use_scan=False, **ro):
    item, padded = make_branchy(NB, mixed, use_scan)
    xs, keys, tab, th, wq = inputs(NB)
    AX = (0, 0, None, None)
    f = tm.vmap(item, outputs={"Q": tm.Sum((NB,), F)}, in_axes=AX, backend=b, **ro)
    r = reference(padded, outputs={"Q": tm.Sum((NB,), F)}, in_axes=AX)
    g = grads(loss_of(f, wq), (xs, keys, tab, th))
    info = dict(_reverse.LAST)
    return g, grads(loss_of(r, wq), (xs, keys, tab, th)), info


@case("switch in the body, table read in one branch (sparse)")
def _(b):
    g, gr, info = branchy_case(b, 256)
    if not info["sparse"]:
        raise Fail(f"instrument {info}")
    return close(g, gr, "switch")


@case("scan inside the body reading the table (sparse)")
def _(b):
    g, gr, info = branchy_case(b, 256, use_scan=True)
    if not info["sparse"]:
        raise Fail(f"instrument {info}")
    return close(g, gr, "scan")


@case("MIXED use (reads + jnp.sum(tab*tab)): dense detected, still exact")
def _(b):
    g, gr, info = branchy_case(b, 256, mixed=True)
    if not any(s == 256 for _, s in info["dense"]):
        raise Fail(f"instrument: the mixed table must be on the dense path: {info}")
    return close(g, gr, "mixed")


@case("NEGATIVE CONTROL: mixed use with dense detection disabled MUST differ", ["lockstep"])
def _(b):
    g, gr, _ = branchy_case(b, 256, mixed=True)
    gb, _, info = branchy_case(b, 256, mixed=True, _rev_nodense=True)
    if any(s == 256 for _, s in info["dense"]):
        raise Fail(f"instrument: the defect was not applied {info}")
    e = relerr(gb["tab"], gr["tab"])
    if e < 100 * TOL:
        raise Fail(f"defect not detected: tab rel err {e:.3g} (correct run {relerr(g['tab'], gr['tab']):.2g})")
    return f"defect gives tab rel err {e:.3g} (correct run {relerr(g['tab'], gr['tab']):.1e})"


@case("SECOND ORDER through sparse table reads (NB=64): hessian, jacrev(jacrev), jvp(grad) vs dense")
def _(b):
    NB, K = 64, 20
    item, padded = make_walk(NB, K)
    xs, keys, tab, th, wq = inputs(NB)
    AX = (0, 0, None, None)
    f = tm.vmap(item, outputs={"Q": tm.Sum((NB,), F)}, in_axes=AX, backend=b)
    r = reference(padded, outputs={"Q": tm.Sum((NB,), F)}, in_axes=AX)
    P = jnp.concatenate([tab[:6], th])

    def L(fn):
        def g(p):
            q = fn(xs, keys, tab.at[:6].set(p[:6]) * (1 + 0.1 * p[6]), p[6:]).outputs["Q"]
            return jnp.sum(q * wq) + 0.1 * jnp.sum(q * q) / NB
        return g
    jax.grad(L(f))(P)
    if not _reverse.LAST.get("sparse"):
        raise Fail(f"instrument: the 64-entry table must be on the sparse path: {_reverse.LAST}")
    H = jax.jacfwd(jax.jacfwd(L(r)))(P)
    v = jnp.linspace(-1.0, 1.0, P.size).astype(F)
    got = dict(hessian=jax.jit(jax.hessian(L(f)))(P), jacrev2=jax.jit(jax.jacrev(jax.jacrev(L(f))))(P),
               hvp=jax.jit(lambda p: jax.jvp(jax.grad(L(f)), (p,), (v,))[1])(P))
    return close(got, dict(hessian=H, jacrev2=H, hvp=H @ v), "second order")


# ------------------------------------------------------------------------------------------- toy 3b: token threading
def make_constructs(NB, which):
    """a table read nested in one construct chain inside the top-level loop's body.  item: real while loops;
    padded: the same with static fori loops (scans: jax.grad can differentiate them) -> the reference."""
    K = 12

    def loop(n, f, init, real):
        if real:
            return lax.while_loop(lambda c: c[0] < n, lambda c: (c[0] + 1, f(c[0], c[1])), (0, init))[1]
        return lax.fori_loop(0, n, f, init)

    def step(c, key, tab, th, real):
        k, w, y = c
        bits = kb(key, k)
        i = (bits % jnp.uint32(NB)).astype(jnp.int32)
        rd = lambda j: tab[(i + j) % NB]
        bit = lambda s: ((bits >> jnp.asarray(s).astype(jnp.uint32)) & jnp.uint32(1)) == 0
        if which == "while-in-cond-in-scan":
            def sb(a, s):
                return lax.cond(bit(s), lambda a: loop(2, lambda m, v: v + 0.1 * rd(m + s), a, real),
                                lambda a: a * 0.99, a), None
            v = lax.scan(sb, w, jnp.arange(2))[0]
        elif which == "cond-in-while":
            v = loop(3, lambda m, v: lax.cond(bit(m), lambda v: v + 0.1 * rd(m), lambda v: v * 0.98, v), w, real)
        elif which == "scan-in-cond-in-while":
            v = loop(2, lambda m, v: lax.cond(bit(m + 4), lambda v: lax.scan(
                lambda a, s: (a + 0.05 * rd(s + m), None), v, jnp.arange(3))[0], lambda v: v, v), w, real)
        elif which == "jit-in-while":
            h = jax.jit(lambda v, m: v + 0.1 * rd(m) * th[0])
            v = loop(2, lambda m, v: h(v, m), w, real)
        elif which == "custom_jvp on the whole table (dense)":
            v = w + 0.1 * jax.nn.relu(tab - 1.0)[i]
        else:
            raise ValueError(which)
        return (k + 1, w * 0.97 + 0.01 * v, y + 0.05 * jnp.sin(3 * y + th[1])), i, v * w

    def item(t, x, key, tab, th):
        def body(c):
            s, t = c
            s2, i, v = step(s, key, tab, th, True)
            return s2, t.add("Q", i, v)
        (k, w, y), t = lax.while_loop(lambda c: c[0][0] < K, body, ((0, x, 0.5 * x), t))
        return t, (w, y)

    def padded(t, x, key, tab, th):
        def body(c, _):
            s, t = c
            s2, i, v = step(s, key, tab, th, False)
            return (s2, t.add("Q", i, v)), None
        ((k, w, y), t), _ = lax.scan(body, ((0, x, 0.5 * x), t), None, length=K)
        return t, (w, y)
    return item, padded


for which in ("while-in-cond-in-scan", "cond-in-while", "scan-in-cond-in-while", "jit-in-while",
              "custom_jvp on the whole table (dense)"):
    @case(f"token threading: table read in {which}")
    def _(b, which=which):
        NB = 256
        item, padded = make_constructs(NB, which)
        xs, keys, tab, th, wq = inputs(NB)
        AX = (0, 0, None, None)
        f = tm.vmap(item, outputs={"Q": tm.Sum((NB,), F)}, in_axes=AX, backend=b)
        r = reference(padded, outputs={"Q": tm.Sum((NB,), F)}, in_axes=AX)
        g = grads(loss_of(f, wq), (xs, keys, tab, th))
        info = dict(_reverse.LAST)
        dense = which.endswith("(dense)")
        on_sparse = any(s == NB for _, s in info["sparse"])
        if on_sparse == dense:
            raise Fail(f"instrument: the {NB}-entry table should be on the {'dense' if dense else 'sparse'} path: "
                       f"{info}")
        return close(g, grads(loss_of(r, wq), (xs, keys, tab, th)), which) + (" (dense)" if dense else " (sparse)")


# ----------------------------------------------------------------------------------------- toy 4: ints, keys, closures
@case("typed key carried + int per-item input + closed-over th (symbolic zeros)")
def _(b):
    NB = 64

    def mk(th_c, padded):
        def item(t, x, i0, key):
            def step(c):
                k, w, key = c
                key, sub = jax.random.split(key)
                u = jax.random.uniform(sub, dtype=F)
                i = (i0 + k * 7) % NB
                return (k + 1, w * (0.8 + 0.2 * u * th_c[0]), key), i, w * u * th_c[1]
            if padded:
                def body(c, _):
                    s, t = c
                    s2, i, v = step(s)
                    return (s2, t.add("Q", i, v)), None
                ((k, w, key), t), _ = lax.scan(body, ((0, x, key), t), None, length=12)
            else:
                def body(c):
                    s, t = c
                    s2, i, v = step(s)
                    return s2, t.add("Q", i, v)
                (k, w, key), t = lax.while_loop(lambda c: c[0][0] < 12, body, ((0, x, key), t))
            return t, (w, i0 * 2)
        return item
    xs = jnp.linspace(0.5, 1.5, N).astype(F)
    ii = jnp.arange(N, dtype=jnp.int32)
    keys = jax.random.split(jax.random.key(3), N)
    wq = jnp.cos(jnp.arange(NB)).astype(F)

    def L(th, x, rev):
        if rev:
            f = tm.vmap(mk(th, False), outputs={"Q": tm.Sum((NB,), F)}, backend=b)
        else:
            f = reference(mk(th, True), outputs={"Q": tm.Sum((NB,), F)})
        r = f(x, ii, keys)
        return jnp.sum(r.outputs["Q"] * wq) + jnp.sum(r.per_item[0] ** 2)
    th = jnp.array([0.7, 1.3], F)
    g = jax.jit(jax.grad(lambda t, x: L(t, x, True), argnums=(0, 1)))(th, xs)
    gr = jax.jit(jax.grad(lambda t, x: L(t, x, False), argnums=(0, 1)))(th, xs)
    return close(dict(th=g[0], x=g[1]), dict(th=gr[0], x=gr[1]), "keys/ints/closure")


@case("K=0 items (loop never runs) mixed with long ones")
def _(b):
    NB = 64
    item, padded = make_walk(NB, 40)
    xs, keys, tab, th, wq = inputs(NB)

    def item0(t, x, key, tab, th):           # odd items: zero-length path
        def body(c):
            (k, w, y), t = c
            i = (kb(key, k) % jnp.uint32(NB)).astype(jnp.int32)
            return (k + 1, w * 0.99, y), t.add("Q", i, w * tab[i] * th[0])
        L = jnp.where(x > 1.0, 0, 20)
        (k, w, y), t = lax.while_loop(lambda c: c[0][0] < L, body, ((0, x, x), t))
        return t, (w, y)

    def pad0(t, x, key, tab, th):
        def body(c, _):
            (k, w, y), t = c
            act = k < jnp.where(x > 1.0, 0, 20)
            i = (kb(key, k) % jnp.uint32(NB)).astype(jnp.int32)
            return (mask_tree(act, (k + 1, w * 0.99, y), (k, w, y)),
                    t.add("Q", jnp.where(act, i, -1), w * tab[i] * th[0])), None
        ((k, w, y), t), _ = lax.scan(body, ((0, x, x), t), None, length=20)
        return t, (w, y)
    AX = (0, 0, None, None)
    f = tm.vmap(item0, outputs={"Q": tm.Sum((NB,), F)}, in_axes=AX, backend=b)
    r = reference(pad0, outputs={"Q": tm.Sum((NB,), F)}, in_axes=AX)
    return close(grads(loss_of(f, wq), (xs, keys, tab, th)), grads(loss_of(r, wq), (xs, keys, tab, th)), "K=0")


@case("activity: weight-only parameter differentiates only the active carry")
def _(b):
    NB = 16

    def item(t, x, key, th):
        def body(c):
            (k, pos, d, w), t = c                  # pos/d: geometry-like state independent of th
            u = jax.random.uniform(jax.random.fold_in(key, k), dtype=F)
            d2 = jnp.cos(d + u)
            pos2 = pos + d2
            i = (jnp.abs(pos2 * 3).astype(jnp.int32)) % NB
            return (k + 1, pos2, d2, w * th), t.add("Q", i, w)
        (k, pos, d, w), t = lax.while_loop(lambda c: c[0][0] < 15, body, ((0, x, x * 0.3, x * 0 + 1), t))
        return t

    def padded(t, x, key, th):
        def body(c, _):
            (k, pos, d, w), t = c
            u = jax.random.uniform(jax.random.fold_in(key, k), dtype=F)
            d2 = jnp.cos(d + u)
            pos2 = pos + d2
            i = (jnp.abs(pos2 * 3).astype(jnp.int32)) % NB
            return ((k + 1, pos2, d2, w * th), t.add("Q", i, w)), None
        (_, t), _ = lax.scan(body, ((0, x, x * 0.3, x * 0 + 1), t), None, length=15)
        return t
    xs = jnp.linspace(0.5, 1.5, N).astype(F)
    keys = jax.random.split(jax.random.PRNGKey(1), N)
    wq = jnp.cos(jnp.arange(NB)).astype(F)
    f = tm.vmap(item, outputs={"Q": tm.Sum((NB,), F)}, in_axes=(0, 0, None), backend=b)
    r = reference(padded, outputs={"Q": tm.Sum((NB,), F)}, in_axes=(0, 0, None))
    th = F(0.95)
    g = jax.jit(jax.grad(lambda t: jnp.sum(f(xs, keys, t).outputs["Q"] * wq)))(th)
    lp = _reverse.LAST["loops"][0]
    gr = jax.jit(jax.grad(lambda t: jnp.sum(r(xs, keys, t).outputs["Q"] * wq)))(th)
    if lp["n_active"] != 1:
        raise Fail(f"activity: expected 1 active carry (w), got {lp}")
    return close(dict(th=g), dict(th=gr), "activity") + f" active {lp['n_active']}/{lp['n_float']}"


@case("max_steps truncation + n_active under reverse (capped loop, inactive items)")
def _(b):
    NB, K, CAP = 64, 40, 25
    item, _ = make_walk(NB, K)

    def padded(t, x, key, tab, th):                    # the capped walk: at most CAP steps
        L = jnp.minimum((K // 2 + kb(key, 999_999) % jnp.uint32(K + 1)).astype(jnp.int32), CAP)

        def body(c, _):
            (k, w, y), t = c
            act = k < L
            i = (kb(key, k) % jnp.uint32(NB)).astype(jnp.int32)
            v = w * tab[i] * (1 + 0.1 * y * y) * th[0]
            s2 = (k + 1, w * (0.999 + 0.001 * jnp.cos(y * th[2])), y + 0.05 * jnp.sin(3 * y + th[1]))
            return (mask_tree(act, s2, (k, w, y)), t.add("Q", jnp.where(act, i, -1), v)), None
        ((k, w, y), t), _ = lax.scan(body, ((0, x, 0.5 * x), t), None, length=CAP)
        return t, (w, y)
    xs, keys, tab, th, wq = inputs(NB)
    AX = (0, 0, None, None)
    f = tm.vmap(item, outputs={"Q": tm.Sum((NB,), F)}, in_axes=AX, backend=b, max_steps=CAP)
    r = reference(padded, outputs={"Q": tm.Sum((NB,), F)}, in_axes=AX)
    nt = int(f(xs, keys, tab, th).stats["n_truncated"])
    if nt == 0:
        raise Fail("instrument: no item was truncated (the cap is not exercised)")
    # truncation poisons the float tally, so every gradient that reads it is NaN (the resource-limit rule)
    gq = jax.jit(jax.grad(lambda t_: jnp.sum(f(xs, keys, tab, t_).outputs["Q"] * wq)))(th)
    if not np.all(np.isnan(np.asarray(gq))):
        raise Fail(f"the gradient of a poisoned tally is not NaN: {np.asarray(gq)}")
    # the per-item results are not poisoned: the replay adjoint through the capped loop stays exact

    def per_item_loss(fn):
        def L(x, ks, tb, t_):
            w, y = fn(x, ks, tb, t_).per_item
            return jnp.sum(w * w) + jnp.sum(jnp.sin(y))
        return L
    msg = close(grads(per_item_loss(f), (xs, keys, tab, th)), grads(per_item_loss(r), (xs, keys, tab, th)), "capped")
    nact = 30

    def La(x, tb, t_, na):
        w, y = f(x, keys, tb, t_, n_active=na).per_item
        return jnp.sum(w[:nact] ** 2) + jnp.sum(jnp.sin(y[:nact]))

    def Lr(x, tb, t_):
        w, y = r(x[:nact], keys[:nact], tb, t_).per_item
        return jnp.sum(w ** 2) + jnp.sum(jnp.sin(y))
    ga = jax.jit(jax.grad(La, argnums=(0, 1, 2)))(xs, tab, th, jnp.int32(nact))
    gr = jax.jit(jax.grad(Lr, argnums=(0, 1, 2)))(xs, tab, th)
    gx = np.asarray(ga[0])
    if np.any(gx[nact:] != 0):
        raise Fail("inactive items got nonzero cotangents")
    msg2 = close(dict(x=gx[:nact], tab=ga[1], th=ga[2]), dict(x=gr[0][:nact], tab=gr[1], th=gr[2]), "n_active")
    return f"{nt} truncated; {msg}; n_active: {msg2}"


def _replay_item(NB):
    def item(t, x):                       # trajectory decided by a sqrt/div-driven continuous state (ulp-sensitive)
        def body(c):
            k, y, w, t = c
            y2 = y + 0.013 * jnp.sqrt(y + 0.1) / (x + 0.37) + 0.001 * jnp.sin(37.0 * y)
            return k + 1, y2, w * 0.999, t.add("Q", (k * 7) % NB, w * y2)
        k, y, w, t = lax.while_loop(lambda c: (c[1] < 3.0) & (c[0] < 5000), body, (0, x * 0.5, x, t))
        return t, (k, y)
    return item


@case("REPLAY GATE: per-item reverse == forward-mode derivative on an ulp-sensitive trajectory (nested and T3); "
      "NEGATIVE CONTROL adjoint math=approx must fail it", ["cuda"])
def _(b):
    """the adjoint differentiates the trajectory the forward kernel ran (what the removed runtime replay check
    compared).  Here the step count of every item hangs on f32 sqrt / div bits, so an adjoint that replays a
    different trajectory gets a different per-item derivative (one step more or fewer is ~1/150 of it), which the
    forward-mode derivative (the fused jvp kernel, the forward's own trajectory) exposes item by item.  The negative
    control compiles only the adjoint with approx div / sqrt: the gate must flag about the items whose step count
    approx changes."""
    if X64:     # the control flips f32 div / sqrt PTX; an f64 item does not use them (the control would be void)
        return "skipped under x64 (the negative control needs f32 arithmetic)"
    NB, n = 64, 4096
    item = _replay_item(NB)
    xs = jnp.linspace(0.11, 1.9, n).astype(F)
    ki = tm.vmap(item, outputs={"Q": tm.Sum((NB,), F)}, backend="cuda", cuda_math="ieee")(xs).per_item[0]
    ka = tm.vmap(item, outputs={"Q": tm.Sum((NB,), F)}, backend="cuda", cuda_math="approx")(xs).per_item[0]
    ndiff = int(np.sum(np.asarray(ki) != np.asarray(ka)))
    if ndiff == 0:
        raise Fail("instrument: approx vs ieee changes no step count here; the negative control would be void")
    s0, out = _reverse.SINGLE, {}
    try:
        for single in ("off", "force"):
            for adj in (None, "approx"):
                f = tm.vmap(item, outputs={"Q": tm.Sum((NB,), F)}, backend="cuda", cuda_math="ieee",
                            **(dict(_adj_math=adj) if adj else {}))
                _reverse.SINGLE = single
                gr = np.asarray(jax.jit(jax.grad(lambda x: jnp.sum(f(x).per_item[1])))(xs))
                if _reverse.LAST_SINGLE.get("used") != (single == "force"):
                    raise Fail(f"instrument: SINGLE={single} but {_reverse.LAST_SINGLE}")
                gf = jax.jit(lambda x: jax.jvp(lambda x_: f(x_).per_item[1], (x,), (jnp.ones_like(x),))[1])(xs)
                gf = np.asarray(gf)
                out[single, adj] = int(np.sum(~(np.abs(gr - gf) <= 2e-4 * np.abs(gf) + 1e-6)))
    finally:
        _reverse.SINGLE = s0
    for single in ("off", "force"):
        if out[single, None]:
            raise Fail(f"{single}: {out[single, None]}/{n} items' reverse derivative differs from forward mode")
        if out[single, "approx"] < ndiff // 2:
            raise Fail(f"NEGATIVE CONTROL ({single}): {out[single, 'approx']} items flagged, {ndiff} change step count")
    return (f"clean 0/{n} (nested and T3); approx changes {ndiff} step counts; control flags nested "
            f"{out['off', 'approx']}, T3 {out['force', 'approx']}")


@case("grad(vmap) of a call == per-member grads (the batching rule's folded / mapped adjoint)")
def _(b):
    NB, n, B = 16, 1024, 2
    item = _replay_item(NB)
    xs = jnp.stack([jnp.linspace(0.11, 1.9, n), jnp.linspace(0.2, 1.7, n)]).astype(F)
    wq = jnp.cos(jnp.arange(B * NB)).astype(F).reshape(B, NB)
    f = tm.vmap(item, outputs={"Q": tm.Sum((NB,), F)}, backend=b, **(dict(cuda_math="ieee") if b == "cuda" else {}))
    g = np.asarray(jax.jit(jax.grad(lambda x2: jnp.sum(jax.vmap(lambda x: f(x).outputs["Q"])(x2) * wq)))(xs))
    ref = np.stack([np.asarray(jax.jit(jax.grad(lambda x, m=m: jnp.sum(f(x).outputs["Q"] * wq[m])))(xs[m]))
                    for m in range(B)])
    if not np.isfinite(g).all():
        raise Fail("grad(vmap) gradient not finite")
    return close(dict(g=g), dict(g=ref), "grad(vmap) vs per-member grad")


@case("AUTO C: 16 for a small state, 64 for a ~300 B one, less beyond; bitwise == explicit C")
def _(b):
    """T5: rev_checkpoints unset picks C from the stored state's bytes (_reverse.auto_checkpoints); the recorded loop
    says so; per-item cotangents are bitwise those of the same explicit C (C changes work, never values)."""
    R = _reverse
    want = {}
    for sb in (12, 300, 400, 4096):
        c = R.AUTO_C_SMALL if sb < R.AUTO_BIG_STATE else R.AUTO_C_MAX
        while c > R.AUTO_C_MIN and (3 * c + 1) * sb > R.AUTO_CKPT_BYTES:
            c //= 2
        want[sb] = c
    got = {sb: R.auto_checkpoints(sb) for sb in want}
    if got != want or got[12] != R.AUTO_C_SMALL or got[300] != R.AUTO_C_MAX or not got[300] >= got[400] >= got[4096]:
        raise Fail(f"auto_checkpoints rule: {got} (want {want}: C_SMALL for a small state, C_MAX for a 300 B one)")
    g, gr, info, _ = walk_case(b, 64, 40)
    lp = info["loops"][0]
    if not lp.get("auto") or lp["checkpoints"] != R.auto_checkpoints(lp["state_bytes"]):
        raise Fail(f"instrument: loop record {lp}")
    gc, _, infoc, _ = walk_case(b, 64, 40, rev_checkpoints=lp["checkpoints"])
    if infoc["loops"][0].get("auto"):
        raise Fail("instrument: explicit rev_checkpoints recorded as auto")
    nx = int((np.asarray(g["x"]) != np.asarray(gc["x"])).sum())
    if nx:
        raise Fail(f"auto vs explicit C={lp['checkpoints']}: {nx} per-item cotangents differ")
    return (f"C={lp['checkpoints']} (state {lp['state_bytes']} B, ckpt {lp['ckpt_bytes']} B); "
            + close(g, gr, "auto C vs padded"))


@case("walk K in [3, 9]: head path (K <= C) and head + replay (K > C) mixed in one call, == grad(padded)")
def _(b):
    """lanes of one call on both sides of K = C (and K = C exactly): the head sweep and the refilled tapes"""
    g, gr, _, _ = walk_case(b, 64, 6, rev_checkpoints=8)          # K in [3, 9]: spans C = 8
    g2, _, _, _ = walk_case(b, 64, 6, rev_checkpoints=2)          # K > C^2 for most items
    nx = int((np.asarray(g["x"]) != np.asarray(g2["x"])).sum())
    if b == "cuda" and nx:
        raise Fail(f"cuda: C=8 vs C=2 moved {nx} per-item cotangents")
    return f"C=8 vs C=2 per-item differ {nx}; " + close(g, gr, "C=8") + "; " + close(g2, gr, "C=2")


@case("live state: a carry the step never reads is not stored; grad == padded")
def _(b):
    """_rewrite._live_carries: a per-step diagnostic the body overwrites without reading, and a pass-through
    constant, are not in the checkpoint state (restored as their initial values, which nothing reads)."""
    NB = 16

    def step(c, key, tab, th):
        k, w, y, last, cst = c
        i = (kb(key, k) % jnp.uint32(NB)).astype(jnp.int32)
        v = w * tab[i] * th[0] * cst
        return (k + 1, w * (0.99 + 0.01 * jnp.cos(y * th[2])), y + 0.05 * jnp.sin(3 * y + th[1]), v, cst), i, v

    def item(t, x, key, tab, th):
        L = (5 + kb(key, 7) % jnp.uint32(20)).astype(jnp.int32)

        def body(c):
            s, t = c
            s2, i, v = step(s, key, tab, th)
            return s2, t.add("Q", i, v)
        (k, w, y, last, cst), t = lax.while_loop(lambda c: c[0][0] < L, body, ((0, x, 0.5 * x, x * 0, x * 2), t))
        return t, (w, y + last)

    def padded(t, x, key, tab, th):
        L = (5 + kb(key, 7) % jnp.uint32(20)).astype(jnp.int32)

        def body(c, _):
            s, t = c
            act = s[0] < L
            s2, i, v = step(s, key, tab, th)
            return (mask_tree(act, s2, s), t.add("Q", jnp.where(act, i, -1), v)), None
        ((k, w, y, last, cst), t), _ = lax.scan(body, ((0, x, 0.5 * x, x * 0, x * 2), t), None, length=25)
        return t, (w, y + last)
    xs, keys, tab, th, wq = inputs(NB)
    AX = (0, 0, None, None)
    f = tm.vmap(item, outputs={"Q": tm.Sum((NB,), F)}, in_axes=AX, backend=b)
    r = reference(padded, outputs={"Q": tm.Sum((NB,), F)}, in_axes=AX)
    g = grads(loss_of(f, wq), (xs, keys, tab, th))
    lp = _reverse.LAST["loops"][0]
    gr = grads(loss_of(r, wq), (xs, keys, tab, th))
    if not lp["n_live"] < lp["n_state"]:
        raise Fail(f"instrument: every state carry stored ({lp})")
    return f"stored {lp['n_live']}/{lp['n_state']} state carries; " + close(g, gr, "live")


@case("FILL PATTERN: every uninitialised per-thread declaration 0x00 vs 0xFF -> bitwise-equal forward and per-item "
      "cotangents", ["cuda"])
def _(b):
    # The reverse program's checkpoint buffers are tallymap_uninit (declared with no fill) and copies of them are
    # elided (codegen.Emitter.assign).  Correct only if no slot is read before it is written: filling EVERY
    # uninitialised per-thread declaration with two different bytes must not move a single bit of what the program
    # computes (per-item results and per-item cotangents; the tally sums and broadcast-operand cotangents are
    # atomics, whose float order is the hardware's, so those are compared within tolerance).
    from tallymap import _diagnostics as D
    from tallymap.backends.cuda import codegen as CG
    out = {}
    try:
        for fb in (0x00, 0xFF):
            CG.FILL = fb
            D.KERNELS.clear()
            g, gr, info, (f, xs, keys, tab, th, wq, AX, item) = walk_case(b, 64, 100, rev_checkpoints=3)
            ks = tm.diagnostics()["kernels"]
            r = f(xs, keys, tab, th)
            srcs = [open(k["source"]).read() for k in ks]
            if not any("tallymap_uninit" in k["prims"] for k in ks):
                raise Fail(f"instrument: no kernel uses tallymap_uninit ({[k['prims'] for k in ks]})")
            if not all(f"sj_f[sj_e] = {fb};" in sc for sc in srcs):
                raise Fail(f"instrument: fill {fb:#x} is not in every kernel source")
            out[fb] = (g, r, gr)
    finally:
        CG.FILL = None
    (g0, r0, gr), (g1, r1, _) = out[0x00], out[0xFF]
    for nm, a, c in (("per-item w", r0.per_item[0], r1.per_item[0]), ("per-item y", r0.per_item[1], r1.per_item[1]),
                     ("cotangent x", g0["x"], g1["x"])):
        a, c = np.asarray(a), np.asarray(c)
        if not np.array_equal(a, c):
            raise Fail(f"{nm}: {int((a != c).sum())}/{a.size} differ between fills 0x00 and 0xFF")
    close(dict(Q=r1.outputs["Q"]), dict(Q=r0.outputs["Q"]), "Q 0xFF vs 0x00")
    return "per-item + cotangent x bitwise; " + close(g1, g0, "0xFF vs 0x00") + "; " + close(g0, gr, "vs padded")


# ------------------------------------------------------------------------------------------- T3: single-loop adjoint
def _with_single(mode, fn):
    s0 = _reverse.SINGLE
    _reverse.SINGLE = mode
    try:
        return fn()
    finally:
        _reverse.SINGLE = s0


@case("T3 forced: per-item cotangents bitwise == the nested-loop adjoint (walk K<=C, K>C^2; nested); T3 used")
def _(b):
    out = []
    for nm, mk, NB, ro in (("walk40", lambda: make_walk(64, 40), 64, {}), ("walk200C3", lambda: make_walk(64, 200), 64,
                           dict(rev_checkpoints=3)), ("nested", lambda: make_nested(8), 8, {})):
        item, padded = mk()
        xs, keys, tab, th, wq = inputs(NB)
        f = tm.vmap(item, outputs={"Q": tm.Sum((NB,), F)}, in_axes=(0, 0, None, None), backend=b, **ro)
        gs = []
        for mode in ("off", "force"):
            gs.append(_with_single(mode, lambda: grads(loss_of(f, wq), (xs, keys, tab, th))))
            if _reverse.LAST_SINGLE.get("used") != (mode == "force"):
                raise Fail(f"{nm}: SINGLE={mode} but {_reverse.LAST_SINGLE}")
        nx = int((np.asarray(gs[0]["x"]) != np.asarray(gs[1]["x"])).sum())
        if nx:
            raise Fail(f"{nm}: {nx} per-item cotangents differ between T3 and the nested-loop adjoint")
        out.append(f"{nm}: " + close(gs[1], gs[0], nm))
    return "; ".join(out)


@case("T3 auto: a small step keeps the nested-loop adjoint (T3_MIN_EQNS)")
def _(b):
    item, _ = make_walk(64, 10)
    xs, keys, tab, th, wq = inputs(64)
    f = tm.vmap(item, outputs={"Q": tm.Sum((64,), F)}, in_axes=(0, 0, None, None), backend=b)
    _with_single("auto", lambda: grads(loss_of(f, wq), (xs, keys, tab, th)))
    w = dict(_reverse.LAST_SINGLE)
    if w.get("used") or "T3_MIN_EQNS" not in str(w.get("why")):
        raise Fail(f"auto on a ~30-equation step: {w}")
    return f"small step: {w['why']}"


@case("max_steps under jax.grad: ONE launch (no forward); tally gradient NaN when truncated, per-item gradient exact "
      "(nested and T3)")
def _(b):
    def item(t, x, th):
        k, w, t = lax.while_loop(lambda c: c[0] < (x * 10).astype(jnp.int32),
                                 lambda c: (c[0] + 1, c[1] * th, c[2].add("Q", 0, c[1] * c[1])), (0, x, t))
        return t, w
    xs = jnp.linspace(0.5, 1.5, 37).astype(F)
    th = F(0.97)
    res, msg = {}, []
    for mode in ("off", "force"):
        for cap in (64, 10):
            f = tm.vmap(item, outputs={"Q": tm.Sum((4,), F)}, in_axes=(0, None), backend=b, max_steps=cap)
            gq_f = jax.jit(jax.grad(lambda t_: jnp.sum(f(xs, t_).outputs["Q"])))
            if b == "cuda":     # what runs: the custom calls left in the optimised program
                hlo = _with_single(mode, lambda: gq_f.lower(th).compile().as_text())
                nl = hlo.count('custom_call_target="tallymap_cuda_launch"')
                if nl != 1:
                    raise Fail(f"{mode}: jax.grad of a linear loss runs {nl} kernels, want 1 (the adjoint)")
            gq = float(_with_single(mode, lambda: gq_f(th)))
            if _reverse.LAST_SINGLE.get("used") != (mode == "force"):
                raise Fail(f"SINGLE={mode} but {_reverse.LAST_SINGLE}")
            gy = _with_single(mode, lambda: jax.jit(jax.grad(lambda t_: jnp.sum(f(xs, t_).per_item ** 2)))(th))
            r = tm.reference(item, outputs={"Q": tm.Sum((4,), F)}, in_axes=(0, None), max_steps=cap)
            close(dict(g=gy), dict(g=jax.jacfwd(lambda t_: jnp.sum(r(xs, t_).per_item ** 2))(th)), "per-item grad")
            res[mode, cap] = gq
        if not np.isfinite(res[mode, 64]) or not np.isnan(res[mode, 10]):
            raise Fail(f"{mode}: tally gradient: no truncation {res[mode, 64]} (want finite), truncated "
                       f"{res[mode, 10]} (want NaN)")
        msg.append(f"{mode}: cap 64 finite {res[mode, 64]:.4g}, cap 10 NaN")
    # forward mode: the tangent kernel poisons its own tangent sums (jvp and jacfwd, fused and lanes)
    f = tm.vmap(item, outputs={"Q": tm.Sum((4,), F)}, in_axes=(0, None), backend=b, max_steps=10)
    tq = np.asarray(jax.jvp(lambda t_: f(xs, t_).outputs["Q"], (th,), (F(1.0),))[1])
    jq = np.asarray(jax.jacfwd(lambda t2: f(xs, t2[0] * t2[1]).outputs["Q"])(jnp.array([th, 1.0], F)))
    if not (np.isnan(tq).all() and np.isnan(jq).all()):
        raise Fail(f"truncated call: tangent sums not NaN (jvp {tq}, jacfwd {jq})")
    return "; ".join(msg) + "; jvp / jacfwd tangents NaN when truncated"


print("\n================ SUMMARY", jax.__version__, "x64" if X64 else "", jax.devices()[0].device_kind)
for n, b, m in RES:
    print(f"{n:58s} {b:8s} {m[:200]}")
print("TOTAL", len(RES), "PASS", sum(m.startswith("PASS") for _, _, m in RES))
