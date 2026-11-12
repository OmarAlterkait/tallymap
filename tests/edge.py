"""Phase-2 edge-case matrix.  python tests/edge.py  (env: BACKENDS=xla,cuda  TEST_X64=1  ONLY=substr)
Prints one line per (case, backend): PASS / FAIL <why> / RAISE <exception> and a summary table."""
import os
import re
import sys
import time
import traceback
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.environ.get("TEST_X64"):
    import jax
    jax.config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import numpy as np
from ab_rule import assert_arms_differ, patched
from jax import lax

import tallymap as tm
from tallymap import reference

if os.environ.get("TX_UNFUSED") == "1":                 # run_all --unfused: the two-node jvp design
    from tallymap import _primitive as _P0
    _P0.FUSE_JVP = False
if os.environ.get("TX_T3") == "1":                      # run_all --t3: the single-loop adjoint wherever it applies
    from tallymap import _reverse as _R0
    _R0.SINGLE = "force"

jax.config.update("jax_default_matmul_precision", "highest")
GPU = jax.default_backend() == "gpu"
BACKENDS = os.environ.get("BACKENDS", "xla,auto").split(",")
ONLY = os.environ.get("ONLY", "")
F = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
print("jax", jax.__version__, jax.__file__, jax.devices(), "x64", jax.config.jax_enable_x64, flush=True)
RES = []
NB = 8


class Fail(Exception):
    pass


def close(a, b, rtol=2e-4, atol=1e-6, what=""):
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        if x.shape != y.shape:
            raise Fail(f"{what} shape {x.shape} vs {y.shape}")
        err = np.max(np.abs(x - y)) if x.size else 0.0
        tol = rtol * (np.max(np.abs(y)) if y.size else 0.0) + atol
        if not err <= tol:
            raise Fail(f"{what} max err {err:.3g} > tol {tol:.3g}")


def raises(exc, f, contains=""):
    try:
        f()
    except exc as e:
        if contains and contains not in str(e):
            raise Fail(f"raised {type(e).__name__} but message lacks {contains!r}: {str(e)[:150]}")
        return str(e)
    raise Fail(f"did not raise {exc.__name__}")


def case(name, backends=None):
    def deco(fn):
        for b in (backends or BACKENDS):
            if b not in BACKENDS or (ONLY and ONLY not in name):
                continue
            t0 = time.perf_counter()
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    r = fn(b)
                msg = "PASS" + (f" ({r})" if isinstance(r, str) else "")
                ws = [str(x.message) for x in w if "tallymap" in str(x.message)]
                if ws:
                    msg += f" [warn: {ws[0][:90]}]"
            except Fail as e:
                msg = f"FAIL {e}"
            except Exception as e:
                msg = f"RAISE {type(e).__name__}: {str(e)[:200]}"
                traceback.print_exc()
            RES.append((name, b, msg))
            print(f"{name:52s} {b:8s} {msg}  [{time.perf_counter() - t0:.1f}s]", flush=True)
        return fn
    return deco


def V(fn, b, **kw):
    kw.setdefault("outputs", {"Q": tm.Sum((NB,), F)})
    return tm.vmap(fn, backend=b, **kw)


def R(fn, **kw):
    kw.setdefault("outputs", {"Q": tm.Sum((NB,), F)})
    kw.pop("ad", None)
    return reference(fn, **kw)


def bits(key, k):
    return jax.random.bits(jax.random.fold_in(key, k))


# integer-exact control flow (RNG bits), smooth float weights: backends differ by ulps only
def walk(t, x, key, tab, th):
    def body(c):
        k, w, t = c
        b = bits(key, k)
        i = (b % NB).astype(jnp.int32)
        t = t.add("Q", i, w * tab[i] * th)
        return k + 1, w * (0.7 + 0.2 * th), t
    k, w, t = lax.while_loop(lambda c: (bits(key, c[0] + 1000) % 5 != 0) | (c[0] == 0), body, (0, x, t))
    return t, (w, k)


N = 37
xs = jnp.linspace(0.5, 1.5, N).astype(F)
keys = jax.random.split(jax.random.PRNGKey(0), N)
tab = jnp.arange(1.0, NB + 1).astype(F)
TH = F(0.9)
AX = (0, 0, None, None)


@case("forward vs dense reference")
def _(b):
    close(V(walk, b, in_axes=AX)(xs, keys, tab, TH), R(walk, in_axes=AX)(xs, keys, tab, TH))


@case("emit inside nested while / cond / switch")
def _(b):
    def item(t, x, key):
        def outer(c):
            k, t = c

            def inner(d):
                j, t = d
                t = t.add("Q", (j + k) % NB, x * (j + 1))
                return j + 1, t
            _, t = lax.while_loop(lambda d: d[0] < (bits(key, k) % 3).astype(jnp.int32), inner, (0, t))
            t = lax.cond(bits(key, k + 50) % 2 == 0, lambda t: t.add("Q", 0, x), lambda t: t, t)
            t = lax.switch((bits(key, k + 99) % 3).astype(jnp.int32),
                           [lambda t: t.add("Q", 1, 2 * x), lambda t: t.add("Q", 2, x), lambda t: t], t)
            return k + 1, t
        _, t = lax.while_loop(lambda c: c[0] < 4, outer, (0, t))
        return t
    close(V(item, b)(xs, keys), R(item)(xs, keys))


@case("OOB and negative emit indices are dropped")
def _(b):
    def item(t, x):
        t = t.add("Q", -1, x)
        t = t.add("Q", NB, x)
        t = t.add("Q", 10 ** 6, x)
        t = t.add("Q", jnp.array([-3, 0, 7, 8]), jnp.array([1.0, 2.0, 3.0, 4.0], F) * x)
        return t
    q = V(item, b)(xs).outputs["Q"]
    want = np.zeros(NB)
    want[0], want[7] = 2 * float(xs.sum()), 3 * float(xs.sum())
    close(q, want)


@case("2-D output, tuple index, per-dim OOB dropped")
def _(b):
    def item(t, x):
        t = t.add("M", (1, 2), x)
        t = t.add("M", (3, 0), x)      # row 3 out of range (shape 3x4): dropped, NOT wrapped to (2, 4+0)
        t = t.add("M", (0, -1), x)     # negative: dropped
        return t
    m = tm.vmap(item, outputs={"M": tm.Sum((3, 4), F)}, backend=b)(xs).outputs["M"]
    want = np.zeros((3, 4))
    want[1, 2] = float(xs.sum())
    close(m, want)


@case("spawn without Queue / spawn outside", ["xla"])
def _(b):
    raises(ValueError, lambda: V(lambda t, x: t.spawn(x), b)(xs), "Queue")
    def item(t, x):
        return t.spawn(x, when=x > 1.0)
    raises(ValueError, lambda: jax.jit(lambda x: R(item)(x))(xs), "Queue")


@case("API: unknown or renamed options / backends and removed aliases raise", ["xla"])
def _(b):
    f = lambda t, x: t.add("Q", 0, x)
    for kw in (dict(math="ieee"), dict(min_blocks=4), dict(block=8), dict(rev_slots=4), dict(rev_remat=2),
               dict(ad="forward"), dict(bogus=1), dict(_occ_cap=2), dict(_replay_check=False),
               dict(cuda_min_blocks=2), dict(_rev_phase="skip"), dict(_tree_phase="fwd"),
               dict(rev_step_segments=8), dict(lockstep_block=8), dict(_schedule="step"), dict(_cuda_inplace=False),
               dict(_lanes="off"), dict(_fold=False), dict(_rev_sparse_min=1), dict(_rev_nodense=True),
               dict(_rev_gen_capacity=8), dict(_tree_neg="detach")):
        raises(ValueError, lambda: V(f, b, **kw), "unknown option")
    for bk in ("lockstep", "jax"):                                     # 'lockstep' is the old name of 'xla'
        raises(ValueError, lambda: tm.vmap(f, outputs={"Q": tm.Sum((NB,), F)}, backend=bk), "unknown backend")
    raises(TypeError, lambda: tm.Sum((NB,), F, exact=True), "exact")    # renamed fixed_point=
    r = tm.vmap(lambda t, x: t.spawn(x, when=x > 9.0).add("Q", 0, x), outputs={"Q": tm.Sum((NB,), F)},
                spawn=tm.Queue(4), backend=b)(xs)
    if set(r.stats) != {"fixed_point_overflow", "dropped_emits", "spawned", "children_dropped", "children_truncated",
                        "n_truncated"}:
        raise Fail(f"stats keys {sorted(r.stats)}")
    from tallymap import _tally
    for mod, name in ((tm, "emit"), (tm, "spawn"), (tm, "check"), (tm, "reverse_stats"), (_tally, "emit"),
                      (_tally, "spawn")):
        if hasattr(mod, name):
            raise Fail(f"tm.{name} still exists")


@case("forked / dropped / read tally rejected")
def _(b):
    def fork(t, x):
        t1 = t.add("Q", 0, x)  # noqa: F841  (the fork under test)
        t2 = t.add("Q", 1, x)
        return t2
    raises(tm.TallyError, lambda: V(fork, b)(xs), "used twice")

    def drop(t, x):
        def body(c):
            k, t = c
            t.add("Q", 0, x)           # result discarded
            return k + 1, t
        return lax.while_loop(lambda c: c[0] < 2, body, (0, t))[1]
    m = raises(tm.TallyError, lambda: V(drop, b)(xs))      # message depends on JAX's loop-invariant hoisting

    def read(t, x):
        return t, t._a["Q"].sum()
    raises(tm.TallyError, lambda: V(read, b)(xs), "reads a Tally")
    return m[:60]


@case("nested tm.vmap: inner item writing the OUTER tally rejected")
def _(b):
    def outer(t, x):
        inner = V(lambda ti, y: t.add("Q", 0, y), b)
        inner(jnp.stack([x, x]))
        return t
    raises(Exception, lambda: V(outer, b)(xs))


@case("nested tm.vmap (self-contained inner)")
def _(b):
    def outer(t, x):
        r = tm.vmap(lambda ti, y: ti.add("Q", 0, y * x), outputs={"Q": tm.Sum((NB,), F)}, backend="xla")(
            jnp.stack([x, 2 * x]))
        return t.add("Q", 3, r.outputs["Q"][0])
    if b == "auto":
        raises(tm.UnsupportedError, lambda: V(outer, b)(xs), "backend='xla'")
        b = "xla"
    q = V(outer, b)(xs).outputs["Q"]
    close(q[3], 3 * float((xs ** 2).sum()))


@case("multiple top-level loops + fori(scan) + no loop + loop in cond")
def _(b):
    def item(t, x, key):
        t = t.add("Q", 0, x)                                                     # no loop
        t = lax.fori_loop(0, 3, lambda i, t: t.add("Q", i, x), t)                # static fori -> scan
        k, t = lax.while_loop(lambda c: c[0] < 2, lambda c: (c[0] + 1, c[1].add("Q", 4, x)), (0, t))
        k, t = lax.while_loop(lambda c: c[0] < 3, lambda c: (c[0] + 1, c[1].add("Q", 5, x)), (0, t))
        n = (bits(key, 0) % 4).astype(jnp.int32)
        t = lax.fori_loop(0, n, lambda i, t: t.add("Q", 6, x), t)               # dynamic fori -> while
        t = lax.cond(x > 1.0, lambda t: lax.while_loop(lambda c: c[0] < 2, lambda c: (c[0] + 1, c[1].add("Q", 7, x)),
                                                       (0, t))[1], lambda t: t, t)
        return t
    close(V(item, b, max_steps=10)(xs, keys), R(item, max_steps=10)(xs, keys))


@case("pytree in/out, int & bool per-item results, int tally")
def _(b):
    def item(t, p, key):
        x, y = p["x"], p["y"]
        k, t = lax.while_loop(lambda c: c[0] < 3, lambda c: (c[0] + 1, c[1].add("C", c[0], 1).add("Q", c[0], x)),
                              (0, t))
        return t, {"k": k, "big": x > 1.0, "z": (x * y, -y)}
    out = {"Q": tm.Sum((NB,), F), "C": tm.Sum((NB,), jnp.int32)}
    p = {"x": xs, "y": xs[::-1]}
    A = tm.vmap(item, outputs=out, backend=b)(p, keys)
    Rr = reference(item, outputs=out)(p, keys)
    close(A.outputs, Rr.outputs)
    same = lambda k: np.array_equal(A.per_item[k], Rr.per_item[k])
    if not (same("k") and same("big")):
        raise Fail("int/bool per-item results differ")
    if A.per_item["k"].dtype != Rr.per_item["k"].dtype or A.per_item["big"].dtype != jnp.bool_:
        raise Fail(f"dtypes {A.per_item['k'].dtype} {A.per_item['big'].dtype}")
    close(A.per_item["z"], Rr.per_item["z"])
    close(A.outputs["C"], np.array([N, N, N, 0, 0, 0, 0, 0]))


@case("N=0, N=1, N not a multiple of block")
def _(b):
    f = V(walk, b, in_axes=AX, xla_batch_size=8)
    for n in (0, 1, 13):
        A = f(xs[:n], keys[:n], tab, TH)
        if n == 0:
            close(A.outputs["Q"], np.zeros(NB))
            if A.per_item[0].shape != (0,):
                raise Fail("N=0 per-item shape")
        else:
            close(A, R(walk, in_axes=AX)(xs[:n], keys[:n], tab, TH), what=f"N={n}")


@case("N >= 2^31 rejected with a clear error", ["xla"])
def _(b):
    f = V(lambda t, x: t.add("Q", 0, x), b)
    raises(ValueError, lambda: jax.eval_shape(f, jax.ShapeDtypeStruct((2 ** 31,), F)), "2^31")


@case("large N (2^22 items, ids folded into one key)", ["auto"])
def _(b):
    n = 1 << 22
    ids = jnp.arange(n, dtype=jnp.uint32)

    def item(t, i, key):
        k = jax.random.fold_in(key, i)
        return t.add("Q", (jax.random.bits(k) % NB).astype(jnp.int32), 1.0)
    f = tm.vmap(item, outputs={"Q": tm.Sum((NB,), jnp.int32)}, in_axes=(0, None), backend=b)
    q = f(ids, jax.random.PRNGKey(3)).outputs["Q"]
    if int(q.sum()) != n:
        raise Fail(f"count {int(q.sum())} != {n}")
    return f"bins {np.asarray(q).min()}..{np.asarray(q).max()}"


@case("max_steps: truncated items reported; float tallies poisoned, int tallies and per-item results kept")
def _(b):
    def item(t, x):
        k, t = lax.while_loop(lambda c: c[0] < (x * 10).astype(jnp.int32),
                              lambda c: (c[0] + 1, c[1].add("Q", 0, 1.0).add("C", 1, 1)), (0, t))
        return t, k
    out = {"Q": tm.Sum((NB,), F), "C": tm.Sum((NB,), jnp.int32)}
    A = tm.vmap(item, outputs=out, backend=b, max_steps=10)(xs)
    Rr = reference(item, outputs=out, max_steps=10)(xs)
    want = np.asarray(xs * 10).astype(int) > 10
    if not want.any():
        raise Fail("instrument: no item was truncated (the cap is not exercised)")
    if not np.array_equal(np.asarray(A.truncated), want) or int(A.stats["n_truncated"]) != want.sum():
        raise Fail(f"truncated {np.asarray(A.truncated).sum()} vs {want.sum()}")
    if not np.all(np.isnan(np.asarray(A.outputs["Q"]))):
        raise Fail(f"a truncated item did not poison the float tally: {np.asarray(A.outputs['Q'])}")
    if not np.all(np.isnan(np.asarray(Rr.outputs["Q"]))):
        raise Fail("the dense capped reference does not apply the same rule")
    if not np.array_equal(np.asarray(A.outputs["C"]), np.asarray(Rr.outputs["C"])):
        raise Fail(f"int tally {np.asarray(A.outputs['C'])} vs {np.asarray(Rr.outputs['C'])}")
    close(A.per_item, Rr.per_item, what="per-item")
    B = tm.vmap(item, outputs=out, backend=b, max_steps=64)(xs)          # a cap no item hits: nothing is poisoned
    if int(B.stats["n_truncated"]) != 0:
        raise Fail("truncated under a cap no item hits")
    close(B.outputs, reference(item, outputs=out)(xs).outputs, what="uncapped")
    return f"{want.sum()} of {N} truncated: Q NaN, C and per-item intact"


@case("max_steps: a poisoned tally gives NaN tangents and NaN cotangents, per-item gradients stay exact")
def _(b):
    def item(t, x, th):
        k, w, t = lax.while_loop(lambda c: c[0] < (x * 10).astype(jnp.int32),
                                 lambda c: (c[0] + 1, c[1] * th, c[2].add("Q", 0, c[1])), (0, x, t))
        return t, w
    f = V(item, b, in_axes=(0, None), max_steps=10)
    r = R(item, in_axes=(0, None), max_steps=10)
    th = F(0.9)
    if int(f(xs, th).stats["n_truncated"]) == 0:
        raise Fail("instrument: no item was truncated")
    tan = jax.jvp(lambda t_: f(xs, t_).outputs["Q"], (th,), (F(1),))[1]
    if not np.all(np.isnan(np.asarray(tan))):
        raise Fail(f"the jvp tangent of a poisoned tally is not NaN: {np.asarray(tan)}")
    g = jax.jit(jax.grad(lambda t_: jnp.sum(f(xs, t_).outputs["Q"])))(th)
    if not np.isnan(float(g)):
        raise Fail(f"the cotangent of a poisoned tally is not NaN: {float(g)}")
    L = lambda fn: lambda t_: jnp.sum(fn(xs, t_).per_item ** 2)
    close(jax.jit(jax.grad(L(f)))(th), jax.jacfwd(L(r))(th), what="per-item grad")
    return "tangent NaN, cotangent NaN, per-item grad exact"


@case("max_steps + n_active: inactive items never count as truncated")
def _(b):
    def item(t, x):
        k, t = lax.while_loop(lambda c: c[0] < 20, lambda c: (c[0] + 1, c[1].add("Q", 0, x)), (0, t))
        return t, k
    f = V(item, b, max_steps=5)
    A = f(xs, n_active=jnp.int32(0))
    if int(A.stats["n_truncated"]) != 0 or bool(np.asarray(A.truncated).any()):
        raise Fail("an inactive item counted as truncated")
    close(A.outputs["Q"], np.zeros(NB), what="no active item")
    B = f(xs, n_active=jnp.int32(3))
    if int(B.stats["n_truncated"]) != 3 or int(np.asarray(B.truncated)[3:].sum()) != 0:
        raise Fail(f"n_truncated {int(B.stats['n_truncated'])} (want 3), flags {np.asarray(B.truncated).astype(int)}")
    if not np.all(np.isnan(np.asarray(B.outputs["Q"]))):
        raise Fail("3 truncated active items did not poison the float tally")
    return "n_active=0: 0 truncated, finite zero tally; n_active=3: 3 truncated, poisoned"


@case("stats['dropped_emits']: off-by-one binning counted; in-range emits count 0; values unchanged")
def _(b):
    def mk(off):
        def item(t, x, i):
            return t.add("Q", i + off, x).add("Q", i - off, x)      # off=1: one past the last bin, and negative
        return item
    idx = jnp.arange(N, dtype=jnp.int32) % NB
    out = {"Q": tm.Sum((NB,), F)}
    A = tm.vmap(mk(0), outputs=out, backend=b)(xs, idx)
    if int(A.stats["dropped_emits"]) != 0:
        raise Fail(f"in-range emits counted {int(A.stats['dropped_emits'])} drops")
    want = int(np.sum(np.asarray(idx) + 1 >= NB) + np.sum(np.asarray(idx) - 1 < 0))
    f1 = tm.vmap(mk(1), outputs=out, backend=b)
    r1 = reference(mk(1), outputs=out)
    B = f1(xs, idx)
    if int(B.stats["dropped_emits"]) != want:
        raise Fail(f"dropped_emits {int(B.stats['dropped_emits'])} != {want}")
    # counting changes no value: the tally, its jvp tangent, its gradient and jax.vmap of it are the reference's
    close(B.outputs, r1(xs, idx).outputs, what="tally")
    L = lambda fn: lambda s: jnp.sum(fn(xs * s, idx).outputs["Q"] * jnp.arange(1.0, NB + 1))
    close(jax.jvp(L(f1), (F(1),), (F(1),))[1], jax.jvp(L(r1), (F(1),), (F(1),))[1], what="jvp")
    close(jax.jit(jax.grad(L(f1)))(F(1)), jax.jacfwd(L(r1))(F(1)), what="grad")
    ss = jnp.array([0.5, 1.0, 2.0], F)
    close(jax.vmap(lambda s: f1(xs * s, idx).outputs["Q"])(ss),
          jax.vmap(lambda s: r1(xs * s, idx).outputs["Q"])(ss), what="vmap")
    msg = f"{want} of {2 * N} emits dropped"
    if b == "auto" or jax.config.jax_enable_x64:                   # fixed_point=True needs x64 on the xla backend
        E = tm.vmap(mk(1), outputs={"Q": tm.Sum((NB,), F, fixed_point=True, scale=2.0 ** 20)}, backend=b)(xs, idx)
        if int(E.stats["dropped_emits"]) != want or int(E.stats["fixed_point_overflow"]) != 0:
            raise Fail(f"fixed-point tally: dropped_emits {int(E.stats['dropped_emits'])} (want {want}), overflow "
                       f"{int(E.stats['fixed_point_overflow'])}")
        msg += "; fixed-point tally too"
    return msg


@case("dual lanes: the lane guard decides T on the unbatched program; a refused T-lane program is never built",
      ["xla"])
def _(b):
    # Traced with backend='cuda' (the guard is the cuda one), never lowered, so it runs on every host.  Every
    # dual_lanes_prog call made by _dual_lanes is recorded with the lane state it really has, which must be T x the
    # unbatched one (the linearity the decision rests on); P lanes over a loop array of S elements per lane.
    from tallymap import _primitive as P_
    orig, built = P_.dual_lanes_prog, []

    def rec(p, T, bpi, bbc, shared=(0, 0)):
        q, lt, ly = orig(p, T, bpi, bbc, shared)
        if sys._getframe(1).f_code.co_name != "_dual_lanes":      # (a fused node rebuilding its chosen lanes)
            return q, lt, ly
        lt0 = [s_.kind != "spawn" and k >= shared[0] for k, s_ in enumerate(p.tallies)]
        m1 = P_._taint_max(p.cj.jaxpr, lt0 + list(bpi) + list(bbc), False)[0]
        built.append((T, P_._taint_max(q.cj.jaxpr, list(lt) + list(bpi) + list(bbc), False)[0], m1))
        return q, lt, ly

    def prog(S, P):
        def item(t, x, th):
            def body(c):
                k, a, t = c
                a = a * (1.0 + 0.01 * th[k % P]) + x
                return k + 1, a, t.add("Q", k % NB, jnp.sum(a))
            return lax.while_loop(lambda c: c[0] < 3, body, (0, jnp.full((S,), x, F), t))[2]
        f = tm.vmap(item, outputs={"Q": tm.Sum((NB,), F)}, in_axes=(0, None), backend="auto")
        return lambda th: f(xs, th).outputs["Q"]
    # (S, P) -> the lane width the guard must choose: under the limit (all P lanes), chunked, refused (no lanes)
    want = {(10, 16): 16, (100, 16): 4, (300, 4): None}
    out = []
    P_.dual_lanes_prog = rec
    try:
        for (S, P), w in want.items():
            built.clear()
            jx = str(jax.make_jaxpr(jax.jacfwd(prog(S, P)))(jnp.ones((P,), F)))
            got = sorted(set(int(v) for v in re.findall(r"dual_lanes(\d+)", jx)))
            if got != ([w] if w else []):
                raise Fail(f"S={S} P={P}: lanes {got} in the jaxpr, expected {w}")
            Ts = [T for T, _, _ in built]
            if Ts != ([w] if w else []):
                raise Fail(f"S={S} P={P}: T-lane programs built for T={Ts}, expected {[w] if w else []} "
                           "(a refused or re-chunked width must not be built)")
            for T, mT, m1 in built:
                if mT != T * m1:
                    raise Fail(f"S={S} P={P} T={T}: lane state {mT} != T x unbatched {m1}")
            out.append(f"S={S} P={P}: T={w}")
    finally:
        P_.dual_lanes_prog = orig
    return "; ".join(out)


@case("done at step 0 / NaN in the predicate")
def _(b):
    def item(t, x):
        k, t = lax.while_loop(lambda c: c[0] < x, lambda c: (c[0] + 1.0, c[1].add("Q", 0, 1.0)), (F(0), t))
        return t, k
    x = jnp.array([0.0, -1.0, np.nan, 2.5, np.inf * 0], F)
    A = V(item, b, max_steps=5)(x)
    Rr = R(item, max_steps=5)(x)
    close(A.outputs, Rr.outputs)
    close(A.per_item, Rr.per_item)
    if bool(A.truncated.any()):
        raise Fail("NaN/zero-trip items flagged truncated")


@case("jvp + jacfwd vs dense reference (table, closure, per-item)")
def _(b):
    f = V(walk, b, in_axes=AX)
    r = R(walk, in_axes=AX)
    close(jax.jvp(lambda th: f(xs, keys, tab, th).outputs["Q"], (TH,), (F(1),))[1],
          jax.jvp(lambda th: r(xs, keys, tab, th).outputs["Q"], (TH,), (F(1),))[1], what="jvp th")
    p = jnp.concatenate([tab, jnp.array([TH])])
    J = jax.jit(jax.jacfwd(lambda p: f(xs, keys, p[:NB], p[NB]).outputs["Q"]))(p)
    Jr = jax.jacfwd(lambda p: r(xs, keys, p[:NB], p[NB]).outputs["Q"])(p)
    close(J, Jr, what="jacfwd table+th")
    Jx = jax.jacfwd(lambda x: f(x, keys, tab, TH).outputs["Q"])(xs)
    Jxr = jax.jacfwd(lambda x: r(x, keys, tab, TH).outputs["Q"])(xs)
    close(Jx, Jxr, what="jacfwd per-item")


@case("closed-over tracers are differentiated (jvp, jacfwd, grad, vmap)")
def _(b):
    def make(th):
        def item(t, x, key):
            return walk(t, x, key, tab * th, th)        # th captured by CLOSURE
        return tm.vmap(item, outputs={"Q": tm.Sum((NB,), F)}, backend=b)

    def ref(th):
        return R(lambda t, x, key: walk(t, x, key, tab * th, th))(xs, keys).outputs["Q"].sum()
    g_ref = jax.jacfwd(ref)(TH)
    close(jax.jvp(lambda th: make(th)(xs, keys).outputs["Q"].sum(), (TH,), (F(1),))[1], g_ref, what="jvp")
    close(jax.grad(lambda th: make(th)(xs, keys).outputs["Q"].sum())(TH), g_ref, what="grad")
    close(jax.jit(jax.grad(lambda th: make(th)(xs, keys).outputs["Q"].sum()))(TH), g_ref, what="jit grad")
    ths = jnp.array([0.8, 0.9, 1.0], F)
    close(jax.vmap(lambda th: make(th)(xs, keys).outputs["Q"].sum())(ths), jax.vmap(ref)(ths), what="vmap")
    close(jax.vmap(jax.grad(lambda th: make(th)(xs, keys).outputs["Q"].sum()))(ths),
          jax.vmap(jax.jacfwd(ref))(ths), what="vmap(grad)")


@case("reverse: grad wrt table / per-item / closure, per-item results in the loss")
def _(b):
    fr = V(walk, b, in_axes=AX)
    r = R(walk, in_axes=AX)
    w = jnp.arange(1.0, NB + 1).astype(F)

    def loss(f, x, tb, th):
        A = f(x, keys, tb, th)
        return (A.outputs["Q"] * w).sum() + (A.per_item[0] ** 2).sum()
    g = jax.grad(lambda x, tb, th: loss(fr, x, tb, th), argnums=(0, 1, 2))(xs, tab, TH)
    gr = (jax.jacfwd(lambda x: loss(r, x, tab, TH))(xs), jax.jacfwd(lambda tb: loss(r, xs, tb, TH))(tab),
          jax.jacfwd(lambda th: loss(r, xs, tab, th))(TH))
    for a, c, n in zip(g, gr, ("per-item x", "table", "theta")):
        close(a, c, what=n)


@case("reverse through NESTED while with emits in the inner loop")
def _(b):
    def item(t, x, th):
        def outer(c):
            k, y, t = c

            def inner(d):
                j, z, t = d
                return j + 1, z * th, t.add("Q", j, z)
            _, y, t = lax.while_loop(lambda d: d[0] < k + 1, inner, (0, y, t))
            return k + 1, y * 0.9, t
        return lax.while_loop(lambda c: c[0] < 3, outer, (0, x, t))[2]
    f = V(item, b, in_axes=(0, None))
    r = R(item, in_axes=(0, None))
    w = jnp.arange(1.0, NB + 1).astype(F)
    g = jax.grad(lambda x, th: (f(x, th).outputs["Q"] * w).sum(), argnums=(0, 1))(xs, TH)
    gr = jax.jacfwd(lambda x, th: (r(x, th).outputs["Q"] * w).sum(), argnums=(0, 1))(xs, TH)
    close(g, gr)


@case("integer inputs (float0) and symbolic zeros")
def _(b):
    ii = jnp.arange(N, dtype=jnp.int32)

    def item(t, x, i, th):
        return t.add("Q", i % NB, x * th), i * 2
    f = V(item, b, in_axes=(0, 0, None))
    r = R(item, in_axes=(0, 0, None))
    close(jax.jvp(lambda th: f(xs, ii, th).outputs["Q"], (TH,), (F(1),))[1],
          jax.jvp(lambda th: r(xs, ii, th).outputs["Q"], (TH,), (F(1),))[1])
    fr = V(item, b, in_axes=(0, 0, None))
    close(jax.grad(lambda th: fr(xs, ii, th).outputs["Q"].sum())(TH), float(xs.sum()))
    t = jax.jvp(lambda th: f(xs, ii, th).per_item, (TH,), (F(1),))[1]
    if t.dtype != jax.dtypes.float0:
        raise Fail(f"int per-item tangent dtype {t.dtype}")


@case("jit / vmap(theta) / vmap(jacfwd) around tm.vmap")
def _(b):
    f = V(walk, b, in_axes=AX)
    r = R(walk, in_axes=AX)
    ths = jnp.array([0.7, 0.9, 1.1], F)
    close(jax.jit(jax.vmap(lambda th: f(xs, keys, tab, th).outputs["Q"]))(ths),
          jax.vmap(lambda th: r(xs, keys, tab, th).outputs["Q"])(ths), what="vmap")
    close(jax.vmap(jax.jacfwd(lambda th: f(xs, keys, tab, th).outputs["Q"]))(ths),
          jax.vmap(jax.jacfwd(lambda th: r(xs, keys, tab, th).outputs["Q"]))(ths), what="vmap(jacfwd)")
    xb = jnp.stack([xs, 2 * xs])
    close(jax.vmap(lambda x: f(x, keys, tab, TH).outputs["Q"])(xb),
          jax.vmap(lambda x: r(x, keys, tab, TH).outputs["Q"])(xb), what="vmap over per-item batch")


@case("tm.vmap inside lax.scan (optimiser) + kernel reuse when N changes")
def _(b):
    f = V(walk, b, in_axes=AX)

    def step(th, _):
        g = jax.jacfwd(lambda th: f(xs, keys, tab, th).outputs["Q"].sum())(th)
        return th - 1e-3 * g, g
    th, gs = jax.jit(lambda th: lax.scan(step, th, None, length=4))(TH)
    if not np.isfinite(np.asarray(gs)).all():
        raise Fail("non-finite")
    if b == "auto":
        from tallymap.backends import kernel as CB
        n0 = len(CB._KERNELS)
        for n in (5, 9, 33):
            f(xs[:n], keys[:n], tab, TH).outputs["Q"].block_until_ready()
        if len(CB._KERNELS) != n0:
            raise Fail(f"{len(CB._KERNELS) - n0} new kernels for new N (N must be a runtime argument)")
        return f"{n0} kernels, 0 new for 3 new N"


@case("n_active: fixed capacity, traced active count (no retrace, items >= n skipped)")
def _(b):
    f = V(walk, b, in_axes=AX)
    g = jax.jit(lambda n: f(xs, keys, tab, TH, n_active=n))
    for n in (0, 5, 37):
        A = g(jnp.int32(n))
        want = R(walk, in_axes=AX)(xs[:n], keys[:n], tab, TH).outputs if n else {"Q": np.zeros(NB)}
        close(A.outputs, want, what=f"n={n}")
        if n < N and float(np.abs(np.asarray(A.per_item[0][n:])).max()) != 0.0:
            raise Fail("skipped items have non-zero per-item results")
    return "3 active counts, one traced program"


@case("typed keys (jax.random.key) vs raw uint32 keys")
def _(b):
    tk = jax.random.split(jax.random.key(0), N)
    rk = jax.random.key_data(tk)

    def item_t(t, key):
        return t.add("Q", (jax.random.bits(key) % NB).astype(jnp.int32), jax.random.uniform(key, dtype=F)), key

    def item_r(t, key):
        return item_t(t, jax.random.wrap_key_data(key))[0], key
    A = V(item_t, b)(tk)
    Bq = V(item_r, b)(rk)
    close(A.outputs, Bq.outputs)
    close(A.outputs, R(item_t)(tk).outputs)
    if not np.array_equal(jax.random.key_data(A.per_item), rk):
        raise Fail("typed key per-item result")


@case("complex arithmetic inside an item")
def _(b):
    def item(t, x):
        z = jnp.exp(1j * x.astype(jnp.complex64)) * (1 + 2j)
        return t.add("Q", 0, jnp.abs(z).astype(F)).add("Q", 1, jnp.real(z * z).astype(F))
    close(V(item, b)(xs), R(item)(xs))


@case("debug.print raises clearly; pure_callback -> cuda unsupported (cuda and auto raise, pointing to xla)")
def _(b):
    def dbg(t, x):
        jax.debug.print("x={x}", x=x)
        return t.add("Q", 0, x)
    raises(Exception, lambda: V(dbg, b)(xs), "effect")

    def cb(t, x):
        y = jax.pure_callback(lambda v: np.asarray(v * 2), jax.ShapeDtypeStruct((), F), x, vmap_method="sequential")
        return t.add("Q", 0, y)
    if b == "auto":
        raises(tm.UnsupportedError, lambda: V(cb, "auto")(xs), "backend='xla'")
        raises(tm.UnsupportedError, lambda: V(cb, "auto")(xs), "backend='xla'")
        q = V(cb, "xla")(xs).outputs["Q"]
    else:
        q = V(cb, b)(xs).outputs["Q"]
    close(q[0], 2 * float(xs.sum()))


@case("in-place updates: a dead array is updated in place; an update that reads it is not (overlapping copies)")
def _(b):
    """x.at[k + arange(3)].set(...) with a traced k is a SCATTER, lax.dynamic_update_slice a DUS: the kernel backend
    writes both into a dead operand's storage.  Each step also copies x[k-1 : k+2] onto x[k : k+3] -- the update
    READS the storage it writes, so an element-by-element write in place would copy x[k-1] three times -- and
    accumulates into a second array in place.  The loop is NESTED (a single top-level while is split into step
    functions whose parameters are never mutated, and nothing would be updated in place).  Copies are exact:
    per-item results must equal jax.vmap bitwise.  Instrument (cuda): strictly fewer 12-element arrays are
    declared than with codegen.INPLACE = False, i.e. the in-place rule fired."""
    import re
    S = 12

    def one(x, k0):
        def inner(c):
            i, h, a = c
            k = 1 + (k0 + i) % (S - 4)
            h = lax.dynamic_update_slice(h, lax.dynamic_slice(h, (k - 1,), (3,)), (k,))
            h = h.at[k + 1 + jnp.arange(3)].set(lax.dynamic_slice(h, (k,), (3,)) * 2)
            a = a.at[k].set(h[k - 1] + a[k]).at[(k + 5) % S].add(x)
            return i + 1, h, a

        def outer(c):
            m, h, a = c
            _, h, a = lax.while_loop(lambda c: c[0] < 5, inner, (m, h, a))
            return m + 1, h, a
        _, h, a = lax.while_loop(lambda c: c[0] < 2, outer, (0, jnp.arange(S, dtype=F) * x, jnp.zeros((S,), F)))
        return h, a

    def item(t, x, k0):
        h, a = one(x, k0)
        return t.add("Q", 0, h[0] + a[1]), (h, a)
    xs = jnp.linspace(0.5, 1.5, 64).astype(F)
    ks = jnp.arange(64, dtype=jnp.int32) % 7
    want = jax.jit(jax.vmap(one))(xs, ks)

    def run(**o):
        tm.diagnostics(reset=True)
        r = tm.vmap(item, outputs={"Q": tm.Sum((1,), F)}, backend=b, **o)(xs, ks)
        for got, w, nm in zip(r.per_item, want, ("h", "a")):
            if not np.array_equal(np.asarray(got), np.asarray(w)):
                bad = int((np.asarray(got) != np.asarray(w)).sum())
                raise Fail(f"{nm} ({o or 'default'}): {bad} elements differ from jax.vmap "
                           "(an update read storage it had already overwritten)")
        ks_ = tm.diagnostics()["kernels"]
        src = open(ks_[-1]["source"]).read() if ks_ else ""
        return len(re.findall(r"\b(?:float|double) \w+\[12\];", src))
    n_in = run()
    if b != "auto":
        return ""
    from tallymap.backends.kernel import codegen as CG
    with patched(CG, INPLACE=False):
        n_cp = run()
    if not n_in < n_cp:
        raise Fail(f"instrument: {n_in} 12-element arrays declared in place, {n_cp} copying: the rule did not fire")
    return f"12-element arrays: {n_in} in place, {n_cp} copying"


# (helpers of the next cell, defined first: @case runs its cell at decoration time)
def _tangent_overflow(b, x2):
    """(G7) only the TANGENT overflows: emit x*(s0-1)*1e15 + x*s1 at s = (1, 1) -> primal x (finite), tangent
    column 0 x*1e15 (unrepresentable), column 1 x.  On every jvp path the column-0 tangent bin is NaN, the column-1
    bin and every primal bin are finite, and the primal is bitwise the plain call's."""
    from tallymap import _primitive as P_
    out = {"Q": tm.Sum((NB,), F, fixed_point=True, scale=2.0 ** 20)}
    s0 = jnp.ones((2,), F)
    eye = jnp.eye(2, dtype=F)

    def mk(**o):
        g = tm.vmap(lambda t, x, s: t.add("Q", 0, x * (s[0] - 1) * 1e15 + x * s[1]), outputs=out,
                    in_axes=(0, None), backend=b, **o)
        return lambda s: g(x2, s).outputs["Q"]
    ref = np.asarray(mk()(s0))
    if not np.isfinite(ref).all():
        raise Fail(f"the primal overflowed: {ref[:2]}")

    def check(nm, prim, tans):
        prim, tans = np.asarray(prim), np.asarray(tans)
        if not np.array_equal(prim, np.broadcast_to(ref, prim.shape)):
            raise Fail(f"{nm}: primal not bitwise the plain call (finite {np.isfinite(prim).all()})")
        t0, t1 = tans[0], tans[1] if tans.shape[0] > 1 else None
        if not (np.isnan(t0[0]) and np.isfinite(t0[1:]).all()):
            raise Fail(f"{nm}: tangent-only overflow not poisoned in its bin: {t0[:2]}")
        if t1 is not None:
            close(t1, ref, what=f"{nm}: column 1")
    paths = []
    fused0 = P_.FUSE_JVP
    try:
        for fused in (True, False):
            P_.FUSE_JVP = fused
            tag = "fused" if fused else "unfused"
            f = mk()
            p, t = jax.jvp(f, (s0,), (eye[0],))
            check(f"jvp {tag}", p, t[None])
            p, t = jax.vmap(lambda d: jax.jvp(f, (s0,), (d,)))(eye)          # jacfwd's mechanism, with the primal
            check(f"jacfwd lanes {tag}", p, t)
            with patched(P_, LANES="off"):                                    # lanes refused (DTRAX's regime)
                p, t = jax.vmap(lambda d: jax.jvp(f, (s0,), (d,)))(eye)
            check(f"jacfwd lanes-refused {tag}", p, t)
            check(f"jacfwd() {tag}", ref, jax.jacfwd(f)(s0).T)
            p, lin = jax.linearize(f, s0)
            check(f"linearize {tag}", p, lin(eye[0])[None])
            paths.append(tag)
    finally:
        P_.FUSE_JVP = fused0
    return "tangent-only overflow poisons its bin, primal bitwise: jvp/jacfwd(lanes, refused)/linearize x " + \
        "/".join(paths)


def _f32_bins(b, x2, k2):
    fl = tm.vmap(walk, outputs={"Q": tm.Sum((NB,), F)}, in_axes=AX, backend=b)
    f1 = np.asarray(fl(x2, k2, tab, TH).outputs["Q"])
    f2 = np.asarray(fl(x2[::-1], k2[::-1], tab, TH).outputs["Q"])
    return f"f32 atomics order-dependent bins: {int((f1 != f2).sum())}/{NB}"


@case("fixed_point=True: order-independent, run-to-run bitwise; overflow counted, poisons its own bin "
      "(primal and tangent)")
def _(b):
    if b == "xla" and not jax.config.jax_enable_x64:
        raises(NotImplementedError, lambda: tm.vmap(walk, outputs={"Q": tm.Sum((NB,), F, fixed_point=True)}, in_axes=AX,
                                                      backend=b)(xs, keys, tab, TH), "x64")
        return "needs x64 on xla (raises clearly)"
    out = {"Q": tm.Sum((NB,), F, fixed_point=True, scale=2.0 ** 20)}
    n = 4096
    x2 = jnp.linspace(0.1, 3.0, n).astype(F)
    k2 = jax.random.split(jax.random.PRNGKey(5), n)
    f = tm.vmap(walk, outputs=out, in_axes=AX, backend=b)
    a1 = np.asarray(f(x2, k2, tab, TH).outputs["Q"])
    a2 = np.asarray(f(x2[::-1], k2[::-1], tab, TH).outputs["Q"])          # different item order
    if not np.array_equal(a1, a2):
        raise Fail("fixed-point output depends on item order")
    big = tm.vmap(lambda t, x: t.add("Q", 0, x * 1e15), outputs=out, backend=b)(x2)
    if int(big.stats["fixed_point_overflow"]) != n:
        raise Fail(f"overflow count {int(big.stats['fixed_point_overflow'])} != {n}")
    # an overflow poisons the BIN it was emitted to, nothing else: other bins, other fixed-point outputs and float
    # outputs stay as they are
    mix = {"Q": tm.Sum((NB,), F, fixed_point=True, scale=2.0 ** 20),
           "E": tm.Sum((2,), F, fixed_point=True, scale=2.0 ** 20), "P": tm.Sum((1,), F)}

    def ovf(t, x, s):
        return t.add("Q", 0, x * s).add("Q", 1, x).add("E", 1, x).add("P", 0, x)
    g = tm.vmap(ovf, outputs=mix, in_axes=(0, None), backend=b)
    r = g(x2, F(1e15))
    q = np.asarray(r.outputs["Q"])
    if not (np.isnan(q[0]) and np.isfinite(q[1:]).all() and np.isfinite(np.asarray(r.outputs["E"])).all()):
        raise Fail(f"exact_overflow={int(r.stats['fixed_point_overflow'])}: want only Q[0] NaN, got Q {q[:3]} "
                   f"E {np.asarray(r.outputs['E'])}")
    close(r.outputs["P"][0], float(x2.sum()))
    ok = g(x2, F(1.0))
    if int(ok.stats["fixed_point_overflow"]) != 0 or not np.isfinite(np.asarray(ok.outputs["Q"])).all():
        raise Fail("a fixed-point call with no overflow was poisoned")
    return _tangent_overflow(b, x2) + "; " + _f32_bins(b, x2, k2)


@case("fixed_point: cuda == xla bitwise; jvp == jacfwd column bitwise", ["auto"])
def _(b):
    if not jax.config.jax_enable_x64:
        return "skipped (needs x64 for the xla side)"
    out = {"Q": tm.Sum((NB,), F, fixed_point=True, scale=2.0 ** 24)}
    fc = tm.vmap(walk, outputs=out, in_axes=AX, backend="auto")
    fx = tm.vmap(walk, outputs=out, in_axes=AX, backend="xla")
    a, c = np.asarray(fc(xs, keys, tab, TH).outputs["Q"]), np.asarray(fx(xs, keys, tab, TH).outputs["Q"])
    p = jnp.concatenate([tab, jnp.array([TH])])
    jv = jax.jvp(lambda th: fc(xs, keys, tab, th).outputs["Q"], (TH,), (F(1),))[1]
    J = jax.jacfwd(lambda p: fc(xs, keys, p[:NB], p[NB]).outputs["Q"])(p)
    if not np.array_equal(a, c):
        raise Fail(f"fixed-point cuda != xla in {int((a != c).sum())}/{NB} bins")
    if not np.array_equal(np.asarray(jv), np.asarray(J[:, NB])):
        raise Fail("fixed-point: jvp != jacfwd column bitwise")
    # int64 tallies (u64 atomics): per-item values above 2^32 exercise the high word
    big = (jnp.arange(N, dtype=jnp.int64) + 1) * (1 << 33)
    q = tm.vmap(lambda t, v: t.add("I", (v >> 33) % NB, v), outputs={"I": tm.Sum((NB,), jnp.int64)},
                backend="auto")(big).outputs["I"]
    want = np.zeros(NB, np.int64)
    np.add.at(want, (np.asarray(big) >> 33) % NB, np.asarray(big))
    if not np.array_equal(np.asarray(q), want):
        raise Fail(f"int64 tally {np.asarray(q)} != {want}")
    return f"cuda==xla bitwise {NB}/{NB} bins; jvp==jacfwd column bitwise; int64 tally exact"


@case("spawn: cascade counts, overflow counted, children truncated, jvp")
def _(b):
    def item(t, gen, x):
        t = t.add("Q", gen, x)
        t = t.spawn((gen + 1, x * 0.5), when=gen < 3)
        t = t.spawn((gen + 1, x * 0.25), when=gen < 3)
        return t
    out = {"Q": tm.Sum((NB,), F)}
    g0 = jnp.zeros((N,), jnp.int32)
    A = tm.vmap(item, outputs=out, spawn=tm.Queue(1 << 12), backend=b)(g0, xs)
    S = float(xs.sum())
    want = np.array([S, 0.75 * S, 0.75 ** 2 * S, 0.75 ** 3 * S, 0, 0, 0, 0])
    close(A.outputs["Q"], want, what="cascade")
    if int(A.stats["spawned"]) != N * (2 + 4 + 8) or int(A.stats["children_dropped"]) != 0:
        raise Fail(f"spawned {int(A.stats['spawned'])} dropped {int(A.stats['children_dropped'])}")
    Bq = tm.vmap(item, outputs=out, spawn=tm.Queue(16), backend=b)(g0, xs)
    tot = int(Bq.stats["spawned"]) + int(Bq.stats["children_dropped"])
    if int(Bq.stats["children_dropped"]) == 0:
        raise Fail("tiny queue dropped nothing")
    if not np.all(np.isnan(np.asarray(Bq.outputs["Q"]))):
        raise Fail(f"dropped children did not NaN-poison the float tally: {np.asarray(Bq.outputs['Q'])}")
    if not np.all(np.isfinite(np.asarray(A.outputs["Q"]))):
        raise Fail("a call that dropped nothing is poisoned")
    jb = jax.jvp(lambda s: tm.vmap(item, outputs=out, spawn=tm.Queue(16), backend=b)(g0, xs * s).outputs["Q"],
                 (F(1),), (F(1),))[1]
    if not np.all(np.isnan(np.asarray(jb))):
        raise Fail("dropped children did not poison the jvp tangent")
    jt = jax.jvp(lambda s: tm.vmap(item, outputs=out, spawn=tm.Queue(1 << 12), backend=b)(g0, xs * s).outputs["Q"],
                 (F(1),), (F(1),))[1]
    close(jt, want, what="jvp")

    def looper(t, gen, x):
        k, t = lax.while_loop(lambda c: c[0] < 100, lambda c: (c[0] + 1, c[1]), (0, t))
        return t.spawn((gen + 1, x), when=gen < 1)
    C = tm.vmap(looper, outputs=out, spawn=tm.Queue(1 << 10), max_steps=5, backend=b)(g0, xs)
    if int(C.stats["children_truncated"]) != N or int(C.stats["n_truncated"]) != N:
        raise Fail(f"children_truncated {int(C.stats['children_truncated'])} n_truncated {int(C.stats['n_truncated'])}")
    if not np.all(np.isnan(np.asarray(C.outputs["Q"]))):
        raise Fail("truncated items / children did not poison the float tally")
    return f"small queue: spawned {int(Bq.stats['spawned'])}, dropped {int(Bq.stats['children_dropped'])} (of {tot})"


@case("reverse with spawn (tree adjoint) == jacfwd; second order through it raises")
def _(b):
    def item(t, gen, x):
        t = t.add("Q", gen, x * x)
        return t.spawn((gen + 1, x * 0.5 + 0.1), when=gen < 2)
    g0 = jnp.zeros((N,), jnp.int32)
    f = tm.vmap(item, outputs={"Q": tm.Sum((NB,), F)}, spawn=tm.Queue(1 << 10), backend=b)
    L = lambda s: jnp.sum(f(g0, xs * s).outputs["Q"] * jnp.arange(1.0, NB + 1))
    close(jax.jit(jax.grad(L))(F(1.1)), jax.jit(jax.jacfwd(L))(F(1.1)), what="grad vs jacfwd")
    return "hessian raises: " + raises(NotImplementedError, lambda: jax.hessian(L)(F(1.1)))[:80]


@case("hessian (jacfwd of jacfwd, and jax.hessian) vs dense reference")
def _(b):
    f = V(walk, b, in_axes=AX)
    r = R(walk, in_axes=AX)
    loss = lambda fn: lambda p: (fn(xs, keys, p[:NB], p[NB]).outputs["Q"] * jnp.arange(1.0, NB + 1)).sum()
    p = jnp.concatenate([tab, jnp.array([TH])])
    Hr = jax.jacfwd(jax.jacfwd(loss(r)))(p)
    close(jax.jit(jax.jacfwd(jax.jacfwd(loss(f))))(p), Hr, what="jacfwd(jacfwd)")
    close(jax.jit(jax.hessian(loss(f)))(p), Hr, what="hessian")


@case("compare protocol (tests/compare_protocol.py): cuda vs xla (CRN, ulp control, Poisson floor)",
      ["auto"])
def _(b):
    from compare_protocol import compare

    def walk_sq(t, x, key, tab, th):               # the floor run: walk with every emitted value squared
        def body(c):
            k, w, t = c
            i = (bits(key, k) % NB).astype(jnp.int32)
            v = w * tab[i] * th
            t = t.add("Q", i, v * v)
            return k + 1, w * (0.7 + 0.2 * th), t
        k, w, t = lax.while_loop(lambda c: (bits(key, c[0] + 1000) % 5 != 0) | (c[0] == 0), body, (0, x, t))
        return t, (w, k)
    n = 1 << 14
    x2 = jnp.linspace(0.1, 3.0, n).astype(F)
    k2 = jax.random.split(jax.random.PRNGKey(7), n)
    ok, rep = compare(V(walk, "auto", in_axes=AX), x2, k2, tab, TH, other=V(walk, "xla", in_axes=AX),
                      floor_fn=V(walk_sq, "xla", in_axes=AX), verbose=False)
    if not ok:
        raise Fail(str(rep))
    return f"flip {rep['flip_rate']:.2e} (ctrl {rep['flip_rate_control']:.2e}); rms_z {rep['Q']['rms_z']:.2e}"


@case("tm.diagnostics(): versions, programs, kernels, reverse counters; rev_stats() is gone")
def _(b):
    if hasattr(tm, "rev_stats"):
        raise Fail("tm.rev_stats still exists (one channel: tm.diagnostics()['reverse'])")
    f = V(walk, b, in_axes=AX)
    d = tm.diagnostics(lambda x, k, t, th: f(x, k, t, th).outputs["Q"], xs, keys, tab, TH)
    for k in ("versions", "platform", "env", "reverse", "programs", "kernels", "stablehlo_sha256"):
        if k not in d:
            raise Fail(f"missing key {k}")
    if not d["programs"] or "while" not in d["programs"][0]["jaxpr"]:
        raise Fail(f"no item jaxpr recorded: {d['programs']}")
    if (b == "auto") != bool(d["kernels"]):
        raise Fail(f"kernels recorded: {d['kernels']}")
    if b == "auto" and not os.path.exists(d["kernels"][0]["source"]):
        raise Fail(f"cached kernel source missing: {d['kernels'][0]['source']}")
    if b == "auto" and not GPU:          # the host target: its stack bound, known at compile time
        res = [k.get("resources") for k in d["kernels"]]
        if not all(k["target"] == "cpu" for k in d["kernels"]) or not all(
                r_ and 0 < r_["stack_bytes"] <= r_["stack_limit"] for r_ in res):
            raise Fail(f"host kernel records: {[(k['target'], k.get('resources')) for k in d['kernels']]}")
    elif b == "auto":                    # once a cuda kernel has run, its measured resources are reported
        jax.block_until_ready(f(xs, keys, tab, TH).outputs["Q"])
        res = [k.get("resources") for k in tm.diagnostics()["kernels"]]
        if not any(r_ and r_["regs"] > 0 and r_["blocks_per_sm"] >= 1 and r_["local_reserved_gib"] >= 0
                   for r_ in res):
            raise Fail(f"no kernel resources after a run: {res}")
    r = tm.diagnostics()["reverse"]
    if set(r) != {"single", "tree"}:
        raise Fail(f"reverse record {r}")
    return f"{len(d['programs'])} program(s), {len(d['kernels'])} kernel(s); jax {d['versions']['jax']}"


@case("A/B rule: tests/ab_rule.assert_arms_differ (identical arms raise; different programs pass)")
def _(b):
    g = lambda **o: (lambda x, k, t, th: V(walk, b, in_axes=AX, **o)(x, k, t, th).outputs["Q"])
    if b == "auto":
        same, diff = (dict(cuda_math="approx"), dict()), (dict(schedule="step"), dict(schedule="item"))
    else:
        same, diff = (dict(cuda_math="ieee"), dict()), (dict(xla_batch_size=8), dict())
    raises(AssertionError, lambda: assert_arms_differ(g(**same[0]), g(**same[1]), xs, keys, tab, TH), "identical")
    (ka, ha), (kb, hb) = assert_arms_differ(g(**diff[0]), g(**diff[1]), xs, keys, tab, TH)
    return f"arms {diff}: kernels {len(ka)}/{len(kb)}, hlo {ha[:8]} vs {hb[:8]}"


@case("tm.diagnostics: the StableHLO sha of a program with a host callback is the same on every lowering")
def _(b):
    f = V(walk, b, in_axes=AX)

    def one(x, k, t, th):
        jax.debug.callback(lambda v: None, th)
        return f(x, k, t, th).outputs["Q"]

    def two(x, k, t, th):                # the control: one more callback is a different program
        jax.debug.callback(lambda v: None, th)
        jax.debug.callback(lambda v: None, x)
        return f(x, k, t, th).outputs["Q"]
    a = [jax.jit(one).lower(xs, keys, tab, TH).as_text() for _ in range(2)]
    if "callback" not in a[0]:
        raise Fail("instrument: the lowered program has no host callback")
    raw_same = a[0] == a[1]
    h1, h2 = (tm.diagnostics(one, xs, keys, tab, TH)["stablehlo_sha256"] for _ in range(2))
    h3 = tm.diagnostics(two, xs, keys, tab, TH)["stablehlo_sha256"]
    if h1 != h2:
        raise Fail(f"two lowerings of one program hash differently: {h1[:12]} vs {h2[:12]}")
    if h3 == h1:
        raise Fail("a program with one more host callback hashes the same")
    return f"sha {h1[:12]} twice, two callbacks {h3[:12]}; raw texts {'equal' if raw_same else 'DIFFER'}"


@case("backend='auto': the kernel backend (on the platform JAX lowers for) raises on an unsupported program, the "
      "error points to backend='xla'; another platform runs the xla backend with a one-time warning; "
      "backend='cuda' (removed) raises", ["xla"])
def _(b):
    from jax import export

    from tallymap import _api
    def cb(t, x):
        y = jax.pure_callback(lambda v: np.asarray(v * 2), jax.ShapeDtypeStruct((), F), x, vmap_method="sequential")
        return t.add("Q", 0, y)
    want = 2 * float(xs.sum())
    m = raises(tm.UnsupportedError, lambda: V(cb, "auto")(xs), "pass backend='xla'")
    close(V(cb, "xla")(xs).outputs["Q"][0], want)
    close(V(walk, "auto", in_axes=AX)(xs, keys, tab, TH), R(walk, in_axes=AX)(xs, keys, tab, TH))

    def dbg(t, x):                      # an effect: the error says no backend runs it
        jax.debug.print("x={x}", x=x)
        return t.add("Q", 0, x)
    raises(tm.UnsupportedError, lambda: V(dbg, "auto")(xs), "on any backend")
    raises(ValueError, lambda: tm.vmap(walk, outputs={"Q": tm.Sum((NB,), F)}, backend="cuda"), "use backend='auto'")
    # a platform the kernel backend does not compile for (lowered, never run: no such device here)
    _api._WARNED.clear()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        line = sys._getframe().f_lineno + 1
        export.export(jax.jit(lambda *a: V(walk, "auto", in_axes=AX)(*a).outputs["Q"]), platforms=["rocm"])(
            xs, keys, tab, TH)
    ws = [x for x in w if "xla backend" in str(x.message)]
    if len(ws) != 1:
        raise Fail(f"expected ONE xla warning, got {[str(x.message) for x in ws]}")
    msg = str(ws[0].message)
    if "not for 'rocm'" not in msg:                           # (the platform being lowered for, by name)
        raise Fail(f"warning text: {msg!r}")
    if os.path.basename(ws[0].filename) != os.path.basename(__file__) or ws[0].lineno != line:
        raise Fail(f"warning attributed to {ws[0].filename}:{ws[0].lineno}, want this file line {line}")
    return f"{jax.devices()[0].platform}: " + m.splitlines()[-1][:70] + " | rocm: " + msg[:60]


@case("error messages name the item function and the source line (tally misuse, effects, unsupported op)")
def _(b):
    here = os.path.basename(__file__) + ":"

    def fork(t, x):
        t1 = t.add("Q", 0, x)  # noqa: F841
        t2 = t.add("Q", 1, x)          # <- the second use of t
        return t2
    m = raises(tm.TallyError, lambda: V(fork, b)(xs), "used twice")
    for want in ("item function", "fork", here):
        if want not in m:
            raise Fail(f"TallyError message lacks {want!r}: {m}")
    line = fork.__code__.co_firstlineno + 2
    if f"{here}{line}" not in m:
        raise Fail(f"TallyError does not point at line {line}: {m}")

    def reader(t, x):
        return t, t._a["Q"].sum() * x
    m2 = raises(tm.TallyError, lambda: V(reader, b)(xs), "reads a Tally")
    if "reader" not in m2 or here not in m2:
        raise Fail(f"read message: {m2}")

    def dbg(t, x):
        jax.debug.print("x={x}", x=x)
        return t.add("Q", 0, x)
    m3 = raises(Exception, lambda: V(dbg, b)(xs), "effect")
    if "dbg" not in m3 or here not in m3:
        raise Fail(f"effects message: {m3}")
    out = [m.split("[")[-1][:60], m3.split("[")[-1][:60]]
    if b == "auto":
        def cb(t, x):
            y = jax.pure_callback(lambda v: np.asarray(v), jax.ShapeDtypeStruct((), F), x, vmap_method="sequential")
            return t.add("Q", 0, y)
        m4 = raises(tm.UnsupportedError, lambda: V(cb, "auto")(xs), "pure_callback")
        if "item function" not in m4 or "cb (" not in m4 or here not in m4:
            raise Fail(f"unsupported-op message: {m4}")
        out.append(m4.split("[")[-1][:60])
    return " | ".join(out)


@case("a tally refusal names the operand, slot and kind; inside a call, the position alignment; a reverse "
      "program's slot legend", ["xla"])
def _(b):
    def reader(t, x):
        return t, t._a["Q"].sum() * x
    m = raises(tm.TallyError, lambda: V(reader, b)(xs), "reads a Tally")
    for want in ("operand 0", "tally slot 0 (kind sum)"):
        if want not in m:
            raise Fail(f"refusal lacks {want!r}: {m}")

    def inner(t, x):
        return t, jax.jit(lambda a: a.sum())(t._a["Q"]) * x
    m2 = raises(tm.TallyError, lambda: V(inner, b)(xs), "reads a Tally")
    for want in ("tally slot 0", "1 equation operand(s) vs 1 sub-jaxpr invar(s)", "slot 0]"):
        if want not in m2:
            raise Fail(f"refusal inside a call lacks {want!r}: {m2}")
    from tallymap._api import TSpec
    from tallymap._reverse import check_slots
    cj = jax.make_jaxpr(lambda t, a: (t, jnp.sum(t) + jnp.sum(a)))(jnp.zeros((8,), F), jnp.ones((8,), F))
    m3 = raises(tm.TallyError, lambda: check_slots(cj, (TSpec("tok3", 8, F, kind="discard"),), "item+vjp"),
                "reads a Tally")
    if "tally slot(s) of item+vjp, in order: 0=tok3(discard,S=8)" not in m3:
        raise Fail(f"no slot legend: {m3}")
    return m.split("; ")[-1][:70]


def _ix(xi):                 # a TRACED table index (a constant index would fold the table read at trace time)
    return (xi * 8).astype(jnp.int32) % 32


@case("diagnostics closed_over_floats: a captured float table is listed, the same table as an operand is not",
      ["xla"])
def _(b):
    tabc = jnp.linspace(0.5, 1.5, 32, dtype=F)
    out = {"Q": tm.Sum((1,), F)}
    cap = tm.vmap(lambda t, xi: t.add("Q", 0, xi * tabc[_ix(xi)]), outputs=out, backend=b)
    opr = tm.vmap(lambda t, xi, tb: t.add("Q", 0, xi * tb[_ix(xi)]), outputs=out, in_axes=(0, None), backend=b)
    pc = tm.diagnostics(lambda x: cap(x).outputs["Q"], xs)["programs"]
    po = tm.diagnostics(lambda x, tb: opr(x, tb).outputs["Q"], xs, tabc)["programs"]
    if not pc or not any(sh == (32,) and n == 32 for r in pc for sh, _, n in r["closed_over_floats"]):
        raise Fail(f"the captured table is not listed: {[r['closed_over_floats'] for r in pc]}")
    if not po or any(r["closed_over_floats"] for r in po):
        raise Fail(f"an operand is listed as closed over: {[r['closed_over_floats'] for r in po]}")
    if any("baked_floats" in r for r in pc):
        raise Fail("the old key baked_floats is still in the record")
    if "closed_over_floats" in tm.diagnostics()["programs"][-1] and "consts" in tm.diagnostics()["programs"][-1]:
        raise Fail("the raw constants leak into the returned record")
    g = jax.jvp(lambda tb: opr(xs, tb).outputs["Q"], (tabc,), (jnp.ones_like(tabc),))[1]
    close(g, jnp.sum(xs)[None], rtol=1e-5)
    return f"captured {pc[0]['closed_over_floats']}, operand []; the operand's tangent reaches the tally"


@case("a located error keeps the original exception's attributes", ["xla"])
def _(b):
    from tallymap._jaxpr import located
    eqn = jax.make_jaxpr(lambda x: x * 2.0)(F(1)).jaxpr.eqns[0]
    e = tm.TallyError("tallymap: refusal")
    e.nbytes, e._tm_eqn = 12345, eqn          # data a caller acts on (dtrax: CodegenError.nbytes)
    n = located(e, eqn, "item function g")
    if type(n) is not tm.TallyError or "item function g" not in str(n) or "refusal" not in str(n):
        raise Fail(f"located: {type(n).__name__}: {n}")
    if getattr(n, "nbytes", None) != 12345 or getattr(n, "_tm_eqn", None) is not eqn:
        raise Fail(f"located dropped the attributes: {sorted(vars(n))}")
    if located(n, eqn) is not n:
        raise Fail("located twice re-wrapped the error")
    return str(n)[-60:]


@case("kernel parameters: the struct has the program's own buffer count; more than 511 buffers run (CUDA large "
      "kernel parameters); over 4094 raises", ["auto"])
def _(b):
    def mk(nt):
        def item(t, x, tabs):
            i = (x * 3).astype(jnp.int32) % 4
            return t.add("Q", 0, x * sum(tb[i] for tb in tabs))
        return tm.vmap(item, outputs={"Q": tm.Sum((1,), F)}, in_axes=(0, None), backend=b)
    msg = []
    for nt in (8, 600, 2000):
        tabs = tuple(jnp.arange(4, dtype=F) * F(1 + j % 7) for j in range(nt))
        d = tm.diagnostics(lambda x, ts: mk(nt)(x, ts).outputs["Q"], xs, tabs)
        src = open(d["kernels"][0]["source"]).read()
        import re
        m = re.search(r"struct SjParams \{ void\* p\[(\d+)\]; \};", src)
        want = int(m.group(1)) if m else -1
        if not nt + 4 <= want <= nt + 16:          # the tables + x + the hoisted / output / counter / stats buffers
            raise Fail(f"{nt} tables: the kernel's parameter struct has {want} pointers")
        got = np.asarray(mk(nt)(xs, tabs).outputs["Q"])        # every table pointer is read: tab_j[i] = i (1 + j % 7)
        i = np.asarray((xs * 3).astype(jnp.int32) % 4)
        want_q = float(np.sum(np.asarray(xs, np.float64) * i) * sum(1 + j % 7 for j in range(nt)))
        close(got, np.array([want_q]), rtol=1e-4, what=f"{nt} tables")
        msg.append(f"{nt}: p[{want}]")
    tabs = tuple(jnp.arange(4, dtype=F) for _ in range(4100))
    m = raises(tm.UnsupportedError, lambda: mk(4100)(xs, tabs), "too many buffers")
    return "; ".join(msg) + "; 4100: " + m[:40]


QT_NK = 8


def _qt_tree(t, gen, x, kind):
    t = t.add("c", kind, jnp.int32(1)).add("s", kind, jnp.sin(x) + gen.astype(x.dtype))
    k2 = ((kind * 7 + gen + 1) % QT_NK).astype(jnp.int32)
    k3 = ((kind * 5 + 3) % QT_NK).astype(jnp.int32)
    t = t.spawn((gen + 1, x * 0.5 + 0.25, k2), when=gen < 5)
    return t.spawn((gen + 1, x * 0.25 + 0.5, k3), when=(gen < 4) & (x > 0.3))


def _qt_flat(t, gen, x, kind):          # no children: every thread lives in the root-claim path
    return t.add("c", kind, jnp.int32(1)).add("s", kind, jnp.cos(x) + gen.astype(x.dtype))


@case("spawn queue termination: every named item runs, launch after launch (cuda == xla, exact counts)",
      ["auto"])
def _(b):
    # dtrax tests_wavefront/test_qterm.py (b65e17a), the one-queue cells: a tree with children in flight while
    # roots are still being claimed, and roots only with N far above the resident grid.  An early exit loses
    # items, and every item is one count in "c": the integer tallies must equal the xla ones exactly.
    out = {"c": tm.Sum((QT_NK,), jnp.int32), "s": tm.Sum((QT_NK,), F)}
    rng = np.random.default_rng(0)
    msg = []
    for fn, n, nm in ((_qt_tree, 20000, "tree"), (_qt_flat, 1 << 20, "roots only")):
        args = (jnp.zeros((n,), jnp.int32), jnp.asarray(rng.random(n), F),
                jnp.asarray(rng.integers(0, QT_NK, n), jnp.int32))
        ref = tm.vmap(fn, outputs=out, spawn=tm.Queue(1 << 20), backend="xla")(*args)
        rc = np.asarray(ref.outputs["c"])
        if int(ref.stats["children_dropped"]) != 0:
            raise Fail(f"{nm}: the xla run dropped {int(ref.stats['children_dropped'])}")
        f = jax.jit(lambda *a, fn=fn: tm.vmap(fn, outputs=out, spawn=tm.Queue(1 << 20), backend=b)(*a))
        for rep in range(3):
            r = f(*args)
            c = np.asarray(r.outputs["c"])
            if not np.array_equal(c, rc) or int(r.stats["children_dropped"]) != 0:
                raise Fail(f"{nm} launch {rep}: {int(c.sum())} items vs {int(rc.sum())}, dropped "
                           f"{int(r.stats['children_dropped'])}")
            if int(r.stats["spawned"]) != int(rc.sum()) - n:
                raise Fail(f"{nm} launch {rep}: spawned {int(r.stats['spawned'])} != {int(rc.sum()) - n}")
            close(r.outputs["s"], ref.outputs["s"], rtol=1e-10 if F == jnp.float64 else 1e-3,   # (sum order)
                  what=f"{nm} launch {rep} float tally")
        msg.append(f"{nm}: {int(rc.sum())} items x3 launches")
    return "; ".join(msg)


@case("spawn queue stress: fan-out >> capacity, repeated; counts consistent; int tally exact, float tally NaN")
def _(b):
    F_OUT, DEPTH, NR = 6, 3, 2048

    def item(t, gen, x):
        t = t.add("C", gen, 1)                            # processed items per generation (exact integer count)
        t = t.add("Q", gen, x)
        for j in range(F_OUT):
            t = t.spawn((gen + 1, x * 0.5), when=gen < DEPTH)
        return t
    out = {"Q": tm.Sum((NB,), F), "C": tm.Sum((NB,), jnp.int32)}
    g0 = jnp.zeros((NR,), jnp.int32)
    x0 = jnp.linspace(0.5, 1.5, NR).astype(F)
    seen = set()
    for cap in (8, 64, 1 << 10):
        f = jax.jit(tm.vmap(item, outputs=out, spawn=tm.Queue(cap), backend=b))
        for rep in range(5):
            r = f(g0, x0)
            c = np.asarray(r.outputs["C"]).astype(np.int64)
            sp, dr = int(r.stats["spawned"]), int(r.stats["children_dropped"])
            if c[0] != NR or c.sum() != NR + sp:
                raise Fail(f"cap {cap} rep {rep}: processed {c.tolist()} != roots + spawned {NR} + {sp}")
            if sp + dr != F_OUT * int(c[:DEPTH].sum()):
                raise Fail(f"cap {cap} rep {rep}: spawned + dropped {sp} + {dr} != attempted "
                           f"{F_OUT * int(c[:DEPTH].sum())}")
            # whether a queue overflows depends on how many threads drain it at once (the CPU target with 4
            # threads never fills 1024 here; 16 threads or a GPU do), so the contract is per outcome: a drop
            # poisons every float bin, no drop is the whole tree with finite tallies
            q = np.asarray(r.outputs["Q"])
            if dr and not np.all(np.isnan(q)):
                raise Fail(f"cap {cap} rep {rep}: dropped {dr} but Q {q[:3]} is not NaN")
            if not dr and (not np.all(np.isfinite(q)) or sp != sum(NR * F_OUT ** g for g in range(1, DEPTH + 1))):
                raise Fail(f"cap {cap} rep {rep}: nothing dropped but spawned {sp}, Q {q[:3]}")
            if cap == 8 and not dr:
                raise Fail(f"cap 8 rep {rep}: a queue of 8 against {NR * F_OUT} first-generation children dropped none")
            seen.add((cap, sp))
    full = tm.vmap(item, outputs=out, spawn=tm.Queue(1 << 20), backend=b)(g0, x0)
    want = [NR * F_OUT ** g for g in range(DEPTH + 1)]
    if np.asarray(full.outputs["C"])[:DEPTH + 1].tolist() != want or int(full.stats["children_dropped"]):
        raise Fail(f"no-drop run: {np.asarray(full.outputs['C']).tolist()} != {want}")
    return f"{len(seen)} distinct (capacity, spawned) outcomes over 15 runs; full tree {sum(want) - NR} children"


@case("per-item keys: fold_in(key, id*stride + step) repeats every 2^32/stride items; the safe patterns do not")
def _(b):
    stride, step = 64, 3
    period = 2 ** 32 // stride                         # 67,108,864 items
    ids = jnp.array([5, 5 + period, 7, 7 + 3 * period], jnp.uint32)
    key = jax.random.PRNGKey(11)

    def item(t, i, key):
        naive = jax.random.bits(jax.random.fold_in(key, i * stride + step))               # uint32 wraps
        safe = jax.random.bits(jax.random.fold_in(jax.random.fold_in(key, i), step))       # id and step separately
        chunk, local = i // period, i % period                                           # chunked calls
        chunked = jax.random.bits(jax.random.fold_in(jax.random.fold_in(key, chunk), local * stride + step))
        return t.add("Q", 0, 1.0), (naive, safe, chunked)
    naive, safe, chunked = (np.asarray(a) for a in V(item, b, in_axes=(0, None))(ids, key).per_item)
    if not (naive[0] == naive[1] and naive[2] == naive[3]):
        raise Fail(f"instrument: the naive keys should collide: {naive}")
    if safe[0] == safe[1] or safe[2] == safe[3] or chunked[0] == chunked[1] or chunked[2] == chunked[3]:
        raise Fail(f"the safe patterns collide: {safe} {chunked}")
    return f"naive key repeats at id + {period}; fold_in(fold_in(key, id), step) and the chunked key do not"


@case("in-place dynamic_update_slice: aliasing hazards == xla == codegen.INPLACE off (bitwise)", ["auto"])
def _(b):
    def item(t, x, key):
        buf0 = jnp.arange(8, dtype=jnp.int32) * x

        def body(c):
            k, buf, acc, t = c
            j = (bits(key, k) % 8).astype(jnp.int32)
            old = lax.dynamic_slice(buf, (j,), (3,))                     # a view of buf ...
            whole = buf                                                  # ... and buf itself, both read AFTER:
            buf2 = lax.dynamic_update_slice(buf, old * 3 + k, ((j + 2) % 8,))
            acc = acc + old.sum(dtype=jnp.int32) + buf2[j] - whole[(j + 1) % 8]
            buf3 = lax.cond(k % 2 == 0,                                  # in place in one branch only
                            lambda q: lax.dynamic_update_index_in_dim(q, q[(j + 3) % 8] + 1, j, 0),
                            lambda q: q, buf2)
            buf4 = lax.dynamic_update_index_in_dim(buf3, acc % 97, (j + 5) % 8, 0)   # carry updated in place
            t = t.add("C", j, acc % 1000)
            return k + 1, buf4, acc + buf3[(j + 4) % 8], t
        k, buf, acc, t = lax.while_loop(lambda c: c[0] < 7, body, (jnp.int32(0), buf0, jnp.int32(0), t))
        return t, (buf, acc)
    out = {"C": tm.Sum((8,), jnp.int32)}
    ii = jnp.arange(N, dtype=jnp.int32) % 5 + 1
    from tallymap.backends.kernel import codegen as CG
    run = lambda bk: tm.vmap(item, outputs=out, backend=bk)
    g = lambda a, k: run("auto")(a, k).per_item[0]
    assert_arms_differ(g, g, ii, keys, patch_b=(CG, dict(INPLACE=False)))    # instrument: in-place updates happened
    rs = [run("auto")(ii, keys)]
    with patched(CG, INPLACE=False):
        rs.append(run("auto")(ii, keys))
    rs.append(run("xla")(ii, keys))
    for r in rs[1:]:
        for a, c in zip(jax.tree_util.tree_leaves(rs[0]), jax.tree_util.tree_leaves(r)):
            if not np.array_equal(np.asarray(a), np.asarray(c)):
                raise Fail("in-place cuda result differs from the copying cuda / xla result")
    return "bitwise equal; in-place and copying cubins differ"


@case("buffer assignment (while init alias, cond destination, lazy write-back) == rule off == copying cuda == "
      "xla", ["auto"])
def _(b):
    """the dead initial carry IS the carry storage, the cond output is written into its dead operand (one branch
    updates it in place, the other returns a lazy whole-array select over it: v5), and the lazy body result goes
    straight back into the carry.  The loop is NESTED: a single top-level while is split into step functions whose
    parameters are never mutated, where the rule (rightly) never fires.  Arms: the rule patched off (the control:
    strictly more arrays must be declared without it), and codegen.INPLACE = False (every liveness-based reuse
    off)."""
    import re

    from tallymap import _primitive as _P
    from tallymap.backends import kernel as CB
    from tallymap.backends.kernel import codegen as CG
    from tallymap.backends.kernel import op_table as OT

    def item(t, x, key):
        def outer(c):
            m, acc, t = c
            buf0 = jnp.zeros(32, F).at[0].set(x + m.astype(F))          # a per-item Local, dead after the loop

            def body(c):
                k, buf, t = c
                j = (bits(key, k) % 32).astype(jnp.int32)
                buf2 = lax.cond(k % 3 == 0, lambda q: q.at[j].add(1.0),
                                lambda q: jnp.where(q > 0.5, q * 0.5, q + 0.25), buf)
                t = t.add("C", j % NB, (buf2[j] > 0.75).astype(jnp.int32))
                return k + 1, buf2 * 1.5 + x, t
            k, buf, t = lax.while_loop(lambda c: c[0] < 9, body, (jnp.int32(0), buf0, t))
            return m + 1, acc + buf, t
        m, acc, t = lax.while_loop(lambda c: c[0] < 2, outer, (jnp.int32(0), jnp.zeros(32, F), t))
        return t, acc
    out = {"C": tm.Sum((NB,), jnp.int32)}
    xs = (jnp.arange(N) % 7 * 0.125).astype(F)

    def run(bk, off=False, inplace=True):
        saved = (OT._cond_dest, OT._init_alias, CG.Emitter.elementwise_self)
        CB._KERNELS.clear()
        _P._EAGER.clear()
        CG.INPLACE = inplace
        if off:
            OT._cond_dest = lambda em, eqn, ops, bodies: [None] * len(eqn.outvars)
            OT._init_alias = lambda em, inits, ivars, pt, others: [None] * len(inits)
            CG.Emitter.elementwise_self = lambda self, dst, x: False
        try:
            tm.diagnostics(reset=True)
            r = tm.vmap(item, outputs=out, backend=bk)(xs, keys)
            ks = tm.diagnostics()["kernels"]
        finally:
            OT._cond_dest, OT._init_alias, CG.Emitter.elementwise_self = saved
            CG.INPLACE = True
            CB._KERNELS.clear()
            _P._EAGER.clear()
        src = open(ks[-1]["source"]).read() if ks else ""
        return r, len(re.findall(r"\b(?:float|double) \w+\[32\];", src))
    (ra, na), (ro, no), (rc, nc), (rl, _) = run("auto"), run("auto", off=True), run("auto", inplace=False), \
        run("xla")
    if not na < no:
        raise Fail(f"instrument: {na} 32-element arrays declared with buffer assignment, {no} with the rule off")
    for r_, what in ((ro, "rule-off"), (rc, "copying")):
        for a, c in zip(jax.tree_util.tree_leaves(ra), jax.tree_util.tree_leaves(r_)):
            if not np.array_equal(np.asarray(a), np.asarray(c)):
                raise Fail(f"buffer-assigned cuda result differs from the {what} cuda result")
    if not np.array_equal(np.asarray(ra.outputs["C"]), np.asarray(rl.outputs["C"])):
        raise Fail("integer tally differs from xla")
    close(ra.per_item, rl.per_item, 1e-6, 0.0, "per-item vs xla")
    return f"bitwise vs rule-off and copying cuda; 32-element arrays declared {na} / rule off {no} / copying {nc}"


@case("$CXX: a missing or wrong host C++ compiler is a tallymap error naming the variable and the fix",
      ["xla"])
def _(b):
    import shutil

    from tallymap.backends.kernel import compile as C
    old = os.environ.get("CXX")
    which = shutil.which
    try:
        os.environ["CXX"] = "/nonexistent/g++-not-here"
        m = raises(RuntimeError, C.host_cxx, "$CXX")
        os.environ.pop("CXX", None)
        shutil.which = lambda *a, **k: None                  # no g++ / c++ on PATH either
        m2 = raises(RuntimeError, C.host_cxx, "$CXX")
    finally:
        shutil.which = which
        os.environ.pop("CXX", None)
        if old is not None:
            os.environ["CXX"] = old
    for name, msg in (("wrong $CXX", m), ("no compiler", m2)):
        for wantstr in ("tallymap:", "CXX", "Fix:"):
            if wantstr not in msg:
                raise Fail(f"{name} message lacks {wantstr!r}: {msg}")
    if C.host_cxx() is None:                                 # the real environment still resolves one
        raise Fail("host_cxx() returned None")
    return f"{os.path.basename(C.host_cxx())}; both errors name $CXX and the fix"


@case("op table: names are JAX primitives (or explained); supported_primitives() and docs/primitives.md current",
      ["xla"])
def _(b):
    from jax.extend.core import Primitive
    from jax.extend.core import primitives as P

    from tallymap.backends.kernel import canon, census, op_table
    jaxnames = {v.name for v in vars(P).values() if isinstance(v, Primitive)}
    ours = set(op_table.RULES) | set(canon.SUGAR)
    bad = sorted(ours - jaxnames - set(census.NOT_IN_JAX_EXTEND))
    if bad:
        raise Fail(f"op-table names that are neither in jax.extend.core.primitives nor explained: {bad} "
                   "(a JAX rename?)")
    rows = tm.supported_primitives()
    if {r["name"] for r in rows} != ours | {"tallymap_emit", "tallymap_spawn"}:
        raise Fail("supported_primitives() does not list the op table")
    keys = {k for r in rows for k in r}
    if keys != {"name", "how"}:
        raise Fail(f"supported_primitives() rows have keys {sorted(keys)}, want name and how only")
    hows = {r["how"] for r in rows}
    if hows - {"rule", "canonicalised", "tally"}:
        raise Fail(f"unknown 'how' values {sorted(hows)}")
    doc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "primitives.md")
    if open(doc).read() != census.primitives_table():
        raise Fail("docs/primitives.md is stale: regenerate it with tallymap.backends.kernel.census.primitives_table()")
    exported_unused = sorted(jaxnames - ours)
    return (f"{len(ours)} names, {len(ours & jaxnames)} exported by jax.extend; {len(exported_unused)} exported, "
            "not ours")


@case("fixed_point=True on cuda without x64: equals the host fixed-point sum bitwise, order-independent", ["auto"])
def _(b):
    n, scale = 1 << 15, 2.0 ** 20
    v = jnp.asarray(np.random.default_rng(3).uniform(-3, 3, n), jnp.float32)
    i = jnp.arange(n, dtype=jnp.int32)
    f = tm.vmap(lambda t, x, j: t.add("Q", j % NB, x).add("Q", (j + 1) % NB, x * 3.0),
                outputs={"Q": tm.Sum((NB,), jnp.float32, fixed_point=True, scale=scale)}, backend="auto")
    a1, a2 = np.asarray(f(v, i).outputs["Q"]), np.asarray(f(v[::-1], i[::-1]).outputs["Q"])
    acc = np.zeros(NB, np.int64)
    vv = np.asarray(v).astype(np.float64)
    np.add.at(acc, np.arange(n) % NB, np.round(vv * scale).astype(np.int64))
    np.add.at(acc, (np.arange(n) + 1) % NB, np.round(np.asarray(v * 3.0).astype(np.float64) * scale).astype(np.int64))
    want = (acc.astype(np.float64) * (1.0 / scale)).astype(np.float32)
    if not (np.array_equal(a1, a2) and np.array_equal(a1, want)):
        raise Fail(f"fixed-point sum: order-independent {np.array_equal(a1, a2)}, == host {np.array_equal(a1, want)}")
    return f"{NB}/{NB} bins bitwise equal to the host int64 sum (x64 off)"


@case("closed-over PYTREE of tracers (dict of params) is hoisted: grad / jacfwd / hessian vs dense")
def _(b):
    P0 = {"a": F(0.9), "b": {"c": jnp.linspace(0.5, 1.5, NB).astype(F), "d": F(0.3)}}

    def make(vm, p):
        def item(t, x, key):
            def body(c):
                k, w, t = c
                i = (bits(key, k) % NB).astype(jnp.int32)
                return k + 1, w * (0.7 + 0.2 * p["a"]), t.add("Q", i, w * p["b"]["c"][i] + p["b"]["d"] ** 2)
            k, w, t = lax.while_loop(lambda c: (bits(key, c[0] + 1000) % 5 != 0) | (c[0] == 0), body, (0, x, t))
            return t, w
        return vm(item)(xs, keys)
    sm = lambda it: tm.vmap(it, outputs={"Q": tm.Sum((NB,), F)}, backend=b)
    rm = lambda it: reference(it, outputs={"Q": tm.Sum((NB,), F)})
    def loss(vm):
        def L(p):
            r = make(vm, p)
            return jnp.sum(r.outputs["Q"] * jnp.arange(1.0, NB + 1)) + jnp.sum(r.per_item)
        return L
    close(jax.jit(jax.grad(loss(sm)))(P0), jax.jacfwd(loss(rm))(P0), what="grad")
    close(jax.jit(jax.jacfwd(loss(sm)))(P0), jax.jacfwd(loss(rm))(P0), what="jacfwd")
    sub = lambda vm: lambda q: loss(vm)({"a": q[0], "b": {"c": P0["b"]["c"], "d": q[1]}})
    q0 = jnp.stack([P0["a"], P0["b"]["d"]])
    close(jax.jit(jax.hessian(sub(sm)))(q0), jax.jacfwd(jax.jacfwd(sub(rm)))(q0), what="hessian")


@case("version policy: the tested versions, a warning only above the newest tested", ["xla"])
def _(b):
    from tallymap._compat import _check_jax_version
    if tm.TESTED_JAX != ("0.6.2", "0.7.2", "0.8.3", "0.9.2", "0.10.2", "0.11.2", "0.12.0.1"):
        raise Fail(f"TESTED_JAX {tm.TESTED_JAX}")
    if hasattr(tm, "check_jax_version") or "check_jax_version" in tm.__all__:
        raise Fail("check_jax_version is still public")
    st = _check_jax_version(warn=False)
    if st != "ok":
        raise Fail(f"this JAX ({jax.__version__}) is {st}: the matrix should run on tested versions")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        got = [_check_jax_version(v) for v in ("0.6.2", "0.13.0.dev20270101", "0.4.38", "0.12.1")]
    if got != ["ok", "newer", "ok", "newer"] or len(w) != 2 or "newer" not in str(w[0].message):
        raise Fail(f"policy {got}, warnings {[str(x.message) for x in w]}")
    return f"jax {jax.__version__}: ok; {len(tm.TESTED_JAX)} tested; 0.13 dev warns, 0.4.38 does not"


@case("eager calls: items that differ only in a host callback are different programs (cache key)", ["xla"])
def _(b):
    def make(k):
        def cb(t, x):
            y = jax.pure_callback(lambda v: np.asarray(v * k, dtype=v.dtype), jax.ShapeDtypeStruct((), F), x,
                                  vmap_method="sequential")
            return t.add("Q", 0, y)
        return cb
    s = float(xs.sum())
    q2 = float(V(make(2.0), b)(xs).outputs["Q"][0])
    q3 = float(V(make(3.0), b)(xs).outputs["Q"][0])
    close(np.array([q2, q3]), np.array([2 * s, 3 * s]), what="callback results")


@case("bitcast_convert_type between widths (shape-changing) on cuda == XLA bitwise", ["auto"])
def _(b):
    u = jax.random.bits(jax.random.PRNGKey(4), (N,), jnp.uint32)
    x = lax.bitcast_convert_type(u, jnp.float32)

    def casts(x, u):
        h = lax.bitcast_convert_type(x, jnp.uint16)                       # f32 -> u16[2]
        b8 = lax.bitcast_convert_type(u, jnp.uint8)                       # u32 -> u8[4]
        back = lax.bitcast_convert_type(b8, jnp.float32)                  # u8[4] -> f32
        hb = lax.bitcast_convert_type(h, jnp.uint32)                      # u16[2] -> u32
        return h, b8, back, hb
    got = V(lambda t, x, u: (t.add("Q", 0, 1.0), casts(x, u)), b)(x, u).per_item
    want = jax.jit(jax.vmap(casts))(x, u)
    for g, w in zip(got, want):
        if not np.array_equal(np.asarray(g).view(np.uint8), np.asarray(w).view(np.uint8)):
            raise Fail(f"bitcast differs from XLA: {np.asarray(g).dtype} {np.asarray(g).shape}")
    if jax.config.jax_enable_x64:
        d = jnp.asarray(np.random.default_rng(1).normal(size=N))
        f = V(lambda t, d: (t.add("Q", 0, 1.0), lax.bitcast_convert_type(lax.bitcast_convert_type(d, jnp.uint32),
                                                                           jnp.float64)), b)
        if not np.array_equal(np.asarray(f(d).per_item), np.asarray(d)):
            raise Fail("f64 -> u32[2] -> f64 round trip differs")
    return "f32<->u16[2], u32<->u8[4] bitwise" + (", f64<->u32[2] round trip" if jax.config.jax_enable_x64 else "")


@case("long items: no typedef name collision, no lazy-expression blowup", ["auto"])
def _(b):
    from tallymap.backends.kernel.codegen import _RESERVED, Emitter
    e = object.__new__(Emitter)                    # the counter is shared by every prefix: walk it past 8/16/32/64
    for p in ("i", "u", "v"):
        e.uid = 0
        got = [e.fresh(p) for _ in range(80)]
        if set(got) & _RESERVED:
            raise Fail(f"fresh({p!r}) produced {sorted(set(got) & _RESERVED)}")

    def item(t, x):                                # 40 stages over a 64-wide array (> UNROLL); jnp.roll is a concat
        #                                            of two slices of v, so a lazily inlined v doubles every stage
        v = x * jnp.arange(64, dtype=F)
        for k in range(40):
            v = jnp.sin(v) * (1.0 + 0.01 * k) + jnp.roll(v, 1)
        return t.add("Q", 0, jnp.sum(v))
    close(V(item, b)(xs).outputs["Q"], R(item)(xs).outputs["Q"], rtol=1e-3)
    return "fresh() skips reserved names; a 40-stage roll chain compiles and matches the reference"

@case("custom_jvp rule closing over an item-computed value: a clear UnsupportedError (JAX's staging limit); the "
      "advised fix (an explicit argument) differentiates")
def _(b):
    def mk(explicit):
        def item(t, x, th):
            tab = jnp.stack([x, x * 2.0, x + 1.0])         # computed inside the item, not differentiated

            @jax.custom_jvp
            def f(a, tb):
                return jnp.zeros_like(a)

            @f.defjvp
            def f_jvp(p, tng):
                (a, tb), (da, _) = p, tng
                return f(a, tb), da * jnp.sum(tb if explicit else tab) * jnp.cos(a)
            y = f(th * x, tab if explicit else jnp.zeros(3, F)) + jnp.sin(x)
            return t.add("Q", 0, y), y
        return item
    jv = lambda it: jax.jvp(lambda q: V(it, b, in_axes=(0, None))(xs, q).outputs["Q"], (TH,), (F(1.0),))[1]
    m = raises(tm.UnsupportedError, lambda: jv(mk(False)), "closure_convert")

    want = jax.vmap(lambda x: jax.jvp(lambda q: jnp.sin(x) + 0 * q, (TH,), (F(1.0),))[1] +
                    jnp.sum(jnp.stack([x, x * 2.0, x + 1.0])) * jnp.cos(TH * x) * x)(xs).sum()
    close(jv(mk(True))[0], want, rtol=1e-4)

    def outer(q, w):                                 # NOT a leak: the item closes over the LIVE outer jit's w
        it = lambda t, x: (t.add("Q", 0, x * w * q), x * w)
        return V(it, b)(xs).outputs["Q"]
    close(jax.jit(outer)(TH, F(2.0))[0], TH * 2.0 * xs.sum(), rtol=1e-5)
    close(jax.jit(jax.grad(lambda q, w: outer(q, w).sum()))(TH, F(2.0)), 2.0 * xs.sum(), rtol=1e-5)
    return "leak: " + m.split(".")[1][:60] + "...; explicit argument: jvp matches; outer-jit closure runs"


@case("jnp.linalg.solve / inv inside an item: cuda raises the ordinary UnsupportedError (no rule)", ["auto"])
def _(b):
    def mk(op):
        def item(t, x):
            A = jnp.eye(4, dtype=F) * (2.0 + x) + 0.1
            y = op(A)
            return t.add("Q", 0, jnp.sum(y)), y
        return item
    out = []
    for name, op in (("solve", lambda A: jnp.linalg.solve(A, jnp.ones(4, F))), ("inv", lambda A: jnp.linalg.inv(A)[0])):
        m = raises(tm.UnsupportedError, lambda op=op: V(mk(op), b)(xs),
                   "is not supported by the tallymap kernel backend")
        prim = re.search(r"primitive '(\w+)' is not supported", m)
        out.append(f"{name}: {prim.group(1) if prim else m[:60]}")
    return "; ".join(out)


@case("step schedule: a large array loop carry is updated in place (no per-step copy), same result", ["auto"])
def _(b):
    from tallymap.backends.kernel.kernel import STEP_INPLACE_MIN
    H = 2 * STEP_INPLACE_MIN

    def item(t, x, key, th):
        def body(c):
            k, w, h = c
            i = k % H                     # (a dynamic_update_slice: the in-place write the step schedule keeps in
            h = lax.dynamic_update_index_in_dim(h, h[i] + w * th, i, 0)   # place; scatter always copies)
            return k + 1, w * F(0.9), h
        k, w, h = lax.while_loop(lambda c: (bits(key, c[0]) % 7 != 0) | (c[0] < 3), body,
                                 (0, x, jnp.zeros((H,), F)))
        return t.add("Q", (k % NB).astype(jnp.int32), jnp.sum(h * jnp.arange(H, dtype=F)))
    f = V(item, b, in_axes=(0, 0, None), schedule="step")
    d = tm.diagnostics(lambda x, k, th: f(x, k, th).outputs["Q"], xs, keys, TH)
    k0 = d["kernels"][-1]
    src = open(k0["source"]).read()
    if k0.get("schedule") != "step":
        raise Fail(f"the step schedule did not run ({k0.get('schedule')}): the cell does not test the in-place carry")
    if re.search(rf"e < {H}; \+\+e\) k\d+\[e\] = n\d+\[e\]", src):
        raise Fail(f"the {H}-element carry is still copied back every step")
    close(f(xs, keys, TH), R(item, in_axes=(0, 0, None))(xs, keys, TH))
    return f"{H}-element carry: no per-step copy"



@case("jacrev with max_steps set (a laned adjoint's per-lane truncation flags) == jacfwd")
def _(b):
    def item(t, x, key, th):
        L = 2 + (bits(key, 7) % 19).astype(jnp.int32)

        def body(c):
            k, w, t = c
            i = (bits(key, k) % NB).astype(jnp.int32)
            return k + 1, w * th[i % 4] + 0.125, t.add("Q", i, w)
        return lax.while_loop(lambda c: c[0] < L, body, (jnp.int32(0), x, t))[2]
    th = jnp.linspace(0.5, 0.9, 4).astype(F)
    for cap, trunc in ((64, False), (10, True)):
        f = V(item, b, in_axes=(0, 0, None), max_steps=cap)
        Q = lambda th: f(xs, keys, th).outputs["Q"]
        jr, jf = jax.jit(jax.jacrev(Q))(th), jax.jit(jax.jacfwd(Q))(th)
        if trunc:
            if not (np.all(np.isnan(np.asarray(jr))) and np.all(np.isnan(np.asarray(jf)))):
                raise Fail(f"max_steps={cap} truncates: jacrev / jacfwd should be NaN")
        else:
            close(jr, jf, what=f"jacrev vs jacfwd, max_steps={cap}")
    return "max_steps 64: jacrev == jacfwd; 10 (truncating): both NaN"


@case("typed PRNG key (jax.random.key) as a mapped input of a spawning call: value and grad == raw uint32 key")
def _(b):
    def particle(t, p, th):
        def body(c):
            s, n, t = c
            k = jax.random.fold_in(s["key"], n)
            u = jax.random.uniform(k)
            t = t.add("Q", (u * NB).astype(jnp.int32), s["E"] * th)
            t = t.spawn(dict(E=s["E"] * 0.5, key=jax.random.fold_in(k, 1), g=s["g"] + 1), when=(u < 0.3) & (s["g"] < 3))
            return dict(E=s["E"] * 0.8, key=s["key"], g=s["g"]), n + 1, t
        return lax.while_loop(lambda c: c[1] < 5, body, (p, jnp.int32(0), t))[2]
    raw = jax.random.split(jax.random.PRNGKey(1), N)
    got = {}
    for kind, k in (("typed", jax.random.wrap_key_data(raw)), ("raw", raw)):
        p0 = dict(E=xs, key=k, g=jnp.zeros((N,), jnp.int32))
        g = V(particle, b, in_axes=(0, None), spawn=tm.Queue(1 << 12))
        got[kind] = (jax.jit(lambda th: g(p0, th).outputs["Q"])(TH),
                     jax.jit(jax.grad(lambda th: g(p0, th).outputs["Q"].sum()))(TH))
    close(got["typed"], got["raw"], what="typed vs raw key")
    return "typed key == raw key, value and grad"


@case("cond nested 4 deep: each item equation interpreted about once (not 3^depth)", ["xla"])
def _(b):
    from tallymap.backends import xla as XB
    n = [0]
    bind0 = XB.bind

    def counting(e, ins):
        n[0] += 1
        return bind0(e, ins)

    def item(t, x):
        def blk(v, d):
            if d == 0:
                return jnp.sin(v) * 1.01 + 0.1
            return lax.cond(v > 0.1 * d, lambda v: blk(v, d - 1), lambda v: blk(v * 0.9, d - 1), v)

        def body(c):
            i, v, t = c
            v = blk(v, 4)
            return i + 1, v, t.add("Q", i % NB, v)
        return lax.while_loop(lambda c: c[0] < 6, body, (jnp.int32(0), x, t))[2]
    XB.bind = counting
    try:
        got = jax.jit(lambda x: V(item, b)(x).outputs["Q"])(xs)
    finally:
        XB.bind = bind0
    close(got, R(item)(xs).outputs["Q"], what="nested cond vs reference")
    neq = 2 ** 5 * 3 + 8          # leaves of the binary cond tree, a few ops each, plus the loop: an upper bound
    if n[0] > 2 * neq:
        raise Fail(f"{n[0]} interpretations of primitive equations for a ~{neq}-equation item (3^depth re-tracing?)")
    return f"{n[0]} primitive interpretations"


@case("f32 tally accuracy: the xla backend's sum is about as close to the exact sum as plain JAX's", ["xla"])
def _(b):
    if F != jnp.float32:
        return "f32 only (x64 run)"

    @jax.tree_util.register_pytree_node_class
    class Dense:                                     # a plain-JAX tally: the item's own (8,) array
        def __init__(self, h):
            self.h = h

        def add(self, name, i, v):
            return Dense(self.h.at[i].add(v))

        def tree_flatten(self):
            return (self.h,), None

        @classmethod
        def tree_unflatten(cls, aux, ch):
            return cls(ch[0])

    def item(t, x):
        def body(c):
            i, v, t = c
            for _ in range(8):
                v = jnp.sin(v) * 1.01 + 0.1
            return i + 1, v, t.add("Q", i % NB, v)
        return lax.while_loop(lambda c: (c[0] < 6) & (c[1] < 3.0), body, (jnp.int32(0), x, t))[2]
    x = jnp.linspace(-1.0, 1.0, 4096).astype(F)
    got = np.asarray(jax.jit(lambda x: V(item, b)(x).outputs["Q"])(x), np.float64)
    per = np.asarray(jax.jit(jax.vmap(lambda xi: item(Dense(jnp.zeros(NB, F)), xi).h))(x))     # (N, 8) f32
    exact = per.astype(np.float64).sum(0)
    plain = np.asarray(jnp.asarray(per).sum(0), np.float64)                                     # XLA's f32 reduction
    e_x, e_p = np.max(np.abs(got - exact)), np.max(np.abs(plain - exact))
    if e_x > 8 * e_p + 1e-6 * np.max(np.abs(exact)):
        raise Fail(f"xla f32 error {e_x:.3g} vs plain JAX {e_p:.3g} (|sum| {np.max(np.abs(exact)):.4g}): "
                   "serial accumulation into one accumulator?")
    return f"error {e_x:.3g} vs plain JAX {e_p:.3g} ({e_x / max(e_p, 1e-30):.1f}x), |sum| {np.max(np.abs(exact)):.4g}"


@case("erf_inv in the kernel: f64 inverts XLA's f64 erf/erfc to a few ulps across (-1, 1) incl. the tails, matches "
      "two published values; erf_inv(+-1) = +-inf, |y| > 1 NaN; cpu: f32 within 2 ulp of the f64 result rounded",
      ["auto"])
def _(b):
    x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        t = np.concatenate([np.linspace(-0.999, 0.999, 4001), 1 - np.logspace(-15, -3, 200),
                            -(1 - np.logspace(-15, -3, 50)), [0.0, 0.5, 0.9, 1.0, -1.0, 1.5, -2.0]])
        g = lambda dt: jax.jit(tm.vmap(lambda tt, y: (tt, jax.scipy.special.erfinv(y)), outputs={"_": tm.Sum((1,), dt)},
                                       backend=b))
        y64 = jnp.asarray(t, jnp.float64)
        x = np.asarray(g(jnp.float64)(y64).per_item)
        fin = np.abs(t) < 1
        lo, hi = fin & (np.abs(t) < 0.5), fin & (np.abs(t) >= 0.5)
        e_lo = np.max(np.abs(np.asarray(jax.scipy.special.erf(x[lo])) - t[lo]) / np.spacing(np.abs(t[lo]) + 1e-300))
        # the tails: compare 1 - |y| (exact in f64 for these y) with erfc(|x|)
        q = 1 - np.abs(t[hi])
        e_hi = np.max(np.abs(np.asarray(jax.scipy.special.erfc(np.abs(x[hi]))) - q) / q)
        if not (e_lo <= 4 and e_hi <= 1e-13):
            raise Fail(f"round trip: |erf(x) - y| {e_lo:.2f} ulp (|y| < 0.5), tail rel {e_hi:.2e}")
        ref = {0.5: 0.4769362762044699, 0.9: 1.1630871536766743}
        for y, xr in ref.items():
            v = x[np.nonzero(t == y)[0][0]]
            if abs(v - xr) > 4 * np.spacing(xr):
                raise Fail(f"erfinv({y}) = {v!r}, published {xr!r}")
        sp = {1.0: np.inf, -1.0: -np.inf}
        for y, v in sp.items():
            if x[np.nonzero(t == y)[0][0]] != v:
                raise Fail(f"erfinv({y}) = {x[np.nonzero(t == y)[0][0]]}")
        if not np.isnan(x[np.abs(t) > 1]).all():
            raise Fail("erfinv(|y| > 1) is not NaN")
        if GPU:                          # (cuda's f32 erfinvf is libdevice's own; ops.py bounds it at 2 ulp)
            return f"f64 round trip {e_lo:.2f} ulp, tail rel {e_hi:.1e}"
        y32 = jnp.asarray(t, jnp.float32)
        x32 = np.asarray(g(jnp.float32)(y32).per_item)
        r32 = np.asarray(g(jnp.float64)(jnp.asarray(np.asarray(y32), jnp.float64)).per_item).astype(np.float32)
        fin32 = np.isfinite(r32)
        u32 = np.max(np.abs(x32[fin32].view(np.int32).astype(np.int64) - r32[fin32].view(np.int32).astype(np.int64)))
        if u32 > 2 or not np.array_equal(x32[~fin32], r32[~fin32], equal_nan=True):
            raise Fail(f"f32 erf_inv {u32} ulp from the f64 result rounded (bound 2, libdevice's) or a special differs")
        return f"f64 round trip {e_lo:.2f} ulp, tail rel {e_hi:.1e}; f32 within {u32} ulp of f64 rounded"
    finally:
        jax.config.update("jax_enable_x64", x64)


@case("cpu target: the spawn queue's spin-wait with every team thread running at once, oversubscribed (64 threads) "
      "and single-threaded, == xla (counts exact, tallies to tolerance)", ["auto"])
def _(b):
    if GPU:
        return "n/a on cuda (the CPU target's thread team)"
    def shower(t, x):
        t = t.add("Q", jnp.clip((x * NB).astype(jnp.int32), 0, NB - 1), x)
        return t.spawn(x * 0.5, when=x > 0.05).spawn(x * 0.25, when=x > 0.2)
    x0 = jnp.linspace(0.1, 1.0, 512).astype(F)
    mk = lambda bk: jax.jit(tm.vmap(shower, outputs={"Q": tm.Sum((NB,), F)}, spawn=tm.Queue(1 << 15), backend=bk))
    ref = mk("xla")(x0)
    old = os.environ.get("OMP_NUM_THREADS")
    out = []
    try:
        for nt in ("64", "1"):
            os.environ["OMP_NUM_THREADS"] = nt          # (read by the shim at every launch)
            r = mk(b)(x0)
            for k in ("spawned", "children_dropped"):
                if int(r.stats[k]) != int(ref.stats[k]):
                    raise Fail(f"{nt} threads: {k} {int(r.stats[k])} vs xla {int(ref.stats[k])}")
            close(r.outputs["Q"], ref.outputs["Q"], what=f"{nt} threads")
            out.append(f"{nt} threads spawned {int(r.stats['spawned'])}")
    finally:
        if old is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = old
    return "; ".join(out)

print("\n================ SUMMARY", jax.__version__, "x64" if jax.config.jax_enable_x64 else "",
      jax.devices()[0].device_kind)
for n, b, m in RES:
    print(f"{n:52s} {b:8s} {m[:160]}")
print("TOTAL", len(RES), "PASS", sum(m.startswith("PASS") for _, _, m in RES))
