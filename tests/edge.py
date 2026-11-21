"""Phase-2 edge-case matrix.  python tests/edge.py  (env: BACKENDS=xla,cuda  TEST_X64=1  ONLY=substr)
Prints one line per (case, backend): PASS / FAIL <why> / RAISE <exception> and a summary table."""
import math
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
# an emit a fixed_point=True output of dtype F cannot represent: past 2^63 / scale for f64 fixed point; for the exact
# float32 sum (which represents every finite f32) a non-finite one.  TOVF: a factor that makes an f32 item value
# overflow to inf (an exact f32 tangent then has a non-finite emit) and is past 2^63 / 2^20 for f64
OVF = F(1e15) if jax.config.jax_enable_x64 else F(np.inf)
TOVF = F(1e15) if jax.config.jax_enable_x64 else F(3e38)
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
    E = tm.vmap(mk(1), outputs={"Q": tm.Sum((NB,), F, fixed_point=True, scale=2.0 ** 20)}, backend=b)(xs, idx)
    if int(E.stats["dropped_emits"]) != want or int(E.stats["fixed_point_overflow"]) != 0:
        raise Fail(f"fixed-point tally: dropped_emits {int(E.stats['dropped_emits'])} (want {want}), overflow "
                   f"{int(E.stats['fixed_point_overflow'])}")
    return msg + "; fixed-point tally too"


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
        n0, b0 = len(CB._KERNELS), CB.BUILDS[0]
        for n in (5, 9, 33):
            f(xs[:n], keys[:n], tab, TH).outputs["Q"].block_until_ready()
        if CB.BUILDS[0] != b0:
            raise Fail(f"{CB.BUILDS[0] - b0} new kernels for new N (N must be a runtime argument)")
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
    """(G7) only the TANGENT overflows: emit x*(s0-1)*TOVF + x*s1 at s = (1, 1) -> primal x (finite), tangent
    column 0 x*TOVF (unrepresentable; inf in f32), column 1 x.  On every jvp path the column-0 tangent bin is NaN,
    the column-1 bin and every primal bin are finite, and the primal is bitwise the plain call's."""
    from tallymap import _primitive as P_
    out = {"Q": tm.Sum((NB,), F, fixed_point=True, scale=2.0 ** 20)}
    s0 = jnp.ones((2,), F)
    eye = jnp.eye(2, dtype=F)

    def mk(**o):
        g = tm.vmap(lambda t, x, s: t.add("Q", 0, x * (s[0] - 1) * TOVF + x * s[1]), outputs=out,
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
    out = {"Q": tm.Sum((NB,), F, fixed_point=True, scale=2.0 ** 20)}
    n = 4096
    x2 = jnp.linspace(0.1, 3.0, n).astype(F)
    k2 = jax.random.split(jax.random.PRNGKey(5), n)
    f = tm.vmap(walk, outputs=out, in_axes=AX, backend=b)
    a1 = np.asarray(f(x2, k2, tab, TH).outputs["Q"])
    a2 = np.asarray(f(x2[::-1], k2[::-1], tab, TH).outputs["Q"])          # different item order
    if not np.array_equal(a1, a2):
        raise Fail("fixed-point output depends on item order")
    big = tm.vmap(lambda t, x: t.add("Q", 0, x * OVF), outputs=out, backend=b)(x2)
    if int(big.stats["fixed_point_overflow"]) != n:
        raise Fail(f"overflow count {int(big.stats['fixed_point_overflow'])} != {n}")
    # an overflow poisons the BIN it was emitted to, nothing else: other bins, other fixed-point outputs and float
    # outputs stay as they are
    mix = {"Q": tm.Sum((NB,), F, fixed_point=True, scale=2.0 ** 20),
           "E": tm.Sum((2,), F, fixed_point=True, scale=2.0 ** 20), "P": tm.Sum((1,), F)}

    def ovf(t, x, s):
        return t.add("Q", 0, x * s).add("Q", 1, x).add("E", 1, x).add("P", 0, x)
    g = tm.vmap(ovf, outputs=mix, in_axes=(0, None), backend=b)
    r = g(x2, OVF)
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
    if not jax.config.jax_enable_x64:
        return f"cuda==xla bitwise {NB}/{NB} bins; jvp==jacfwd column bitwise"
    # int64 tallies (u64 atomics): per-item values above 2^32 exercise the high word
    big = (jnp.arange(N, dtype=jnp.int64) + 1) * (1 << 33)
    q = tm.vmap(lambda t, v: t.add("I", (v >> 33) % NB, v), outputs={"I": tm.Sum((NB,), jnp.int64)},
                backend="auto")(big).outputs["I"]
    want = np.zeros(NB, np.int64)
    np.add.at(want, (np.asarray(big) >> 33) % NB, np.asarray(big))
    if not np.array_equal(np.asarray(q), want):
        raise Fail(f"int64 tally {np.asarray(q)} != {want}")
    return f"cuda==xla bitwise {NB}/{NB} bins; jvp==jacfwd column bitwise; int64 tally exact"


def _exact_bins(qs, idx, nbins, scale, dt):
    """the exact fixed-point sum of the emits qs (Python ints, already round(v * scale)) into bins idx, as the
    output dtype dt: the int total / scale rounded once to float64 (Fraction), then cast -> (outputs, int totals)."""
    from fractions import Fraction
    tot = [0] * nbins
    for q, i in zip(qs, idx):
        tot[i] += q
    return np.array([float(Fraction(t) / Fraction(scale)) for t in tot]).astype(dt), tot


@case("fixed_point: the carry word -- a bin past 2^31 at scale 2^32 is exact (primal, jvp, linearize; privatised "
      "and global bins), deterministic; non-carrying bins as before; a total beyond the output type is counted")
def _(b):
    n, sc, GB = 4096, 2.0 ** 32, 1024           # GB bins: over PRIV_BYTES, so G is a global (not privatised) tally
    rng = np.random.default_rng(7)
    mag = rng.uniform(1e6, 2e8, n)
    bin_ = np.arange(n) % 4
    # bin 0: all positive (total ~1e11 >> 2^31), bin 1: all negative, bin 2: +-pairs (the running sum crosses 2^31
    # and back: carries that cancel), bin 3: small values (no carry ever)
    v = np.where(bin_ == 0, mag, np.where(bin_ == 1, -mag, np.where(bin_ == 2, np.where((np.arange(n) // 4) % 2 == 0,
                 mag, -mag * 0.999), mag * 1e-9)))
    x = jnp.asarray(v, F)
    xv = np.asarray(x, np.float64)                 # exactly the emitted values (F), so q = v * 2^32 exactly
    qs = [int(np.round(a * sc)) for a in xv]
    ib = jnp.asarray(bin_, jnp.int32)
    out = {"Q": tm.Sum((NB,), F, fixed_point=True), "G": tm.Sum((GB,), F, fixed_point=True)}

    def item(t, x, i, s):
        y = x * s
        return t.add("Q", i, y).add("G", i * 255, y)
    g = tm.vmap(item, outputs=out, in_axes=(0, 0, None), backend=b)
    wq, tot = _exact_bins(qs, bin_, NB, sc, np.dtype(F))
    wg, _ = _exact_bins(qs, bin_ * 255, GB, sc, np.dtype(F))
    if not (abs(tot[0]) >= 2 ** 63 and abs(tot[1]) >= 2 ** 63 and abs(tot[3]) < 2 ** 63):
        raise Fail(f"test setup: totals {[t / 2 ** 63 for t in tot[:4]]} x 2^63")

    def chk(nm, r, st=None):
        q, gg = (np.asarray(r[k]) for k in ("Q", "G"))
        if not (np.array_equal(q, wq) and np.array_equal(gg, wg)):
            raise Fail(f"{nm}: not the exact sum: Q {q[:4]} want {wq[:4]}; G bins equal "
                       f"{int((gg == wg).sum())}/{GB}")
        if st is not None and int(st["fixed_point_overflow"]) != 0:
            raise Fail(f"{nm}: fixed_point_overflow {int(st['fixed_point_overflow'])} (nothing overflowed)")
    one = F(1.0)
    r1 = g(x, ib, one)
    chk("primal", r1.outputs, r1.stats)
    # non-carrying bin 3: today's formula, (double)(int64 total) / scale cast, bitwise
    if np.asarray(r1.outputs["Q"])[3] != np.asarray(np.float64(np.int64(tot[3])) * (1.0 / sc)).astype(F):
        raise Fail("non-carrying bin differs from the plain int64 readout")
    r2 = g(x[::-1], ib[::-1], one)                 # another item order: other carries, the same net carry
    if not all(np.array_equal(np.asarray(r1.outputs[k]), np.asarray(r2.outputs[k])) for k in out):
        raise Fail("carrying fixed-point sum depends on the item order")
    r3 = g(x, ib, one)
    if not all(np.array_equal(np.asarray(r1.outputs[k]), np.asarray(r3.outputs[k])) for k in out):
        raise Fail("carrying fixed-point sum differs run to run")
    f = lambda s: g(x, ib, s).outputs                                     # noqa: E731
    p, t = jax.jvp(f, (one,), (one,))
    chk("jvp primal", p)
    chk("jvp tangent", t)
    p, lin = jax.linearize(f, one)
    chk("linearize tangent", lin(one))
    # a total that does not fit the output type: counted (stats['fixed_point_overflow']); a float bin is NaN
    # (and on the linearize path, where the tangent node's stats have no output to reach, its bin is NaN)
    gb = tm.vmap(lambda t, x, s: t.add("A", 0, x * s).add("A", 1, x * 1e-30), outputs={"A": tm.Sum(
                 (2,), jnp.float32, fixed_point=True, scale=2.0 ** -80)}, in_axes=(0, None), backend=b)
    x4, s1 = jnp.full((4,), 3e38, jnp.float32), jnp.float32(1.0)
    big = gb(x4, s1)
    a = np.asarray(big.outputs["A"])
    if not (np.isnan(a[0]) and np.isfinite(a[1]) and int(big.stats["fixed_point_overflow"]) == 1):
        raise Fail(f"f32 total beyond FLT_MAX: {a}, overflow {int(big.stats['fixed_point_overflow'])} (want NaN, 1)")
    a = np.asarray(jax.linearize(lambda s: gb(x4, s).outputs["A"], s1)[1](s1))
    if not (np.isnan(a[0]) and np.isfinite(a[1])):
        raise Fail(f"linearize: f32 tangent total beyond FLT_MAX not NaN: {a}")
    ii = tm.vmap(lambda t, x: t.add("I", 0, x).add("I", 1, x // 4), outputs={"I": tm.Sum((2,), jnp.int32,
                 fixed_point=True, scale=1.0)}, backend=b)(jnp.full((4,), 1 << 30, jnp.int32))
    if int(ii.stats["fixed_point_overflow"]) != 1 or int(np.asarray(ii.outputs["I"])[1]) != 1 << 30:
        raise Fail(f"int32 total 2^32: overflow {int(ii.stats['fixed_point_overflow'])} (want 1), "
                   f"{np.asarray(ii.outputs['I'])}")
    return (f"exact: bins at {tot[0] / 2 ** 63:+.0f} / {tot[1] / 2 ** 63:+.0f} x 2^63 (primal, jvp, linearize, "
            f"{NB} privatised + {GB} global bins), order- and run-independent; out-of-type totals counted")


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


@case("tm.history(): the last program's closed_over_floats and the kernel records, nothing rendered")
def _(b):
    tabc = jnp.linspace(0.5, 1.5, 32, dtype=F)
    out = {"Q": tm.Sum((1,), F)}
    cap = tm.vmap(lambda t, xi: t.add("Q", 0, xi * tabc[_ix(xi)]), outputs=out, backend=b)
    opr = tm.vmap(lambda t, xi, tb: t.add("Q", 0, xi * tb[_ix(xi)]), outputs=out, in_axes=(0, None), backend=b)
    tm.diagnostics(reset=True)
    jax.block_until_ready(cap(xs).outputs["Q"])
    h = tm.history()
    if set(h) != {"programs", "kernels"}:
        raise Fail(f"history keys {sorted(h)}")
    p = h["programs"][-1]
    if set(p) != {"tag", "closed_over_floats"} or [sh for sh, _, _ in p["closed_over_floats"]] != [(32,)]:
        raise Fail(f"last program record {p}")
    jax.block_until_ready(opr(xs, tabc).outputs["Q"])
    h2 = tm.history()
    if len(h2["programs"]) != 2 or h2["programs"][-1]["closed_over_floats"]:
        raise Fail(f"an operand is listed as closed over: {h2['programs']}")
    d = tm.diagnostics()                 # the same history, the expensive way
    if [r["closed_over_floats"] for r in d["programs"]] != [r["closed_over_floats"] for r in h2["programs"]]:
        raise Fail("history() and diagnostics() disagree on closed_over_floats")
    if [(k["tag"], k["schedule"], k["hash"]) for k in h2["kernels"]] != \
            [(k["tag"], k["schedule"], k["hash"]) for k in d["kernels"]]:
        raise Fail("history() and diagnostics() disagree on the kernels")
    if (b == "auto") != (len(h2["kernels"]) == 2):
        raise Fail(f"kernel records {h2['kernels']}")
    h2["kernels"].clear()
    if b == "auto" and not tm.history()["kernels"]:
        raise Fail("history() returned the library's own list")
    return f"{len(h2['programs'])} programs, {len(tm.history()['kernels'])} kernels: {p['closed_over_floats']}"


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


@case("integer ranges: index normalisation / clamps elided == ranges off (bitwise) == xla", ["auto"])
def _(b):
    """kernel/ranges.py: an index provably inside its bounds (iota, fori_loop counter, clamp, argmax, unsigned
    source, a counter narrowed by its loop predicate) loses JAX's `i < 0 ? i + n : i` and the clamp; an index that
    may be negative (a decreasing counter, a raw input) keeps both.  Arms: ranges.ENABLE = False (every index
    normalised and clamped as before; the cubins must differ: bitwise equal results) and xla (integers bitwise,
    floats close)."""
    ftab = jnp.arange(40).astype(F) * F(0.37) - F(3.0)
    itab = (jnp.arange(24, dtype=jnp.int32) * 7) % 11 - 3

    def item(t, x, key):
        u = (bits(key, 0) % 200).astype(jnp.uint8)
        m = jnp.argmax(jnp.sin(ftab[:12] * x.astype(F)))
        s = ftab[m] + ftab[u % 40] + ftab[jnp.clip(x - 3, 0, 39)]

        def body(i, c):                                          # fori_loop: 0 <= i < 9
            a, w = c
            a = a + itab[i * 2] * itab[9 - 1 - i + 2]             # 9 - 1 - i: decreasing, still >= 0
            return a, w.at[i].set(a)                             # a scatter at a ranged index
        a, w = lax.fori_loop(0, 9, body, (jnp.int32(0), jnp.zeros(9, jnp.int32)))

        def wb(c):                                               # a counter that goes negative: the wrap stays
            k, acc = c
            return k - 1, acc + itab[k] + itab[k % 24]
        k, acc = lax.while_loop(lambda c: c[0] > -5, wb, (x % 7, jnp.int32(0)))
        neg = itab[x - 20] + itab[jnp.minimum(x, 23)]             # x - 20 may be negative: the wrap stays
        sel = jnp.where(jnp.arange(4) < 9, jnp.arange(4) * 3, -1)  # a decided compare -> the case
        t = t.add("Q", (u % NB).astype(jnp.int32), F(1))   # (an exact sum: float atomics have no fixed order)
        return t, (s, a, w, acc, k, neg, sel.sum() + itab[jnp.arange(24)[x % 24]] + m)
    ii = jnp.arange(N, dtype=jnp.int32) % 31
    from tallymap.backends.kernel import ranges as RG
    run = lambda bk: V(item, bk)(ii, keys)
    g = lambda a, k: V(item, "auto")(a, k).per_item[0]
    assert_arms_differ(g, g, ii, keys, patch_b=(RG, dict(ENABLE=False)))   # instrument: the ranges fired
    on = run("auto")
    with patched(RG, ENABLE=False):
        off = run("auto")
    for a, c in zip(jax.tree_util.tree_leaves(on), jax.tree_util.tree_leaves(off)):
        if not np.array_equal(np.asarray(a), np.asarray(c)):
            raise Fail("the ranged cuda result differs from the ranges-off cuda result")
    x = run("xla")
    for a, c in zip(jax.tree_util.tree_leaves(on), jax.tree_util.tree_leaves(x)):
        a, c = np.asarray(a), np.asarray(c)
        if a.dtype.kind in "iub" and not np.array_equal(a, c):
            raise Fail("an integer result differs from xla")
    close(on, x, what="vs xla")
    return "bitwise equal to the ranges-off cubin (which differs); integers == xla"


@case("float facts: min / max of proven non-NaN operands (no NaN test; FMNMX without a zero tie) == facts off "
      "(bitwise) == xla", ["auto"])
def _(b):
    """ranges.float_facts: operands from integer conversions, literals, the NaN guard where(x != x, c, x), selects
    and min / max chains lose sj_fmax's NaN test (and are one FMNMX where the -0 / +0 tie cannot occur); operands
    that may be NaN (a table with NaN, inf, -0) keep it.  Arms: ranges.ENABLE = False (the cubins must differ) and
    xla, bitwise (min / max / select / convert are exact)."""
    sp = jnp.array([0.0, -0.0, np.nan, np.inf, -np.inf, 1.5, -2.5, 3.0], F)

    def item(t, x, key):
        u = sp[(bits(key, 0) % 8).astype(jnp.int32)]                       # may be NaN / inf / -0
        xi = (x % 7 - 3).astype(F)                                         # integer -> float: non-NaN
        g = jnp.where(u != u, F(0.25), u)                                  # NaN guard: non-NaN
        a = jnp.maximum(xi, F(0))                                          # literal +0 second: tie possible
        c = jnp.maximum(F(0), xi)                                          # literal +0 first: FMNMX
        d = jnp.minimum(jnp.maximum(g, F(-1)), F(2))                       # clamp of a guarded value
        e = jnp.maximum(u, xi)                                             # may be NaN: sj_fmax
        f = jnp.minimum(jnp.where(x > 3, xi, F(-0.0)), g)                  # select of non-NaN values
        t = t.add("Q", (x % NB).astype(jnp.int32), F(1))
        return t, (a, c, d, e, f, jnp.maximum(-xi, F(-0.0)))
    ii = jnp.arange(N, dtype=jnp.int32) % 17
    from tallymap.backends.kernel import ranges as RG
    run = lambda bk: V(item, bk)(ii, keys)
    g_ = lambda a, k: V(item, "auto")(a, k).per_item[0]
    assert_arms_differ(g_, g_, ii, keys, patch_b=(RG, dict(ENABLE=False)))   # instrument: the facts fired
    rs = [run("auto")]
    with patched(RG, ENABLE=False):
        rs.append(run("auto"))
    rs.append(run("xla"))
    for a, c in zip(jax.tree_util.tree_leaves(rs[0]), jax.tree_util.tree_leaves(rs[1])):
        if np.asarray(a).tobytes() != np.asarray(c).tobytes():
            raise Fail("a min / max result differs (bits) from the facts-off cuda result")
    for a, c in zip(jax.tree_util.tree_leaves(rs[0]), jax.tree_util.tree_leaves(rs[2])):
        a, c = np.asarray(a), np.asarray(c)
        if not (np.array_equal(a, c, equal_nan=a.dtype.kind == "f") and np.array_equal(np.signbit(a), np.signbit(c))
                if a.dtype.kind == "f" else np.array_equal(a, c)):
            raise Fail("a min / max result differs from xla (value, NaN or zero sign)")
    return "bits equal to the facts-off cubin (which differs); values, NaN and zero signs equal to xla"


def _ssa_variants(th):
    """value facts across an in-place write: in each, p = a NaN test of the per-item array a (computed into its own
    array), then a is updated IN PLACE (a is dead after it) to a2 with a NaN in it, then the NaN guard pattern is
    applied to p and a2 -- which is NOT a guard of a2 (p tested the old a), so the NaN must survive the max.  A fact
    keyed on the storage instead of the value claims a2's select non-NaN and the max drops the NaN (1 instead of
    NaN).  The in-place writers: dynamic_update_slice / scatter, a cond destination, a while carry (nested: an item's
    single top-level while runs as step functions)."""
    nan = F(np.nan)
    arr = lambda x: jnp.sin(x * th[0] * jnp.arange(1, 9, dtype=F)) + x * th[1]     # 8 elements: its own array

    def dus(x):
        a = arr(x)
        p = a != a
        a2 = lax.dynamic_update_slice(a, jnp.full((1,), nan, F), (0,))
        return jnp.maximum(F(1), jnp.where(p, F(0), a2))           # where(x != x, c, x) shape, on a stale x

    def scatter(x):
        a = arr(x)
        p = jnp.isfinite(a)
        a2 = a.at[(x * 7).astype(jnp.int32) % 8].set(nan)
        return jnp.maximum(F(1), jnp.where(p, a2, F(0)))           # where(isfinite(x), x, c) shape

    def cond(x):
        a = arr(x)
        p = a != a
        a2 = lax.cond(x > 0.3, lambda u: u.at[2].set(nan), lambda u: u * th[1], a)
        return jnp.maximum(F(1), jnp.where(p, F(0), a2))

    def carry(x):
        def outer(c):
            m, acc = c
            a = arr(x + m.astype(F))
            p = a != a
            a2 = lax.while_loop(lambda d: d[0] < 2, lambda d: (d[0] + 1, d[1].at[d[0] + 3].set(nan)),
                                (jnp.int32(0), a))[1]
            return m + 1, acc + jnp.maximum(F(1), jnp.where(p, F(0), a2))
        return lax.while_loop(lambda c: c[0] < 1, outer, (jnp.int32(0), jnp.zeros(8, F)))[1]
    return {"dus": dus, "scatter": scatter, "cond destination": cond, "while carry": carry}


@case("facts per SSA value: a NaN guard on a value updated in place since is no guard (dus, scatter, cond dest, "
      "while carry) == jax", ["auto", "xla"])
def _(b):
    """float facts (ranges.float_facts) must be facts about a VALUE.  Each variant (_ssa_variants) tests the old
    contents of an array, updates the array in place with a NaN and applies the guard pattern to the new value:
    nothing proves it non-NaN, so maximum(1, .) must give NaN where plain JAX does.  Default options."""
    xs = jnp.linspace(0.1, 1.0, N).astype(F)
    th = jnp.array([0.9, 1.3], F)
    bad = []
    for nm, g in _ssa_variants(th).items():
        def item(t, x, g=g):
            return t.add("Q", 0, x), g(x)
        got = np.asarray(V(item, b)(xs).per_item)
        if b == "auto":                     # instrument: the write was in place (the copying kernel differs)
            from tallymap.backends.kernel import codegen as CG
            assert_arms_differ(V(item, b), V(item, b), xs, patch_b=(CG, {"INPLACE": False}))
        ref = np.asarray(jax.jit(jax.vmap(g))(xs))
        if not np.array_equal(np.isnan(got), np.isnan(ref)):
            bad.append(f"{nm} ({int(np.isnan(ref).sum())} NaN in jax, {int(np.isnan(got).sum())} here)")
        else:
            close(np.nan_to_num(got), np.nan_to_num(ref), what=nm)
    if bad:
        raise Fail("NaN positions differ from jax: " + "; ".join(bad))
    return "all four variants == jax.vmap (NaN positions exactly)"


@case("cuda_math='fast' (opt-in): a different, FMA-contracted kernel; results close to 'approx' and xla", ["auto"])
def _(b):
    """the fast mode compiles its own kernel (multiply-adds contracted, the guarded reciprocal 1 / where(|d| < eps,
    eps, d) a div.approx); its results agree with the default mode
    and xla to float tolerance, forward and jvp.  The CPU target ignores it (IEEE): there the two modes are equal."""
    def item(t, x, key):
        def body(k, c):
            a, v = c
            r = jax.random.uniform(jax.random.fold_in(key, k), (3,), F)
            v = v * F(0.9) + r * x
            d = r[1] - F(0.5)                                            # a guarded reciprocal (fast division)
            inv = F(1) / jnp.where(jnp.abs(d) < F(1e-3), F(1e-3), d)
            return a + jnp.dot(v, r) / (F(1.5) + r[0]) + F(1e-3) * jnp.minimum(jnp.abs(inv), F(1e3)), v
        a, v = lax.fori_loop(0, 8, body, (F(0), jnp.zeros(3, F)))
        t = t.add("Q", (bits(key, 99) % NB).astype(jnp.int32), a)
        return t, (a, v)
    run = lambda bk, **o: V(item, bk, **o)(xs, keys)
    ref = run("auto")
    fast = run("auto", cuda_math="fast")
    close(fast, ref, rtol=1e-4, what="fast vs approx")
    close(fast, run("xla"), rtol=1e-4, what="fast vs xla")
    tj = jax.jvp(lambda s: V(item, "auto", cuda_math="fast")(xs * s, keys).per_item[0], (F(1),), (F(1),))
    rj = jax.jvp(lambda s: V(item, "auto")(xs * s, keys).per_item[0], (F(1),), (F(1),))
    close(tj, rj, rtol=1e-4, what="jvp fast vs approx")
    if GPU:
        g = lambda o: (lambda a, k: V(item, "auto", **o)(a, k).per_item[0])
        from ab_rule import artifacts
        ka, kb = artifacts(g({}), xs, keys)[0], artifacts(g(dict(cuda_math="fast")), xs, keys)[0]
        if ka == kb:
            raise Fail("cuda_math='fast' compiled the same kernel as 'approx'")
        d = tm.diagnostics(g(dict(cuda_math="fast")), xs, keys)["kernels"]
        srcs = [open(k["source"]).read() for k in d if k.get("source") and os.path.exists(k["source"])]
        if srcs and not any("sj_div_approx(" in s for s in srcs):
            raise Fail("the guarded reciprocal did not get the fast division")
        same = all(np.asarray(p).tobytes() == np.asarray(q).tobytes()
                   for p, q in zip(jax.tree_util.tree_leaves(fast.per_item), jax.tree_util.tree_leaves(ref.per_item)))
        return f"close; per-item results bitwise {'equal' if same else 'different'} (FMA rounds differently)"
    return "close (CPU: the mode is ignored)"


@case("staged tables: small broadcast / constant tables read from shared memory == SHARED_TABLES off == xla "
      "(bitwise)", ["auto"])
def _(b):
    """kernel._stage_tables: small tables (a broadcast f32 input, a closed-over int32 constant, a bool table, an odd
    byte count) are copied into shared memory per block and read there; a table over SHARED_TABLE_MAX stays global.
    Arms: kernel.SHARED_TABLES = 0 (the cubins must differ) and xla; gathers and integer arithmetic are exact."""
    import importlib
    KM = importlib.import_module("tallymap.backends.kernel.kernel")
    ftab = jnp.arange(37).astype(F) * F(0.75) - F(4.0)            # broadcast input (in_axes None)
    ctab = jnp.asarray((np.arange(50, dtype=np.int32) * 13) % 29 - 7)        # a closed-over constant
    btab = jnp.asarray(np.arange(19) % 3 == 0)                                # bool
    u8t = jnp.asarray((np.arange(7) * 37 % 251).astype(np.uint8))             # 7 bytes: not a multiple of 4
    big = jnp.asarray(np.arange(4096, dtype=np.int32) * 3)                    # 16 KB: stays in global memory

    def item(t, x, key, tab):
        def body(k, acc):
            i = (bits(key, k) % 37).astype(jnp.int32)
            v = tab[i] * F(2) + ctab[(i + k) % 50].astype(F)
            v = jnp.where(btab[i % 19], v, -v) + u8t[i % 7].astype(F) + big[(i * 97 + k) % 4096].astype(F)
            return acc + v
        acc = lax.fori_loop(0, 6, body, F(0))
        t = t.add("Q", (x % NB).astype(jnp.int32), F(1))
        return t, acc
    ii = jnp.arange(N, dtype=jnp.int32) % 23
    run = lambda bk: V(item, bk, in_axes=(0, 0, None))(ii, keys, ftab)
    g = lambda a, k, tb: V(item, "auto", in_axes=(0, 0, None))(a, k, tb).per_item
    if GPU:                                  # (the CPU target stages nothing: the host has no per-block memory)
        assert_arms_differ(g, g, ii, keys, ftab, patch_b=(KM, dict(SHARED_TABLES=0)))   # instrument: staged
    on = run("auto")
    with patched(KM, SHARED_TABLES=0):
        off = run("auto")
    xl = run("xla")
    for a, c in zip(jax.tree_util.tree_leaves(on), jax.tree_util.tree_leaves(off)):
        if np.asarray(a).tobytes() != np.asarray(c).tobytes():
            raise Fail("the staged-table result differs (bits) from the global-table result")
    close(on, xl, what="vs xla")
    return "bitwise equal to the unstaged cubin (which differs); == xla"


@case("guarded update: jnp.where(p, x.at[i].set(v), x) as one guarded store == codegen.GUARDED off == xla "
      "(bitwise)", ["auto"])
def _(b):
    """codegen.guarded_updates: a stack push under a condition inside a loop (in place), the negated form
    where(p, x, x.at[i].add(v)), a predicate that reads the array itself (evaluated before the store), a
    dynamic_update_slice window, an old array still live after the select (the guarded store goes to a copy), and an
    elementwise mask (not fused).  Arms: GUARDED = False (the copy + whole-array select; the cubins must differ) and
    xla; integer programs, so all bitwise."""
    def item(t, x, key):
        def body(c):
            k, sp, st, acc = c
            b_ = bits(key, k)
            push = (b_ % 3) != 0
            st = jnp.where(push, st.at[jnp.minimum(sp, 15)].set(k * 7 + x), st)           # push under a condition
            sp = jnp.where(push, jnp.minimum(sp + 1, 16), jnp.maximum(sp - 1, 0))
            st = jnp.where((b_ % 5) == 0, st, st.at[(b_ % 16).astype(jnp.int32)].add(3))  # negated, scatter-add
            st = jnp.where(st[0] > st[1], st.at[0].set(st[1]), st)                      # predicate reads st
            acc = acc + st[jnp.maximum(sp - 1, 0)]
            return k + 1, sp, st, acc
        k, sp, st, acc = lax.while_loop(lambda c: c[0] < 11, body,
                                        (jnp.int32(0), jnp.int32(0), jnp.zeros(16, jnp.int32), jnp.int32(0)))
        w = jnp.arange(8, dtype=jnp.int32) * x
        w2 = jnp.where(x % 2 == 0, lax.dynamic_update_slice(w, jnp.array([5, 6, 7], jnp.int32), (x % 6,)), w)
        live = jnp.arange(6, dtype=jnp.int32) + x
        l2 = jnp.where(x > 3, live.at[x % 6].set(-1), live)                            # `live` read again below
        m = jnp.where(jnp.arange(6) < x % 6, live.at[2].set(9), live)                    # elementwise mask
        t = t.add("Q", (acc % NB).astype(jnp.int32), F(1))
        return t, (st, sp, acc, w2, l2 + live, m)
    ii = jnp.arange(N, dtype=jnp.int32) % 13
    from tallymap.backends.kernel import codegen as CG
    run = lambda bk: V(item, bk)(ii, keys)
    g = lambda a, k: V(item, "auto")(a, k).per_item[0]
    assert_arms_differ(g, g, ii, keys, patch_b=(CG, dict(GUARDED=False)))   # instrument: the fusion fired
    rs = [run("auto")]
    with patched(CG, GUARDED=False):
        rs.append(run("auto"))
    rs.append(run("xla"))
    for r in rs[1:]:
        for a, c in zip(jax.tree_util.tree_leaves(rs[0]), jax.tree_util.tree_leaves(r)):
            if not np.array_equal(np.asarray(a), np.asarray(c)):
                raise Fail("the guarded-update result differs from the unfused cuda / xla result")
    return "bitwise equal to the unfused cubin (which differs) and to xla"


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
        saved = (OT._may_overwrite, CG.Emitter.elementwise_self)
        CB._KERNELS.clear()
        _P._EAGER.clear()
        CG.INPLACE = inplace
        if off:     # the one aliasing rule answers only dus / scatter (in place stays on); loop / cond: never
            OT._may_overwrite = lambda em, X, paths: (sys._getframe(1).f_code.co_name in ("_dus", "_scatter")
                                                      and saved[0](em, X, paths))
            CG.Emitter.elementwise_self = lambda self, dst, x: False
        try:
            tm.diagnostics(reset=True)
            r = tm.vmap(item, outputs=out, backend=bk)(xs, keys)
            ks = tm.diagnostics()["kernels"]
        finally:
            OT._may_overwrite, CG.Emitter.elementwise_self = saved
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


@case("while init that is ALSO a loop constant (the body reads the initial value): == copying cuda == xla", ["auto"])
def _(b):
    """one per-item array v is a while loop's initial carry AND a constant its body reads (the single-loop adjoint's
    restart from its initial state has this shape).  The init-alias rule (v3) must not make v the carry storage:
    the body would read the updated carry where it reads v.  Top-level loop and a loop nested in another one (the
    two code paths of the kernel backend's top-level while); the arm codegen.INPLACE = False copies every carry."""
    from tallymap.backends.kernel import codegen as CG

    def item(t, x):
        v = jnp.sin(x * jnp.arange(1, 5, dtype=F))
        v = jnp.where(v > 0.3, v * jnp.cos(v), 2.0 * v)            # a materialised per-item array

        def run(v):
            def body(c):
                k, w = c
                return k + 1, 0.5 * w + v[::-1]                      # reads v, the initial carry
            return lax.while_loop(lambda c: c[0] < 5, body, (jnp.int32(0), v))[1]

        def outer(c):
            m, a = c
            u = v + m.astype(F)
            return m + 1, a + lax.while_loop(lambda c: c[0] < 3, lambda c: (c[0] + 1, c[1] * 0.5 + u), (0, u))[1]
        w1 = run(v)
        w2 = lax.while_loop(lambda c: c[0] < 2, outer, (jnp.int32(0), jnp.zeros(4, F)))[1]
        return t.add("Q", jnp.arange(4), w1), (w1, w2)
    xs = jnp.linspace(0.1, 1.0, N).astype(F)
    out = {"Q": tm.Sum((4,), F)}
    ra = tm.vmap(item, outputs=out, backend="auto")(xs)
    with patched(CG, INPLACE=False):
        rc = tm.vmap(item, outputs=out, backend="auto")(xs)
    rx = tm.vmap(item, outputs=out, backend="xla")(xs)
    val = lambda r: [r.outputs["Q"], *r.per_item]
    close(val(ra), val(rc), 1e-5, 1e-6, "in-place cuda vs copying cuda (the body read an updated init?)")
    close(val(ra), val(rx), 1e-5, 1e-6, "cuda vs xla")
    return "== copying cuda == xla"


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
    if {r["name"] for r in rows} != ours | {"tallymap_emit", "tallymap_spawn", "tallymap_claim"}:
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


def _units(v):
    """float32 values -> their exact values as Python ints in units of 2^-149 (every finite f32 is one)."""
    b = np.asarray(v, np.float32).view(np.uint32).astype(np.int64)
    e, mant = (b >> 23) & 255, b & 0x7FFFFF
    m, p = np.where(e > 0, mant | 0x800000, mant), np.maximum(e - 1, 0)
    return [(-int(mi) if s else int(mi)) << int(pi) for s, mi, pi in zip(b >> 31, m, p)]


def _f32_round(T):
    """the correctly rounded float32 (nearest, ties to even) of the integer T x 2^-149: the exact sum's reference."""
    if T == 0:
        return np.float32(0.0)
    m = abs(T)
    sh = max(m.bit_length() - 24, 0)
    q, r = divmod(m, 1 << sh)
    if sh and (2 * r > (1 << sh) or (2 * r == (1 << sh) and q & 1)):
        q += 1
    with np.errstate(over="ignore"):
        v = np.float32(math.ldexp(float(q), sh - 149))
    return -v if T < 0 else v


def _exact_ref(vals, bins, nbins):
    tot = [0] * nbins
    for u, i in zip(_units(vals), np.asarray(bins)):
        tot[int(i)] += u
    return np.array([_f32_round(t) for t in tot], np.float32)


@case("fixed_point=True float32 on the kernel backend: the correctly rounded exact sum (host integer reference) "
      "bitwise, order-independent", ["auto"])
def _(b):
    n = 1 << 15
    v = jnp.asarray(np.random.default_rng(3).uniform(-3, 3, n), jnp.float32)
    i = jnp.arange(n, dtype=jnp.int32)
    f = tm.vmap(lambda t, x, j: t.add("Q", j % NB, x).add("Q", (j + 1) % NB, x * 3.0),
                outputs={"Q": tm.Sum((NB,), jnp.float32, fixed_point=True)}, backend="auto")
    a1, a2 = np.asarray(f(v, i).outputs["Q"]), np.asarray(f(v[::-1], i[::-1]).outputs["Q"])
    vv = np.asarray(v)
    want = _exact_ref(np.r_[vv, np.asarray(v * 3.0)], np.r_[np.arange(n) % NB, (np.arange(n) + 1) % NB], NB)
    if not (np.array_equal(a1, a2) and np.array_equal(a1, want)):
        raise Fail(f"exact sum: order-independent {np.array_equal(a1, a2)}, == host {np.array_equal(a1, want)}")
    return f"{NB}/{NB} bins bitwise the correctly rounded exact sum (x64 {jax.config.jax_enable_x64})"


@case("fixed_point=True float32 is exact: 65536 x 1e-11 into one bin -- primal, a jvp tangent beside an O(1) primal "
      "(Q32 rounded both to 0), a reverse cotangent")
def _(b):
    out = {"s": tm.Sum((1,), jnp.float32, fixed_point=True)}
    n = 65536
    xs = jnp.full((n,), 1e-11, jnp.float32)
    want = _f32_round(sum(_units(xs)))
    f = tm.vmap(lambda t, x: t.add("s", 0, x), outputs=out, backend=b)
    p = np.asarray(f(xs).outputs["s"])[0]
    g = tm.vmap(lambda t, x, th: t.add("s", 0, 1.0 + th * x), outputs=out, in_axes=(0, None), backend=b)
    pj, tj = jax.jvp(lambda th: g(xs, th).outputs["s"][0], (jnp.float32(0.0),), (jnp.float32(1.0),))
    h = tm.vmap(lambda t, x, tab: t.add("s", 0, tab[0] * x + tab[1]), outputs=out, in_axes=(0, None), backend=b)
    gr = np.asarray(jax.grad(lambda tab: h(xs, tab).outputs["s"][0])(jnp.ones(2, jnp.float32)))
    got = (p, np.float32(tj), np.float32(gr[0]))
    if not all(x == want for x in got) or float(pj) != n or float(gr[1]) != n:
        raise Fail(f"primal / tangent / cotangent {got}, want {want!r}; jvp primal {float(pj)}, d/dtab1 {gr[1]}")
    return f"primal, tangent and cotangent {float(want)!r} (the f32 of the exact sum)"


@case("exact f32 sum: carries and borrows under contention -- mixed signs across 64-bit word boundaries, totals "
      "crossing zero, cancellation to 0 and to a subnormal; bitwise the host reference in any item order, run to run, "
      "for the primal and the jvp tangent, privatised and global bins")
def _(b):
    rng = np.random.default_rng(11)
    GB = 1024                                  # over PRIV_BYTES: a global tally; Q (NB bins) is privatised
    # pairs +v / -v of the same bin (they cancel exactly, in a random order: the running total crosses zero, borrows
    # ripple to the top word) with exponents whose 24-bit piece straddles a 64-bit word (p % 64 > 40), subnormals and
    # values up to 2^127 (|v| >= 2^96 counted); plus small residuals that make the totals; bin NB-2 gets pairs only
    # (total 0), bin NB-1 2^100 - 2^100 + 2^-149 (total the smallest subnormal)
    m = 1 << 14
    e = rng.choice(np.r_[np.arange(42, 65), np.arange(106, 129), np.arange(170, 193), np.arange(234, 255), [0, 0]], m)
    bits = (e.astype(np.uint32) << 23) | rng.integers(0, 1 << 23, m).astype(np.uint32)
    pv = (bits | (rng.integers(0, 2, m).astype(np.uint32) << 31)).view(np.float32)
    pb = rng.integers(0, NB - 1, m)
    rv = (rng.uniform(0.5, 1.0, m) * 2.0 ** rng.integers(-40, 40, m) * rng.choice([-1.0, 1.0], m)).astype(np.float32)
    rb = rng.integers(0, NB - 2, m)
    tiny = np.array([2.0 ** 100, -(2.0 ** 100), 2.0 ** -149], np.float32)
    vals = np.r_[pv, -pv, rv, tiny]
    bins = np.r_[pb, pb, rb, [NB - 1] * 3]
    perm = rng.permutation(len(vals))
    vals, bins = vals[perm], bins[perm]
    gbin = bins * (GB // NB) + 3
    # XLA:CPU flushes subnormal operands to zero (x * s is 0 for a subnormal x there), so on the xla backend on a CPU
    # the emitted values, and so the reference, have the subnormals flushed; the kernel backend keeps them
    ftz = b == "xla" and not GPU
    rvals = np.where(np.abs(vals) < np.float32(2.0 ** -126), np.float32(0.0), vals) if ftz else vals
    wq, wg = _exact_ref(rvals, bins, NB), _exact_ref(rvals, gbin, GB)
    if not (wq[NB - 2] == 0 and wq[NB - 1] == np.float32(0.0 if ftz else 2.0 ** -149) and np.isfinite(wq).all()):
        raise Fail(f"test setup: reference {wq}")
    out = {"Q": tm.Sum((NB,), jnp.float32, fixed_point=True), "G": tm.Sum((GB,), jnp.float32, fixed_point=True)}
    g = tm.vmap(lambda t, x, i, j, s: t.add("Q", i, x * s).add("G", j, x * s), outputs=out, in_axes=(0, 0, 0, None),
                backend=b)
    x, i, j, one = jnp.asarray(vals), jnp.asarray(bins, jnp.int32), jnp.asarray(gbin, jnp.int32), jnp.float32(1.0)

    def chk(nm, r):
        q, gg = np.asarray(r["Q"]), np.asarray(r["G"])
        if not (np.array_equal(q.view(np.uint32), wq.view(np.uint32))
                and np.array_equal(gg.view(np.uint32), wg.view(np.uint32))):
            dq = np.flatnonzero(q.view(np.uint32) != wq.view(np.uint32))
            dg = np.flatnonzero(gg.view(np.uint32) != wg.view(np.uint32))
            raise Fail(f"{nm}: Q bins {dq.tolist()} got {q[dq].tolist()} want {wq[dq].tolist()}; G bins "
                       f"{dg.tolist()[:8]} got {gg[dg[:8]].tolist()} want {wg[dg[:8]].tolist()}")
    r1 = g(x, i, j, one)
    chk("primal", r1.outputs)
    if int(r1.stats["fixed_point_overflow"]) != 0:
        raise Fail(f"fixed_point_overflow {int(r1.stats['fixed_point_overflow'])}")
    chk("primal, items reversed", g(x[::-1], i[::-1], j[::-1], one).outputs)
    chk("primal, rerun", g(x, i, j, one).outputs)
    p, t = jax.jvp(lambda s: g(x, i, j, s).outputs, (one,), (one,))
    chk("jvp primal", p)
    chk("jvp tangent", t)
    if b == "xla":                             # the carry normalisation (xla._xnorm), run many times in the call
        from tallymap.backends import xla as X
        with patched(X, XNORM=64, XNORM32=64):
            chk("primal, normalised every 64 pieces", tm.vmap(
                lambda t, x, i, j, s: t.add("Q", i, x * s).add("G", j, x * s), outputs=out, in_axes=(0, 0, 0, None),
                backend=b, xla_batch_size=4096)(x, i, j, one).outputs)
    return (f"{len(vals)} emits, {NB} privatised + {GB} global bins bitwise the reference (incl. 0 and 2^-149) in 3 "
            "orders and the jvp" + (" (XLA:CPU: subnormal emits flushed to 0 by XLA)" if ftz else ""))


@case("exact f32 reverse cotangents of broadcast operands (sparse table and dense parameter): carries / borrows, "
      "cancellation, subnormals; bitwise the host reference in any item order, run to run, normalised and chunked "
      "(the xla backend without x64 keeps int32 words of 8-bit digits)")
def _(b):
    rng = np.random.default_rng(12)
    TB = 64                                    # >= _reverse.SPARSE_MIN: a sparse table (its reads emit cotangents)
    m = 1 << 12
    e = rng.choice(np.r_[np.arange(1, 30), np.arange(100, 140), np.arange(200, 255), [0, 0]], m)
    bits_ = (e.astype(np.uint32) << 23) | rng.integers(0, 1 << 23, m).astype(np.uint32)
    pv = (bits_ | (rng.integers(0, 2, m).astype(np.uint32) << 31)).view(np.float32)
    pb = rng.integers(0, TB - 2, m)
    rv = (rng.uniform(0.5, 1.0, m) * 2.0 ** rng.integers(-40, 40, m) * rng.choice([-1.0, 1.0], m)).astype(np.float32)
    tiny = np.array([2.0 ** 100, -(2.0 ** 100), 2.0 ** -149], np.float32)
    vals = np.r_[pv, -pv, rv, tiny]
    bins = np.r_[pb, pb, rng.integers(0, TB - 2, m), [TB - 1] * 3]
    perm = rng.permutation(len(vals))
    vals, bins = vals[perm], bins[perm]
    ftz = b == "xla" and not GPU               # (XLA:CPU flushes subnormal operands: x * ct is 0 there)
    rvals = np.where(np.abs(vals) < np.float32(2.0 ** -126), np.float32(0.0), vals) if ftz else vals
    want = _exact_ref(rvals, bins, TB)
    want2 = _exact_ref(np.r_[rvals, rvals], np.r_[np.zeros(len(vals), int), np.ones(len(vals), int)], 2)
    if not (want[TB - 2] == 0 and want[TB - 1] == np.float32(0.0 if ftz else 2.0 ** -149)):
        raise Fail(f"test setup: reference {want[TB - 2:]}")

    def grads(x, i, **kw):
        f = tm.vmap(lambda t, x, i, tab, p: t.add("s", 0, x * tab[i] + x * p[0] + x * p[1]),
                    outputs={"s": tm.Sum((1,), jnp.float32)}, in_axes=(0, 0, None, None), backend=b, **kw)
        g = jax.grad(lambda tab, p: f(x, i, tab, p).outputs["s"][0], argnums=(0, 1))(
            jnp.ones(TB, jnp.float32), jnp.ones(2, jnp.float32))
        return np.asarray(g[0]), np.asarray(g[1])

    x, i = jnp.asarray(vals), jnp.asarray(bins, jnp.int32)
    runs = [("in order", grads(x, i)), ("rerun", grads(x, i)), ("items reversed", grads(x[::-1], i[::-1])),
            ("batches of 1000", grads(x, i, xla_batch_size=1000) if b == "xla" else grads(x, i))]
    if b == "xla":                             # many normalisations, and emits cut into chunks (_xfmt period / K)
        from tallymap.backends import xla as X
        with patched(X, XNORM=64, XNORM32=64):
            runs.append(("normalised every 64 pieces", grads(x, i, xla_batch_size=2048)))
    for nm, (gt, gp) in runs:
        if not (np.array_equal(gt.view(np.uint32), want.view(np.uint32))
                and np.array_equal(gp.view(np.uint32), want2.view(np.uint32))):
            d = np.flatnonzero(gt.view(np.uint32) != want.view(np.uint32))
            raise Fail(f"{nm}: table bins {d.tolist()[:8]} got {gt[d[:8]].tolist()} want {want[d[:8]].tolist()}; "
                       f"param {gp.tolist()} want {want2.tolist()}")
    return f"{len(vals)} contributions: {TB} table + 2 parameter cotangents bitwise the reference in {len(runs)} runs"


def _x64_both(fn):
    """fn() with jax_enable_x64 off, then on -> (its leaves as numpy, off; on); the setting is restored."""
    x0 = jax.config.jax_enable_x64
    out = []
    try:
        for on in (False, True):
            jax.config.update("jax_enable_x64", on)
            out.append([np.asarray(v) for v in jax.tree_util.tree_leaves(fn())])
    finally:
        jax.config.update("jax_enable_x64", x0)
    return out


def _same_bits(a, b_):
    """the same values bit for bit (NaN: NaN in both, any payload)."""
    a, b_ = np.asarray(a), np.asarray(b_)
    if a.shape != b_.shape or a.dtype != b_.dtype:
        return False
    if np.issubdtype(a.dtype, np.floating) or a.dtype == jnp.bfloat16:
        na, nb = np.isnan(a.astype(np.float32)), np.isnan(b_.astype(np.float32))
        u = {2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize]
        return bool(np.array_equal(na, nb) and np.array_equal(a[~na].view(u), b_[~nb].view(u)))
    return bool(np.array_equal(a, b_))


@case("exact Sum without x64 on the xla backend: bitwise the x64 result, the kernel backend's and the host reference "
      "-- primal, jvp tangent, grad and a Records field; carries, cancellation, subnormals, a non-finite emit and a "
      "total past FLT_MAX (NaN, counted the same)", ["xla"])
def _(b):
    rng = np.random.default_rng(21)
    m = 1 << 12
    e = rng.choice(np.r_[np.arange(0, 30), np.arange(100, 140), np.arange(200, 255)], m)
    bits_ = (e.astype(np.uint32) << 23) | rng.integers(0, 1 << 23, m).astype(np.uint32)
    pv = (bits_ | (rng.integers(0, 2, m).astype(np.uint32) << 31)).view(np.float32)
    pb = rng.integers(0, NB - 3, m)
    rv = (rng.uniform(0.5, 1.0, m) * 2.0 ** rng.integers(-40, 40, m) * rng.choice([-1.0, 1.0], m)).astype(np.float32)
    # bin NB-3: 2^100 - 2^100 + 2^-149 (the smallest subnormal); NB-2: an inf emit; NB-1: 3 x 3e38 (past FLT_MAX)
    sv = np.array([2.0 ** 100, -(2.0 ** 100), 2.0 ** -149, np.inf, 3e38, 3e38, 3e38], np.float32)
    sb = np.array([NB - 3] * 3 + [NB - 2] + [NB - 1] * 3)
    fv, fb = np.r_[pv, -pv, rv, sv[:3]], np.r_[pb, pb, rng.integers(0, NB - 3, m), sb[:3]]     # (finite bins only)
    vals, bins = np.r_[fv, sv[3:]], np.r_[fb, sb[3:]]
    perm = rng.permutation(len(vals))
    vals, bins = vals[perm], bins[perm].astype(np.int32)
    out = {"Q": tm.Sum((NB,), jnp.float32, fixed_point=True)}
    want = _exact_ref(fv, fb, NB)
    want[NB - 2:] = np.nan
    # XLA:CPU flushes subnormal operands in its arithmetic (x * th): the reference of an item that multiplies
    ftz = not GPU
    fz = lambda v: np.where(np.abs(v) < np.float32(2.0 ** -126), np.float32(0.0), v) if ftz else v   # noqa: E731
    wantz = _exact_ref(fz(fv), fb, NB)
    wantz[NB - 2:] = np.nan
    prim = lambda bk, **kw: tm.vmap(lambda t, x, i: t.add("Q", i, x), outputs=out, backend=bk, **kw)  # noqa: E731
    run = lambda bk, **kw: (lambda r: (r.outputs["Q"], r.stats["fixed_point_overflow"]))(  # noqa: E731
        prim(bk, **kw)(vals, bins))
    g = tm.vmap(lambda t, x, i, th: t.add("Q", i, x * th), outputs=out, in_axes=(0, 0, None), backend=b)
    one = np.float32(1.0)
    w = np.r_[np.ones(NB - 2, np.float32), np.zeros(2, np.float32)]
    fin = np.isfinite(vals)
    gx, gi = vals[np.abs(vals) < 1e38], bins[np.abs(vals) < 1e38]             # (grad: no inf * 0)
    rec = tm.vmap(lambda t, x, th: t.append("r", {"q": x * th}), in_axes=(0, None),
                  outputs={"r": tm.Records({"q": jnp.float32}, capacity=1 << 14)}, backend=b)
    sel = (np.arange(1 << 14) < fin.sum()).astype(np.float32)
    off, on = _x64_both(lambda: (
        run(b), run(b, xla_batch_size=1000),
        jax.jvp(lambda th: g(vals, bins, th).outputs["Q"], (one,), (one,)),
        jax.grad(lambda th: jnp.sum(g(gx, gi, th).outputs["Q"] * w))(one),
        jax.jvp(lambda th: rec(vals[fin], th).outputs["r"]["q"], (one,), (one,)),
        jax.grad(lambda th: jnp.sum(rec(vals[fin], th).outputs["r"]["q"] * sel))(one)))
    names = ["primal", "overflow count", "primal, batches of 1000", "its count", "jvp primal", "jvp tangent", "grad",
             "Records q", "Records q tangent", "Records grad"]
    for nm, x, y in zip(names, off, on):
        if not _same_bits(x, y):
            raise Fail(f"{nm}: x64 off {x.ravel()[:8]} != on {y.ravel()[:8]}")
    if not (_same_bits(off[0], want) and int(off[1]) == 2):
        raise Fail(f"primal {off[0]} != the exact sums {want}, or overflow count {int(off[1])} != 2")
    if not (_same_bits(off[5], wantz) or _same_bits(off[5], want)):
        raise Fail(f"jvp tangent {off[5]} != the exact sums {want}")
    gw = _exact_ref(fz(gx[gi < NB - 2]), np.zeros(int((gi < NB - 2).sum()), int), 1)[0]
    if not _same_bits(off[6], np.float32(gw)):
        raise Fail(f"grad {off[6]!r} != the exact sum {gw!r}")
    from tallymap.backends import xla as X
    for pb in (1 << 30, 0):                           # the carry step fused / a word row at a time (xla.FUSED_WORDS)
        with patched(X, XNORM32=64, FUSED_WORDS=pb):
            z = _x64_both(lambda: run(b, xla_batch_size=2048))[0]
        if not (_same_bits(z[0], want) and int(z[1]) == 2):
            raise Fail(f"normalised every 64 values (FUSED_WORDS {pb}): {z[0]} != {want}")
    msg = "x64 off == on bitwise (primal, 2 batchings, jvp, grad, Records q / tangent / grad), == the host reference"
    if "auto" in BACKENDS:
        k = [np.asarray(v) for v in run("auto")]
        if not (_same_bits(k[0], off[0]) and int(k[1]) == int(off[1])):
            raise Fail(f"kernel backend {k[0]} != xla without x64 {off[0]}")
        msg += ", == the kernel backend"
    return msg


@case("fixed_point=True float16 / bfloat16 / integer outputs without x64 on the xla backend: bitwise the x64 result "
      "(values, NaN bins, stats; double rounding, carries, dropped emits, totals past the type) and the kernel "
      "backend's; a scale that is not a power of two raises without x64", ["xla"])
def _(b):
    rng = np.random.default_rng(22)
    n = 4096
    fl = lambda lo, hi: (rng.uniform(-1, 1, n) * 2.0 ** rng.integers(lo, hi, n))   # noqa: E731
    cases = [(jnp.float16, 2.0 ** 20, fl(-30, 16)), (jnp.float16, 2.0 ** 40, fl(-10, 16)),
             (jnp.bfloat16, 2.0 ** 10, np.r_[fl(-60, 52)[:-8], [1e30, -1e30, 3e38, 2.0 ** 52] * 2]),
             (jnp.bfloat16, 2.0 ** -3, fl(-20, 30)), (jnp.int32, 1.0, rng.integers(-2 ** 31, 2 ** 31, n)),
             (jnp.int32, 2.0 ** 30, rng.integers(-2 ** 31, 2 ** 31, n)),
             (jnp.int32, 2.0 ** 4, rng.integers(-99, 99, n)),
             (jnp.int16, 2.0 ** -2, rng.integers(-2 ** 15, 2 ** 15, n)), (jnp.uint32, 1.0, rng.integers(0, 2 ** 32, n)),
             (jnp.uint32, 2.0 ** -1, rng.integers(0, 2 ** 20, n)), (jnp.int8, 1.0, rng.integers(-128, 128, n))]
    done = []
    for dt, sc, v in cases:
        x = np.asarray(v).astype(dt)
        i = rng.integers(0, NB, n).astype(np.int32)
        i[: n // 2] = 0                                # (bin 0: half the emits: totals past the type, carries)
        f = tm.vmap(lambda t, x, i: t.add("Q", i, x), outputs={"Q": tm.Sum((NB,), dt, fixed_point=True, scale=sc)},
                    backend=b)
        r = lambda f=f, x=x, i=i: (lambda o: (o.outputs["Q"], o.stats["fixed_point_overflow"]))(f(x, i))  # noqa
        off, on = _x64_both(r)
        nm = f"{np.dtype(dt).name} scale 2^{int(np.log2(sc))}"
        if not (_same_bits(off[0], on[0]) and int(off[1]) == int(on[1])):
            d = np.flatnonzero(off[0] != on[0])
            raise Fail(f"{nm}: x64 off {off[0][d]} != on {on[0][d]} (bins {d.tolist()}), overflow {int(off[1])} "
                       f"vs {int(on[1])}")
        fk = None
        if "auto" in BACKENDS:
            try:
                fk = tm.vmap(lambda t, x, i: t.add("Q", i, x), backend="auto",
                             outputs={"Q": tm.Sum((NB,), dt, fixed_point=True, scale=sc)})(x, i)
            except Exception as e:                    # (float16 / bfloat16: not kernel dtypes)
                if "not supported by the kernel backend" not in str(e):
                    raise
        if fk is not None:
            # an integer bin whose total does not fit the type is counted on both backends, its value unspecified
            # (xla saturates, the kernel's double -> int conversion does not): compare the bins that fit
            m = np.ones(NB, bool)
            if np.issubdtype(np.dtype(dt), np.integer):
                q = np.round(np.asarray(x, np.float64) * sc)              # (exact: a power-of-two scale)
                tot = [sum(int(u) for u in q[i == k]) / sc for k in range(NB)]
                ii = np.iinfo(dt)
                m = np.array([ii.min <= s < ii.max + 1 for s in tot])
            ko = np.asarray(fk.outputs["Q"])
            if not (_same_bits(ko[m], off[0][m]) and int(fk.stats["fixed_point_overflow"]) == int(off[1])):
                raise Fail(f"{nm}: kernel {np.asarray(fk.outputs['Q'])} != xla {off[0]}, overflow "
                           f"{int(fk.stats['fixed_point_overflow'])} vs {int(off[1])}")
        done.append(f"{nm} ({int(off[1])} counted{'' if fk is not None else ', no kernel'})")
    x0 = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        raises(NotImplementedError, lambda: tm.vmap(lambda t, x: t.add("Q", 0, x), backend=b, outputs={
            "Q": tm.Sum((1,), jnp.float16, fixed_point=True, scale=1000.0)})(np.ones(4, np.float16)), "power-of-two")
    finally:
        jax.config.update("jax_enable_x64", x0)
    return "x64 off == on" + (" == kernel" if "auto" in BACKENDS else "") + ": " + ", ".join(done)


@case("signed-zero ties and NaN of lax.max / lax.min (f32; f64 under x64), operands unknown or proven non-NaN: bitwise "
      "XLA's own jnp.maximum / jnp.minimum (+0 for max(-0, +0), -0 for min(+0, -0), NaN propagates)")
def _(b):
    done = []
    for dt in [jnp.float32] + ([jnp.float64] if jax.config.jax_enable_x64 else []):
        a = np.array([-0.0, 0.0, -0.0, 0.0, np.nan, 1.0, np.nan, -0.0, 3.0], dt)
        c = np.array([0.0, -0.0, -0.0, 0.0, 1.0, np.nan, np.nan, 2.0, 3.0], dt)

        def ops(x, y, dt=dt):
            gx, gy = jnp.where(x != x, dt(0), x), jnp.where(y != y, dt(0), y)     # NaN guards: proven non-NaN
            return (jnp.maximum(x, y), jnp.minimum(x, y), jnp.maximum(gx, gy), jnp.minimum(gx, gy),
                    jnp.maximum(gx, dt(-0.0)), jnp.minimum(dt(0.0), gy), jnp.maximum(dt(-0.0), gy))
        got = tm.vmap(lambda t, x, y: (t.add("Q", 0, dt(1)), ops(x, y)), outputs={"Q": tm.Sum((1,), dt)},
                      backend=b)(a, c).per_item
        want = jax.jit(jax.vmap(ops))(a, c)
        for k, (g, w) in enumerate(zip(got, want)):
            if not _same_bits(g, w):
                raise Fail(f"{np.dtype(dt).name} op {k}: {np.asarray(g)} != XLA {np.asarray(w)} (signbits "
                           f"{np.signbit(np.asarray(g)).astype(int)} vs {np.signbit(np.asarray(w)).astype(int)})")
        done.append(np.dtype(dt).name)
    return f"{', '.join(done)}: 7 max / min forms x 9 operand pairs bitwise XLA"


@case("exact f32 sum: device memory per bin on the kernel backend is 40 B of accumulator + 1 flag bit + the output",
      ["auto"])
def _(b):
    B = 1 << 20
    vm = tm.vmap(lambda t, x, i: t.add("h", i, x), outputs={"h": tm.Sum((B,), jnp.float32, fixed_point=True)},
                 backend="auto")
    x, i = jnp.ones(4096, jnp.float32), (jnp.arange(4096, dtype=jnp.int32) * 997) % B
    ma = jax.jit(lambda x, i: vm(x, i).outputs["h"]).lower(x, i).compile().memory_analysis()
    if ma is None:
        return "skipped (no memory analysis on this platform)"
    tmp, out = int(ma.temp_size_in_bytes), int(ma.output_size_in_bytes)
    if not (out == 4 * B and 40 * B <= tmp <= 40 * B + B // 8 + (1 << 16)):
        raise Fail(f"{B} bins: output {out} B (want {4 * B}), scratch {tmp} B (want 40 B / bin + flag bits: "
                   f"{40 * B + B // 8})")
    return f"{B} bins: output {out / B:.2f} B / bin, scratch {tmp / B:.3f} B / bin"


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
    # the weekly nightly cell runs a JAX DEV build, which is newer than every tested version by construction; any
    # other JAX newer than TESTED_JAX means a release the matrix does not list yet
    if st != "ok" and not (st == "newer" and "dev" in jax.__version__):
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


@case("an array element read, then the array updated, then the read used: updated in place, not copied", ["auto"])
def _(b):
    # x = a[j]; a = a.at[k].set(f(x)); acc += x -- the read is a lazy value still live at the update, so before the
    # fix the emitter copied the whole 512-element carry twice per loop step to keep x's old value
    S = 512

    def one(x, k0):
        def body(c):
            k, a, acc = c
            j = (k * 37 + k0) % S
            r = a[j]
            return k + 1, a.at[(j * 11 + 5) % S].set(r * 0.5 + x), acc + r
        _, a, acc = lax.while_loop(lambda c: c[0] < 64, body, (0, jnp.arange(S, dtype=F) * x, F(0.0)))
        return acc + a[3]

    def item(t, x, k0):
        y = one(x, k0)
        return t.add("Q", 0, y), y
    xs_ = jnp.linspace(0.5, 1.5, 64).astype(F)
    ks = jnp.arange(64, dtype=jnp.int32) % 13
    tm.diagnostics(reset=True)
    r = tm.vmap(item, outputs={"Q": tm.Sum((1,), F)}, backend=b)(xs_, ks)
    close(r.per_item, jax.jit(jax.vmap(one))(xs_, ks), what="per-item")
    ks_ = tm.diagnostics()["kernels"]
    src = open(ks_[-1]["source"]).read()
    n = len(re.findall(r"\b(?:float|double) \w+\[512\];", src))
    if n != 1:
        raise Fail(f"{n} {S}-element arrays declared (want 1: the carry, updated in place)")
    return "one 512-element array (the carry, updated in place)"


@case("a loop carry rebuilt from OTHER elements of itself (an elimination): two buffers swapped, no copy back",
      ["auto"])
def _(b):
    # each pivot step reads row p and column p of the old matrix, so its new value cannot be written in place:
    # before, every iteration computed it into a fresh array and copied that back into the carry.  (24 x 26: larger
    # than a thread's 255 registers, the carries the rule applies to)
    R_, C_ = 24, 26

    def elim(x, k0, n):
        A = (jnp.arange(R_ * C_, dtype=F).reshape(R_, C_) % 7 + 1.0) * x + jnp.eye(R_, C_, dtype=F) * (5.0 + k0)

        def piv(p, A):
            col = A[:, p] / A[p, p]
            return A - jnp.where(jnp.arange(R_)[:, None] == p, 0.0, col[:, None] * A[p, :][None, :])
        A = lax.fori_loop(0, n, piv, A)                       # a scan: static trip count
        _, A = lax.while_loop(lambda c: c[0] < k0, lambda c: (c[0] + 1, c[1] * 0.5 + c[1][(c[0] + 1) % R_][None, :]),
                              (0, A))                          # a while loop: run-time trip count 0..4
        return jnp.sum(A ** 2)

    xs_ = jnp.linspace(0.5, 1.5, 64).astype(F)
    ks = (jnp.arange(64) % 5).astype(F)
    out = []
    for n in (24, 23):                                         # even and odd trip counts
        def item(t, x, k0, n=n):
            y = elim(x, k0, n)
            return t.add("Q", 0, y), y
        want = jax.jit(jax.vmap(lambda x, k: elim(x, k, n)))(xs_, ks)

        def copies(item=item, want=want):
            tm.diagnostics(reset=True)
            r = tm.vmap(item, outputs={"Q": tm.Sum((1,), F)}, backend=b)(xs_, ks)
            close(r.per_item, want, rtol=1e-5, what=f"per-item, {n} pivots")
            src = open(tm.diagnostics()["kernels"][-1]["source"]).read()
            # element copies inside a loop between two arrays at the same index (a copy back)
            return len(re.findall(r"^\s*(\w+)\[(.+)\] = (\w+)\[\2\];$", src, re.M))
        n_in = copies()
        from tallymap.backends.kernel import codegen as CG
        with patched(CG, INPLACE=False):
            n_cp = copies()
        # one copy remains: after the while loop, when its last value is in the second buffer (odd trips; an odd
        # scan adds one more, which this count cannot tell from a copy back: that case is checked for its values)
        if n % 2 == 0 and not (n_in <= 1 and n_in < n_cp):
            raise Fail(f"{n} pivots: {n_in} array copy statements with in-place updates on, {n_cp} off: the carry "
                       "is still copied back every iteration")
        out.append(f"{n} pivots: {n_in} copies vs {n_cp}")
    return "; ".join(out)


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


@case("kernel cache temp names: two nodes with the same pid get different build directories (the cache dir is shared)",
      ["xla"])
def _(b):
    import tempfile

    from tallymap.backends.kernel import compile as C
    gethost, getpid = C.socket.gethostname, C.os.getpid
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "k_0123.so")
        try:
            C.os.getpid = lambda: 4242
            names = {}
            for host in ("nodeA", "nodeB"):
                C.socket.gethostname = lambda host=host: host
                names[host] = C.tmp_dir(p)
            again = C.tmp_dir(p)                               # nodeB, same pid again
        finally:
            C.socket.gethostname, C.os.getpid = gethost, getpid
        if len({names["nodeA"], names["nodeB"], again}) != 3:
            raise Fail(f"temp names collide: {names}, again {again}")
        for host, n in [*names.items(), ("nodeB", again)]:
            if not (os.path.isdir(n) and os.path.dirname(n) == d and os.path.basename(n).startswith(".k_0123.so.tmp")
                    and host in n and "4242" in n):
                raise Fail(f"{host}: {n!r} is not a new directory .<name>.tmp... beside the path, naming the host "
                           "and pid")
    return f"{os.path.basename(names['nodeA'])} != {os.path.basename(names['nodeB'])}"


@case("host kernel cache: a missing -fstack-usage file is an error, a cached .so without its .su is rebuilt, and "
      "another node's .su for the same kernel is never taken", ["xla"])
def _(b):
    import subprocess
    import tempfile

    from tallymap.backends.kernel import compile as C
    from tallymap.backends.kernel import host as H
    calls, write_su = [], [False]

    def fake_cxx(cmd, **kw):                                   # writes -o <tmp>, and the .su as GCC 11 names it
        calls.append(cmd)
        out, cc = cmd[cmd.index("-o") + 1], cmd[-3]
        open(out, "w").close()
        if write_su[0]:
            with open(f"{out}-{os.path.basename(cc)[:-3]}.su", "w") as f:
                f.write("k.cc:1:6:void sj_main()\t64\tstatic\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    saved = (H.compiler_identity, C.host_cxx, H.stack_limit, H.subprocess.run, os.environ.get("TALLYMAP_CACHE"))
    try:
        os.environ["TALLYMAP_CACHE"] = tempfile.mkdtemp(prefix="tm_su_", dir=os.environ.get("LSCRATCH") or None)
        H.compiler_identity, C.host_cxx, H.stack_limit = (lambda: "fake"), (lambda: "fake-c++"), (lambda: 1 << 20)
        H.subprocess.run = fake_cxx
        src = "// edge: missing .su\n"
        raises(RuntimeError, lambda: H.compile_so(src), "-fstack-usage")          # no .su written: not skipped
        write_su[0] = True
        n0 = len(calls)
        _, info = H.compile_so(src)                            # the .so from above is cached, its .su is not
        if len(calls) != n0 + 1 or info["cached"] or info["stack"] != 64:
            raise Fail(f"a cached .so without its .su: compiles {len(calls) - n0}, info {info}")
        _, info2 = H.compile_so(src)
        if len(calls) != n0 + 1 or not info2["cached"] or info2["stack"] != 64:
            raise Fail(f"with both files cached: compiles {len(calls) - n0 - 1}, info {info2}")
        tmp = calls[-1][calls[-1].index("-o") + 1]
        if C.socket.gethostname() not in tmp:
            raise Fail(f"the host kernel's temp name {tmp!r} does not name the host")
        # another node building the same kernel at the same time: its .su (here a huge bound) sits in the cache dir
        # while this build runs; this build must take only its own .su and leave the other one alone
        other = []

        def racing_cxx(cmd, **kw):
            h = os.path.basename(cmd[-3])[2:-3]
            other.append(os.path.join(os.path.dirname(cmd[-3]), f"k_{h}.so.tmp.nodeB.4242.00000000-k_{h}.su"))
            with open(other[0], "w") as f:
                f.write("k.cc:1:6:void sj_main()\t999999999\tstatic\n")
            return fake_cxx(cmd, **kw)
        H.subprocess.run = racing_cxx
        _, info3 = H.compile_so("// edge: a concurrent build on another node\n")
        if info3["stack"] != 64 or not os.path.exists(other[0]):
            raise Fail(f"took another build's .su: stack {info3['stack']}, its file left {os.path.exists(other[0])}")
    finally:
        H.compiler_identity, C.host_cxx, H.stack_limit, H.subprocess.run, old = saved
        if old is None:
            os.environ.pop("TALLYMAP_CACHE", None)
        else:
            os.environ["TALLYMAP_CACHE"] = old
    return "missing .su raises; .so without .su rebuilt; both cached -> no compile; another build's .su left alone"


@case("cuda frame check: a cubin with no readable frame record is an error, not a pass", ["xla"])
def _(b):
    # NVRTC writes a frame record for every function, a 0-byte frame included (sm_75..sm_90), so no record at all
    # means the cubin was not read; every real kernel compiled by the cuda cells passes through this same check
    from tallymap.backends.kernel import compile as C
    elf_no_sections = b"\x7fELF\x02\x01\x01" + bytes(0x40 - 7)
    for what, cub in (("not an ELF", b"not a cubin"), ("an ELF64 with no sections", elf_no_sections)):
        if C.frame_bytes(cub) != {}:
            raise Fail(f"{what}: frame_bytes {C.frame_bytes(cub)}")
        raises(RuntimeError, lambda cub=cub: C.check_frame(cub), "frame size")
    return "no record -> RuntimeError"

# ------------------------------------------------------------------ destination aliasing (op_table._may_overwrite)
# Every in-place write of the kernel backend (dus / scatter into the operand, a while carry in its initial value's
# storage, a cond output in an operand's storage) asks one rule: a storage may be overwritten only if no read of
# its old contents remains, on any path.  Each case builds one hazard of that rule around a per-item array and
# checks, on the kernel backend: the in-place kernel compiles to a different program than the copying one
# (codegen.INPLACE = False; else the comparison is void) and equals it and xla forward; and the jvp, jax.vmap of a
# jvp over ONE tangent (the lane program that hid the cond slot bug) and over two, and the grad (the adjoint's
# replays) equal xla's.
def _alias_arms(item):
    from tallymap.backends.kernel import codegen as CG
    xs = jnp.linspace(0.1, 1.0, N).astype(F)
    th = jnp.array([0.9, 1.3], F)
    tv = jnp.array([0.7, -0.4], F)
    wq = 1.0 + 0.25 * jnp.arange(8, dtype=F)
    out = {"Q": tm.Sum((8,), F)}
    f = {bk: tm.vmap(item, outputs=out, in_axes=(0, None), backend=bk) for bk in ("auto", "xla")}
    assert_arms_differ(f["auto"], f["auto"], xs, th, patch_b=(CG, {"INPLACE": False}))
    val = lambda r: [r.outputs["Q"], r.per_item]
    ra = val(f["auto"](xs, th))
    with patched(CG, INPLACE=False):
        rc = val(f["auto"](xs, th))
    rx = val(f["xla"](xs, th))
    close(ra, rc, 1e-5, 1e-6, "in-place vs copying kernel")
    close(ra, rx, 1e-4, 1e-5, "kernel vs xla")

    def L(bk):
        def g(th):
            r = f[bk](xs, th)
            return jnp.sum(r.outputs["Q"] * wq) + jnp.sum(r.per_item)
        return g
    d = {}
    for bk in ("auto", "xla"):
        g = L(bk)
        d[bk] = dict(jvp=jax.jvp(g, (th,), (tv,))[1],
                     vmap1=jax.vmap(lambda t: jax.jvp(g, (th,), (t,))[1])(tv[None])[0],
                     vmap2=jax.vmap(lambda t: jax.jvp(g, (th,), (t,))[1])(jnp.stack([tv, -0.5 * tv])),
                     grad=jax.jit(jax.grad(g))(th))
    close(d["auto"]["vmap1"], d["auto"]["jvp"], 1e-4, 1e-5, "kernel: vmap of one tangent vs the plain jvp")
    for k in d["auto"]:
        close(d["auto"][k], d["xla"][k], 1e-4, 1e-5, f"kernel vs xla {k}")
    return "in-place == copying == xla; jvp, vmap of 1 / 2 tangents, grad == xla"


def _arr(x, th):
    """a per-item array in a Local of its own (a scatter into a lazy value is a copy)"""
    return jnp.sin(x * th[0] * jnp.arange(1, 9, dtype=F)).at[0].set(x * th[1])


@case("alias: cond with one array in two operand slots and two outputs of its shape", ["auto"])
def _(b):
    """one dead array fills both operand slots; both outputs have its shape.  At most one output may be written
    into its storage (0a784c6: the rule counted slots, so both were, and the second overwrote the first)."""
    def item(t, x, th):
        a = _arr(x, th)
        o1, o2 = lax.cond(x > 0.5, lambda u, v: (u * 2.0, v + th[0]), lambda u, v: (v - th[1], u * th[0]), a, a)
        return t.add("Q", jnp.arange(8), o1 + 3.0 * o2), jnp.sum(o1 * o2)
    return _alias_arms(item)


@case("alias: cond swapping two dead operands (parallel destination writes)", ["auto"])
def _(b):
    """(a, c) -> (c, a) in one branch, element-wise updates in the other: each output's own operand is a valid
    destination alone, but in the swapping branch each result reads the other's destination."""
    def item(t, x, th):
        a, c = _arr(x, th), _arr(0.5 * x + 0.1, th)
        o1, o2 = lax.cond(x > 0.4, lambda u, v: (v, u), lambda u, v: (u * th[0], v * th[1]), a, c)
        return t.add("Q", jnp.arange(8), o1 - 2.0 * o2), jnp.sum(o1 * jnp.arange(8, dtype=F)) + jnp.sum(o2)
    return _alias_arms(item)


@case("alias: while with one array as two initial carries and a body constant", ["auto"])
def _(b):
    """v starts two carries and is read by the body every iteration (the single-loop adjoint's restart from its
    initial state has this shape, 2e83400): v may be neither carry's storage.  Nested in a loop (an item's single
    top-level while runs as step functions, where nothing is aliased)."""
    def item(t, x, th):
        v = _arr(x, th)

        def outer(c):
            m, acc = c

            def body(c):
                k, p, q = c
                return k + 1, 0.5 * p + v[::-1] * th[0], q * th[1] - 0.25 * p
            _, p, q = lax.while_loop(lambda c: c[0] < 4, body, (jnp.int32(0), v, v))
            return m + 1, acc + p - q
        acc = lax.while_loop(lambda c: c[0] < 2, outer, (jnp.int32(0), jnp.zeros(8, F)))[1]
        return t.add("Q", jnp.arange(8), acc), jnp.sum(acc * acc)
    return _alias_arms(item)


@case("alias: loop constant that is a lazy view of a per-item array (frozen by storage)", ["auto"])
def _(b):
    """a scan (and a while nested in a while) re-reads its constant w, an identity view of the uint32 array U
    (random_key_data of random_wrap: it reads U at its own index), every step; a loop nested in the body starts
    from w.  U's storage must not become that loop's carry: freezing w freezes the storage it reads.  (7562c37:
    frozen frames held Local NAMES only, the lazy-init rule made U the carry, and the next step read the updated U:
    the in-place kernel was 59% / 76% off.)"""
    def nested(w):
        return lax.while_loop(lambda d: d[0] < 3, lambda d: (d[0] + 1, d[1] * jnp.uint32(3) + jnp.uint32(1)),
                              (0, w))[1]

    def item(t, x, th):
        U = (jnp.arange(16, dtype=jnp.uint32).reshape(8, 2) * (x * 100).astype(jnp.uint32)).at[0, 0].set(7)
        w = jax.random.key_data(jax.random.wrap_key_data(U))
        h = lambda: (nested(w) % jnp.uint32(1000)).astype(F)[:, 0]

        def body(acc, _):
            return (acc + h() * th[0]).at[0].add(th[1]), None
        acc, _ = lax.scan(body, jnp.zeros(8, F), None, length=3)

        def outer(c):
            m, a = c
            return m + 1, a + lax.while_loop(lambda c: c[0] < 3, lambda c: (c[0] + 1, c[1] + h() * th[1]),
                                             (0, jnp.zeros(8, F)))[1]
        acc2 = lax.while_loop(lambda c: c[0] < 2, outer, (jnp.int32(0), jnp.zeros(8, F)))[1]
        return t.add("Q", jnp.arange(8), acc + acc2), jnp.sum(acc * acc2)
    return _alias_arms(item)


@case("alias: in-place updates whose operand is read during or after them (dus, scatter, captured reads)", ["auto"])
def _(b):
    """in a loop: a scatter whose update reads another element of its operand, a dynamic_update_slice whose update
    is a slice of its own operand (both must copy), and an element read before an update and used after it (read
    captured, the update in place)."""
    def item(t, x, th):
        def body(i, a):
            j = (i * 3) % 8
            y = a[j]
            a = a.at[j].set(a[(j + 1) % 8] * th[0] + 0.5)
            a = lax.dynamic_update_slice(a, a[1:4] * th[1], (i % 5,))
            return a * 0.9 + 0.01 * y
        a = lax.fori_loop(0, (x * 6).astype(jnp.int32) + 2, body, _arr(x, th))
        return t.add("Q", jnp.arange(8), a), jnp.sum(a)
    return _alias_arms(item)


@case("alias: cond in a while body writing the carry, the initial array a body constant (reverse mode)", ["auto"])
def _(b):
    """the cond's destination is the loop carry, updated in place in one branch and element-wise in the other;
    the initial array is also read by the body (as the adjoint's restart reads it)."""
    def item(t, x, th):
        a0 = _arr(x, th)

        def body(c):
            k, a = c
            a = lax.cond(k % 2 == 0, lambda u: u.at[k % 8].add(th[0]), lambda u: jnp.where(u > 0, u * th[1], u), a)
            return k + 1, a + 0.01 * a0
        a = lax.while_loop(lambda c: c[0] < (x * 5).astype(jnp.int32) + 2, body, (jnp.int32(0), a0))[1]
        return t.add("Q", jnp.arange(8), a), jnp.sum(a * a0)
    return _alias_arms(item)


# subnormal f32 emits (a GPU's f32 atomic add flushes subnormal inputs and results to zero; tallymap must not):
# (bin, value) pairs whose per-bin sums are exact in any order.  Bin 0: 4 x 1.4e-44 (lumen's report: 0 in a float Sum
# and a Records field); 1: two normals cancelling to a subnormal; 2: a normal + a subnormal; 3: one subnormal; 4: a
# negative subnormal total; 5: 1 + 2^-149 (rounds to 1); 6: 3 x 2^-110 (small normals); 7: -0.0 only.
_TN = 2.0 ** -126
_SUBN = [(0, 1.4e-44)] * 4 + [(1, 1.5 * _TN), (1, -_TN), (2, _TN), (2, 2.0 ** -140), (3, 2.0 ** -140), (4, -3e-44),
                              (4, 1e-45), (5, 1.0), (5, 1e-45)] + [(6, 2.0 ** -110)] * 3 + [(7, -0.0)]
_SUBN_I = jnp.asarray([i for i, _ in _SUBN], jnp.int32)
_SUBN_X = jnp.asarray(np.array([x for _, x in _SUBN], np.float32))


def _subn_skip(b):
    """the xla backend on a CPU follows XLA:CPU, which flushes subnormals in its own arithmetic: -> a skip message
    when XLA's own reduce of two 2^-149 is 0 there (else None: the cell runs)."""
    if b != "xla" or GPU:
        return None
    s = float(jax.jit(jnp.sum)(jnp.full((2,), 1e-45, jnp.float32)))
    return None if s != 0.0 else ("skipped: XLA:CPU flushes subnormals (its own jnp.sum of 2 x 2^-149 is 0.0); the "
                                  "xla backend follows XLA's platform arithmetic")


def _bitwise(a, b_):
    a, b_ = np.asarray(a, np.float32), np.asarray(b_, np.float32)
    return a.shape == b_.shape and np.array_equal(a.view(np.uint32), b_.view(np.uint32))


@case("subnormal f32 emits survive: Sum, exact Sum (fixed_point=True) and a Records field, privatised and global "
      "bins, bitwise the exact sum (= XLA:GPU's own reduce)")
def _(b):
    if (m := _subn_skip(b)):
        return m
    want = _exact_ref(np.asarray(_SUBN_X), np.asarray(_SUBN_I), NB)
    if GPU:      # XLA's own arithmetic on a GPU keeps subnormals: its dense reduce is the same exact sums
        dense = jax.jit(lambda x, i: jnp.sum(jnp.where(i[:, None] == jnp.arange(NB), x[:, None], 0.0), 0))(
            _SUBN_X, _SUBN_I)
        if not _bitwise(dense, want):
            raise Fail(f"XLA's dense reduce {np.asarray(dense)} != the exact sums {want}")
    done = []
    for S in (NB, 2048):                              # NB: privatised (shared memory, then the global flush); 2048
        for fp in (False, True):
            f = tm.vmap(lambda t, x, i: t.add("s", i, x), outputs={"s": tm.Sum((S,), jnp.float32, fixed_point=fp)},
                        backend=b)
            got = np.asarray(f(_SUBN_X, _SUBN_I).outputs["s"])[:NB]
            if not _bitwise(got, want):
                raise Fail(f"Sum({S}, fixed_point={fp}) {got} != the exact sums {want}")
            done.append(f"Sum{S}/fp={fp}")
    f = tm.vmap(lambda t, x: t.append("r", {"q": x}), outputs={"r": tm.Records({"q": jnp.float32}, capacity=32)},
                backend=b)
    q = np.asarray(f(_SUBN_X).outputs["r"]["q"])
    wantq = np.zeros(32, np.float32)
    wantq[:len(_SUBN)] = np.where(np.asarray(_SUBN_X) == 0, np.float32(0.0), np.asarray(_SUBN_X))    # -0.0 -> +0.0
    if not _bitwise(q, wantq):
        raise Fail(f"Records q {q[:len(_SUBN)]} != the emitted values {wantq[:len(_SUBN)]}")
    if jax.config.jax_enable_x64:                     # f64: the values times 2^-900 (f64 subnormals), exact
        f = tm.vmap(lambda t, x, i: t.add("s", i, x.astype(jnp.float64) * 2.0 ** -900),
                    outputs={"s": tm.Sum((2048,), jnp.float64)}, backend=b)
        got = np.asarray(f(_SUBN_X, _SUBN_I).outputs["s"])[:NB]
        if not np.array_equal(got, want.astype(np.float64) * 2.0 ** -900):
            raise Fail(f"f64 Sum {got} != {want.astype(np.float64) * 2.0 ** -900}")
        done.append("f64")
    return f"{', '.join(done)}, Records: bitwise"


@case("subnormal f32 emits survive in jvp tangents and grad cotangents (Sum, exact Sum, Records)")
def _(b):
    if (m := _subn_skip(b)):
        return m
    X, I = np.asarray(_SUBN_X), np.asarray(_SUBN_I)
    want = _exact_ref(X, I, NB)
    w = jnp.asarray(np.isin(np.arange(NB), [0, 3, 4]), jnp.float32)          # bins whose values are all subnormal
    gwant = _exact_ref(X[np.isin(I, [0, 3, 4])], np.zeros(int(np.isin(I, [0, 3, 4]).sum()), int), 1)[0]
    done = []
    for S in (NB, 2048):
        for fp in (False, True):
            f = tm.vmap(lambda t, x, i, th: t.add("s", i, x * th), in_axes=(0, 0, None),
                        outputs={"s": tm.Sum((S,), jnp.float32, fixed_point=fp)}, backend=b)
            p, tg = jax.jvp(lambda th: f(_SUBN_X, _SUBN_I, th).outputs["s"][:NB], (jnp.float32(1.0),),
                            (jnp.float32(1.0),))
            g = jax.grad(lambda th: jnp.sum(f(_SUBN_X, _SUBN_I, th).outputs["s"][:NB] * w))(jnp.float32(1.0))
            if not (_bitwise(p, want) and _bitwise(tg, want) and _bitwise(g, gwant)):
                raise Fail(f"Sum({S}, fixed_point={fp}): primal {np.asarray(p)}, tangent {np.asarray(tg)} (want "
                           f"{want}); grad {float(g)!r} (want {float(gwant)!r})")
            done.append(f"Sum{S}/fp={fp}")
    f = tm.vmap(lambda t, x, th: t.append("r", {"q": x * th}), in_axes=(0, None),
                outputs={"r": tm.Records({"q": jnp.float32}, capacity=32)}, backend=b)
    p, tg = jax.jvp(lambda th: f(_SUBN_X, th).outputs["r"]["q"], (jnp.float32(1.0),), (jnp.float32(1.0),))
    wantq = np.zeros(32, np.float32)
    wantq[:len(_SUBN)] = np.where(X == 0, np.float32(0.0), X)
    sub = np.zeros(32, np.float32)
    sub[:len(_SUBN)] = np.abs(X) < _TN                                       # the subnormal records
    g = jax.grad(lambda th: jnp.sum(f(_SUBN_X, th).outputs["r"]["q"] * sub))(jnp.float32(1.0))
    rg = _exact_ref(X[np.abs(X) < _TN], np.zeros(int((np.abs(X) < _TN).sum()), int), 1)[0]
    if not (_bitwise(p, wantq) and _bitwise(np.where(np.asarray(tg) == 0, np.float32(0), tg), wantq)
            and _bitwise(g, rg)):
        raise Fail(f"Records: primal {np.asarray(p)[:len(_SUBN)]}, tangent {np.asarray(tg)[:len(_SUBN)]} (want "
                   f"{wantq[:len(_SUBN)]}); grad {float(g)!r} (want {float(rg)!r})")
    return f"{', '.join(done)}, Records: primal, tangent and cotangent bitwise"


# ------------------------------------------------------------------------------------------- memory: module caches
def _gc():
    import gc
    jax.clear_caches()
    gc.collect()


@case("[mem] canon cache: weak on the jaxpr, bounded across fresh jaxprs, never another object's entry", ["xla"])
def _(b):
    import weakref

    from jax.extend.core import ClosedJaxpr, jaxpr_as_fun

    from tallymap.backends.kernel import canon
    x = jnp.arange(3, dtype=F)

    def mk(v):
        a = jnp.full((3,), v, F)
        return jax.make_jaxpr(lambda x: x ** 2 + a)(x)          # integer_pow: sugar, canonicalised
    _gc()
    n0 = len(canon._CACHE)
    refs = []
    for i in range(40):
        cj = mk(float(i))
        out = canon.canonical(cj)
        if any(e.primitive.name == "integer_pow" for e in out.jaxpr.eqns):
            raise Fail("integer_pow survived canonicalisation")
        if not _bitwise(jaxpr_as_fun(out)(x)[0], x ** 2 + F(i)):
            raise Fail(f"canonical form of jaxpr {i} computes the wrong values")
        refs.append(weakref.ref(cj.jaxpr))
        del cj, out
    _gc()
    alive, grown = sum(r() is not None for r in refs), len(canon._CACHE) - n0
    if alive or grown > 0:
        raise Fail(f"after 40 fresh jaxprs: {alive} still alive, cache grew by {grown} (an unbounded strong cache)")
    # one Jaxpr, two constant sets: each gets a form with its OWN constants
    cj = mk(1.0)
    c5 = jnp.full((3,), 5.0, F)
    o1, o2 = canon.canonical(cj), canon.canonical(ClosedJaxpr(cj.jaxpr, [c5]))
    if not all(p is q for p, q in zip(o1.consts, cj.consts)) or not all(
            p is q for p, q in zip(canon.canonical(cj).consts, cj.consts)):
        raise Fail("a repeated canonicalisation does not carry its own constants")
    if o2.consts[0] is not c5 or not _bitwise(jaxpr_as_fun(o2)(x)[0], x ** 2 + 5):
        raise Fail("canonical() returned the constants of another ClosedJaxpr sharing the Jaxpr")
    if not _bitwise(jaxpr_as_fun(canon.canonical(cj))(x)[0], x ** 2 + 1):
        raise Fail("canonical() of the first ClosedJaxpr changed after a second constant set")
    # an id reused by a new jaxpr after the old one died: never the old entry
    hits = 0
    for i in range(20):
        cj = mk(float(i))
        k = id(cj.jaxpr)
        del cj
        _gc()
        cj2 = mk(float(i + 1000))
        hits += id(cj2.jaxpr) == k
        if not _bitwise(jaxpr_as_fun(canon.canonical(cj2))(x)[0], x ** 2 + F(i + 1000)):
            raise Fail(f"a new jaxpr (id reused: {id(cj2.jaxpr) == k}) got a stale canonical form")
        del cj2
    return f"40 fresh jaxprs: 0 alive, cache +0; own constants per ClosedJaxpr; {hits}/20 id reuses, all fresh"


@case("[mem] ranges memo: weak on the jaxpr, the constants' ranges are part of the fact set", ["xla"])
def _(b):
    import weakref

    from jax.extend.core import ClosedJaxpr

    from tallymap.backends.kernel import ranges
    j = jax.make_jaxpr(lambda k: k + jnp.int32(1))(jnp.int32(0)).jaxpr
    j = j.replace(constvars=list(j.invars), invars=[])          # k: an int32 scalar constant
    c3 = ClosedJaxpr(j, [np.int32(3)])
    r3 = ranges.jaxpr_ranges(c3, [])
    r100 = ranges.jaxpr_ranges(ClosedJaxpr(j, [np.int32(100)]), [])
    if tuple(r3[0]) != (4, 4) or tuple(r100[0]) != (101, 101):
        raise Fail(f"ranges of k + 1: k = 3 -> {r3}, k = 100 -> {r100} (want (4, 4), (101, 101)): a memo keyed on "
                   "the jaxpr alone returns another constant set's facts")
    wr = weakref.ref(c3.jaxpr)       # (the memo's key: the Jaxpr; on JAX >= 0.12 a ClosedJaxpr is a Jaxpr: c3 itself)
    n0 = len(ranges._MEMO)
    del j, c3
    _gc()
    if wr() is not None or len(ranges._MEMO) >= n0:
        raise Fail("the ranges memo keeps a dead jaxpr alive")
    return "k = 3 -> (4, 4), k = 100 -> (101, 101); entry gone with its jaxpr"


@case("[mem] repeated eager launches reuse the item program (no rewrite re-trace); a changed closure is seen")
def _(b):
    from tallymap import _api
    n = [0]
    real = _api.retrace

    def counting(*a, **k):
        n[0] += 1
        return real(*a, **k)
    xs = jnp.arange(16, dtype=F)
    tab = [jnp.arange(NB, dtype=F)]

    def item(t, x):
        c = jnp.asarray([1.0, 2.0, 3.0], F)                     # a small array made at every trace (a new object)
        def body(s):
            i, t = s
            return i + 1, t.add("Q", i % NB, x * c[i % 3] + tab[0][i % NB])
        return lax.while_loop(lambda s: s[0] < 3, body, (jnp.int32(0), t))[1]

    def rec_item(t, x):
        return t.append("r", {"v": x * 2}, when=x > 3)
    fs = dict(sum=V(item, b), one=tm.vmap(rec_item, outputs={"r": tm.Records({"v": F}, capacity=32)}, backend=b),
              ctf=tm.vmap(rec_item, outputs={"r": tm.Records({"v": F}, capacity=None)}, backend=b))
    get = dict(sum=lambda r: r.outputs["Q"], one=lambda r: r.outputs["r"]["v"], ctf=lambda r: r.outputs["r"]["v"])
    _api.retrace = counting
    try:
        out = []
        for k, f in fs.items():
            first = np.asarray(get[k](f(xs)))
            n0 = n[0]
            for _ in range(5):
                if not _bitwise(get[k](f(xs)), first):
                    raise Fail(f"{k}: a repeated launch differs")
            if n[0] != n0:
                raise Fail(f"{k}: {n[0] - n0} rewrite re-traces over 5 repeated launches (want 0)")
            out.append(k)
        # the closure now holds another table: the next launch must use it (never a stale program)
        want = np.asarray(fs["sum"](xs).outputs["Q"])
        tab[0] = tab[0] + 100
        got = np.asarray(fs["sum"](xs).outputs["Q"])
        ref = want + 100 * np.bincount(np.arange(3) % NB, minlength=NB)[:NB] * len(xs)
        if not np.allclose(got, ref, rtol=1e-6):
            raise Fail(f"a changed closed-over table was not seen: {got} vs {ref}")
    finally:
        _api.retrace = real
    return f"{', '.join(out)}: 0 re-traces over 5 launches, bitwise; a changed closure is seen"


@case("[mem] fresh closures: the program and kernel caches stay bounded; an evicted kernel rebuilds", ["auto"])
def _(b):
    from tallymap import _api
    from tallymap.backends import kernel as CB
    xs = jnp.arange(8, dtype=F)
    saved = CB.KERNEL_CACHE
    CB.KERNEL_CACHE = 3

    def f(i):                                    # a new literal: a new program and a new kernel per i
        return V(lambda t, x: t.add("Q", 0, x * x + F(i)), b)(xs).outputs["Q"]
    try:
        first = np.asarray(f(0))
        b0 = CB.BUILDS[0]
        for i in range(1, 7):
            f(i)
        if len(CB._KERNELS) > 3 or len(_api._PROGS) > _api.PROG_CACHE:
            raise Fail(f"kernels {len(CB._KERNELS)} (bound 3), programs {len(_api._PROGS)}")
        from tallymap import _primitive as _P
        _P._EAGER.clear()                        # (its compiled launches would not ask for the kernel again)
        again = np.asarray(f(0))                 # evicted: rebuilt (from the disk cache), same bits
        if not _bitwise(again, first) or CB.BUILDS[0] - b0 != 7:
            raise Fail(f"evicted kernel: builds {CB.BUILDS[0] - b0} (want 7), bitwise {_bitwise(again, first)}")
    finally:
        CB.KERNEL_CACHE = saved
    return f"7 programs: {len(CB._KERNELS)} kernels kept (bound 3); the evicted one rebuilt bitwise"


print("\n================ SUMMARY", jax.__version__, "x64" if jax.config.jax_enable_x64 else "",
      jax.devices()[0].device_kind)
for n, b, m in RES:
    print(f"{n:52s} {b:8s} {m[:160]}")
print("TOTAL", len(RES), "PASS", sum(m.startswith("PASS") for _, _, m in RES))
