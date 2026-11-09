"""Phase-2 edge-case matrix.  python tests/edge.py  (env: BACKENDS=lockstep,cuda  TEST_X64=1  ONLY=substr)
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
from ab_rule import assert_arms_differ
from jax import lax

import tallymap as tm
from tallymap import reference

if os.environ.get("TX_UNFUSED") == "1":                 # run_all --unfused: the two-node jvp design
    from tallymap import _primitive as _P0
    _P0.FUSE_JVP = False

jax.config.update("jax_default_matmul_precision", "highest")
GPU = jax.default_backend() == "gpu"
BACKENDS = os.environ.get("BACKENDS", "lockstep,cuda" if GPU else "lockstep").split(",")
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


@case("spawn without Queue / spawn outside", ["lockstep"])
def _(b):
    raises(ValueError, lambda: V(lambda t, x: t.spawn(x), b)(xs), "Queue")
    def item(t, x):
        return t.spawn(x, when=x > 1.0)
    raises(ValueError, lambda: jax.jit(lambda x: R(item)(x))(xs), "Queue")


@case("API: unknown or renamed options / backends and removed aliases raise", ["lockstep"])
def _(b):
    f = lambda t, x: t.add("Q", 0, x)
    for kw in (dict(math="ieee"), dict(min_blocks=4), dict(block=8), dict(rev_slots=4), dict(rev_remat=2),
               dict(ad="forward"), dict(bogus=1), dict(_occ_cap=2), dict(_replay_check=False),
               dict(cuda_min_blocks=2), dict(_rev_phase="skip"), dict(_tree_phase="fwd"),
               dict(rev_step_segments=8)):
        raises(ValueError, lambda: V(f, b, **kw), "unknown option")
    raises(ValueError, lambda: tm.vmap(f, outputs={"Q": tm.Sum((NB,), F)}, backend="xla"), "unknown backend")
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
        r = tm.vmap(lambda ti, y: ti.add("Q", 0, y * x), outputs={"Q": tm.Sum((NB,), F)}, backend="lockstep")(
            jnp.stack([x, 2 * x]))
        return t.add("Q", 3, r.outputs["Q"][0])
    if b == "cuda":
        raises(tm.UnsupportedError, lambda: V(outer, b)(xs), "backend='lockstep'")
        b = "lockstep"
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
    f = V(walk, b, in_axes=AX, lockstep_block=8)
    for n in (0, 1, 13):
        A = f(xs[:n], keys[:n], tab, TH)
        if n == 0:
            close(A.outputs["Q"], np.zeros(NB))
            if A.per_item[0].shape != (0,):
                raise Fail("N=0 per-item shape")
        else:
            close(A, R(walk, in_axes=AX)(xs[:n], keys[:n], tab, TH), what=f"N={n}")


@case("N >= 2^31 rejected with a clear error", ["lockstep"])
def _(b):
    f = V(lambda t, x: t.add("Q", 0, x), b)
    raises(ValueError, lambda: jax.eval_shape(f, jax.ShapeDtypeStruct((2 ** 31,), F)), "2^31")


@case("large N (2^22 items, ids folded into one key)", ["cuda"] if GPU else ["lockstep"])
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
    if b == "cuda" or jax.config.jax_enable_x64:                   # exact=True needs x64 on the lockstep backend
        E = tm.vmap(mk(1), outputs={"Q": tm.Sum((NB,), F, exact=True, scale=2.0 ** 20)}, backend=b)(xs, idx)
        if int(E.stats["dropped_emits"]) != want or int(E.stats["exact_overflow"]) != 0:
            raise Fail(f"exact tally: dropped_emits {int(E.stats['dropped_emits'])} (want {want}), overflow "
                       f"{int(E.stats['exact_overflow'])}")
        msg += "; exact tally too"
    return msg


@case("dual lanes: the lane guard decides T on the unbatched program; a refused T-lane program is never built",
      ["lockstep"])
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
        f = tm.vmap(item, outputs={"Q": tm.Sum((NB,), F)}, in_axes=(0, None), backend="cuda")
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
    if b == "cuda":
        from tallymap.backends import cuda as CB
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


@case("debug.print raises clearly; pure_callback -> cuda unsupported (cuda and auto raise, pointing to lockstep)")
def _(b):
    def dbg(t, x):
        jax.debug.print("x={x}", x=x)
        return t.add("Q", 0, x)
    raises(Exception, lambda: V(dbg, b)(xs), "effect")

    def cb(t, x):
        y = jax.pure_callback(lambda v: np.asarray(v * 2), jax.ShapeDtypeStruct((), F), x, vmap_method="sequential")
        return t.add("Q", 0, y)
    if b == "cuda":
        raises(tm.UnsupportedError, lambda: V(cb, "cuda")(xs), "backend='lockstep'")
        raises(tm.UnsupportedError, lambda: V(cb, "auto")(xs), "backend='lockstep'")
        q = V(cb, "lockstep")(xs).outputs["Q"]
    else:
        q = V(cb, b)(xs).outputs["Q"]
    close(q[0], 2 * float(xs.sum()))


@case("exact=True: order-independent, run-to-run bitwise; overflow counted and poisons the exact sums")
def _(b):
    if b == "lockstep" and not jax.config.jax_enable_x64:
        raises(NotImplementedError, lambda: tm.vmap(walk, outputs={"Q": tm.Sum((NB,), F, exact=True)}, in_axes=AX,
                                                      backend=b)(xs, keys, tab, TH), "x64")
        return "needs x64 on lockstep (raises clearly)"
    out = {"Q": tm.Sum((NB,), F, exact=True, scale=2.0 ** 20)}
    n = 4096
    x2 = jnp.linspace(0.1, 3.0, n).astype(F)
    k2 = jax.random.split(jax.random.PRNGKey(5), n)
    f = tm.vmap(walk, outputs=out, in_axes=AX, backend=b)
    a1 = np.asarray(f(x2, k2, tab, TH).outputs["Q"])
    a2 = np.asarray(f(x2[::-1], k2[::-1], tab, TH).outputs["Q"])          # different item order
    if not np.array_equal(a1, a2):
        raise Fail("exact output depends on item order")
    big = tm.vmap(lambda t, x: t.add("Q", 0, x * 1e15), outputs=out, backend=b)(x2)
    if int(big.stats["exact_overflow"]) != n:
        raise Fail(f"overflow count {int(big.stats['exact_overflow'])} != {n}")
    # an overflow is a resource limit: every exact float output is NaN (also its tangent); a non-exact one is not
    mix = {"Q": tm.Sum((NB,), F, exact=True, scale=2.0 ** 20), "E": tm.Sum((2,), F, exact=True, scale=2.0 ** 20),
           "P": tm.Sum((1,), F)}

    def ovf(t, x, s):
        return t.add("Q", 0, x * s).add("E", 1, x).add("P", 0, x)
    g = tm.vmap(ovf, outputs=mix, in_axes=(0, None), backend=b)
    r = g(x2, F(1e15))
    if not (np.isnan(np.asarray(r.outputs["Q"])).all() and np.isnan(np.asarray(r.outputs["E"])).all()):
        raise Fail(f"exact_overflow={int(r.stats['exact_overflow'])} did not poison the exact sums: "
                   f"Q {np.asarray(r.outputs['Q'])[:2]} E {np.asarray(r.outputs['E'])}")
    close(r.outputs["P"][0], float(x2.sum()))
    tan = jax.jvp(lambda s: g(x2, s).outputs["Q"], (F(1e15),), (F(1),))[1]
    if not np.isnan(np.asarray(tan)).all():
        raise Fail("the tangent of an overflowed exact sum is not NaN")
    ok = g(x2, F(1.0))
    if int(ok.stats["exact_overflow"]) != 0 or not np.isfinite(np.asarray(ok.outputs["Q"])).all():
        raise Fail("an exact call with no overflow was poisoned")
    fl = tm.vmap(walk, outputs={"Q": tm.Sum((NB,), F)}, in_axes=AX, backend=b)
    f1 = np.asarray(fl(x2, k2, tab, TH).outputs["Q"])
    f2 = np.asarray(fl(x2[::-1], k2[::-1], tab, TH).outputs["Q"])
    return f"f32 atomics order-dependent bins: {int((f1 != f2).sum())}/{NB}"


@case("exact: cuda == lockstep bitwise; jvp == jacfwd column bitwise", ["cuda"])
def _(b):
    if not jax.config.jax_enable_x64:
        return "skipped (needs x64 for the lockstep side)"
    out = {"Q": tm.Sum((NB,), F, exact=True, scale=2.0 ** 24)}
    fc = tm.vmap(walk, outputs=out, in_axes=AX, backend="cuda")
    fx = tm.vmap(walk, outputs=out, in_axes=AX, backend="lockstep")
    a, c = np.asarray(fc(xs, keys, tab, TH).outputs["Q"]), np.asarray(fx(xs, keys, tab, TH).outputs["Q"])
    p = jnp.concatenate([tab, jnp.array([TH])])
    jv = jax.jvp(lambda th: fc(xs, keys, tab, th).outputs["Q"], (TH,), (F(1),))[1]
    J = jax.jacfwd(lambda p: fc(xs, keys, p[:NB], p[NB]).outputs["Q"])(p)
    if not np.array_equal(a, c):
        raise Fail(f"exact cuda != lockstep in {int((a != c).sum())}/{NB} bins")
    if not np.array_equal(np.asarray(jv), np.asarray(J[:, NB])):
        raise Fail("exact: jvp != jacfwd column bitwise")
    # int64 tallies (u64 atomics): per-item values above 2^32 exercise the high word
    big = (jnp.arange(N, dtype=jnp.int64) + 1) * (1 << 33)
    q = tm.vmap(lambda t, v: t.add("I", (v >> 33) % NB, v), outputs={"I": tm.Sum((NB,), jnp.int64)},
                backend="cuda")(big).outputs["I"]
    want = np.zeros(NB, np.int64)
    np.add.at(want, (np.asarray(big) >> 33) % NB, np.asarray(big))
    if not np.array_equal(np.asarray(q), want):
        raise Fail(f"int64 tally {np.asarray(q)} != {want}")
    return f"cuda==lockstep bitwise {NB}/{NB} bins; jvp==jacfwd column bitwise; int64 tally exact"


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
    if int(A.stats["spawned"]) != N * (2 + 4 + 8) or int(A.stats["dropped"]) != 0:
        raise Fail(f"spawned {int(A.stats['spawned'])} dropped {int(A.stats['dropped'])}")
    Bq = tm.vmap(item, outputs=out, spawn=tm.Queue(16), backend=b)(g0, xs)
    tot = int(Bq.stats["spawned"]) + int(Bq.stats["dropped"])
    if int(Bq.stats["dropped"]) == 0:
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
    return f"small queue: spawned {int(Bq.stats['spawned'])}, dropped {int(Bq.stats['dropped'])} (of {tot})"


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


@case("compare protocol (tests/compare_protocol.py): cuda vs lockstep (CRN, ulp control, Poisson floor)",
      ["cuda"])
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
    ok, rep = compare(V(walk, "cuda", in_axes=AX), x2, k2, tab, TH, other=V(walk, "lockstep", in_axes=AX),
                      floor_fn=V(walk_sq, "lockstep", in_axes=AX), verbose=False)
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
    if (b == "cuda") != bool(d["kernels"]):
        raise Fail(f"kernels recorded: {d['kernels']}")
    if b == "cuda" and not os.path.exists(d["kernels"][0]["source"]):
        raise Fail(f"cached .cu missing: {d['kernels'][0]['source']}")
    if b == "cuda":                      # once a kernel has run, its measured resources are reported
        jax.block_until_ready(f(xs, keys, tab, TH).outputs["Q"])
        res = [k.get("resources") for k in tm.diagnostics()["kernels"]]
        if not any(r_ and r_["regs"] > 0 and r_["blocks_per_sm"] >= 1 and r_["local_reserved_gib"] >= 0
                   for r_ in res):
            raise Fail(f"no kernel resources after a run: {res}")
    r = tm.diagnostics()["reverse"]
    if set(r) != {"replay", "tree"} or set(r["replay"]) != {"calls", "mismatch", "last"}:
        raise Fail(f"reverse counters {r}")
    return f"{len(d['programs'])} program(s), {len(d['kernels'])} kernel(s); jax {d['versions']['jax']}"


@case("A/B rule: tests/ab_rule.assert_arms_differ (identical arms raise; different programs pass)")
def _(b):
    g = lambda **o: (lambda x, k, t, th: V(walk, b, in_axes=AX, **o)(x, k, t, th).outputs["Q"])
    if b == "cuda":
        same, diff = (dict(cuda_math="approx"), dict()), (dict(_schedule="step"), dict(_schedule="item"))
    else:
        same, diff = (dict(cuda_math="ieee"), dict()), (dict(lockstep_block=8), dict())
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


@case("backend='auto': GPU raises on an unsupported program (the error points to backend='lockstep'); "
      "elsewhere lockstep with a one-time warning", ["lockstep"])
def _(b):
    from tallymap import _api
    def cb(t, x):
        y = jax.pure_callback(lambda v: np.asarray(v * 2), jax.ShapeDtypeStruct((), F), x, vmap_method="sequential")
        return t.add("Q", 0, y)
    want = 2 * float(xs.sum())
    if GPU:
        m = raises(tm.UnsupportedError, lambda: V(cb, "auto")(xs), "pass backend='lockstep'")
        close(V(cb, "lockstep")(xs).outputs["Q"][0], want)
        close(V(walk, "auto", in_axes=AX)(xs, keys, tab, TH), R(walk, in_axes=AX)(xs, keys, tab, TH))

        def dbg(t, x):                      # an effect: the error says no backend runs it
            jax.debug.print("x={x}", x=x)
            return t.add("Q", 0, x)
        raises(tm.UnsupportedError, lambda: V(dbg, "auto")(xs), "on any backend")
        return "GPU: " + m.splitlines()[-1][:90]
    _api._WARNED.clear()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        line = sys._getframe().f_lineno + 1
        close(V(cb, "auto")(xs).outputs["Q"][0], want)
        close(V(walk, "auto", in_axes=AX)(xs, keys, tab, TH), R(walk, in_axes=AX)(xs, keys, tab, TH))
    ws = [x for x in w if "lockstep backend" in str(x.message)]
    if len(ws) != 1:
        raise Fail(f"expected ONE lockstep warning, got {[str(x.message) for x in ws]}")
    msg = str(ws[0].message)
    if msg != "tallymap: no CUDA GPU on this platform; running the lockstep backend (plain JAX)":
        raise Fail(f"CPU warning text: {msg!r}")
    if os.path.basename(ws[0].filename) != os.path.basename(__file__) or ws[0].lineno != line:
        raise Fail(f"warning attributed to {ws[0].filename}:{ws[0].lineno}, want this file line {line}")
    _api._WARNED.clear()
    with warnings.catch_warnings(record=True) as w2:          # under jax.jit too (the caller is JAX, not the user)
        warnings.simplefilter("always")
        jline = sys._getframe().f_lineno + 1
        jax.jit(V(cb, "auto"))(xs).outputs["Q"].block_until_ready()
    wj = [x for x in w2 if "lockstep backend" in str(x.message)]
    if len(wj) != 1 or os.path.basename(wj[0].filename) != os.path.basename(__file__) or wj[0].lineno != jline:
        raise Fail(f"jit: {[(x.filename, x.lineno) for x in wj]}, want this file line {jline}")
    m = raises(tm.UnsupportedError, lambda: V(walk, "cuda", in_axes=AX)(xs, keys, tab, TH), "needs a CUDA GPU")
    return "CPU: " + msg[:70] + " | cuda: " + m[:50]


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
    if b == "cuda":
        def cb(t, x):
            y = jax.pure_callback(lambda v: np.asarray(v), jax.ShapeDtypeStruct((), F), x, vmap_method="sequential")
            return t.add("Q", 0, y)
        m4 = raises(tm.UnsupportedError, lambda: V(cb, "cuda")(xs), "pure_callback")
        if "item function" not in m4 or "cb (" not in m4 or here not in m4:
            raise Fail(f"unsupported-op message: {m4}")
        out.append(m4.split("[")[-1][:60])
    return " | ".join(out)


@case("a tally refusal names the operand, slot and kind; inside a call, the position alignment; a reverse "
      "program's slot legend", ["lockstep"])
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


@case("diagnostics baked_floats: a captured float table is listed, the same table as an operand is not",
      ["lockstep"])
def _(b):
    tabc = jnp.linspace(0.5, 1.5, 32, dtype=F)
    out = {"Q": tm.Sum((1,), F)}
    cap = tm.vmap(lambda t, xi: t.add("Q", 0, xi * tabc[_ix(xi)]), outputs=out, backend=b)
    opr = tm.vmap(lambda t, xi, tb: t.add("Q", 0, xi * tb[_ix(xi)]), outputs=out, in_axes=(0, None), backend=b)
    pc = tm.diagnostics(lambda x: cap(x).outputs["Q"], xs)["programs"]
    po = tm.diagnostics(lambda x, tb: opr(x, tb).outputs["Q"], xs, tabc)["programs"]
    if not pc or not any(sh == (32,) and n == 32 for r in pc for sh, _, n in r["baked_floats"]):
        raise Fail(f"the captured table is not listed: {[r['baked_floats'] for r in pc]}")
    if not po or any(r["baked_floats"] for r in po):
        raise Fail(f"an operand is listed as baked: {[r['baked_floats'] for r in po]}")
    if "baked_floats" in tm.diagnostics()["programs"][-1] and "consts" in tm.diagnostics()["programs"][-1]:
        raise Fail("the raw constants leak into the returned record")
    g = jax.jvp(lambda tb: opr(xs, tb).outputs["Q"], (tabc,), (jnp.ones_like(tabc),))[1]
    close(g, jnp.sum(xs)[None], rtol=1e-5)
    return f"captured {pc[0]['baked_floats']}, operand []; the operand's tangent reaches the tally"


@case("a located error keeps the original exception's attributes", ["lockstep"])
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


@case("kernel parameters: a program with more than 504 buffers runs (CUDA large kernel parameters); over 2048 "
      "raises", ["cuda"])
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
        want = 504 if nt <= 500 else 2048
        if f"#define SJ_MAXP {want}" not in src:
            raise Fail(f"{nt} tables: the kernel source does not declare SJ_MAXP {want}")
        got = np.asarray(mk(nt)(xs, tabs).outputs["Q"])        # every table pointer is read: tab_j[i] = i (1 + j % 7)
        i = np.asarray((xs * 3).astype(jnp.int32) % 4)
        want_q = float(np.sum(np.asarray(xs, np.float64) * i) * sum(1 + j % 7 for j in range(nt)))
        close(got, np.array([want_q]), rtol=1e-4, what=f"{nt} tables")
        msg.append(f"{nt}: SJ_MAXP {want}")
    tabs = tuple(jnp.arange(4, dtype=F) for _ in range(2100))
    m = raises(tm.UnsupportedError, lambda: mk(2100)(xs, tabs), "too many buffers")
    return "; ".join(msg) + "; 2100: " + m[:40]


QT_NK = 8


def _qt_tree(t, gen, x, kind):
    t = t.add("c", kind, jnp.int32(1)).add("s", kind, jnp.sin(x) + gen.astype(x.dtype))
    k2 = ((kind * 7 + gen + 1) % QT_NK).astype(jnp.int32)
    k3 = ((kind * 5 + 3) % QT_NK).astype(jnp.int32)
    t = t.spawn((gen + 1, x * 0.5 + 0.25, k2), when=gen < 5)
    return t.spawn((gen + 1, x * 0.25 + 0.5, k3), when=(gen < 4) & (x > 0.3))


def _qt_flat(t, gen, x, kind):          # no children: every thread lives in the root-claim path
    return t.add("c", kind, jnp.int32(1)).add("s", kind, jnp.cos(x) + gen.astype(x.dtype))


@case("spawn queue termination: every named item runs, launch after launch (cuda == lockstep, exact counts)",
      ["cuda"])
def _(b):
    # dtrax tests_wavefront/test_qterm.py (b65e17a), the one-queue cells: a tree with children in flight while
    # roots are still being claimed, and roots only with N far above the resident grid.  An early exit loses
    # items, and every item is one count in "c": the integer tallies must equal the lockstep ones exactly.
    out = {"c": tm.Sum((QT_NK,), jnp.int32), "s": tm.Sum((QT_NK,), F)}
    rng = np.random.default_rng(0)
    msg = []
    for fn, n, nm in ((_qt_tree, 20000, "tree"), (_qt_flat, 1 << 20, "roots only")):
        args = (jnp.zeros((n,), jnp.int32), jnp.asarray(rng.random(n), F),
                jnp.asarray(rng.integers(0, QT_NK, n), jnp.int32))
        ref = tm.vmap(fn, outputs=out, spawn=tm.Queue(1 << 20), backend="lockstep")(*args)
        rc = np.asarray(ref.outputs["c"])
        if int(ref.stats["dropped"]) != 0:
            raise Fail(f"{nm}: the lockstep reference dropped {int(ref.stats['dropped'])}")
        f = jax.jit(lambda *a, fn=fn: tm.vmap(fn, outputs=out, spawn=tm.Queue(1 << 20), backend=b)(*a))
        for rep in range(3):
            r = f(*args)
            c = np.asarray(r.outputs["c"])
            if not np.array_equal(c, rc) or int(r.stats["dropped"]) != 0:
                raise Fail(f"{nm} launch {rep}: {int(c.sum())} items vs {int(rc.sum())}, dropped "
                           f"{int(r.stats['dropped'])}")
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
            sp, dr = int(r.stats["spawned"]), int(r.stats["dropped"])
            if c[0] != NR or c.sum() != NR + sp:
                raise Fail(f"cap {cap} rep {rep}: processed {c.tolist()} != roots + spawned {NR} + {sp}")
            if sp + dr != F_OUT * int(c[:DEPTH].sum()):
                raise Fail(f"cap {cap} rep {rep}: spawned + dropped {sp} + {dr} != attempted "
                           f"{F_OUT * int(c[:DEPTH].sum())}")
            if dr == 0 or not np.all(np.isnan(np.asarray(r.outputs["Q"]))):
                raise Fail(f"cap {cap} rep {rep}: dropped {dr}, Q {np.asarray(r.outputs['Q'])[:3]}")
            seen.add((cap, sp))
    full = tm.vmap(item, outputs=out, spawn=tm.Queue(1 << 20), backend=b)(g0, x0)
    want = [NR * F_OUT ** g for g in range(DEPTH + 1)]
    if np.asarray(full.outputs["C"])[:DEPTH + 1].tolist() != want or int(full.stats["dropped"]):
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


@case("in-place dynamic_update_slice: aliasing hazards == lockstep == _cuda_inplace=False (bitwise)", ["cuda"])
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
    run = lambda bk, **o: tm.vmap(item, outputs=out, backend=bk, **o)
    g = lambda **o: (lambda a, k: run("cuda", **o)(a, k).per_item[0])
    assert_arms_differ(g(), g(_cuda_inplace=False), ii, keys)      # instrument: in-place updates did happen
    rs = [run("cuda")(ii, keys), run("cuda", _cuda_inplace=False)(ii, keys), run("lockstep")(ii, keys)]
    for r in rs[1:]:
        for a, c in zip(jax.tree_util.tree_leaves(rs[0]), jax.tree_util.tree_leaves(r)):
            if not np.array_equal(np.asarray(a), np.asarray(c)):
                raise Fail("in-place cuda result differs from the copying cuda / lockstep result")
    return "bitwise equal; in-place and copying cubins differ"


@case("buffer assignment (while init alias, cond destination, lazy write-back) == rule off == copying cuda == "
      "lockstep", ["cuda"])
def _(b):
    """the dead initial carry IS the carry storage, the cond output is written into its dead operand (one branch
    updates it in place, the other returns a lazy whole-array select over it: v5), and the lazy body result goes
    straight back into the carry.  The loop is NESTED: a single top-level while is split into step functions whose
    parameters are never mutated, where the rule (rightly) never fires.  Arms: the rule patched off (the control:
    strictly more arrays must be declared without it), and _cuda_inplace=False (every liveness-based reuse off)."""
    import re

    from tallymap import _primitive as _P
    from tallymap.backends import cuda as CB
    from tallymap.backends.cuda import codegen as CG
    from tallymap.backends.cuda import op_table as OT

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

    def run(bk, off=False, **o):
        saved = (OT._cond_dest, OT._init_alias, CG.Emitter.elementwise_self)
        CB._KERNELS.clear()
        _P._EAGER.clear()
        if off:
            OT._cond_dest = lambda em, eqn, ops, bodies: [None] * len(eqn.outvars)
            OT._init_alias = lambda em, inits, ivars, pt, others: [None] * len(inits)
            CG.Emitter.elementwise_self = lambda self, dst, x: False
        try:
            tm.diagnostics(reset=True)
            r = tm.vmap(item, outputs=out, backend=bk, **o)(xs, keys)
            ks = tm.diagnostics()["kernels"]
        finally:
            OT._cond_dest, OT._init_alias, CG.Emitter.elementwise_self = saved
            CB._KERNELS.clear()
            _P._EAGER.clear()
        src = open(ks[-1]["source"]).read() if ks else ""
        return r, len(re.findall(r"\b(?:float|double) \w+\[32\];", src))
    (ra, na), (ro, no), (rc, nc), (rl, _) = run("cuda"), run("cuda", off=True), run("cuda", _cuda_inplace=False), \
        run("lockstep")
    if not na < no:
        raise Fail(f"instrument: {na} 32-element arrays declared with buffer assignment, {no} with the rule off")
    for r_, what in ((ro, "rule-off"), (rc, "copying")):
        for a, c in zip(jax.tree_util.tree_leaves(ra), jax.tree_util.tree_leaves(r_)):
            if not np.array_equal(np.asarray(a), np.asarray(c)):
                raise Fail(f"buffer-assigned cuda result differs from the {what} cuda result")
    if not np.array_equal(np.asarray(ra.outputs["C"]), np.asarray(rl.outputs["C"])):
        raise Fail("integer tally differs from lockstep")
    close(ra.per_item, rl.per_item, 1e-6, 0.0, "per-item vs lockstep")
    return f"bitwise vs rule-off and copying cuda; 32-element arrays declared {na} / rule off {no} / copying {nc}"


@case("$CXX: a missing or wrong host C++ compiler is a tallymap error naming the variable and the fix",
      ["lockstep"])
def _(b):
    import shutil

    from tallymap.backends.cuda import compile as C
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
      ["lockstep"])
def _(b):
    from jax.extend.core import Primitive
    from jax.extend.core import primitives as P

    from tallymap.backends.cuda import canon, census, op_table
    jaxnames = {v.name for v in vars(P).values() if isinstance(v, Primitive)}
    from tallymap import _foreign
    ours = set(op_table.RULES) | set(canon.SUGAR) | set(_foreign._generic())
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
    if hows - {"rule", "canonicalised", "expanded", "tally"}:
        raise Fail(f"unknown 'how' values {sorted(hows)}")
    doc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "primitives.md")
    if open(doc).read() != census.primitives_table():
        raise Fail("docs/primitives.md is stale: regenerate it with tallymap.backends.cuda.census.primitives_table()")
    exported_unused = sorted(jaxnames - ours)
    return (f"{len(ours)} names, {len(ours & jaxnames)} exported by jax.extend; {len(exported_unused)} exported, "
            "not ours")


@case("exact=True on cuda without x64: equals the host fixed-point sum bitwise, order-independent", ["cuda"])
def _(b):
    n, scale = 1 << 15, 2.0 ** 20
    v = jnp.asarray(np.random.default_rng(3).uniform(-3, 3, n), jnp.float32)
    i = jnp.arange(n, dtype=jnp.int32)
    f = tm.vmap(lambda t, x, j: t.add("Q", j % NB, x).add("Q", (j + 1) % NB, x * 3.0),
                outputs={"Q": tm.Sum((NB,), jnp.float32, exact=True, scale=scale)}, backend="cuda")
    a1, a2 = np.asarray(f(v, i).outputs["Q"]), np.asarray(f(v[::-1], i[::-1]).outputs["Q"])
    acc = np.zeros(NB, np.int64)
    vv = np.asarray(v).astype(np.float64)
    np.add.at(acc, np.arange(n) % NB, np.round(vv * scale).astype(np.int64))
    np.add.at(acc, (np.arange(n) + 1) % NB, np.round(np.asarray(v * 3.0).astype(np.float64) * scale).astype(np.int64))
    want = (acc.astype(np.float64) * (1.0 / scale)).astype(np.float32)
    if not (np.array_equal(a1, a2) and np.array_equal(a1, want)):
        raise Fail(f"exact sum: order-independent {np.array_equal(a1, a2)}, == host {np.array_equal(a1, want)}")
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


@case("version policy: the tested versions, a warning only above the newest tested", ["lockstep"])
def _(b):
    from tallymap._compat import _check_jax_version
    if tm.TESTED_JAX != ("0.5.3", "0.6.2", "0.7.2", "0.8.3", "0.9.2", "0.10.2", "0.11.2"):
        raise Fail(f"TESTED_JAX {tm.TESTED_JAX}")
    if hasattr(tm, "check_jax_version") or "check_jax_version" in tm.__all__:
        raise Fail("check_jax_version is still public")
    st = _check_jax_version(warn=False)
    if st != "ok":
        raise Fail(f"this JAX ({jax.__version__}) is {st}: the matrix should run on tested versions")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        got = [_check_jax_version(v) for v in ("0.6.2", "0.12.0.dev20261014", "0.4.38", "0.11.3")]
    if got != ["ok", "newer", "ok", "newer"] or len(w) != 2 or "newer" not in str(w[0].message):
        raise Fail(f"policy {got}, warnings {[str(x.message) for x in w]}")
    return f"jax {jax.__version__}: ok; {len(tm.TESTED_JAX)} tested; 0.12 dev warns, 0.4.38 does not"


@case("eager calls: items that differ only in a host callback are different programs (cache key)", ["lockstep"])
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


@case("bitcast_convert_type between widths (shape-changing) on cuda == XLA bitwise", ["cuda"])
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


@case("long items: no typedef name collision, no lazy-expression blowup", ["cuda"])
def _(b):
    from tallymap.backends.cuda.codegen import _RESERVED, Emitter
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


@case("jnp.linalg.solve / inv inside an item (lu, custom_linear_solve, triangular_solve): fwd, jvp, grad vs dense")
def _(b):
    def mk(n):
        def item(t, x, th):
            i = jnp.arange(n, dtype=F)
            A = jnp.eye(n, dtype=F) * (2.0 + x) + th * jnp.cos(i[:, None] * x + i[None, :]) * 0.3
            A = A[::-1] if n > 3 else A                    # needs pivoting
            y = jnp.linalg.solve(A, jnp.sin(i * th + x))
            z = jnp.linalg.inv(A) @ jnp.ones(n, F)
            return t.add("Q", (x * 7).astype(jnp.int32) % NB, jnp.sum(y) + z[0] * th), y[0]
        return item
    out = []
    for n in (3, 4, 12):
        f, r = V(mk(n), b, in_axes=(0, None)), R(mk(n), in_axes=(0, None))
        close(f(xs, TH).outputs["Q"], r(xs, TH).outputs["Q"], rtol=2e-4)
        close(f(xs, TH).per_item, r(xs, TH).per_item, rtol=2e-4)
        jv = lambda g: jax.jvp(lambda q: g(xs, q).outputs["Q"], (TH,), (F(1.0),))[1]
        close(jv(f), jv(r), rtol=1e-3, what=f"jvp n={n}")
        gr = lambda g: jax.grad(lambda q: jnp.sum(g(xs, q).outputs["Q"] ** 2) + jnp.sum(g(xs, q).per_item))(TH)
        close(gr(f), gr(r), rtol=1e-3, what=f"grad n={n}")
        out.append(n)
    return f"n = {out}: forward, jvp and grad match the dense reference"


@case("lockstep narrowing: a while loop on a narrowed lane axis gives the masked interpreter's results exactly",
      ["lockstep"])
def _(b):
    """The A/B control of backends/lockstep.py's narrowing (_lockstep_compact=False is the masked interpreter).

    The default widths need thousands of lanes, so the widths are forced here (8 and 5 lanes, the second one not
    a divisor of N).

    Integer tallies, per-item integers and stats must be IDENTICAL on every platform.  Float tallies are bitwise
    identical on a CPU, where a scatter-add runs in index order, but a GPU orders the updates of one scatter by
    itself, so a different lane count sums them in a different order: there the contract is one ulp, which is
    the same caveat the block width already carries (design.md 1d).  Measured on a 2080 Ti: forward bitwise, the
    jvp 9.7e-08 relative."""
    ftol = dict(rtol=1e-5, atol=0) if GPU else dict(rtol=0, atol=0)

    def exact(a, bb, what):
        """integer and bool leaves identical, float leaves to ftol."""
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(bb)):
            if np.asarray(x).dtype.kind in "iub":
                if not np.array_equal(np.asarray(x), np.asarray(y)):
                    raise Fail(f"{what}: integer result differs")
            else:
                close(x, y, what=what, **ftol)

    def sw(t, x, key, th):
        """a batched switch with one expensive branch, inside a divergent loop"""
        def body(c):
            k, y, t = c
            i = (bits(key, k) % 3).astype(jnp.int32)
            y, t = lax.switch(i, [lambda y, t: (y * 0.9 * th, t.add("Q", k % NB, y)),
                                  lambda y, t: (y + 0.1, t.add("C", (k + 1) % NB, 1)),
                                  lambda y, t: (y + 0.01 * jnp.sum(jnp.sin(jnp.arange(16, dtype=F) * y * th)),
                                                t.add("Q", (k + 2) % NB, y))], y, t)
            return k + 1, y, t
        k, y, t = lax.while_loop(lambda c: (bits(key, c[0] + 7) % 4 != 0) & (c[0] < 9), body, (0, x, t))
        return t, (y, k)

    def spawner(t, gen, x):
        t = t.add("Q", gen % NB, x).add("C", gen % NB, 1)
        return t.spawn((gen + 1, x * 0.5), when=(gen < 3) & (x > 0.7))
    outs = {"Q": tm.Sum((NB,), F), "C": tm.Sum((NB,), jnp.int32)}
    g0 = jnp.zeros((N,), jnp.int32)
    cases = [("walk", walk, AX, (xs, keys, tab, TH), dict(outputs={"Q": tm.Sum((NB,), F)})),
             ("switch", sw, (0, 0, None), (xs, keys, TH), dict(outputs=outs)),
             ("spawn", spawner, (0, 0), (g0, xs), dict(outputs=outs, spawn=tm.Queue(1 << 10)))]
    opts = [dict(_lockstep_compact=8), dict(_lockstep_compact=5)]
    for name, item, ax, args, kw in cases:
        ref = tm.vmap(item, backend=b, in_axes=ax, _lockstep_compact=False, **kw)
        base = ref(*args)
        for opt in opts:
            f = tm.vmap(item, backend=b, in_axes=ax, **kw, **opt)
            A = f(*args)
            exact(A.outputs, base.outputs, f"{name} {opt} outputs")
            exact(A.per_item, base.per_item, f"{name} {opt} per-item")
            for k, v in base.stats.items():
                if not np.array_equal(np.asarray(A.stats[k]), np.asarray(v)):
                    raise Fail(f"{name} {opt} stats[{k}] {A.stats[k]} != {v}")
            if name == "spawn":
                continue                      # reverse mode through spawn is the tree adjoint (its own suite)
            loss = lambda g, q: jnp.sum(g(*args[:-1], q).outputs["Q"] ** 2)
            th = args[-1]
            close(jax.grad(lambda q: loss(f, q))(th), jax.grad(lambda q: loss(ref, q))(th),
                  what=f"{name} {opt} grad", **ftol)
            close(jax.jvp(lambda q: loss(f, q), (th,), (F(1),))[1],
                  jax.jvp(lambda q: loss(ref, q), (th,), (F(1),))[1], what=f"{name} {opt} jvp", **ftol)
            close(jax.vmap(lambda q: loss(f, q))(jnp.stack([th, th * F(1.1)])),
                  jax.vmap(lambda q: loss(ref, q))(jnp.stack([th, th * F(1.1)])),
                  what=f"{name} {opt} vmap", **ftol)
    return (f"{len(cases)} items x {len(opts)} widths: outputs, per-item, stats, grad, jvp and vmap "
            + ("within 1e-5 (GPU scatter order)" if GPU else "all BITWISE"))


print("\n================ SUMMARY", jax.__version__, "x64" if jax.config.jax_enable_x64 else "",
      jax.devices()[0].device_kind)
for n, b, m in RES:
    print(f"{n:52s} {b:8s} {m[:160]}")
print("TOTAL", len(RES), "PASS", sum(m.startswith("PASS") for _, _, m in RES))
