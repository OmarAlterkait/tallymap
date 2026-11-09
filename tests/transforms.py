"""Transformation composability matrix.  python tests/transforms.py
env: BACKENDS=lockstep,cuda  TEST_X64=1  ONLY=substr  NDEV=2 (pmap/shard_map cells; CPU: XLA_FLAGS host device count)
     TX_UNFUSED=1 (the two-node jvp design that runs when JAX drops the partial-eval registries)

A BIG function contains tm.vmap calls with pre/post-processing, a second tallymap call fed by the first one's output,
and (in some cells) tallymap inside lax.scan / fori / cond / while.  Every cell applies the SAME transformation to the
tallymap program and to the dense reference program (tallymap.reference = jax.vmap + sum).  JAX
cannot reverse-differentiate the reference's while loops, so reverse-mode cells are compared with the equivalent
FORWARD-mode transformation of the reference (grad <-> jacfwd, hessian <-> jacfwd(jacfwd), ...).
Each cell prints: PASS (numbers agree) | RAISES-OK (expected, clear error) | FAIL (wrong numbers) | RAISE (unexpected).
"""
import os
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
from jax import lax

import tallymap as tm
from tallymap import reference

jax.config.update("jax_default_matmul_precision", "highest")
if os.environ.get("TX_UNFUSED") == "1":
    from tallymap import _primitive as _C0
    _C0.FUSE_JVP = False
if os.environ.get("TX_T3") == "1":                      # run_all --t3: the single-loop adjoint wherever it applies
    from tallymap import _reverse as _R0
    _R0.SINGLE = "force"
NEG = os.environ.get("TEST_NEG", "")      # negative controls: the matrix MUST report FAIL in the affected cells
if NEG:
    from tallymap import _primitive as _C
    if NEG == "tangent":                      # every jvp tangent of an tallymap call 0.1% too large
        _orig_t = _C._tangents_out
        _C._tangents_out = lambda p, o, t: [x * 1.001 if not _C._is_zero(x) else x for x in _orig_t(p, o, t)]
    elif NEG == "cotangent":                  # every replay-adjoint cotangent 0.1% too large
        _orig_v = _C._vjp_call
        _C._vjp_call = lambda *a: tuple([None if g is None else g * 1.001 for g in gs] for gs in _orig_v(*a))
    elif NEG == "fold":                       # folded batches: member b's emits shifted to member (b+1)%B
        _orig_f = _C.fold_prog
        def _bad_fold(p, B, bbc):
            q = _orig_f(p, B, bbc)
            f = jax.extend.core.jaxpr_as_fun(q.cj)
            nT, nP = p.nT, p.nP
            def g(*a):
                a = list(a)
                a[nT + nP] = (a[nT + nP] + 1) % B
                return f(*a)
            avals = [jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype) for v in q.cj.jaxpr.invars]
            return _C.dataclasses.replace(q, cj=jax.make_jaxpr(g)(*avals))
        _C.fold_prog = _bad_fold
    print("NEGATIVE CONTROL", NEG, flush=True)
GPU = jax.default_backend() == "gpu"
BACKENDS = os.environ.get("BACKENDS", "lockstep,cuda" if GPU else "lockstep").split(",")
ONLY = os.environ.get("ONLY", "")
F = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
print("jax", jax.__version__, jax.__file__, jax.devices(), "x64", jax.config.jax_enable_x64, flush=True)
RES = []
NB = 8


class Fail(Exception):
    pass


def close(a, b, rtol=3e-4, atol=1e-5, what=""):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    if len(la) != len(lb):
        raise Fail(f"{what} {len(la)} leaves vs {len(lb)}")
    for x, y in zip(la, lb):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        if x.shape != y.shape:
            raise Fail(f"{what} shape {x.shape} vs {y.shape}")
        if not np.isfinite(x).all():
            raise Fail(f"{what} non-finite")
        err = np.max(np.abs(x - y)) if x.size else 0.0
        tol = rtol * (np.max(np.abs(y)) if y.size else 0.0) + atol
        if not err <= tol:
            raise Fail(f"{what} max err {err:.3g} > tol {tol:.3g} (|ref| {np.max(np.abs(y)):.3g})")
    return f"|ref| {max(float(np.max(np.abs(np.asarray(y)))) for y in lb if np.size(y)):.3g}" if lb else ""


class Expected(Exception):
    pass


def expect_raise(f, contains=""):
    try:
        f()
    except Exception as e:
        if contains and contains not in str(e):
            raise Fail(f"raised {type(e).__name__} without {contains!r}: {str(e)[:200]}")
        raise Expected(f"{type(e).__name__}: {str(e).splitlines()[0][:150]}")
    raise Fail("did not raise")


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
                msg = "PASS" + (f" ({r})" if isinstance(r, str) and r else "")
                ws = [str(x.message) for x in w if "tallymap" in str(x.message)]
                if ws:
                    msg += f" [warn: {ws[0][:90]}]"
            except Expected as e:
                msg = f"RAISES-OK {e}"
            except Fail as e:
                msg = f"FAIL {e}"
            except Exception as e:
                msg = f"RAISE {type(e).__name__}: {str(e)[:300]}"
                traceback.print_exc()
            RES.append((name, b, msg))
            print(f"{name:44s} {b:8s} {msg}  [{time.perf_counter() - t0:.1f}s]", flush=True)
        return fn
    return deco


def bits(key, k):
    return jax.random.bits(jax.random.fold_in(key, k))


# integer-exact control flow (RNG bits), smooth float weights in theta = (th0, th1): backends differ by ulps only
def walk(t, x, key, tab, th):
    def body(c):
        k, w, t = c
        b = bits(key, k)
        i = (b % NB).astype(jnp.int32)
        t = t.add("Q", i, w * tab[i] * th[0] + th[1] * th[1])
        return k + 1, w * (0.7 + 0.2 * th[0]) + 0.01 * jnp.sin(th[1] * w), t
    k, w, t = lax.while_loop(lambda c: (bits(key, c[0] + 1000) % 5 != 0) | (c[0] == 0), body, (0, x, t))
    return t, (w, k)


N = 37
xs = jnp.linspace(0.5, 1.5, N).astype(F)
keys = jax.random.split(jax.random.PRNGKey(0), N)
keys2 = jax.random.split(jax.random.PRNGKey(1), N)
tab = jnp.arange(1.0, NB + 1).astype(F)
TH = jnp.array([0.9, 0.3], F)
V2 = jnp.array([0.7, -0.4], F)
WQ = jnp.linspace(0.5, 2.0, NB).astype(F)
AX = (0, 0, None, None)
OUT = {"Q": tm.Sum((NB,), F)}


def S(b, fn=walk, **kw):
    kw.setdefault("outputs", OUT)
    kw.setdefault("in_axes", AX)
    return tm.vmap(fn, backend=b, **kw)


def R(fn=walk, **kw):
    kw.setdefault("outputs", OUT)
    kw.setdefault("in_axes", AX)
    return reference(fn, **kw)


def big_vec(f, th, x=xs):
    """pre-processing -> tallymap call -> its output feeds a 2nd tallymap call (as a table and a scale) ->
    post-processing."""
    tb = tab * jnp.exp(0.1 * th[1])
    x2 = x * (1 + 0.05 * th[0])
    A = f(x2, keys, tb, th)
    q = A.outputs["Q"]
    Bq = f(x2 * (q.sum() / 50.0), keys2, q / q.sum() * NB, th * 0.5).outputs["Q"]
    w = A.per_item[0]
    return jnp.concatenate([jnp.sin(q * 0.1) * WQ, Bq * 0.01, jnp.array([jnp.sum(w ** 2) * 0.1])])


def big(f, th, x=xs):
    return jnp.sum(big_vec(f, th, x) * jnp.linspace(1.0, 2.0, 2 * NB + 1).astype(F))


def in_scan(f, th):
    def step(c, k):
        y = big(f, c * (1 + 0.1 * k))
        return c * 0.9 + 0.01 * y, y
    c, ys = lax.scan(step, th, jnp.arange(2.0).astype(F))
    return jnp.sum(c) + jnp.sum(ys)


def in_fori(f, th):
    return lax.fori_loop(0, 2, lambda i, c: c * 0.9 + 0.02 * big(f, c), th).sum()


def in_cond(f, th):
    return lax.cond(th[0] > 0.5, lambda t: big(f, t), lambda t: big(f, t * 2.0) * 3.0, th)


def jac_ref(g):
    return jax.jacfwd(g)


# ------------------------------------------------------------------------------------------------ cells
@case("jit")
def _(b):
    return close(jax.jit(lambda t: big_vec(S(b), t))(TH), big_vec(R(), TH))


@case("eager (no jit)")
def _(b):
    return close(big_vec(S(b), TH), big_vec(R(), TH))


@case("grad")
def _(b):
    return close(jax.jit(jax.grad(lambda t: big(S(b), t)))(TH), jac_ref(lambda t: big(R(), t))(TH))


@case("grad eager")
def _(b):
    return close(jax.grad(lambda t: big(S(b), t))(TH), jac_ref(lambda t: big(R(), t))(TH))


@case("value_and_grad")
def _(b):
    v, g = jax.jit(jax.value_and_grad(lambda t: big(S(b), t)))(TH)
    close(v, big(R(), TH), what="value")
    return close(g, jac_ref(lambda t: big(R(), t))(TH), what="grad")


@case("vjp (vector output)")
def _(b):
    ct = jnp.linspace(-1.0, 1.0, 2 * NB + 1).astype(F)
    y, f_vjp = jax.vjp(lambda t: big_vec(S(b), t), TH)
    close(y, big_vec(R(), TH), what="primal")
    return close(jax.jit(f_vjp)(ct)[0], ct @ jac_ref(lambda t: big_vec(R(), t))(TH), what="vjp")


@case("jvp")
def _(b):
    y, dy = jax.jit(lambda t, v: jax.jvp(lambda s: big_vec(S(b), s), (t,), (v,)))(TH, V2)
    yr, dyr = jax.jvp(lambda s: big_vec(R(), s), (TH,), (V2,))
    close(y, yr, what="primal")
    return close(dy, dyr, what="tangent")


@case("linearize (+ jit of the linear fn)")
def _(b):
    y, f_lin = jax.linearize(lambda t: big_vec(S(b), t), TH)
    yr, dyr = jax.jvp(lambda s: big_vec(R(), s), (TH,), (V2,))
    close(y, yr, what="primal")
    close(f_lin(V2), dyr, what="lin")
    return close(jax.jit(f_lin)(V2), dyr, what="jit(lin)")


@case("linearize residuals = operands (no per-step / per-item-tally residuals)")
def _(b):
    _, f_lin = jax.linearize(lambda t: big_vec(S(b), t), TH)
    cj = jax.make_jaxpr(f_lin)(V2)
    shapes = sorted({tuple(np.shape(c)) for c in cj.consts}, key=lambda x: -int(np.prod(x)))
    big_res = [sh for sh in shapes if int(np.prod(sh)) > 4 * N]
    if big_res:
        raise Fail(f"residuals larger than 4N elements: {big_res}")
    return f"{len(cj.consts)} residuals, largest {shapes[:3]}"


@case("linear_transpose of linearize")
def _(b):
    ct = jnp.linspace(-1.0, 1.0, 2 * NB + 1).astype(F)
    _, f_lin = jax.linearize(lambda t: big_vec(S(b), t), TH)
    fT = jax.linear_transpose(f_lin, TH)
    return close(fT(ct)[0], ct @ jac_ref(lambda t: big_vec(R(), t))(TH))


@case("jacfwd")
def _(b):
    return close(jax.jit(jax.jacfwd(lambda t: big_vec(S(b), t)))(TH), jac_ref(lambda t: big_vec(R(), t))(TH))


@case("jacrev")
def _(b):
    return close(jax.jit(jax.jacrev(lambda t: big_vec(S(b), t)))(TH), jac_ref(lambda t: big_vec(R(), t))(TH))


H_REF = {}


def href():
    if "h" not in H_REF:
        H_REF["h"] = jax.jacfwd(jax.jacfwd(lambda t: big(R(), t)))(TH)
    return H_REF["h"]


@case("hessian (jacfwd(jacrev))")
def _(b):
    return close(jax.jit(jax.hessian(lambda t: big(S(b), t)))(TH), href())


@case("jacfwd(jacfwd)")
def _(b):
    return close(jax.jit(jax.jacfwd(jax.jacfwd(lambda t: big(S(b), t))))(TH), href())


@case("jacrev(jacfwd)")
def _(b):
    return close(jax.jit(jax.jacrev(jax.jacfwd(lambda t: big(S(b), t))))(TH), href())


@case("jvp-of-grad (hvp, fwd-over-rev)")
def _(b):
    g = jax.grad(lambda t: big(S(b), t))
    return close(jax.jit(lambda t, v: jax.jvp(g, (t,), (v,))[1])(TH, V2), href() @ V2)


@case("grad-of-jvp (rev-over-fwd)")
def _(b):
    f = lambda t, v: jax.jvp(lambda s: big(S(b), s), (t,), (v,))[1]
    return close(jax.jit(jax.grad(f))(TH, V2), href() @ V2)


@case("jax.test_util.check_grads(order=2, fwd + rev) on the smooth toy (finite differences)")
def _(b):
    from jax.test_util import check_grads
    # the walk's tally + per-item results: smooth in theta (control flow from RNG bits only), O(1) curvature, so a
    # central difference at eps = 1e-3 (f32) / 1e-4 (x64) is accurate to the check's tolerance
    eps = 1e-4 if jax.config.jax_enable_x64 else 1e-3

    def f(t):
        r = S(b)(xs, keys, tab, t)
        return jnp.concatenate([r.outputs["Q"], r.per_item[0]])
    try:
        check_grads(f, (TH,), order=2, modes=("fwd", "rev"), eps=eps)
    except AssertionError as e:
        raise Fail("check_grads: " + " ".join(str(e).split())[:300])
    return f"order 2, fwd + rev, eps {eps}"


@case("fused jvp: one tallymap_jvp eqn under jvp, split under linearize")
def _(b):
    from tallymap import _primitive
    j = str(jax.make_jaxpr(lambda t, v: jax.jvp(lambda s: big_vec(S(b), s), (t,), (v,)))(TH, V2))
    _, f_lin = jax.linearize(lambda t: big_vec(S(b), t), TH)
    jl = str(jax.make_jaxpr(f_lin)(V2))
    nj, nl = j.count("tallymap_jvp"), jl.count("tallymap_jvp")
    if _primitive.FUSE_JVP and (nj == 0 or nl != 0):
        raise Fail(f"fused={_primitive.FUSE_JVP}: tallymap_jvp eqns under jvp {nj}, in the linear jaxpr {nl}")
    return (f"FUSE_JVP={_primitive.FUSE_JVP}: jvp jaxpr {nj} tallymap_jvp / {j.count('tallymap_call')} tallymap_call; "
            f"linear jaxpr {nl} tallymap_jvp")


@case("fused jacfwd: primal + T tangent lanes in ONE tallymap eqn per call (has_aux primal used)")
def _(b):
    from tallymap import _primitive

    def g(fn):
        def h(t):
            y = big_vec(fn, t)
            return y, y
        return jax.jacfwd(h, has_aux=True)
    js = str(jax.make_jaxpr(g(S(b)))(TH))
    n = js.count("tallymap_call[") + js.count("tallymap_jvp[")
    want = 2 if _primitive.FUSE_JVP else 4      # big_vec: 2 tallymap calls, one fused eqn each (unfused: 2 eqns)
    if n != want:
        raise Fail(f"FUSE_JVP={_primitive.FUSE_JVP}: {n} tallymap eqns under jacfwd, expected {want}")
    J, y = jax.jit(g(S(b)))(TH)
    Jr, yr = g(R())(TH)
    close(y, yr, what="primal")
    return close(J, Jr, what="jac") + f"; {n} tallymap eqns"


@case("chunked dual lanes (_lanes=3, T=7 -> 3+3+1 padded): jacfwd with primal, jacrev(jacfwd)")
def _(b):
    P7 = jnp.concatenate([tab[:5], TH])

    def g(fn):
        def h(p):
            y = big_vec(fn, p[5:], x=xs) + jnp.sum(fn(xs, keys, tab.at[:5].set(p[:5]), p[5:]).outputs["Q"] * WQ)
            return y, y
        return h
    J, y = jax.jit(jax.jacfwd(g(S(b, _lanes=3)), has_aux=True))(P7)
    Jr, yr = jax.jacfwd(g(R()), has_aux=True)(P7)
    close(y, yr, what="primal")
    m = close(J, Jr, what="jacfwd")
    js = str(jax.make_jaxpr(jax.jacfwd(g(S(b, _lanes=3)), has_aux=True))(P7))
    if "+dual_lanes3" not in js:
        raise Fail("instrument: no 3-lane chunk program was built")
    H = jax.jit(jax.jacrev(lambda p: jax.jacfwd(lambda q: g(S(b, _lanes=3))(q)[0][-1])(p)))(P7)
    Hr = jax.jacfwd(jax.jacfwd(lambda q: g(R())(q)[0][-1]))(P7)
    return m + "; " + close(H, Hr, what="jacrev(jacfwd)")


for pol in ("none", "nothing_saveable", "everything_saveable"):
    @case(f"grad(checkpoint(jvp)) policy={pol} (pe_custom split)")
    def _(b, pol=pol):
        P = None if pol == "none" else getattr(jax.checkpoint_policies, pol)
        f = lambda fn: lambda t, v: jax.checkpoint(lambda s: jax.jvp(lambda u: big(fn, u), (s,), (v,))[1], policy=P)(t)
        return close(jax.jit(jax.grad(f(S(b))))(TH, V2), href() @ V2)


@case("grad of scan whose body takes a jvp of tallymap (scan partial eval)")
def _(b):
    def f(fn, th):
        def step(c, _):
            y, dy = jax.jvp(lambda s: big(fn, s), (c,), (V2,))
            return c * 0.9 + 0.02 * dy * jnp.ones_like(c), y
        c, ys = lax.scan(step, th, None, length=2)
        return jnp.sum(c) + jnp.sum(ys)
    return close(jax.jit(jax.grad(lambda t: f(S(b), t)))(TH), jax.jacfwd(lambda t: f(R(), t))(TH))


@case("grad-of-grad (rev-over-rev)")
def _(b):
    f = lambda t: jnp.vdot(jax.grad(lambda s: big(S(b), s))(t), V2)
    return close(jax.jit(jax.grad(f))(TH), href() @ V2)


@case("vmap over theta")
def _(b):
    ths = jnp.stack([TH, TH * 1.1, TH * 0.8])
    return close(jax.jit(jax.vmap(lambda t: big_vec(S(b), t)))(ths), jax.vmap(lambda t: big_vec(R(), t))(ths))


@case("vmap over events (per-item batch)")
def _(b):
    xb = jnp.stack([xs, 1.3 * xs])
    return close(jax.jit(jax.vmap(lambda x: big_vec(S(b), TH, x)))(xb), jax.vmap(lambda x: big_vec(R(), TH, x))(xb))


@case("vmap(grad) over theta")
def _(b):
    ths = jnp.stack([TH, TH * 1.1, TH * 0.8])
    return close(jax.jit(jax.vmap(jax.grad(lambda t: big(S(b), t))))(ths),
                 jax.vmap(jac_ref(lambda t: big(R(), t)))(ths))


@case("grad(vmap) (batched in the loss)")
def _(b):
    ths = jnp.stack([TH, TH * 1.1])
    f = lambda fn: lambda t: jnp.sum(jax.vmap(lambda s: big(fn, s))(ths * t))
    return close(jax.jit(jax.grad(f(S(b))))(TH), jac_ref(f(R()))(TH))


for pol in ("none", "nothing_saveable", "everything_saveable", "dots_saveable"):
    @case(f"grad(checkpoint) policy={pol}")
    def _(b, pol=pol):
        P = None if pol == "none" else getattr(jax.checkpoint_policies, pol)
        g = jax.jit(jax.grad(jax.checkpoint(lambda t: big(S(b), t), policy=P)))(TH)
        return close(g, jac_ref(lambda t: big(R(), t))(TH))


@case("checkpoint inside: grad(scan(checkpoint(step)))")
def _(b):
    def f(fn, th):
        step = jax.checkpoint(lambda c, k: (c * 0.9 + 0.02 * big(fn, c), None))
        return lax.scan(step, th, jnp.arange(2))[0].sum()
    return close(jax.jit(jax.grad(lambda t: f(S(b), t)))(TH), jac_ref(lambda t: f(R(), t))(TH))


@case("tallymap inside lax.scan: grad + jacfwd")
def _(b):
    close(jax.jit(jax.grad(lambda t: in_scan(S(b), t)))(TH), jac_ref(lambda t: in_scan(R(), t))(TH), what="grad")
    return close(jax.jit(jax.jacfwd(lambda t: in_scan(S(b), t)))(TH), jac_ref(lambda t: in_scan(R(), t))(TH),
                 what="jacfwd")


@case("tallymap inside fori_loop (static -> scan): grad")
def _(b):
    return close(jax.jit(jax.grad(lambda t: in_fori(S(b), t)))(TH), jac_ref(lambda t: in_fori(R(), t))(TH))


@case("tallymap inside lax.cond: grad + hessian")
def _(b):
    close(jax.jit(jax.grad(lambda t: in_cond(S(b), t)))(TH), jac_ref(lambda t: in_cond(R(), t))(TH), what="grad")
    return close(jax.jit(jax.hessian(lambda t: in_cond(S(b), t)))(TH),
                 jax.jacfwd(jax.jacfwd(lambda t: in_cond(R(), t)))(TH), what="hessian")


@case("tallymap inside while_loop: fwd + jvp; grad raises (JAX)")
def _(b):
    def f(fn, th):
        return lax.while_loop(lambda c: c[0] < 2, lambda c: (c[0] + 1, c[1] * 0.9 + 0.02 * big(fn, c[1])),
                              (0, th))[1].sum()
    close(jax.jit(lambda t: f(S(b), t))(TH), f(R(), TH), what="fwd")
    close(jax.jvp(lambda t: f(S(b), t), (TH,), (V2,))[1], jax.jvp(lambda t: f(R(), t), (TH,), (V2,))[1], what="jvp")
    expect_raise(lambda: jax.grad(lambda t: f(S(b), t))(TH), "while")


@case("optimiser: lax.scan of grad steps")
def _(b):
    def opt(gfn, th):
        def step(c, _):
            g = gfn(c)
            return c - 1e-3 * g, jnp.sum(g)
        return lax.scan(step, th, None, length=3)
    a = jax.jit(lambda t: opt(jax.grad(lambda s: big(S(b), s)), t))(TH)
    r = opt(jac_ref(lambda s: big(R(), s)), TH)
    return close(a, r)


@case("user custom_jvp wrapper around tallymap")
def _(b):
    def make(fn):
        @jax.custom_jvp
        def g(t):
            return big(fn, t)

        @g.defjvp
        def g_jvp(p, tn):
            return jax.jvp(lambda s: big(fn, s), p, tn)
        return g
    close(jax.jit(jax.grad(make(S(b))))(TH), jac_ref(lambda t: big(R(), t))(TH), what="grad")
    return close(jax.jit(jax.jacfwd(make(S(b))))(TH), jac_ref(lambda t: big(R(), t))(TH), what="jacfwd")


@case("user custom_vjp wrapper around tallymap")
def _(b):
    def make(fn):
        @jax.custom_vjp
        def g(t):
            return big(fn, t)

        def fwd(t):
            y, vj = jax.vjp(lambda s: big(fn, s), t)
            return y, t

        def bwd(t, ct):
            return jax.vjp(lambda s: big(fn, s), t)[1](ct)
        g.defvjp(fwd, bwd)
        return g
    return close(jax.jit(jax.grad(make(S(b))))(TH), jac_ref(lambda t: big(R(), t))(TH))


@case("eval_shape / make_jaxpr (no backend work)")
def _(b):
    s = jax.eval_shape(lambda t: big_vec(S(b), t), TH)
    if s.shape != (2 * NB + 1,):
        raise Fail(f"eval_shape {s}")
    j = jax.make_jaxpr(jax.grad(lambda t: big(S(b), t)))(TH)
    n = str(j).count("tallymap_call")
    return f"grad jaxpr has {n} tallymap_call eqns"


@case("AOT lower/compile")
def _(b):
    c = jax.jit(lambda t: big_vec(S(b), t)).lower(TH).compile()
    close(c(TH), big_vec(R(), TH), what="fwd")
    c2 = jax.jit(jax.grad(lambda t: big(S(b), t))).lower(TH).compile()
    return close(c2(TH * 1.05), jac_ref(lambda t: big(R(), t))(TH * 1.05), what="grad at a new point")


@case("jax.export (serialize/deserialize roundtrip)")
def _(b):
    from jax import export
    f = jax.jit(lambda t: big_vec(S(b), t))
    try:
        exp = export.export(f)(TH)
    except Exception as e:
        if b == "cuda" or (b == "auto" and GPU):
            err = e
            expect_raise(lambda: (_ for _ in ()).throw(err), "custom call")
        raise
    try:
        exp2, note = export.deserialize(exp.serialize()), "serialized roundtrip"
    except ImportError as e:            # flatbuffers missing in this environment
        exp2, note = exp, f"in-process only ({str(e)[:40]})"
    return note + "; " + close(exp2.call(TH), big_vec(R(), TH))


@case("jax.export cuda with the custom-call check disabled", ["cuda"])
def _(b):
    from jax import export
    f = jax.jit(lambda t: big_vec(S(b), t))
    exp = export.export(f, disabled_checks=[export.DisabledSafetyCheck.custom_call("tallymap_cuda_launch")])(TH)
    try:
        exp2, note = export.deserialize(exp.serialize()), "serialized roundtrip"
    except ImportError as e:
        exp2, note = exp, f"in-process only ({str(e)[:40]})"
    return note + "; " + close(exp2.call(TH), big_vec(R(), TH))


@case("donate_argnums")
def _(b):
    f = jax.jit(lambda t, x: big_vec(S(b), t, x), donate_argnums=(1,))
    out = f(TH, xs * 1.0)
    return close(out, big_vec(R(), TH))


@case("disable_jit: fwd + grad + jacfwd")
def _(b):
    with jax.disable_jit():
        y = big_vec(S(b), TH)
        g = jax.grad(lambda t: big(S(b), t))(TH)
        J = jax.jacfwd(lambda t: big_vec(S(b), t))(TH)
    close(y, big_vec(R(), TH), what="fwd")
    close(g, jac_ref(lambda t: big(R(), t))(TH), what="grad")
    return close(J, jac_ref(lambda t: big_vec(R(), t))(TH), what="jacfwd")


@case("debug_nans on (clean program)")
def _(b):
    with jax.debug_nans(True):
        g = jax.jit(jax.grad(lambda t: big(S(b), t)))(TH)
    return close(g, jac_ref(lambda t: big(R(), t))(TH))


@case("debug_nans on (NaN produced inside the item)")
def _(b):
    def bad(t, x):
        return t.add("Q", 0, jnp.log(x - 1.0))           # NaN for x < 1
    f = tm.vmap(bad, outputs=OUT, backend=b)
    with jax.debug_nans(True):
        expect_raise(lambda: jax.jit(lambda s: f(xs * s).outputs["Q"])(F(1.0)).block_until_ready(), "nan")


@case("pmap over devices (theta per device)")
def _(b):
    nd = jax.local_device_count()
    if nd < 2:
        return f"skipped: {nd} device"
    ths = jnp.stack([TH * (1 + 0.1 * i) for i in range(nd)])
    a = jax.pmap(lambda t: big_vec(S(b), t))(ths)
    close(a, jax.vmap(lambda t: big_vec(R(), t))(ths), what="pmap fwd")
    g = jax.pmap(jax.grad(lambda t: big(S(b), t)))(ths)
    return close(g, jax.vmap(jac_ref(lambda t: big(R(), t)))(ths), what="pmap grad")


@case("shard_map over devices (items sharded)")
def _(b):
    nd = jax.local_device_count()
    if nd < 2:
        return f"skipped: {nd} device"
    try:
        from jax import shard_map as smap  # jax >= 0.7: default check_vma=True
    except ImportError:
        from jax.experimental.shard_map import shard_map as smap  # 0.5.3: default check_rep=True
    kw = {}
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P
    mesh = Mesh(np.array(jax.local_devices()[:nd]), ("d",))
    n = (N // nd) * nd
    f = S(b)

    def local(x, k, th):
        q = f(x, k, tab, th).outputs["Q"]
        return lax.psum(q, "d")
    g = smap(local, mesh=mesh, in_specs=(P("d"), P("d"), P()), out_specs=P(), **kw)
    note = "default checks"
    try:
        q = jax.jit(g)(xs[:n], keys[:n], TH)
    except Exception as e:
        if "check_rep" not in str(e) and "replication" not in str(e):
            raise
        note = f"check_rep=False needed ({type(e).__name__}: {str(e)[:80]})"
        g = smap(local, mesh=mesh, in_specs=(P("d"), P("d"), P()), out_specs=P(), check_rep=False)
        q = jax.jit(g)(xs[:n], keys[:n], TH)
    close(q, R()(xs[:n], keys[:n], tab, TH).outputs["Q"], what="fwd")
    gr = jax.jit(jax.grad(lambda t: jnp.sum(g(xs[:n], keys[:n], t) * WQ)))(TH)
    gref = jac_ref(lambda t: jnp.sum(R()(xs[:n], keys[:n], tab, t).outputs["Q"] * WQ))(TH)
    return note + "; " + close(gr, gref, what="grad")


@case("per-item results + truncated under grad / jacrev")
def _(b):
    def item(t, x, th):
        k, w, t = lax.while_loop(lambda c: c[1] > 0.2, lambda c: (c[0] + 1, c[1] * th, c[2].add("Q", c[0] % NB, c[1])),
                                 (0, x, t))
        return t, (w, k)
    f = tm.vmap(item, outputs=OUT, in_axes=(0, None), max_steps=6, backend=b)
    r = reference(item, outputs=OUT, in_axes=(0, None), max_steps=6)
    A, Rr = f(xs, F(0.8)), r(xs, F(0.8))
    nt = int(A.stats["n_truncated"])
    if nt != int(Rr.stats["n_truncated"]):
        raise Fail("n_truncated")
    if nt == 0:
        raise Fail("instrument: no item was truncated (the cap is not exercised)")
    # a truncated item poisons the float tally, so every gradient that reads it is NaN (both implementations)
    gq = jax.jit(jax.grad(lambda th: jnp.sum(f(xs, th).outputs["Q"] * WQ)))(F(0.8))
    gqr = jax.jacfwd(lambda th: jnp.sum(r(xs, th).outputs["Q"] * WQ))(F(0.8))
    if not (np.isnan(float(gq)) and np.isnan(float(gqr))):
        raise Fail(f"the gradient of a poisoned tally is not NaN ({float(gq)}, {float(gqr)})")
    # the per-item results are not poisoned: their gradients agree with the dense reference
    loss = lambda fn: lambda th: jnp.sum(fn(xs, th).per_item[0] ** 2)
    close(jax.jit(jax.grad(loss(f)))(F(0.8)), jax.jacfwd(loss(r))(F(0.8)), what="grad")
    J = jax.jit(jax.jacrev(lambda th: f(xs, th).per_item[0]))(F(0.8))
    return close(J, jax.jacfwd(lambda th: r(xs, th).per_item[0])(F(0.8)), what="jacrev per-item") + f"; {nt} truncated"


@case("spawn: jvp/jacfwd and grad (tree adjoint)")
def _(b):
    def item(t, gen, x):
        t = t.add("Q", gen, x)
        return t.spawn((gen + 1, x * 0.5), when=gen < 2)
    g0 = jnp.zeros((N,), jnp.int32)
    f = tm.vmap(item, outputs=OUT, spawn=tm.Queue(1 << 10), backend=b)
    jt = jax.jacfwd(lambda s: f(g0, xs * s).outputs["Q"])(F(1.0))
    S0 = float(xs.sum())
    close(jt, np.array([S0, 0.5 * S0, 0.25 * S0, 0, 0, 0, 0, 0]), what="jacfwd")
    return close(jax.jit(jax.grad(lambda s: jnp.sum(f(g0, xs * s).outputs["Q"] * WQ)))(F(1.0)),
                 jax.jacfwd(lambda s: jnp.sum(f(g0, xs * s).outputs["Q"] * WQ))(F(1.0)), what="grad vs jacfwd")


@case("closed-over tracer (param captured by closure) under grad/hessian")
def _(b):
    def mk(fn_maker, th):
        return fn_maker(lambda t, x, key: walk(t, x, key, tab * th[0], th))(xs, keys).outputs["Q"]
    sm = lambda it: tm.vmap(it, outputs=OUT, backend=b)
    rm = lambda it: reference(it, outputs=OUT)
    lossS = lambda th: jnp.sum(mk(sm, th) * WQ)
    lossR = lambda th: jnp.sum(mk(rm, th) * WQ)
    close(jax.jit(jax.grad(lossS))(TH), jax.jacfwd(lossR)(TH), what="grad")
    return close(jax.jit(jax.hessian(lossS))(TH), jax.jacfwd(jax.jacfwd(lossR))(TH), what="hessian")


@case("integer / bool operands and outputs (float0) under grad + jacrev")
def _(b):
    ii = jnp.arange(N, dtype=jnp.int32)

    def item(t, x, i, th):
        return t.add("Q", i % NB, x * th).add("C", i % NB, 1), (i * 2, x * th > 1.0, x * th)
    out = {"Q": tm.Sum((NB,), F), "C": tm.Sum((NB,), jnp.int32)}
    f = tm.vmap(item, outputs=out, in_axes=(0, 0, None), backend=b)
    r = reference(item, outputs=out, in_axes=(0, 0, None))
    loss = lambda fn: lambda th: jnp.sum(fn(xs, ii, th).outputs["Q"]) + jnp.sum(fn(xs, ii, th).per_item[2])
    close(jax.grad(loss(f))(F(0.9)), jax.jacfwd(loss(r))(F(0.9)), what="grad")
    J = jax.jacrev(lambda th: f(xs, ii, th).outputs["Q"])(F(0.9))
    close(J, jax.jacfwd(lambda th: r(xs, ii, th).outputs["Q"])(F(0.9)), what="jacrev")
    t = jax.jvp(lambda th: f(xs, ii, th).per_item[0], (F(0.9),), (F(1.0),))[1]
    if t.dtype != jax.dtypes.float0:
        raise Fail(f"int tangent dtype {t.dtype}")


@case("grad w.r.t. per-item inputs AND tables (argnums)")
def _(b):
    loss = lambda fn: lambda x, tb, th: (jnp.sum(fn(x, keys, tb, th).outputs["Q"] * WQ)
                                         + jnp.sum(fn(x, keys, tb, th).per_item[0]))
    g = jax.jit(jax.grad(loss(S(b)), argnums=(0, 1, 2)))(xs, tab, TH)
    gr = jax.jacfwd(loss(R()), argnums=(0, 1, 2))(xs, tab, TH)
    return close(g, gr)


@case("recompilation: jit cache stable across calls / scan reuse")
def _(b):
    f = jax.jit(lambda t: big(S(b), t))
    for k in range(3):
        f(TH * (1 + 0.01 * k)).block_until_ready()
    g = jax.jit(jax.grad(lambda t: big(S(b), t)))
    for k in range(3):
        g(TH * (1 + 0.01 * k)).block_until_ready()
    if f._cache_size() != 1 or g._cache_size() != 1:
        raise Fail(f"cache sizes {f._cache_size()} {g._cache_size()}")
    from tallymap import _primitive
    n0 = len(_primitive._EAGER)
    for k in range(3):
        S(b)(xs, keys, tab, TH * (1 + 0.01 * k)).outputs["Q"].block_until_ready()
    return f"jit caches 1/1; eager cache +{len(_primitive._EAGER) - n0} for 3 eager calls"


print("\n================ SUMMARY", jax.__version__, "x64" if jax.config.jax_enable_x64 else "",
      jax.devices()[0].device_kind, "ndev", jax.local_device_count())
for n, b, m in RES:
    print(f"{n:44s} {b:8s} {m[:200]}")
cnt = lambda p: sum(m.startswith(p) for _, _, m in RES)
print("TOTAL", len(RES), "PASS", cnt("PASS"), "RAISES-OK", cnt("RAISES-OK"), "FAIL", cnt("FAIL"), "RAISE",
      cnt("RAISE "))
