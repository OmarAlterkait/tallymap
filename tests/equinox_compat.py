"""Equinox-ecosystem solvers inside an item.  python tests/equinox_compat.py
env: BACKENDS=lockstep,cuda  TEST_X64=1  ONLY=substr  N=64 (items)

Diffrax, Optimistix, Lineax and Quadax all build on `equinox.internal.while_loop`, whose primitives tallymap
expands at the item trace (tallymap/_foreign.py), plus the small dense factorisations the implicit solvers need
(tallymap/backends/cuda/dense.py).  This suite runs the user's literal solver call inside tm.vmap and compares it
with jax.vmap of the same function.

Each cell is one (item, mode):
  fwd    the per-item result and the tally against jax.vmap(f)
  jvp    jax.jvp of both
  grad   jax.grad of the tallymap program against jax.jacfwd of the SAME program, and against
         jax.grad(jax.vmap(f)) where JAX can take it (it cannot for diffrax's ForwardMode, whose loop is a plain
         lax.while_loop: that gradient exists only because tallymap differentiates the loop itself)

Tolerances are per item, not one global number.  An adaptive solver's step sequence is decided by a comparison of
floating-point numbers, and tallymap's arithmetic is not XLA's bit for bit, so two runs can take DIFFERENT steps
and then agree only to about the solver's own tolerance.  `tol` records what was measured for each item; a
constant-step solver is held to f32 reassociation (1e-6).

This file is skipped when Equinox is not installed.  It sets EQX_ON_ERROR=off, and both halves of that matter:

  * Equinox's DEFAULT (raise) is a host callback, which a kernel cannot make.  One cell checks the message.
  * `nan` is the right choice for a forward-only item, but not here.  MEASURED: with EQX_ON_ERROR=nan,
    jax.grad of a Kvaerno3 solve inside an item is NaN for every item while jacfwd of the same program is right.
    A check inside the implicit solver's COTANGENT solve reports a failure, and Lineax passes throw=True to that
    solve unconditionally ("nowhere we can pipe their error to"), so the only place the failure can go is the
    value.  `off` removes the check; passing root_finder=optimistix.Newton(...) also avoids it, because the
    frozen Jacobian of the default VeryChord is what the cotangent solve trips over.  One cell pins that down.
"""
import os
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("EQX_ON_ERROR", "off")       # must be set BEFORE equinox is imported (see above)
if os.environ.get("TEST_X64"):
    import jax
    jax.config.update("jax_enable_x64", True)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

try:
    import diffrax as dfx
    import equinox as eqx
    import lineax as lx
    import optimistix as optx
    import quadax as qx
except ImportError as e:                                     # pragma: no cover (the normal CI has them)
    print(f"equinox_compat: Equinox is not installed ({e})")
    print("SUITE SKIP")
    sys.exit(0)

import tallymap as tm  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")
GPU = jax.default_backend() == "gpu"
BACKENDS = os.environ.get("BACKENDS", "lockstep,cuda" if GPU else "lockstep").split(",")
ONLY = os.environ.get("ONLY", "")
N = int(os.environ.get("N", 64))
RES = []
print("jax", jax.__version__, "equinox", eqx.__version__, "diffrax", dfx.__version__,
      "optimistix", optx.__version__, "lineax", lx.__version__, "quadax", qx.__version__,
      "| x64", jax.config.jax_enable_x64, jax.devices(), flush=True)


class Fail(Exception):
    pass


def close(a, b, rtol, atol=1e-6, what=""):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    if len(la) != len(lb):
        raise Fail(f"{what} {len(la)} leaves vs {len(lb)}")
    worst = 0.0
    for x, y in zip(la, lb):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        if x.shape != y.shape:
            raise Fail(f"{what} shape {x.shape} vs {y.shape}")
        if not np.isfinite(x).all():
            raise Fail(f"{what} non-finite result")
        scale = float(np.max(np.abs(y))) if y.size else 0.0
        err = float(np.max(np.abs(x - y))) if x.size else 0.0
        if not err <= rtol * scale + atol:
            raise Fail(f"{what} max err {err:.3g} > {rtol:.1e}*{scale:.3g}+{atol:.1e}")
        worst = max(worst, err / max(scale, 1e-300))
    return f"rel {worst:.2e}"


def raises(fn, contains):
    """-> a RAISES-OK message when fn() raises with `contains` in the message."""
    try:
        fn()
    except Exception as e:
        if contains not in str(e):
            raise Fail(f"raised {type(e).__name__} without {contains!r}: {str(e)[:250]}") from None
        return f"raises {type(e).__name__}, message names {contains}"
    raise Fail(f"did not raise (expected a message naming {contains})")


def case(name, backends=None):
    def deco(fn):
        for b in (backends or BACKENDS):
            if b not in BACKENDS or (ONLY and ONLY not in name):
                continue
            t0 = time.perf_counter()
            try:
                r = fn(b)
                msg = "PASS" + (f" ({r})" if isinstance(r, str) and r else "")
            except Fail as e:
                msg = f"FAIL {e}"
            except Exception as e:
                msg = f"RAISE {type(e).__name__}: {str(e)[:300]}"
                traceback.print_exc()
            RES.append((name, b, msg))
            print(f"{name:40s} {b:8s} {msg}  [{time.perf_counter() - t0:.1f}s]", flush=True)
        return fn
    return deco


# =================================================================================== the items
# Each is the literal library call a user writes, as ONE item: un-batched, one trajectory or one solve.
FWD = dfx.ForwardMode()


def _pid(rtol=1e-6, atol=1e-8):
    return dfx.PIDController(rtol=rtol, atol=atol)


def _decay(t, y, p):
    return -p * y


def _vdp(t, y, p):
    return jnp.stack([y[1], p * ((1.0 - y[0] * y[0]) * y[1] - y[0])])


def _lotka(t, y, p):
    return jnp.stack([p * y[0] - y[0] * y[1], y[0] * y[1] - p * y[1]])


def tsit5_const(p):
    return dfx.diffeqsolve(dfx.ODETerm(_decay), dfx.Tsit5(), 0.0, 1.0, 0.1, jnp.asarray(1.0), args=p,
                           adjoint=FWD, max_steps=64).ys[0]


def tsit5_pid(p):
    return dfx.diffeqsolve(dfx.ODETerm(_decay), dfx.Tsit5(), 0.0, 1.0, 0.01, jnp.asarray(1.0), args=p,
                           stepsize_controller=_pid(), adjoint=FWD, max_steps=4096).ys[0]


def lotka(p):
    return dfx.diffeqsolve(dfx.ODETerm(_lotka), dfx.Tsit5(), 0.0, 2.0, 0.01, jnp.array([1.0, 1.0]), args=p,
                           stepsize_controller=_pid(), adjoint=FWD, max_steps=4096).ys[0, 0]


def _kv(solver, p):
    return dfx.diffeqsolve(dfx.ODETerm(_vdp), solver, 0.0, 1.0, 0.01, jnp.array([2.0, 0.0]), args=p,
                           stepsize_controller=_pid(1e-5, 1e-7), adjoint=FWD, max_steps=4096).ys[0, 0]


def kvaerno3(p):
    return _kv(dfx.Kvaerno3(), p)


def kvaerno5(p):
    return _kv(dfx.Kvaerno5(), p)


def kvaerno3_newton(p):
    return _kv(dfx.Kvaerno3(root_finder=optx.Newton(rtol=1e-6, atol=1e-8)), p)


def sde_euler(p):
    bm = dfx.VirtualBrownianTree(0.0, 1.0, tol=1e-3, shape=(), key=jax.random.PRNGKey(0))
    terms = dfx.MultiTerm(dfx.ODETerm(_decay), dfx.ControlTerm(lambda t, y, a: 0.1 * y, bm))
    return dfx.diffeqsolve(terms, dfx.Euler(), 0.0, 1.0, 0.05, jnp.asarray(1.0), args=p,
                           adjoint=FWD, max_steps=256).ys[0]


def event(p):
    ev = dfx.Event(lambda t, y, args, **kw: y - 0.5)
    return dfx.diffeqsolve(dfx.ODETerm(_decay), dfx.Tsit5(), 0.0, 10.0, 0.01, jnp.asarray(1.0), args=p,
                           stepsize_controller=_pid(), event=ev, adjoint=FWD, max_steps=4096).ts[0]


def adj_recursive(p):
    """diffrax's DEFAULT adjoint: a custom_vjp around a treeverse loop."""
    return dfx.diffeqsolve(dfx.ODETerm(_decay), dfx.Tsit5(), 0.0, 1.0, 0.01, jnp.asarray(1.0), args=p,
                           stepsize_controller=_pid(), adjoint=dfx.RecursiveCheckpointAdjoint(),
                           max_steps=4096).ys[0]


def adj_direct(p):
    return dfx.diffeqsolve(dfx.ODETerm(_decay), dfx.Tsit5(), 0.0, 1.0, 0.01, jnp.asarray(1.0), args=p,
                           stepsize_controller=_pid(), adjoint=dfx.DirectAdjoint(), max_steps=4096).ys[0]


def _cubic(x, p):
    return x ** 3 - p


def optx_newton(p):
    return optx.root_find(_cubic, optx.Newton(rtol=1e-8, atol=1e-10), jnp.asarray(1.0), args=p,
                          max_steps=64, throw=False).value


def optx_chord(p):
    return optx.root_find(_cubic, optx.Chord(rtol=1e-8, atol=1e-10), jnp.asarray(1.0), args=p,
                          max_steps=64, throw=False).value


def optx_bisection(p):
    return optx.root_find(_cubic, optx.Bisection(rtol=1e-8, atol=1e-10), jnp.asarray(1.0), args=p,
                          options=dict(lower=0.1, upper=4.0), max_steps=64, throw=False).value


def optx_lsq(p):
    t = jnp.linspace(0.0, 1.0, 8)
    data = 2.0 * jnp.exp(-p * t)
    f = lambda z, d: z[0] * jnp.exp(-z[1] * t) - d
    # value[1] is the fitted decay rate, which tracks p (value[0] is the amplitude, which does not)
    return optx.least_squares(f, optx.LevenbergMarquardt(rtol=1e-6, atol=1e-8), jnp.array([1.0, 1.0]),
                              args=data, max_steps=64, throw=False).value[1]


def optx_minimise(p):
    return optx.minimise(lambda z, a: (z - a) ** 2 + 1.0, optx.BFGS(rtol=1e-6, atol=1e-8), jnp.asarray(0.0),
                         args=p, max_steps=64, throw=False).value


def _mat(p):
    return jnp.array([[3.0 + p, 1.0, 0.5], [1.0, 2.5 + p, 1.0], [0.5, 1.0, 4.0 + p]])


RHS = jnp.array([1.0, 2.0, 3.0])


def lx_lu(p):
    return lx.linear_solve(lx.MatrixLinearOperator(_mat(p)), RHS, solver=lx.LU()).value[0]


def lx_qr(p):
    return lx.linear_solve(lx.MatrixLinearOperator(_mat(p)), RHS, solver=lx.QR()).value[0]


def lx_cholesky(p):
    m = _mat(p)
    op = lx.MatrixLinearOperator(m @ m.T, lx.positive_semidefinite_tag)
    return lx.linear_solve(op, RHS, solver=lx.Cholesky()).value[0]


def lx_cg(p):
    m = _mat(p)
    op = lx.MatrixLinearOperator(m @ m.T, lx.positive_semidefinite_tag)
    return lx.linear_solve(op, RHS, solver=lx.CG(rtol=1e-8, atol=1e-10)).value[0]


def qx_quadgk(p):
    return qx.quadgk(lambda x, a: jnp.exp(-a * x * x), [0.0, 1.0], args=(p,), epsabs=1e-8, epsrel=1e-8)[0]


def qx_quadts(p):
    return qx.quadts(lambda x, a: jnp.exp(-a * x * x), [0.0, 1.0], args=(p,), epsabs=1e-8, epsrel=1e-8)[0]


# name -> (f, p0, spread, modes, rtol fwd/jvp, rtol grad)
ALL = ("fwd", "jvp", "grad")
ITEMS = {
    "dfx_tsit5_const":     (tsit5_const, 1.0, 0.5, ALL, 1e-6, 1e-5),
    "dfx_tsit5_pid":       (tsit5_pid, 1.0, 0.5, ALL, 1e-5, 1e-4),
    "dfx_lotka":           (lotka, 1.5, 0.3, ("fwd", "grad"), 1e-4, 2e-3),
    "dfx_kvaerno3":        (kvaerno3, 10.0, 4.0, ALL, 1e-4, 3e-3),
    "dfx_kvaerno5":        (kvaerno5, 10.0, 4.0, ("fwd", "grad"), 1e-4, 3e-3),
    "dfx_kvaerno3_newton": (kvaerno3_newton, 10.0, 4.0, ("fwd",), 1e-4, 3e-3),
    "dfx_sde_euler":       (sde_euler, 1.0, 0.5, ("fwd", "grad"), 1e-5, 1e-4),
    "dfx_event":           (event, 1.0, 0.5, ("fwd", "grad"), 1e-5, 1e-4),
    "dfx_adj_recursive":   (adj_recursive, 1.0, 0.5, ALL, 1e-5, 1e-4),
    "optx_newton":         (optx_newton, 8.0, 4.0, ALL, 1e-6, 1e-5),
    "optx_chord":          (optx_chord, 1.0, 0.3, ("fwd", "grad"), 1e-6, 1e-5),
    "optx_bisection":      (optx_bisection, 8.0, 4.0, ("fwd",), 1e-6, 1e-5),
    "optx_lsq":            (optx_lsq, 1.0, 0.4, ("fwd", "grad"), 1e-4, 1e-2),
    "optx_minimise":       (optx_minimise, 1.0, 0.5, ("fwd", "grad"), 1e-6, 1e-5),
    "lx_lu":               (lx_lu, 0.5, 0.4, ALL, 1e-6, 1e-5),
    "lx_qr":               (lx_qr, 0.5, 0.4, ("fwd", "grad"), 1e-6, 1e-5),
    "lx_cholesky":         (lx_cholesky, 0.5, 0.4, ("fwd", "grad"), 1e-5, 1e-4),
    "lx_cg":               (lx_cg, 0.5, 0.4, ("fwd", "grad"), 1e-5, 1e-4),
    "qx_quadgk":           (qx_quadgk, 1.0, 0.5, ALL, 1e-6, 1e-5),
    "qx_quadts":           (qx_quadts, 1.0, 0.5, ("fwd", "grad"), 1e-6, 1e-5),
}
# diffrax's ForwardMode is a plain lax.while_loop, so JAX itself cannot reverse-differentiate these:
# the gradient tallymap gives is checked against jacfwd only.
NO_JAX_GRAD = {"dfx_tsit5_const", "dfx_tsit5_pid", "dfx_lotka", "dfx_kvaerno3", "dfx_kvaerno5",
               "dfx_sde_euler", "dfx_event"}
# (item, mode) pairs that must RAISE with advice instead of returning a number.  diffrax's default
# RecursiveCheckpointAdjoint is a custom_vjp, which JAX cannot forward-differentiate at all, and tallymap's
# reverse mode probes the item in forward mode; the error has to name ForwardMode.
RAISES = {("dfx_adj_recursive", "jvp"): "ForwardMode"}
# items whose grad exists but whose jacfwd does not, so the grad can only be checked against jax.grad(jax.vmap)
NO_JACFWD = {"dfx_adj_recursive"}


def item_of(f):
    """f(p) as ONE tallymap item: it also emits its result, so the tally path is exercised too."""
    def it(t, p):
        y = f(p)
        return t.add("S", 0, y), y
    return it


def prog(f, b, **kw):
    return tm.vmap(item_of(f), outputs={"S": tm.Sum((1,))}, backend=b, **kw)


def params(p0, spread):
    return jnp.asarray(p0 + spread * np.linspace(-1.0, 1.0, N), jnp.float32 if not
                       jax.config.jax_enable_x64 else jnp.float64)


def register(name, f, p0, spread, modes, rt, rg):
    ps = params(p0, spread)

    if "fwd" in modes:
        @case(f"{name} fwd")
        def _(b, f=f, ps=ps, rt=rt):
            r = jax.jit(prog(f, b))(ps)
            want = jax.jit(jax.vmap(f))(ps)
            m1 = close(r.per_item, want, rt, what="per_item")
            m2 = close(r.outputs["S"][0], jnp.sum(want, dtype=want.dtype), rt * 4, 1e-4, what="tally")
            return f"{m1}; tally {m2}; steps<={int(np.max(np.asarray(r.truncated, np.int32)))}"

    if "jvp" in modes:
        @case(f"{name} jvp")
        def _(b, f=f, ps=ps, rt=rt, name=name):
            dps = jnp.ones_like(ps)
            g = lambda x: prog(f, b)(x).per_item
            want = RAISES.get((name, "jvp"))
            if want is not None:
                return raises(lambda: jax.jit(lambda a, d: jax.jvp(g, (a,), (d,)))(ps, dps), want)
            y, t = jax.jit(lambda a, d: jax.jvp(g, (a,), (d,)))(ps, dps)
            yr, tr = jax.jit(lambda a, d: jax.jvp(jax.vmap(f), (a,), (d,)))(ps, dps)
            return close((y, t), (yr, tr), rt, what="jvp")

    if "grad" in modes:
        @case(f"{name} grad")
        def _(b, f=f, ps=ps, rg=rg, name=name):
            loss = lambda x: jnp.sum(prog(f, b)(x).per_item)
            want = RAISES.get((name, "grad"))
            if want is not None:
                return raises(lambda: jax.jit(jax.grad(loss))(ps), want)
            g = jax.jit(jax.grad(loss))(ps)
            if name in NO_JACFWD:
                m = "jacfwd does not exist here"
            else:
                m = close(g, jax.jit(jax.jacfwd(loss))(ps), rg, what="grad vs jacfwd")
            if name in NO_JAX_GRAD:
                return m + " (jax.grad of the reference does not exist)"
            gj = jax.jit(jax.grad(lambda x: jnp.sum(jax.vmap(f)(x))))(ps)
            return m + "; vs jax " + close(g, gj, rg, what="grad vs jax.grad")


for _n, (_f, _p0, _s, _m, _rt, _rg) in ITEMS.items():
    register(_n, _f, _p0, _s, _m, _rt, _rg)


# =================================================================================== behaviour cells
@case("truncation: max_steps on a diffrax loop poisons the tally")
def _(b):
    ps = params(10.0, 4.0)
    r = jax.jit(prog(kvaerno3, b, max_steps=8))(ps)
    if not bool(np.any(np.asarray(r.truncated))):
        raise Fail("no item was truncated with max_steps=8")
    if not bool(np.all(np.isnan(np.asarray(r.outputs["S"])))):
        raise Fail("a truncated diffrax solve did not poison the float tally")
    return f"{int(np.sum(np.asarray(r.truncated)))}/{N} truncated, tally NaN"


@case("the foreign primitives are gone from every item program", backends=["lockstep"])
def _(b):
    from tallymap import _foreign
    d = tm.diagnostics(jax.jit(jax.grad(lambda x: jnp.sum(prog(kvaerno3, b)(x).per_item))), params(10.0, 4.0))
    if not d["programs"]:
        raise Fail("no item program was recorded")
    left = sorted({n for n in _foreign.NAMES for p_ in d["programs"] if f" {n} " in p_["jaxpr"]})
    if left:
        raise Fail(f"still present: {left}")
    n = len(d["programs"])
    return f"{n} item program{'' if n == 1 else 's'}, none holding any of the {len(_foreign.NAMES)} foreign names"


@case("EQX_ON_ERROR=nan: an implicit solver's grad is NaN, jacfwd is right", backends=["lockstep"])
def _(b):
    src = (
        "import os, sys; os.environ['EQX_ON_ERROR'] = 'nan'\n"
        "import jax, jax.numpy as jnp, numpy as np, diffrax as dfx, tallymap as tm\n"
        "def vdp(t, y, p): return jnp.stack([y[1], p * ((1.0 - y[0] * y[0]) * y[1] - y[0])])\n"
        "def kv(p):\n"
        "    return dfx.diffeqsolve(dfx.ODETerm(vdp), dfx.Kvaerno3(), 0.0, 1.0, 0.01,\n"
        "        jnp.array([2.0, 0.0]), args=p,\n"
        "        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-7),\n"
        "        adjoint=dfx.ForwardMode(), max_steps=4096).ys[0, 0]\n"
        "f = tm.vmap(lambda t, p: (t.add('S', 0, kv(p)), kv(p)), outputs={'S': tm.Sum((1,))},\n"
        "            backend='lockstep')\n"
        "ps = jnp.asarray(10.0 + 4.0 * np.linspace(-1, 1, 4), jnp.float32)\n"
        "loss = lambda x: jnp.sum(f(x).per_item)\n"
        "g = np.asarray(jax.jit(jax.grad(loss))(ps))\n"
        "d = np.diag(np.asarray(jax.jit(jax.jacfwd(loss))(ps)))\n"
        "print('GRAD_FINITE', int(np.isfinite(g).all()), 'JACFWD_FINITE', int(np.isfinite(d).all()))\n")
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env.pop("EQX_ON_ERROR", None)
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, env=env, timeout=1800)
    out = r.stdout + r.stderr
    if "JACFWD_FINITE 1" not in out:
        raise Fail(f"jacfwd was not finite under EQX_ON_ERROR=nan either: {out[-400:]}")
    if "GRAD_FINITE 0" not in out:
        return "grad is finite under EQX_ON_ERROR=nan now: relax the docs"
    return "grad NaN, jacfwd right, as documented"


@case("eqx.error_if with the default EQX_ON_ERROR says what to set")
def _(b):
    src = (
        "import os, sys; os.environ['EQX_ON_ERROR'] = 'raise'\n"
        "import jax, jax.numpy as jnp, equinox as eqx, tallymap as tm\n"
        "def it(t, p):\n"
        "    y = eqx.error_if(p * 2.0, p < 0.0, 'negative')\n"
        "    return t.add('S', 0, y), y\n"
        f"f = tm.vmap(it, outputs={{'S': tm.Sum((1,))}}, backend={b!r})\n"
        "try:\n"
        "    f(jnp.ones((4,), jnp.float32))\n"
        "except Exception as e:\n"
        "    print('MSG', str(e)); sys.exit(0)\n"
        "print('NO-RAISE')\n")
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env.pop("EQX_ON_ERROR", None)
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, env=env, timeout=1800)
    out = r.stdout + r.stderr
    if "NO-RAISE" in out:
        if b == "cuda":
            raise Fail("a host callback ran inside a cuda kernel")
        return "equinox's raise mode runs here: the lockstep backend can call the host"
    if "EQX_ON_ERROR=nan" not in out:
        raise Fail(f"message without the advice: {out[-400:]}")
    return "message names EQX_ON_ERROR=nan"


@case("diffrax DirectAdjoint grad: a number that is right, or a clear error")
def _(b):
    ps = params(1.0, 0.5)
    loss = lambda x: jnp.sum(prog(adj_direct, b)(x).per_item)
    try:
        g = jax.jit(jax.grad(loss))(ps)
    except Exception as e:
        if "ForwardMode" not in str(e):
            raise Fail(f"{type(e).__name__} without the advice: {str(e)[:300]}")
        return f"{type(e).__name__}: names ForwardMode"
    gf = jax.jit(jax.jacfwd(loss))(ps)
    return "DirectAdjoint grad ran: " + close(g, gf, 1e-4, what="grad vs jacfwd")


@case("a matrix above the dense limit is refused with a reason", backends=["cuda"])
def _(b):
    big = 24

    def it(t, p):
        m = jnp.eye(big, dtype=jnp.float32) * (1.0 + p) + 0.01
        y = lx.linear_solve(lx.MatrixLinearOperator(m), jnp.ones((big,), jnp.float32), solver=lx.LU()).value[0]
        return t.add("S", 0, y), y
    f = tm.vmap(it, outputs={"S": tm.Sum((1,))}, backend="cuda")
    try:
        f(params(0.5, 0.4))
    except tm.UnsupportedError as e:
        if "16" not in str(e):
            raise Fail(f"no size in the message: {str(e)[:300]}")
        return "UnsupportedError names the limit"
    raise Fail("a 24x24 LU inside an item did not raise")


print("\n================ SUMMARY", jax.__version__, "x64" if jax.config.jax_enable_x64 else "",
      jax.devices()[0].device_kind, "N", N)
for _n, _b, _m in RES:
    print(f"{_n:40s} {_b:8s} {_m[:200]}")
_cnt = lambda p: sum(m.startswith(p) for _, _, m in RES)
print("TOTAL", len(RES), "PASS", _cnt("PASS"), "FAIL", _cnt("FAIL"), "RAISE", _cnt("RAISE "))
