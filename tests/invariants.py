"""Invariants the library must keep whatever it does internally.  python tests/invariants.py
env: BACKENDS=lockstep,cuda  TEST_X64=1  ONLY=substr

docs: the consolidation plan's section 9 ("fundamental tests") -- the suite that makes removing internal paths safe.
Sections (cell name prefixes):
  [sched]  schedule invariance: _schedule step/item, _fold, _lanes chunking under jacfwd (T = 1/16/17/33, MAX_LANES =
           16), _cuda_inplace, lockstep vs cuda, item order.  BITWISE wherever the arithmetic is the same program
           (exact and integer tallies, per-item results); a float tally is a sum whose ORDER the schedule decides,
           so it is held to the stated reassociation tolerance FT.
  [key]    cache-key soundness: programs that differ only in a closed-over array's values, a closed-over Python
           scalar, dtype, shape, an option, max_steps or table contents never share a compiled program (one process,
           one fresh TALLYMAP_CUDA_CACHE).  NEGATIVE CONTROLS: the same pairs under a deliberately coarse key
           (jaxpr text only / the real keys' structural digest only / avals only) MUST be caught; the control cell
           PASSES when the stale reuse is detected.
  [tdep]   tangent-dependent control flow: a while loop whose trip count follows the tangent-carrying primal;
           lineax.CG under jacfwd (the _foreign.lanes_unsafe case; skipped without lineax).
  [size]   run-or-raise: dense solve / cholesky / qr and per-item carries on a size ladder either run and match
           jax.vmap, or raise a tallymap error that names the size.  cuda: no silent register-capped kernel.
  [ad]     AD self-consistency without finite differences: <u, J v> by jvp == by vjp; jacfwd == jacrev; the
           gradient does not depend on rev_checkpoints.
  [trunc]  truncation (max_steps): flags, n_truncated, partial integer tallies, poisoned float tallies and per-item
           results agree between cuda, lockstep and the dense reference, with n_active and across cuda schedules.
  (Replay guard negative control: already pinned by tests/reverse.py, cell "REPLAY CHECK: ... NEGATIVE CONTROL
  adjoint math=approx must mismatch -> NaN" (private _adj_math; the gradient must come back NaN, also without
  rev_stats).  Not duplicated here.)

Each cell prints one line: PASS | RAISES-OK (an expected, clear error) | FAIL <why> | RAISE <exception> |
XFAIL <reason> (a KNOWN library defect, pinned: the cell fails the way the reason says) | XPASS (an XFAIL cell that
now passes: the defect is gone, remove the marker -- tests/run_all.py counts XPASS as a failure).  A cell whose
optional dependency is missing prints SKIP and is not counted.
Summary: TOTAL n PASS k RAISES-OK m XFAIL x (then FAIL / RAISE / XPASS counts).
"""
import atexit
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("EQX_ON_ERROR", "off")     # lineax's default error mode is a host callback (equinox_compat.py)
# [key] one FRESH cubin cache for the whole suite: every kernel this process uses is compiled here, so a stale entry
# from another process cannot hide (or fake) a key collision.  Set before tallymap reads it (cache_dir() reads the
# environment at every compile).
_CACHE = tempfile.mkdtemp(prefix="tm_invariants_cache_", dir=os.environ.get("LSCRATCH") or None)
os.environ["TALLYMAP_CUDA_CACHE"] = _CACHE
atexit.register(shutil.rmtree, _CACHE, True)
if os.environ.get("TEST_X64"):
    import jax
    jax.config.update("jax_enable_x64", True)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from ab_rule import assert_arms_differ  # noqa: E402
from jax import lax  # noqa: E402

import tallymap as tm  # noqa: E402
from tallymap import _diagnostics as _D  # noqa: E402
from tallymap import _foreign as _FO  # noqa: E402
from tallymap import _primitive as _P  # noqa: E402
from tallymap import reference  # noqa: E402

if os.environ.get("TX_UNFUSED") == "1":                 # run_all --unfused: the two-node jvp design
    _P.FUSE_JVP = False
jax.config.update("jax_default_matmul_precision", "highest")
X64 = bool(jax.config.jax_enable_x64)
F = jnp.float64 if X64 else jnp.float32
GPU = jax.default_backend() == "gpu"
BACKENDS = os.environ.get("BACKENDS", "lockstep,cuda" if GPU else "lockstep").split(",")
ONLY = os.environ.get("ONLY", "")
# exact=True outputs: the lockstep backend accumulates them in int64, so it needs jax_enable_x64 (edge.py).
EXACT = {"lockstep": X64, "cuda": True}
# FT: a float tally is a SUM whose order the schedule decides (atomics on cuda, block / lane order on lockstep), so
# two schedules agree only up to reassociation.  The sums here have at most a few hundred positive terms of similar
# size: reassociation moves a bin by a few ulps of the bin, 1e-5 (f32) leaves 2 orders of margin, and a LOGIC error
# (an emit lost, doubled or landing in the wrong bin) moves a bin by >= one term / bin total ~ 1e-2.
FT = 1e-12 if X64 else 1e-5
print("jax", jax.__version__, jax.__file__, jax.devices(), "x64", X64, "tallymap", tm.__file__, "cache", _CACHE,
      flush=True)
RES = []
NB = 8


class Fail(Exception):
    pass


class Expected(Exception):
    """an expected, clear error (RAISES-OK)"""


class Skip(Exception):
    """an optional dependency is missing: the cell is not counted"""


def case(name, backends=None, xfail=None):
    """xfail: {backend: reason}.  On that backend the cell is EXPECTED to fail with Fail (a known library defect,
    pinned with its hypothesis): it then prints XFAIL; if it passes it prints XPASS, which fails the run so that the
    marker is removed when the defect is fixed.  An unexpected exception stays RAISE."""
    def deco(fn):
        for b in (backends or BACKENDS):
            if b not in BACKENDS or (ONLY and ONLY not in name):
                continue
            xf = (xfail or {}).get(b)
            t0 = time.perf_counter()
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    r = fn(b)
                msg = "PASS" + (f" ({r})" if isinstance(r, str) and r else "")
                if xf:
                    msg = f"XPASS (marked XFAIL: {xf}) {msg}"
                ws = [str(x.message) for x in w if "tallymap" in str(x.message)]
                if ws:
                    msg += f" [warn: {ws[0][:90]}]"
            except Skip as e:
                print(f"{name:62s} {b:8s} SKIP ({e})  [not counted]", flush=True)
                continue
            except Expected as e:
                msg = f"RAISES-OK {e}" if not xf else f"XPASS (marked XFAIL: {xf}) raised as expected: {e}"
            except Fail as e:
                msg = f"XFAIL ({xf}) -- {e}" if xf else f"FAIL {e}"
            except Exception as e:
                msg = f"RAISE {type(e).__name__}: {str(e)[:300]}"
                traceback.print_exc()
            RES.append((name, b, msg))
            print(f"{name:62s} {b:8s} {msg}  [{time.perf_counter() - t0:.1f}s]", flush=True)
        return fn
    return deco


# ------------------------------------------------------------------------------------------- comparisons
def _leaves(x):
    return [np.asarray(v) for v in jax.tree_util.tree_leaves(x)]


def same(a, b, what):
    """BITWISE: every leaf identical (value and dtype; NaN in the same places)."""
    la, lb = _leaves(a), _leaves(b)
    if len(la) != len(lb):
        raise Fail(f"{what}: {len(la)} leaves vs {len(lb)}")
    for x, y in zip(la, lb):
        if x.shape != y.shape or x.dtype != y.dtype:
            raise Fail(f"{what}: {x.shape} {x.dtype} vs {y.shape} {y.dtype}")
        eq = np.array_equal(x, y, equal_nan=True) if x.dtype.kind in "fc" else np.array_equal(x, y)
        if not eq:
            d = x != y if x.dtype.kind not in "fc" else ~((x == y) | (np.isnan(x) & np.isnan(y)))
            md = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))[d])) if d.any() else 0.0
            raise Fail(f"{what}: not bitwise ({int(d.sum())}/{x.size} differ, max |diff| {md:.3g})")
    return "bitwise"


def near(a, b, rtol, what, atol=0.0):
    """every leaf within rtol * max|b| + atol; NaN in the same places.  -> 'bitwise' or the max relative error."""
    la, lb = _leaves(a), _leaves(b)
    if len(la) != len(lb):
        raise Fail(f"{what}: {len(la)} leaves vs {len(lb)}")
    worst, bit = 0.0, True
    for x, y in zip(la, lb):
        if x.shape != y.shape:
            raise Fail(f"{what}: shape {x.shape} vs {y.shape}")
        x, y = x.astype(np.float64), y.astype(np.float64)
        if not np.array_equal(np.isnan(x), np.isnan(y)):
            raise Fail(f"{what}: NaN pattern differs ({int(np.isnan(x).sum())} vs {int(np.isnan(y).sum())} NaN)")
        m = ~np.isnan(y)
        if not m.any():
            continue
        err = float(np.max(np.abs(x[m] - y[m])))
        scale = float(np.max(np.abs(y[m])))
        bit = bit and err == 0
        if not err <= rtol * scale + atol:
            raise Fail(f"{what}: max err {err:.3g} > {rtol:.1e} * {scale:.3g} + {atol:.1e}")
        worst = max(worst, err / max(scale, 1e-300))
    return "bitwise" if bit else f"rel {worst:.1e}"


def split_res(r):
    """Result -> (float outputs, exact / integer outputs, per-item float leaves, per-item int leaves, stats)"""
    fo = {k: v for k, v in r.outputs.items() if k == "Q"}
    xo = {k: v for k, v in r.outputs.items() if k != "Q"}
    pl = jax.tree_util.tree_leaves(r.per_item)
    pf = [v for v in pl if np.asarray(v).dtype.kind == "f"]
    pi = [v for v in pl if np.asarray(v).dtype.kind != "f"]
    return fo, xo, pf, pi, {k: np.asarray(v) for k, v in r.stats.items()}


def agree(A, B, what, per_item="bitwise", pi_tol=None):
    """two Results of the same item program under different internal choices.  Exact and integer tallies, integer
    per-item results, truncation flags and stats: bitwise.  Float tallies: FT.  Float per-item results: bitwise
    (per_item='bitwise': the same per-item arithmetic) or within pi_tol (another compiler)."""
    fa, xa, pfa, pia, sa = split_res(A)
    fb, xb, pfb, pib, sb = split_res(B)
    same(xa, xb, f"{what} exact/int tallies")
    same(pia, pib, f"{what} per-item int")
    same(A.truncated, B.truncated, f"{what} truncated flags")
    for k in sa:
        if not np.array_equal(sa[k], sb[k]):
            raise Fail(f"{what} stats[{k}] {sa[k]} vs {sb[k]}")
    m = [near(fa, fb, FT, f"{what} float tally")]
    m.append(same(pfa, pfb, f"{what} per-item float") if per_item == "bitwise" else
             near(pfa, pfb, pi_tol, f"{what} per-item float"))
    return f"Q {m[0]}, per-item {m[1]}"


def bits(key, k):
    return jax.random.bits(jax.random.fold_in(key, k))


# Per-item TANGENTS of a laned program vs the single-column jvp.  cuda: each lane executes the same emitted scalar
# arithmetic as the one-column kernel (no FMA contraction, -fmad=false) -> bitwise.  lockstep: the laned program is a
# different XLA program (jax.vmap of the item), and XLA:CPU is free to vectorise / fuse it differently (MEASURED,
# JAX 0.11.2 CPU: 11/37 per-item tangents differ by <= 1.5e-8 for T=1; JAX 0.5.3: bitwise) -> a few ulps.
PI_LANE_TOL = 1e-13 if X64 else 1e-6


def lanes_same(b, a, ref, what):
    return same(a, ref, what) if b == "cuda" else near(a, ref, PI_LANE_TOL, what)


def kernel_hashes():
    return [k["hash"] for k in _D.KERNELS]


# =================================================================================================== [sched]
# A divergent walk: the trip count is decided by the item's RNG bits (integer-exact, so every backend takes the same
# path), a table read per step, emits into a float tally Q, an integer tally C and (where supported) an exact tally
# E, a per-item array h updated in place every step (the _cuda_inplace path), and per-item results (w, k, h).  The
# emitted weights use only IEEE +, * (no div / sqrt / transcendental), so cuda_math does not enter and the exact
# tally can be compared bitwise across compilers.
N1 = 37
XS = jnp.linspace(0.5, 1.5, N1).astype(F)
KEYS = jax.random.split(jax.random.PRNGKey(0), N1)
TAB = (0.5 + 0.25 * jnp.arange(NB)).astype(F)
AX1 = (0, 0, None, None)


def mk_walk(exact):
    def item(t, x, key, tab, th):
        P = th.shape[0]

        def body(c):
            k, w, h, t = c
            i = (bits(key, k) % NB).astype(jnp.int32)
            v = w * tab[i] * th[k % P]
            t = t.add("Q", i, v).add("C", i, 1)
            if exact:
                t = t.add("E", i, v)
            # two dynamic_update_slices: the first copies the loop carry (a carry is never mutated), the second
            # updates that fresh copy IN PLACE -- the path _cuda_inplace=False switches off
            j = i % 4
            h = lax.dynamic_update_index_in_dim(h, h[j] + v, j, 0)
            h = lax.dynamic_update_index_in_dim(h, h[(j + 1) % 4] * 0.5, (j + 1) % 4, 0)
            return k + 1, w * (0.75 + 0.05 * th[(k + 1) % P]), h, t
        k, w, h, t = lax.while_loop(lambda c: (bits(key, c[0] + 1000) % 5 != 0) | (c[0] == 0), body,
                                    (jnp.int32(0), x, jnp.zeros((4,), x.dtype), t))
        return t, (w, k, h)
    return item


def outs1(b, exact=True):
    o = {"Q": tm.Sum((NB,), F), "C": tm.Sum((NB,), jnp.int32)}
    if exact and EXACT[b]:
        o["E"] = tm.Sum((NB,), F, exact=True, scale=2.0 ** 20)
    return o


def walk_fn(b, exact=True, **opts):
    o = outs1(b, exact)
    return tm.vmap(mk_walk("E" in o), outputs=o, in_axes=AX1, backend=b, **opts)


def th_of(T):
    return (0.9 + 0.01 * jnp.arange(T)).astype(F)


@case("[sched] trip counts are divergent (instrument)")
def _(b):
    r = walk_fn(b)(XS, KEYS, TAB, th_of(3))
    k = np.asarray(r.per_item[1])
    if len(set(k.tolist())) < 4:
        raise Fail(f"instrument: trip counts {sorted(set(k.tolist()))} are not divergent enough")
    return f"trip counts {k.min()}..{k.max()} ({len(set(k.tolist()))} distinct); exact tally {'E' in r.outputs}"


@case("[sched] _schedule step == item (cuda)", ["cuda"])
def _(b):
    fs, fi = walk_fn(b, _schedule="step"), walk_fn(b, _schedule="item")
    th = th_of(3)
    assert_arms_differ(lambda t: fs(XS, KEYS, TAB, t).outputs["Q"], lambda t: fi(XS, KEYS, TAB, t).outputs["Q"],
                          th)
    # the same per-item arithmetic in both kernels: per-item results bitwise; Q's atomics order differs: FT
    return agree(fs(XS, KEYS, TAB, th), fi(XS, KEYS, TAB, th), "step vs item")


@case("[sched] _cuda_inplace on == off (cuda)", ["cuda"])
def _(b):
    fa, fb = walk_fn(b), walk_fn(b, _cuda_inplace=False)
    th = th_of(3)
    assert_arms_differ(lambda t: fa(XS, KEYS, TAB, t).per_item[2], lambda t: fb(XS, KEYS, TAB, t).per_item[2], th)
    return agree(fa(XS, KEYS, TAB, th), fb(XS, KEYS, TAB, th), "inplace on vs off")


@case("[sched] lockstep == cuda (exact/int bitwise, float to tol)", ["cuda"])
def _(b):
    th = th_of(3)
    A = walk_fn("cuda", exact=X64)(XS, KEYS, TAB, th)        # (the lockstep exact tally needs x64)
    B = walk_fn("lockstep", exact=X64)(XS, KEYS, TAB, th)
    # Two compilers (NVRTC vs XLA:GPU).  The item uses only IEEE +, * without contraction (cuda: -fmad=false), so the
    # per-item floats are expected to agree to the last bit; held to 1e-6 rather than bitwise because XLA is free to
    # pick its own (exactly rounded, but differently associated) evaluation of a fused expression.
    m = agree(A, B, "cuda vs lockstep", per_item="tol", pi_tol=1e-12 if X64 else 1e-6)
    return m + ("; exact E bitwise" if X64 else "; (no exact tally: lockstep exact needs x64)")


@case("[sched] permuted items: exact/int tallies bitwise, per-item permuted bitwise")
def _(b):
    perm = np.random.default_rng(1).permutation(N1)
    f = walk_fn(b)
    th = th_of(3)
    A = f(XS, KEYS, TAB, th)
    B = f(XS[perm], KEYS[perm], TAB, th)
    Ap = A._replace(per_item=jax.tree_util.tree_map(lambda v: v[perm], A.per_item), truncated=A.truncated[perm])
    m = agree(Ap, B, "permuted")
    return m + ("; exact E bitwise" if "E" in A.outputs else "; (no exact tally: lockstep without x64)")


@case("[sched] _fold on == off under jax.vmap of tm.vmap")
def _(b):
    # exact outputs are never folded (_can_fold: stats['exact_overflow'] is per launch), so this program has none
    ths = jnp.stack([th_of(3) * s for s in (1.0, 1.1, 0.95)])
    fo, fn = walk_fn(b, exact=False), walk_fn(b, exact=False, _fold=False)
    g = lambda f: jax.jit(jax.vmap(lambda th: f(XS, KEYS, TAB, th)))
    jo, jn = str(jax.make_jaxpr(jax.vmap(lambda th: fo(XS, KEYS, TAB, th)))(ths)), \
        str(jax.make_jaxpr(jax.vmap(lambda th: fn(XS, KEYS, TAB, th)))(ths))
    if "folded_batch" not in jo or "folded_batch" in jn:
        raise Fail(f"instrument: folded in default {'folded_batch' in jo}, in _fold=False {'folded_batch' in jn}")
    A, B = g(fo)(ths), g(fn)(ths)
    m = agree(A, B, "fold vs no fold")
    for j in range(ths.shape[0]):                       # and the meaning: member j == the unbatched call at ths[j]
        Cj = fo(XS, KEYS, TAB, ths[j])
        agree(jax.tree_util.tree_map(lambda v: v[j], A), Cj, f"fold member {j} vs unbatched")
    return m + "; every member == its unbatched call"


LANE_VARIANTS = [("auto", {}), ("off", dict(_lanes="off")), ("chunk4", dict(_lanes=4)), ("chunk16", dict(_lanes=16)),
                 ("force", dict(_lanes="force"))]


def _chunk(b, T, opts):
    """the dual-lane width _primitive._dual_lanes picks (None: no lanes)"""
    mode = opts.get("_lanes", "auto")
    if mode == "off":
        return None
    if isinstance(mode, int):
        return _P._balanced(T, min(T, mode))
    if mode == "auto" and b == "cuda":
        return _P._balanced(T, min(T, _P.MAX_LANES))
    return _P._balanced(T, T)


for T in (1, 16, 17, 33):
    @case(f"[sched] jacfwd T={T}: every lane chunking == stacked jvp columns")
    def _(b, T=T):
        th0 = th_of(T)
        ref = None
        msgs = []
        for name, opts in LANE_VARIANTS:
            f = walk_fn(b, **opts)

            def g(th, f=f):
                r = f(XS, KEYS, TAB, th)
                return {k: v for k, v in r.outputs.items() if k != "C"}, (r.per_item[0], r.per_item[2])
            want = _chunk(b, T, opts)
            got = set(int(x) for x in re.findall(r"dual_lanes(\d+)", str(jax.make_jaxpr(jax.jacfwd(g))(th0))))
            if got != ({want} if want else set()):
                raise Fail(f"instrument: {name} built dual lanes {sorted(got)}, expected {want}")
            J = jax.jit(jax.jacfwd(g))(th0)
            if ref is None:      # the meaning: column j of the Jacobian is jax.jvp along e_j (one launch each)
                jv = jax.jit(lambda th, v, g=g: jax.jvp(g, (th,), (v,))[1])
                cols = [jv(th0, jnp.zeros((T,), F).at[j].set(1)) for j in range(T)]
                ref = jax.tree_util.tree_map(lambda *c: jnp.stack(c, -1), *cols)
            o, (w, h) = J
            ro, (rw, rh) = ref
            if "E" in o:          # the exact tangent tally: integer accumulation, order-free -> bitwise
                same(o["E"], ro["E"], f"{name} exact tangent tally")
            mq = near(o["Q"], ro["Q"], FT, f"{name} Q tangent")
            # the per-item tangents: each lane runs the same tangent arithmetic as the single-column jvp
            mp = lanes_same(b, (w, h), (rw, rh), f"{name} per-item tangents")
            msgs.append(f"{name}[{want or '-'}] Q {mq}, per-item {mp}")
        return "; ".join(msgs)


# =================================================================================================== [key]
NK = 64
XK = jnp.linspace(0.5, 1.5, NK).astype(F)


def _np(x):
    return np.asarray(x)


class Pair:
    """two programs that must NOT share a compiled kernel / eager-cache entry.  a, b: () -> Result;
    ok_a, ok_b: Result -> None (raise Fail when the result is not what the math says); differ: (ra, rb) -> bool,
    the instrument that the two results really differ (else the pair is void).  safe: the two programs have the same
    operand avals and tally layout, so a coarse key may reuse a kernel without reading out of bounds (only these
    run under the negative controls)."""

    def __init__(self, name, a, b, ok_a, ok_b, differ, safe=True, cuda_only=False):
        self.name, self.a, self.b, self.ok_a, self.ok_b, self.differ = name, a, b, ok_a, ok_b, differ
        self.safe, self.cuda_only = safe, cuda_only


def _check(cond, msg):
    if not cond:
        raise Fail(msg)


def make_pairs(b):
    P = []
    Q = {"Q": tm.Sum((NB,), F)}

    # 1. a closed-over constant array: same shape / dtype, different VALUES
    def mk_c(cv):
        C = jnp.asarray(cv, F)

        def item(t, x):
            return t.add("Q", jnp.arange(3), x * C), x * C[2]
        return tm.vmap(item, outputs=Q, backend=b)
    fa, fb = mk_c([1.0, 2.0, 3.0]), mk_c([1.0, 2.0, 5.0])

    def ok_c(c2):
        def ok(r):
            same(r.per_item, (_np(XK) * np.asarray(c2, _np(XK).dtype)).astype(_np(XK).dtype), "per-item x*C[2]")
            near(r.outputs["Q"][2], np.sum(_np(XK).astype(np.float64)) * c2, 1e-5, "Q[2]")
        return ok
    P.append(Pair("closed-over array values", lambda: fa(XK), lambda: fb(XK), ok_c(3.0), ok_c(5.0),
                  lambda ra, rb: not np.array_equal(_np(ra.per_item), _np(rb.per_item))))

    # 2. a closed-over Python scalar, ONE ulp apart in the item's dtype
    sa = 0.1
    sb = float(np.nextafter(np.asarray(0.1, F), np.asarray(1.0, F)))

    def mk_s(s):
        def item(t, x):
            y = x * s
            return t.add("Q", 0, y), y
        return tm.vmap(item, outputs=Q, backend=b)
    ga, gb = mk_s(sa), mk_s(sb)

    def ok_s(s):
        def ok(r):
            same(r.per_item, _np(XK) * np.asarray(s, _np(XK).dtype), f"per-item x*{s!r}")
        return ok
    P.append(Pair("closed-over Python scalar (1 ulp apart)", lambda: ga(XK), lambda: gb(XK), ok_s(sa), ok_s(sb),
                  lambda ra, rb: not np.array_equal(_np(ra.per_item), _np(rb.per_item))))

    # 3. dtype: the SAME Python function on float vs int per-item inputs
    def item_d(t, x):
        y = x * 3
        return t.add("Q", 0, y.astype(F)), y
    hd = tm.vmap(item_d, outputs=Q, backend=b)
    xi = jnp.arange(NK, dtype=jnp.int32)
    P.append(Pair("dtype (float vs int32 input)", lambda: hd(XK), lambda: hd(xi),
                  lambda r: same(r.per_item, _np(XK) * np.asarray(3, _np(XK).dtype), "float per-item"),
                  lambda r: same(r.per_item, np.arange(NK, dtype=np.int32) * 3, "int per-item"),
                  lambda ra, rb: _np(ra.per_item).dtype != _np(rb.per_item).dtype, safe=False))

    # 4. shape: the SAME Python function on per-item (2,) vs (3,) inputs
    def item_sh(t, x):
        return t.add("Q", jnp.arange(x.shape[0]), x), x * 2
    hs = tm.vmap(item_sh, outputs=Q, backend=b)
    x2 = jnp.stack([XK * (j + 1) for j in range(2)], 1)
    x3 = jnp.stack([XK * (j + 1) for j in range(3)], 1)

    def ok_sh(x):
        def ok(r):
            same(r.per_item, _np(x) * np.asarray(2, _np(x).dtype), "per-item x*2")
            q = _np(r.outputs["Q"])
            near(q[:x.shape[1]], _np(x).astype(np.float64).sum(0), 1e-5, "Q")
            _check(not q[x.shape[1]:].any(), "Q beyond the per-item length is not zero")
        return ok
    P.append(Pair("shape (per-item (2,) vs (3,))", lambda: hs(x2), lambda: hs(x3), ok_sh(x2), ok_sh(x3),
                  lambda ra, rb: _np(ra.per_item).shape != _np(rb.per_item).shape, safe=False))

    # 5. an option: cuda_math approx vs ieee (f32 div / sqrt; cuda only).  ieee must be correctly rounded (numpy
    #    f32); approx must differ from it somewhere (else the pair is void) but stay within a few ulps.
    if b == "cuda":
        rng = np.random.default_rng(5)
        a32 = jnp.asarray(rng.uniform(0.1, 10.0, 4096), jnp.float32)
        c32 = jnp.asarray(rng.uniform(0.1, 10.0, 4096), jnp.float32)

        def item_m(t, a, c):
            y = a / c
            return t.add("Q", 0, y.astype(F)), (y, jnp.sqrt(a))
        mk_m = lambda m: tm.vmap(item_m, outputs=Q, backend=b, cuda_math=m)
        ma, mb = mk_m("approx"), mk_m("ieee")
        na, nc = _np(a32), _np(c32)

        def ok_ieee(r):
            same(r.per_item, (na / nc, np.sqrt(na)), "ieee per-item vs numpy f32 (correctly rounded)")

        def ok_approx(r):
            y, z = (_np(v) for v in r.per_item)
            ulp = lambda v, w: np.max(np.abs(v.view(np.int32).astype(np.int64) - w.view(np.int32).astype(np.int64)))
            u = max(ulp(y, na / nc), ulp(z, np.sqrt(na)))
            _check(u <= 4, f"approx div/sqrt {u} ulps from correctly rounded")
            _check(u > 0, "instrument: approx is bitwise the correctly rounded result here (pair void)")
        P.append(Pair("option cuda_math approx vs ieee", lambda: ma(a32, c32), lambda: mb(a32, c32), ok_approx,
                      ok_ieee, lambda ra, rb: True, cuda_only=True))

    # 6. max_steps: 8 truncates every item (trip count 10..14), 64 truncates none
    idx = jnp.arange(NK, dtype=jnp.int32)
    L = 10 + np.arange(NK) % 5

    def item_t(t, x, i):
        def body(c):
            k, w, t = c
            return k + 1, w * 0.5 + x, t.add("Q", k % NB, w).add("C", 0, 1)
        k, w, t = lax.while_loop(lambda c: c[0] < 10 + i % 5, body, (jnp.int32(0), x, t))
        return t, k
    QC = {"Q": tm.Sum((NB,), F), "C": tm.Sum((1,), jnp.int32)}
    ta = tm.vmap(item_t, outputs=QC, backend=b, max_steps=8)
    tb = tm.vmap(item_t, outputs=QC, backend=b, max_steps=64)

    def ok_ta(r):
        _check(bool(np.all(_np(r.truncated))) and int(r.stats["n_truncated"]) == NK, "max_steps=8: not all truncated")
        same(r.per_item, np.full(NK, 8, np.int32), "max_steps=8 trip counts")
        _check(int(_np(r.outputs["C"])[0]) == 8 * NK, "max_steps=8 partial integer tally")
        _check(bool(np.all(np.isnan(_np(r.outputs["Q"])))), "max_steps=8: float tally not poisoned")

    def ok_tb(r):
        _check(not np.any(_np(r.truncated)) and int(r.stats["n_truncated"]) == 0, "max_steps=64 truncated an item")
        same(r.per_item, L.astype(np.int32), "max_steps=64 trip counts")
        _check(int(_np(r.outputs["C"])[0]) == int(L.sum()), "max_steps=64 integer tally")
        _check(bool(np.all(np.isfinite(_np(r.outputs["Q"])))), "max_steps=64 float tally not finite")
    P.append(Pair("max_steps 8 vs 64", lambda: ta(XK, idx), lambda: tb(XK, idx), ok_ta, ok_tb,
                  lambda ra, rb: int(ra.stats["n_truncated"]) != int(rb.stats["n_truncated"])))

    # 7. table contents as a broadcast OPERAND: the same compiled program must follow the operand
    t1 = (1.0 + 0.5 * jnp.arange(NB)).astype(F)
    t2 = (3.0 - 0.25 * jnp.arange(NB)).astype(F)
    ii = jnp.asarray(np.arange(NK) % NB, jnp.int32)

    def item_o(t, x, i, tab):
        y = x * tab[i]
        return t.add("Q", i, y), y
    ho = tm.vmap(item_o, outputs=Q, in_axes=(0, 0, None), backend=b)

    def ok_o(tab):
        def ok(r):
            same(r.per_item, _np(XK) * _np(tab)[_np(ii)], "per-item x*tab[i]")
        return ok
    P.append(Pair("table contents (broadcast operand)", lambda: ho(XK, ii, t1), lambda: ho(XK, ii, t2), ok_o(t1),
                  ok_o(t2), lambda ra, rb: not np.array_equal(_np(ra.per_item), _np(rb.per_item))))

    # 8. table contents CLOSED OVER (a 256-entry constant table: the pooled-table / gather path, not a literal)
    big1 = (1.0 + jnp.sin(jnp.arange(256))).astype(F)
    big2 = big1.at[17].add(1.0)
    ij = jnp.asarray((np.arange(NK) * 7 + 17) % 256, jnp.int32)

    def mk_tab(tab):
        def item(t, x, i):
            y = x * tab[i]
            return t.add("Q", i % NB, y), y
        return tm.vmap(item, outputs=Q, backend=b)
    pa, pb = mk_tab(big1), mk_tab(big2)

    def ok_tab(tab):
        def ok(r):
            same(r.per_item, _np(XK) * _np(tab)[_np(ij)], "per-item x*tab[i]")
        return ok
    P.append(Pair("table contents (closed-over 256 entries)", lambda: pa(XK, ij), lambda: pb(XK, ij), ok_tab(big1),
                  ok_tab(big2), lambda ra, rb: not np.array_equal(_np(ra.per_item), _np(rb.per_item))))
    return P


def run_pair(p):
    """A, then B, then A again (eager: the eager cache + the kernel cache), then each under its own jax.jit."""
    ra = p.a()
    p.ok_a(ra)
    rb = p.b()
    p.ok_b(rb)
    _check(p.differ(ra, rb), "instrument: the two programs give the same result (pair void)")
    p.ok_a(p.a())
    p.ok_b(jax.jit(p.b)())
    p.ok_a(jax.jit(p.a)())


def _pair_cell(b, pairs):
    """-> {pair name: None (sound) | failure message}"""
    out = {}
    for p in pairs:
        try:
            run_pair(p)
            out[p.name] = None
        except Fail as e:
            out[p.name] = str(e)[:160]
    return out


@case("[key] programs that differ only in const/scalar/dtype/shape/option/max_steps/table never share a kernel")
def _(b):
    _P._EAGER.clear()
    pairs = make_pairs(b)
    res = _pair_cell(b, pairs)
    bad = {k: v for k, v in res.items() if v}
    if bad:
        raise Fail("; ".join(f"{k}: {v}" for k, v in bad.items()))
    return f"{len(res)} pairs sound: " + ", ".join(res)


def _coarse_kernel_key(kind):
    def text_only(p, opts):                # the plan's "str()-only key": the jaxpr text, nothing else
        return ("NEG", str(p.cj.jaxpr))

    def digest_only(p, opts):              # the real keys' structural part alone (_jaxpr.digest, no consts / options)
        from tallymap._jaxpr import digest
        return ("NEG", digest(p.cj.jaxpr))

    def avals_only(p, opts):               # operand / result avals and the tally layout: no program text at all
        return ("NEG", tuple((tuple(v.aval.shape), str(v.aval.dtype)) for v in p.cj.jaxpr.invars),
                tuple((tuple(v.aval.shape), str(v.aval.dtype)) for v in p.cj.jaxpr.outvars),
                tuple((s.S, str(np.dtype(s.dtype)), tuple(s.lead), s.exact, s.kind) for s in p.tallies),
                p.nP, p.nB, p.trunc, len(p.cj.consts))
    return {"text": text_only, "digest": digest_only, "avals": avals_only}[kind]


# under the coarse key these pairs collide (same text / same avals) and MUST be caught
NEG_EXPECT = {"text": {"closed-over array values", "table contents (closed-over 256 entries)",
                       "option cuda_math approx vs ieee"},
              "digest": {"closed-over array values", "table contents (closed-over 256 entries)",
                         "option cuda_math approx vs ieee"},
              "avals": {"closed-over array values", "table contents (closed-over 256 entries)",
                        "option cuda_math approx vs ieee", "closed-over Python scalar (1 ulp apart)",
                        "max_steps 8 vs 64"}}


@case("[key] _jaxpr.digest: equal where str(jaxpr) is equal, different where it differs or a literal's bytes do",
      ["lockstep"])
def _(b):
    from tallymap._jaxpr import digest
    S, I = jax.ShapeDtypeStruct((4,), F), jax.ShapeDtypeStruct((), jnp.int32)
    D = lambda f, *a: digest(jax.make_jaxpr(f)(*a).jaxpr)
    base = lambda x: jnp.sum(x * x) + jnp.sin(x[0])
    cnd = lambda k: (lambda n, x: lax.cond(n > 0, lambda a: a * k, lambda a: a + 1.0, x))
    wh = lambda k: (lambda x: lax.while_loop(lambda c: c[0] < k, lambda c: c + 1.0, x))
    a1 = float(np.nextafter(np.asarray(0.1, F), np.asarray(1.0, F)))
    same_ = [("traced twice", D(base, S), D(base, S)),
             ("alpha-equivalent", D(lambda x: (lambda a: a + a)(x * 2.0), S),
              D(lambda x: (lambda zz: zz + zz)(x * 2.0), S)),
             ("same cond body", D(cnd(2.0), I, S), D(cnd(2.0), I, S))]
    differ = [("primitive", D(jnp.sin, S), D(jnp.cos, S)),
              ("operand order", D(lambda x: x[0] - x[1], S), D(lambda x: x[1] - x[0], S)),
              ("shape", D(base, S), D(base, jax.ShapeDtypeStruct((5,), F))),
              ("dtype", D(base, S), D(base, jax.ShapeDtypeStruct((4,), jnp.int32))),
              ("slice parameter", D(lambda x: lax.slice(x, (0,), (2,)), S), D(lambda x: lax.slice(x, (1,), (3,)), S)),
              ("cond branch", D(cnd(2.0), I, S), D(cnd(3.0), I, S)),
              ("while condition", D(wh(10.0), S), D(wh(11.0), S)),
              ("scan length", D(lambda x: lax.scan(lambda c, _: (c + 1.0, c), x, length=3)[0], S),
               D(lambda x: lax.scan(lambda c, _: (c + 1.0, c), x, length=4)[0], S)),
              ("float literal 1 ulp apart", D(lambda x: x * 0.1, S), D(lambda x: x * a1, S))]
    bad = [n for n, x, y in same_ if x != y] + [n for n, x, y in differ if x == y]
    if bad:
        raise Fail(f"digest wrong for {bad}")
    return f"{len(same_)} equal, {len(differ)} different"


def _with_power(kind, pairs, expect):
    """the expected pairs under which the coarse key really COLLIDES (both programs get the same coarse key): only
    there can a stale reuse happen, so only there does the control have power.  Not every key collides on every
    JAX: on 0.11 a closed-over array reaches the item jaxpr where _jaxpr.digest hashes its bytes (not as a top-level
    const), so the digest-only key already separates those pairs -- the real keys' structural part is then sound on
    its own.  -> (pairs with power, {pair without power: why})"""
    from tallymap import _diagnostics as D
    ck = _coarse_kernel_key(kind)
    held, orig = [], D.record_program
    D.record_program = lambda q: (held.append(q), orig(q))[1]
    power, none = set(), {}
    try:
        for p in pairs:
            if p.name not in expect:
                continue
            del held[:]
            p.a()
            pa = held[-1]
            del held[:]
            p.b()
            pb = held[-1]
            if ck(pa, {}) == ck(pb, {}):
                power.add(p.name)
            else:
                none[p.name] = f"top-level consts {len(pa.cj.consts)}/{len(pb.cj.consts)}"
    finally:
        D.record_program = orig
    return power, none


def _neg_verdict(kind, res, expect, none):
    caught = {k for k, v in res.items() if v}
    miss = expect - caught
    if miss:
        raise Fail(f"the coarse key was NOT detected for {sorted(miss)}: the [key] test is blind there")
    if not expect:
        raise Skip(f"no power on JAX {jax.__version__}: the {kind}-only key already separates every expected pair "
                   f"({none})")
    return f"stale reuse caught in {sorted(caught)}" + (f"; no collision (no power) for {none}" if none else "")


for _kind in ("text", "digest", "avals"):
    @case(f"[key] NEGATIVE CONTROL: kernel key = {_kind} only must be caught (cuda _KERNELS)", ["cuda"])
    def _(b, kind=_kind):
        from tallymap.backends import cuda as CB
        real = CB._key
        CB._KERNELS.clear()
        _P._EAGER.clear()
        pairs = [p for p in make_pairs(b) if p.safe]
        expect, none = _with_power(kind, pairs, NEG_EXPECT[kind])
        if "option cuda_math approx vs ieee" in NEG_EXPECT[kind]:
            expect.add("option cuda_math approx vs ieee")     # the coarse keys ignore options: always collides
            none.pop("option cuda_math approx vs ieee", None)
        CB._KERNELS.clear()
        _P._EAGER.clear()
        CB._key = _coarse_kernel_key(kind)
        try:
            res = _pair_cell(b, pairs)
        finally:
            CB._key = real
            CB._KERNELS.clear()
            _P._EAGER.clear()
        return _neg_verdict(kind, res, expect, none)

    @case(f"[key] NEGATIVE CONTROL: eager-cache Node.key = {_kind} only must be caught", ["lockstep"])
    def _(b, kind=_kind):
        real = _P.Node.key
        ck = _coarse_kernel_key(kind)
        pairs = [p for p in make_pairs(b) if p.safe]
        expect, none = _with_power(kind, pairs, {k for k in NEG_EXPECT[kind] if "cuda_math" not in k})
        _P._EAGER.clear()
        _P.Node.key = lambda self: (ck(self.prog, {}), self.backend, self.kind)
        try:
            res = _pair_cell(b, pairs)
        finally:
            _P.Node.key = real
            _P._EAGER.clear()
        return _neg_verdict(kind, res, expect, none)


# =================================================================================================== [tdep]
def tdep_item(t, x, th):
    """the trip count follows y, a float that carries the tangent of th: under jvp the loop runs on the PRIMAL y
    (the legitimate case: the trip count is piecewise constant in th, and its tangent is zero)."""
    T = th.shape[0]

    def body(c):
        k, y, w, t = c
        y2 = y + 0.05 * (1.0 + x * th[k % T] * th[k % T])
        w2 = w * (1.0 + 0.01 * th[(k + 1) % T])
        return k + 1, y2, w2, t.add("Q", k % NB, w2 * y2)
    k, y, w, t = lax.while_loop(lambda c: (c[1] < 3.0) & (c[0] < 500), body, (jnp.int32(0), 0.1 * x, x, t))
    return t, (y, w, k)


for T in (3, 17):
    @case(f"[tdep] while on the tangent-carrying primal: jacfwd T={T} == stacked jvp == reference")
    def _(b, T=T):
        th0 = (0.8 + 0.03 * jnp.arange(T)).astype(F)
        f = tm.vmap(tdep_item, outputs={"Q": tm.Sum((NB,), F)}, in_axes=(0, None), backend=b)
        r = reference(tdep_item, outputs={"Q": tm.Sum((NB,), F)}, in_axes=(0, None))
        g = lambda th, f=f: (f(XS, th).outputs["Q"], f(XS, th).per_item[:2])
        gr = lambda th: (r(XS, th).outputs["Q"], r(XS, th).per_item[:2])
        k = np.asarray(f(XS, th0).per_item[2])
        if len(set(k.tolist())) < 3:
            raise Fail(f"instrument: trip counts {sorted(set(k.tolist()))} not divergent")
        J = jax.jit(jax.jacfwd(g))(th0)
        jv = jax.jit(lambda th, v: jax.jvp(g, (th,), (v,))[1])
        cols = [jv(th0, jnp.zeros((T,), F).at[j].set(1)) for j in range(T)]
        S = jax.tree_util.tree_map(lambda *c: jnp.stack(c, -1), *cols)
        m1 = near(J[0], S[0], FT, "Q jacfwd vs jvp columns")
        m2 = lanes_same(b, J[1], S[1], "per-item jacfwd vs jvp columns")
        m3 = near(J, jax.jit(jax.jacfwd(gr))(th0), 1e-10 if X64 else 2e-5, "jacfwd vs reference jacfwd")
        return f"trip counts {k.min()}..{k.max()}; vs jvp: Q {m1}, per-item {m2}; vs reference {m3}"


def _lineax():
    try:
        import lineax as lx
    except ImportError as e:
        raise Skip(f"lineax not installed: {e}") from None
    return lx


B0 = np.array([[2.0, 0.3, 0.1], [0.2, 1.8, 0.4], [0.1, 0.5, 2.2]])
RHS3 = np.array([1.0, -0.5, 2.0])


def cg_item(t, x, th):
    import lineax as lx
    m = jnp.asarray(B0, F) + jnp.diag(th[:3]) * x + th[3] * jnp.ones((3, 3), F) * 0.1
    op = lx.MatrixLinearOperator(m @ m.T, lx.positive_semidefinite_tag)
    y = lx.linear_solve(op, jnp.asarray(RHS3, F), solver=lx.CG(rtol=1e-6, atol=1e-9)).value
    return t.add("Q", jnp.arange(3), y), y


def cg_case(b):
    _lineax()
    th0 = jnp.asarray([0.5, 0.7, 0.9, 0.3], F)
    xs = jnp.linspace(0.5, 1.5, 16).astype(F)
    f = tm.vmap(cg_item, outputs={"Q": tm.Sum((NB,), F)}, in_axes=(0, None), backend=b)
    g = lambda th: (f(xs, th).outputs["Q"], f(xs, th).per_item)
    ref = jax.jit(jax.jacfwd(lambda th: jax.vmap(lambda x: cg_item(_Dense(), x, th)[1])(xs)))(th0)
    jv = jax.jit(lambda th, v: jax.jvp(g, (th,), (v,))[1])
    cols = [jv(th0, jnp.zeros((4,), F).at[j].set(1)) for j in range(4)]
    S = jax.tree_util.tree_map(lambda *c: jnp.stack(c, -1), *cols)
    jxp = str(jax.make_jaxpr(jax.jacfwd(g))(th0))
    J = jax.jit(jax.jacfwd(g))(th0)
    return J, S, ref, "dual_lanes" in jxp


class _Dense:
    """a stand-in Tally for the jax.vmap reference of cg_item (add is a no-op: only the per-item result is used)"""

    def add(self, *a):
        return self


# CG tolerance rtol=1e-6: the tangent solve is a CG solve of its own, so implementations agree to ~1e-5 (f32)
CG_TOL = 1e-6 if X64 else 2e-4


@case("[tdep] lineax.CG under jacfwd, default path (lanes_unsafe: one lane at a time) == jvp columns == jax")
def _(b):
    J, S, ref, lanes = cg_case(b)
    if lanes:
        raise Fail("instrument: the default path built dual lanes for a batched vprim (lanes_unsafe did not fire)")
    m1 = near(J, S, CG_TOL, "jacfwd vs jvp columns")
    m2 = near(J[1], ref, CG_TOL, "per-item jacfwd vs jax.jacfwd(jax.vmap)")
    return f"no dual lanes; vs jvp columns {m1}; vs jax {m2}"


# Was XFAIL on cuda (plan section 4: NaN in lanes 2..T).  It does not reproduce: MEASURED 2026-11-04, bitwise equal to
# the jvp columns on an L40S and a 2080 Ti, JAX 0.5.3 and 0.11.2.  So this cell now pins that forced lanes are right.
@case("[tdep] lineax.CG under jacfwd, dual lanes FORCED (lanes_unsafe bypassed)")
def _(b):
    real = _FO.lanes_unsafe
    _FO.lanes_unsafe = lambda cj: False
    try:
        J, S, ref, lanes = cg_case(b)
    finally:
        _FO.lanes_unsafe = real
    if not lanes:
        raise Fail("instrument: forcing did not build dual lanes")
    bad = [j for j in range(4) if not np.all(np.isfinite(np.asarray(J[1])[..., j]))]
    if bad:
        raise Fail(f"non-finite Jacobian columns {bad} of 4 (lanes forced)")
    m1 = near(J, S, CG_TOL, "jacfwd (forced lanes) vs jvp columns")
    m2 = near(J[1], ref, CG_TOL, "per-item jacfwd (forced lanes) vs jax.jacfwd(jax.vmap)")
    return f"dual lanes forced; vs jvp columns {m1}; vs jax {m2}"


# =================================================================================================== [size]
NS = 64


def _kernel_resources():
    """the measured resources (registers, local bytes per thread: compile.kernel_info, the same numbers
    tm.diagnostics() reports as kernels[i]['resources']) of the cuda kernels recorded in the diagnostics history
    that have run; the callers clear the history first, so these are the kernels of the call under test"""
    from tallymap.backends.cuda.compile import kernel_info
    out = []
    for rec in list(_D.KERNELS):
        info = kernel_info(rec["kid"])
        if info is not None:
            out.append(dict(info, hash=rec["hash"]))
    return out


def _tm_error(e, size):
    """a CLEAR tallymap error for a size limit: tallymap's own exception class (or a message naming tallymap) that
    names the offending size (`size`: the exact text, e.g. '(24, 24)')"""
    return (isinstance(e, tm.UnsupportedError) or "tallymap" in str(e)) and str(size) in str(e)


def run_or_raise(b, item, args, in_axes, ref_fn, rtol, limit=None):
    """-> (message, resources).  Runs tm.vmap(item); it must match ref_fn or raise a tallymap error naming the size
    (`limit`: a string that the message must contain, e.g. '24')."""
    f = tm.vmap(item, outputs={"Q": tm.Sum((1,), F)}, in_axes=in_axes, backend=b)
    _D.KERNELS.clear()
    try:
        r = jax.block_until_ready(f(*args))
    except Exception as e:
        if not _tm_error(e, limit):
            raise Fail(f"raised without naming the size {limit}: {type(e).__name__}: {str(e)[:200]}") from None
        raise Expected(f"{type(e).__name__}: {str(e).splitlines()[0][:160]}") from None
    res = _kernel_resources() if b == "cuda" else []
    want_y = ref_fn(*args)
    m = near(r.per_item, want_y, rtol, "per-item vs jax.vmap")
    near(r.outputs["Q"], jnp.sum(want_y).reshape(1), rtol, "tally vs jax.vmap")
    return m, res


def _res_str(res):
    return "; ".join(f"regs {r['regs']} local {r['local_bytes']} B" for r in res) or "no kernel info"


REG_CAP = 255          # the per-thread register file limit: a kernel AT the cap spills (the plan's "silent 255-reg")


def _solve_inputs(n):
    rng = np.random.default_rng(n)
    M = jnp.asarray(rng.normal(size=(n, n)) * 0.3 + n * np.eye(n), F)
    rhs = jnp.asarray(rng.normal(size=(n,)), F)
    xs = jnp.linspace(0.5, 1.5, NS).astype(F)
    return xs, M, rhs


def solve_item(t, x, M, rhs):
    y = jnp.linalg.solve(M + x * jnp.eye(M.shape[0], dtype=M.dtype), rhs)
    return t.add("Q", 0, jnp.sum(y)), y


def solve_ref(xs, M, rhs):
    return jax.jit(jax.vmap(lambda x: jnp.linalg.solve(M + x * jnp.eye(M.shape[0], dtype=M.dtype), rhs)))(xs)


MAXD = 16              # tallymap.backends.cuda.dense.MAXD (not imported: the CPU suite must not need the cuda backend)
_SOLVE16_XFAIL = ("plan section 4: the n=16 solve (late LU expansion, _foreign.expand_late) compiles silently to a "
                  "register-capped kernel (MEASURED L40S: 255 registers, 4.8 KB local frame; plan: ~19x off the "
                  "trend) instead of running within the trend or raising")
for n in (4, 8, 16, 24):
    @case(f"[size] jnp.linalg.solve n={n}: runs and matches jax.vmap, or raises naming the size",
          xfail={"cuda": _SOLVE16_XFAIL} if n == MAXD else None)
    def _(b, n=n):
        m, res = run_or_raise(b, solve_item, _solve_inputs(n), (0, None, None), solve_ref,
                              1e-9 if X64 else 2e-4, limit=f"({n}, {n})")
        if b == "cuda":
            if not res:
                raise Fail("instrument: no kernel resources recorded")
            if n > MAXD:                    # the contract above MAXD is a clear error, as cholesky / qr give
                raise Fail(f"n={n} > MAXD={MAXD} compiled silently ({_res_str(res)}) instead of raising")
            if any(r["regs"] >= REG_CAP for r in res):
                raise Fail(f"silent register-capped kernel: {_res_str(res)}")
        return f"{m}; {_res_str(res)}" if b == "cuda" else m


def _spd_inputs(n):
    rng = np.random.default_rng(100 + n)
    a = rng.normal(size=(n, n)) * 0.3
    M = jnp.asarray(a @ a.T + n * np.eye(n), F)
    return jnp.linspace(0.5, 1.5, NS).astype(F), M


def chol_item(t, x, M):
    d = jnp.diag(jnp.linalg.cholesky(M + x * jnp.eye(M.shape[0], dtype=M.dtype)))
    return t.add("Q", 0, jnp.sum(d)), d


def chol_ref(xs, M):
    return jax.jit(jax.vmap(lambda x: jnp.diag(jnp.linalg.cholesky(M + x * jnp.eye(M.shape[0], dtype=M.dtype)))))(xs)


def _geqrf(a):
    """lax.linalg.geqrf: the Householder QR primitive dense.py implements (not exported by jax.lax.linalg in the
    tested JAX versions; jnp.linalg.qr binds the `qr` primitive instead, see the cell below)"""
    f = getattr(lax.linalg, "geqrf", None)
    if f is None:
        from jax._src.lax import linalg as _lin
        f = _lin.geqrf
    return f(a)


def qr_item(t, x, M):        # Householder QR through the geqrf primitive (|diag R|: sign-free)
    d = jnp.abs(jnp.diag(_geqrf(M + x * jnp.eye(M.shape[0], dtype=M.dtype))[0]))
    return t.add("Q", 0, jnp.sum(d)), d


def qr_ref(xs, M):
    return jax.jit(jax.vmap(lambda x: jnp.abs(jnp.diag(_geqrf(M + x * jnp.eye(M.shape[0], dtype=M.dtype))[0]))))(xs)


@case("[size] jnp.linalg.qr n=4 (the `qr` primitive): runs and matches jax.vmap, or raises naming the primitive")
def _(b):
    # the cuda backend implements geqrf / ormqr (dense.py) but not the `qr` primitive jnp.linalg.qr binds: today a
    # clear UnsupportedError naming it (a coverage gap, not a wrong answer)
    def item(t, x, M):
        d = jnp.abs(jnp.diag(jnp.linalg.qr(M + x * jnp.eye(4, dtype=M.dtype), mode="r")))
        return t.add("Q", 0, jnp.sum(d)), d
    ref = lambda xs, M: jax.jit(jax.vmap(lambda x: jnp.abs(jnp.diag(jnp.linalg.qr(M + x * jnp.eye(4, dtype=M.dtype),
                                                                                    mode="r")))))(xs)
    m, res = run_or_raise(b, item, _spd_inputs(4), (0, None), ref, 1e-9 if X64 else 2e-4, limit="'qr'")
    return m


for _name, _item, _ref in (("cholesky", chol_item, chol_ref), ("qr (geqrf)", qr_item, qr_ref)):
    for n in (16, 24):
        @case(f"[size] {_name} n={n}: runs and matches jax.vmap, or raises naming the size (MAXD={MAXD})")
        def _(b, n=n, item=_item, ref=_ref):
            m, res = run_or_raise(b, item, _spd_inputs(n), (0, None), ref, 1e-9 if X64 else 2e-4,
                                  limit=f"({n}, {n})")
            if b == "cuda" and n > MAXD:
                raise Fail(f"n={n} > MAXD={MAXD} ran instead of raising")
            return f"{m}; {_res_str(res)}" if b == "cuda" else m


def mk_carry(S):
    def item(t, x, key):
        L = 3 + (bits(key, 0) % 6).astype(jnp.int32)

        def body(c):
            k, h, t = c
            h = h * 0.9 + 0.05 * jnp.flip(h) + x
            return k + 1, h, t.add("Q", 0, h[k % S])
        k, h, t = lax.while_loop(lambda c: c[0] < L, body, (0, jnp.full((S,), x, x.dtype), t))
        return t, h
    return item


def carry_ref(S):
    def one(x, key):
        L = 3 + (bits(key, 0) % 6).astype(jnp.int32)
        h = jnp.full((S,), x, x.dtype)
        return lax.fori_loop(0, L, lambda k, h: h * 0.9 + 0.05 * jnp.flip(h) + x, h)
    return lambda xs, keys: jax.jit(jax.vmap(one))(xs, keys)


for S in (8, 64, 512, 2048):
    @case(f"[size] per-item carry of {S} floats in a while loop: runs and matches jax.vmap, or raises")
    def _(b, S=S):
        keys = jax.random.split(jax.random.PRNGKey(S), NS)
        xs = jnp.linspace(0.5, 1.5, NS).astype(F)
        ref = carry_ref(S)
        f = tm.vmap(mk_carry(S), outputs={"Q": tm.Sum((1,), F)}, backend=b)
        _D.KERNELS.clear()
        try:
            r = jax.block_until_ready(f(xs, keys))
        except Exception as e:
            if not _tm_error(e, f" {S}"):
                raise Fail(f"raised without naming the size {S}: {type(e).__name__}: {str(e)[:200]}") from None
            raise Expected(f"{type(e).__name__}: {str(e).splitlines()[0][:160]}") from None
        res = _kernel_resources() if b == "cuda" else []
        m = near(r.per_item, ref(xs, keys), 1e-12 if X64 else 1e-5, "per-item carry vs jax.vmap")
        return f"{m}; {_res_str(res)}" if b == "cuda" else m


# =================================================================================================== [ad]
NA = 24
NBA = 16


def ad_item(t, x, key, tab, th):
    L = (5 + bits(key, 999) % 36).astype(jnp.int32)          # 5..40 steps: K > C^2 at C=1, 4; K < C^2 at C=16

    def body(c):
        k, w, y, t = c
        i = (bits(key, k) % NBA).astype(jnp.int32)
        v = w * tab[i] * (1.0 + 0.1 * y * y) * th[0]
        y2 = y + 0.05 * jnp.sin(3.0 * y + th[1])
        w2 = w * (0.999 + 0.001 * jnp.cos(y * th[2]))
        return k + 1, w2, y2, t.add("Q", i % NB, v)
    k, w, y, t = lax.while_loop(lambda c: c[0] < L, body, (0, x, 0.5 * x, t))
    return t, (w, y)


def ad_inputs():
    xs = jnp.linspace(0.5, 1.5, NA).astype(F)
    keys = jax.random.split(jax.random.PRNGKey(3), NA)
    tab = (1.0 + 0.5 * jnp.sin(jnp.arange(NBA))).astype(F)
    th = jnp.asarray([0.9, 0.3, 1.7], F)
    return xs, keys, tab, th


def ad_fn(b, **opts):
    f = tm.vmap(ad_item, outputs={"Q": tm.Sum((NB,), F)}, in_axes=(0, 0, None, None), backend=b, **opts)
    xs, keys, tab, th = ad_inputs()

    def G(x, tab, th):                    # (x, tab, th) -> (Q, per-item w, per-item y)
        r = f(x, keys, tab, th)
        return r.outputs["Q"], r.per_item[0], r.per_item[1]
    return G, (xs, tab, th)


AD_TOL = 1e-10 if X64 else 1e-5


def _dot(a, b):
    return sum(float(np.sum(np.asarray(x, np.float64) * np.asarray(y, np.float64)))
               for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


@case("[ad] <u, J v>: jax.jvp == jax.vjp (3 random direction pairs)")
def _(b):
    G, P = ad_fn(b)
    rng = np.random.default_rng(7)
    rnd = lambda t: jax.tree_util.tree_map(lambda a: jnp.asarray(rng.normal(size=np.shape(a)), F), t)
    out = jax.eval_shape(G, *P)
    jvp = jax.jit(lambda p, v: jax.jvp(G, p, v)[1])
    vjp = jax.jit(lambda p, u: jax.vjp(G, *p)[1](u))
    worst = 0.0
    absl = lambda t: jax.tree_util.tree_map(jnp.abs, t)
    for _ in range(3):
        v, u = rnd(P), rnd(out)
        jv = jvp(P, v)
        a = _dot(u, jv)
        c = _dot(vjp(P, u), v)
        # relative to sum |u_i (Jv)_i|: random directions make <u, Jv> a sum of terms of both signs, and rounding
        # is relative to the TERMS, not to their (possibly cancelling) sum
        e = abs(a - c) / max(abs(a), _dot(absl(u), absl(jv)), 1e-30)
        if not e <= AD_TOL:
            raise Fail(f"<u, Jv> {a:.9g} (jvp) vs {c:.9g} (vjp): rel {e:.2e} > {AD_TOL:.0e}")
        worst = max(worst, e)
    return f"max rel {worst:.1e}"


@case("[ad] jacfwd == jacrev (wrt per-item x, table, params)")
def _(b):
    G, P = ad_fn(b)
    Jf = jax.jit(jax.jacfwd(G, argnums=(0, 1, 2)))(*P)
    Jr = jax.jit(jax.jacrev(G, argnums=(0, 1, 2)))(*P)
    return near(Jf, Jr, AD_TOL, "jacfwd vs jacrev", atol=AD_TOL * 1e-3)


@case("[ad] grad identical for rev_checkpoints 1 / 4 / 16 / default")
def _(b):
    rng = np.random.default_rng(11)
    wq = jnp.asarray(rng.normal(size=NB), F)
    gs = {}
    for C in (1, 4, 16, None):
        G, P = ad_fn(b, **({} if C is None else dict(rev_checkpoints=C)))
        L = lambda x, tab, th, G=G: (jnp.sum(G(x, tab, th)[0] * wq) + jnp.sum(G(x, tab, th)[1] ** 2)
                                     + jnp.sum(jnp.sin(G(x, tab, th)[2])))
        gs[C] = jax.jit(jax.grad(L, argnums=(0, 1, 2)))(*P)
    msgs = []
    for C in (1, 4, 16):
        # the per-item cotangents (x) are each item's own replayed adjoint: the checkpoint count changes which states
        # are stored and which are recomputed, never their values -> bitwise.  The table / parameter cotangents are
        # SUMS over items (atomics on cuda: order-dependent) -> FT.
        msgs.append(f"C={C}: x " + same(gs[C][0], gs[None][0], f"C={C} per-item cotangent")
                    + ", tab/th " + near(gs[C][1:], gs[None][1:], FT, f"C={C} summed cotangents"))
    return "; ".join(msgs)


_TOKEN_XFAIL = ("round-2 band-aid audit A5: a reverse program's sparse-table cotangent TOKEN is a discard tally, "
                "i.e. modelled as a VALUE, so JAX's partial evaluation can put it at the same cond output as a "
                "plain array of the same aval (DTRAX item VJP: a 128-entry token and a 128-entry table) and the "
                "merged output's tally mark is a guess; reading it raises 'reads a Tally' here and, past the "
                "checker, in the emitter. Fix: a scatter-add EFFECT primitive with no value (not dtrax's Soft slot, "
                "which only silences the checker)")


@case("[ad] reverse cond: a cotangent token that meets a same-aval array at a cond output is not refused",
      ["lockstep"], xfail={"lockstep": _TOKEN_XFAIL})
def _(b):
    # hand-built, as the reverse program builds it (dtrax tests_wavefront/test_revtables.py T4)
    from tallymap import _rewrite
    from tallymap._tally import emit_p

    def f(t, a):
        def b0(c):
            tt, aa = c
            return emit_p.bind(tt, jnp.int32(0), jnp.asarray(1.0, F)), aa

        def b1(c):
            tt, aa = c
            return aa, tt                       # the filler: branch 1 swaps them
        u, v = lax.cond(a[0] > 0, b0, b1, (t, a))
        return t, jnp.sum(u[1:3]) + jnp.sum(v)  # and the merged outputs are READ
    cj = jax.make_jaxpr(f)(jnp.zeros((8,), F), jnp.ones((8,), F))
    try:
        _rewrite.check_tallies(cj, ["discard"], name="token meets array")
    except tm.TallyError as e:
        raise Fail(f"refused: {str(e).splitlines()[0][:160]}") from None
    return "accepted"


# =================================================================================================== [trunc]
CAP = 10


def trunc_item_of(exact):
    def item(t, x, key):
        L = 2 + (bits(key, 7) % 19).astype(jnp.int32)         # 2..20 steps; CAP = 10 truncates about half

        def body(c):
            k, w, t = c
            i = (bits(key, k) % NB).astype(jnp.int32)
            t = t.add("Q", i, w).add("C", i, 1)
            if exact:
                t = t.add("E", i, w)
            return k + 1, w * 0.75 + 0.125, t
        k, w, t = lax.while_loop(lambda c: c[0] < L, body, (jnp.int32(0), x, t))
        return t, (w, k)
    return item


def trunc_fn(b, exact=True, **opts):
    o = outs1(b, exact)
    return tm.vmap(trunc_item_of("E" in o), outputs=o, backend=b, max_steps=CAP, **opts)


def trunc_check(r, what):
    """the resource-limit rule itself: float tallies NaN (exact included), integer tallies partial and finite"""
    nt = int(r.stats["n_truncated"])
    _check(nt == int(np.sum(np.asarray(r.truncated))), f"{what}: n_truncated {nt} != sum(truncated)")
    _check(0 < nt < len(np.asarray(r.truncated)), f"instrument: {nt} truncated items (want some, not all)")
    for k, v in r.outputs.items():
        if np.asarray(v).dtype.kind == "f":
            _check(bool(np.all(np.isnan(np.asarray(v)))), f"{what}: float tally {k} not poisoned")
    k = np.asarray(r.per_item[1])
    _check(bool(np.all(k[np.asarray(r.truncated)] == CAP)), f"{what}: truncated items did not stop at {CAP}")
    return nt


@case("[trunc] max_steps: flags, n_truncated, partial int tally, NaN floats, per-item == dense reference")
def _(b):
    A = trunc_fn(b)(XS, KEYS)
    nt = trunc_check(A, b)
    Rr = reference(trunc_item_of(False), outputs=outs1(b, exact=False), max_steps=CAP)(XS, KEYS)
    same(A.truncated, Rr.truncated, "truncated flags vs reference")
    # (values only: under x64 tm.reference returns an int32 Sum as int64 -- its jnp.sum promotes -- while tm.vmap
    # keeps the declared int32)
    same(A.outputs["C"], np.asarray(Rr.outputs["C"]).astype(np.asarray(A.outputs["C"]).dtype),
         "partial int tally vs reference")
    same(A.per_item[1], Rr.per_item[1], "per-item trip counts vs reference")
    m = near(A.per_item[0], Rr.per_item[0], 1e-12 if X64 else 1e-6, "per-item w vs reference")
    return f"{nt}/{N1} truncated; per-item w {m}"


@case("[trunc] max_steps: cuda == lockstep (flags, counts, partial tallies, per-item)", ["cuda"])
def _(b):
    A, B = trunc_fn("cuda", exact=X64)(XS, KEYS), trunc_fn("lockstep", exact=X64)(XS, KEYS)
    trunc_check(A, "cuda")
    trunc_check(B, "lockstep")
    # per-item floats: two compilers, IEEE +, * only (see the [sched] lockstep == cuda cell)
    return agree(A, B, "cuda vs lockstep", per_item="tol", pi_tol=1e-12 if X64 else 1e-6)


@case("[trunc] max_steps: _schedule step == item (cuda)", ["cuda"])
def _(b):
    A, B = trunc_fn(b, _schedule="step")(XS, KEYS), trunc_fn(b, _schedule="item")(XS, KEYS)
    trunc_check(A, "step")
    trunc_check(B, "item")
    return agree(A, B, "step vs item")


@case("[trunc] max_steps + n_active: inactive items never truncated, tallies of the active prefix only")
def _(b):
    f = trunc_fn(b)
    g = jax.jit(lambda n: f(XS, KEYS, n_active=n))
    na = 20
    A = g(jnp.int32(na))
    full = trunc_fn(b)(XS[:na], KEYS[:na])
    tr = np.asarray(A.truncated)
    _check(not tr[na:].any(), "an inactive item is reported truncated")
    same(tr[:na], np.asarray(full.truncated), "active-prefix flags vs a call on the prefix")
    _check(int(A.stats["n_truncated"]) == int(full.stats["n_truncated"]), "n_truncated vs a call on the prefix")
    same(A.outputs["C"], full.outputs["C"], "partial int tally vs a call on the prefix")
    same(jax.tree_util.tree_map(lambda v: v[:na], A.per_item), full.per_item, "active per-item vs a call on the prefix")
    k = np.asarray(A.per_item[1])
    _check(not k[na:].any(), "inactive items ran")
    z = g(jnp.int32(0))
    _check(int(z.stats["n_truncated"]) == 0 and all(np.all(np.asarray(v) == 0) for v in z.outputs.values()),
           "n_active=0: something ran or was truncated")
    return f"n_active={na}: {int(A.stats['n_truncated'])} truncated, all in the prefix; n_active=0: zero, finite"


print("\n================ SUMMARY", jax.__version__, "x64" if X64 else "", jax.devices()[0].device_kind)
for n, b, m in RES:
    print(f"{n:62s} {b:8s} {m[:220]}")
cnt = lambda p: sum(m.startswith(p) for _, _, m in RES)
print("TOTAL", len(RES), "PASS", cnt("PASS"), "RAISES-OK", cnt("RAISES-OK"), "XFAIL", cnt("XFAIL"), "FAIL", cnt("FAIL"),
      "RAISE", cnt("RAISE "), "XPASS", cnt("XPASS"))
