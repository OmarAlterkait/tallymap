"""Integer value ranges of the cuda emitter: what the emitted code may assume about an integer value.

A range is (lo, hi), inclusive Python ints bounding EVERY element of a value, or None (nothing known beyond its
dtype).  It is a fact about values, so code that relies on it computes the same bits: a comparison the range decides
is a constant, a select on a constant predicate is the case it picks, and a gather / dynamic_slice / scatter index
inside its bounds needs no clamp (op_table).  JAX's own index normalisation `i < 0 ? i + n : i` (jnp indexing,
x.at[i]) then disappears for an index that is provably >= 0: an iota, an argmax, a value through max / clamp, an
unsigned or bool source, a counter of a fori_loop.

Sources: literals, iota, unsigned / bool / narrow dtypes, argmax / argmin, population_count / clz, and loop
carries (interval widening over the body: a carry whose bound moves is widened to its dtype's bound on that side; a
while whose predicate is `c < b` / `c <= b` / `b > c` on a carry c bounds c's top inside the body).  Integer
arithmetic wraps in the emitted code, so a result whose interval leaves its dtype's range is None.

Two entry points share the per-primitive rules: `eqn_ranges` (one equation, from its operands' ranges: the
emitter's path, Val.rng) and `jaxpr_ranges` (a whole jaxpr without emitting anything: loop invariants).
"""
import weakref

import numpy as np
from jax.extend.core import Literal

from ..._jaxpr import call_jaxpr, cond_branches, is_dropvar, scan_parts, while_parts

ENABLE = True        # TEST-ONLY A/B control: False = no ranges (every index clamped and normalised, as before)
# primitives whose result k IS operand k (measurement markers and the like may register here)
PASS_THROUGH = set()


def _np(dt):
    """numpy dtype, or None (a typed PRNG key, or no dtype: never an integer range)."""
    if dt is None:
        return None
    try:
        return np.dtype(dt)
    except TypeError:
        return None


def dtype_range(dt):
    dt = _np(dt)
    if dt is None:
        return None
    if dt == np.bool_:
        return (0, 1)
    if dt.kind in "iu":
        ii = np.iinfo(dt)
        return (int(ii.min), int(ii.max))
    return None


def _is_int(dt):
    dt = _np(dt)
    if dt is None:
        return False
    return dt.kind in "iu" or dt == np.bool_


def clip(r, dt):
    """r if it fits dtype dt (the emitted integer arithmetic wraps outside it), else None."""
    if r is None:
        return None
    b = dtype_range(dt)
    if b is None or r[0] < b[0] or r[1] > b[1] or r[0] > r[1]:
        return None
    return (int(r[0]), int(r[1]))


def meet(a, b):
    """both facts hold: the intersection (None = unknown)."""
    if a is None:
        return b
    if b is None:
        return a
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if lo <= hi else a


def join(a, b):
    """either holds: the hull."""
    if a is None or b is None:
        return None
    return (min(a[0], b[0]), max(a[1], b[1]))


def full(r, dt):
    """r, or dt's own range when r is unknown (None for floats)."""
    return r if r is not None else dtype_range(dt)


def lit_range(val, dt):
    if not _is_int(dt):
        return None
    a = np.asarray(val)
    if a.size == 0:
        return None
    return (int(a.min()), int(a.max()))


_CMP = {"lt": lambda a, b: (a[1] < b[0], a[0] >= b[1]), "le": lambda a, b: (a[1] <= b[0], a[0] > b[1]),
        "gt": lambda a, b: (a[0] > b[1], a[1] <= b[0]), "ge": lambda a, b: (a[0] >= b[1], a[1] < b[0]),
        "eq": lambda a, b: (a[0] == a[1] == b[0] == b[1], a[1] < b[0] or b[1] < a[0]),
        "ne": lambda a, b: (a[1] < b[0] or b[1] < a[0], a[0] == a[1] == b[0] == b[1])}


def compare(name, a, b, da, db):
    """True / False when the integer ranges decide `a <name> b` for every element, else None."""
    if name not in _CMP or not (_is_int(da) and _is_int(db)):
        return None
    a, b = full(a, da), full(b, db)
    if a is None or b is None:
        return None
    yes, no = _CMP[name](a, b)
    return True if yes else (False if no else None)


def _ew(name, rs, dts, out_dt, params):
    """the range of an elementwise integer op's result, or None."""
    if not _is_int(out_dt):
        return None
    if name == "convert_element_type":
        if not _is_int(dts[0]):
            return None
        return clip(full(rs[0], dts[0]), out_dt)
    if name in _CMP:
        c = compare(name, rs[0], rs[1], dts[0], dts[1])
        return (int(c), int(c)) if c is not None else (0, 1)
    if name == "select_n":
        out = None
        for k, r in enumerate(rs[1:]):
            r = full(r, dts[k + 1])
            out = r if k == 0 else join(out, r)
        return clip(out, out_dt)
    fr = [full(r, d) for r, d in zip(rs, dts)]
    if any(r is None for r in fr):
        return None
    a = fr[0]
    b = fr[1] if len(fr) > 1 else None
    if name in ("add", "add_any"):
        return clip((a[0] + b[0], a[1] + b[1]), out_dt)
    if name == "sub":
        return clip((a[0] - b[1], a[1] - b[0]), out_dt)
    if name == "mul":
        ps = [x * y for x in a for y in b]
        return clip((min(ps), max(ps)), out_dt)
    if name == "neg":
        return clip((-a[1], -a[0]), out_dt)
    if name == "max":
        return (max(a[0], b[0]), max(a[1], b[1]))
    if name == "min":
        return (min(a[0], b[0]), min(a[1], b[1]))
    if name == "abs":
        if a[0] >= 0:
            return a
        return clip((0 if a[1] >= 0 else -a[1], max(-a[0], a[1])), out_dt)
    if name == "and" and np.dtype(out_dt) != np.bool_:
        nn = [r for r in (a, b) if r[0] >= 0]
        return (0, min(r[1] for r in nn)) if nn else None
    if name == "rem" and b[0] > 0:                 # C remainder: the dividend's sign, |r| < |divisor|
        m = b[1] - 1
        return (0, min(a[1], m)) if a[0] >= 0 else (max(a[0], -m), min(max(a[1], 0), m))
    if name == "div" and a[0] >= 0 and b[0] >= 1:
        return (a[0] // b[1], a[1] // b[0])
    if name in ("shift_right_logical", "shift_right_arithmetic") and a[0] >= 0 and b[0] >= 0:
        return (a[0] >> min(b[1], 63), a[1] >> b[0])
    if name in ("population_count", "clz"):
        return (0, np.dtype(out_dt).itemsize * 8)
    return None


_STRUCT = ("broadcast_in_dim", "reshape", "transpose", "slice", "rev", "squeeze", "expand_dims", "dynamic_slice",
           "reduce_max", "reduce_min", "copy", "copy_p", "stop_gradient", "optimization_barrier", "reduce_precision")


def eqn_ranges(eqn, rs, dts, nested=True):
    """ranges of an equation's results from its operands' ranges `rs` (and operand dtypes `dts`); nested jaxprs are
    analysed by jaxpr_ranges (nested=False: none for while / cond / scan / calls, whose emission rules set them).
    -> list, one per outvar."""
    name = eqn.primitive.name
    outs = eqn.outvars
    odt = [_np(getattr(v.aval, "dtype", None)) for v in outs]
    none = [None] * len(outs)
    if not ENABLE:
        return none
    try:
        if name in PASS_THROUGH:
            return [clip(full(r, d), o) if o is not None and _is_int(o) else None for r, d, o in zip(rs, dts, odt)]
        if name in ("reduce_max", "reduce_min", "argmax", "argmin") and any(
                int(eqn.invars[0].aval.shape[a]) == 0 for a in eqn.params.get("axes", ())):
            return none                  # (an empty reduction is its init value, outside the operand's range)
        if name in _STRUCT:
            return [clip(full(rs[0], dts[0]), odt[0]) if odt[0] is not None and _is_int(odt[0]) else None]
        if name == "iota":
            n = int(eqn.params["shape"][eqn.params["dimension"]])
            return [clip((0, max(n - 1, 0)), odt[0])]
        if name in ("argmax", "argmin"):
            n = int(eqn.invars[0].aval.shape[eqn.params["axes"][0]])
            return [clip((0, max(n - 1, 0)), odt[0])]
        if name == "gather":
            fv = eqn.params.get("fill_value")
            mode = str(eqn.params.get("mode")).upper()
            r = full(rs[0], dts[0])
            if "FILL" in mode or "DROP" in mode:
                from .op_table import _fill_default
                r = join(r, lit_range(_fill_default(odt[0]) if fv is None else fv, odt[0]))
            return [clip(r, odt[0])]
        if name in ("scatter", "dynamic_update_slice"):
            k = 2 if name == "scatter" else 1
            return [clip(join(full(rs[0], dts[0]), full(rs[k], dts[k])), odt[0])]
        if name == "concatenate":
            out = full(rs[0], dts[0])
            for r, d in zip(rs[1:], dts[1:]):
                out = join(out, full(r, d))
            return [clip(out, odt[0])]
        if name == "pad":
            return [clip(join(full(rs[0], dts[0]), full(rs[1], dts[1])), odt[0])]
        if name in ("scatter-max", "scatter_max", "scatter-min", "scatter_min"):
            return [clip(join(full(rs[0], dts[0]), full(rs[2], dts[2])), odt[0])]
        if name in ("cummax", "cummin"):
            return [clip(full(rs[0], dts[0]), odt[0])]
        if name == "sort":
            return [clip(full(r, d), o) if o is not None else None for r, d, o in zip(rs, dts, odt)]
        if name in ("population_count", "clz") or name in _CMP or name in _EW_NAMES:
            return [_ew(name, rs, dts, odt[0], eqn.params)]
        if not nested and (name in ("while", "cond", "scan") or call_jaxpr(eqn) is not None):
            return none
        if name == "while":
            return while_ranges(eqn, rs)[1]
        if name == "cond":
            out = None
            for k, br in enumerate(cond_branches(eqn)):
                r = jaxpr_ranges(br, rs[1:])
                out = r if k == 0 else [join(a, b) for a, b in zip(out, r)]
            return out
        if name == "scan":
            return scan_ranges(eqn, rs)
        sub = call_jaxpr(eqn)
        if sub is not None:
            return jaxpr_ranges(sub, rs)
    except Exception:                    # an unexpected parameter layout: no fact (never a wrong one)
        return none
    return none


_EW_NAMES = {"add", "add_any", "sub", "mul", "neg", "max", "min", "abs", "and", "rem", "div", "convert_element_type",
             "select_n", "shift_right_logical", "shift_right_arithmetic"}


def _vrange(v, env):
    if isinstance(v, Literal):
        return lit_range(v.val, v.aval.dtype)
    return env.get(v)


# Jaxpr -> {(input ranges, constant ranges, ENABLE): output ranges}: nested loops are analysed once per fact set.
# Weak on the Jaxpr (an entry dies with its jaxpr; a hit is always the same object), and the constants' ranges are
# part of the fact set: two ClosedJaxprs may share one Jaxpr with different constants.
_MEMO = weakref.WeakKeyDictionary()
_MEMO_FACTS = 64     # fact sets kept per jaxpr (a bound; a jaxpr is analysed under very few)


def jaxpr_ranges(cj, rs, env_out=None):
    """the ranges of a (closed) jaxpr's outputs from its inputs' ranges, nothing emitted."""
    j = getattr(cj, "jaxpr", cj)
    consts = getattr(cj, "consts", [])
    env = {}
    for v, c in zip(j.constvars, consts):
        if _is_int(v.aval.dtype) and np.ndim(c) == 0:
            try:
                env[v] = lit_range(np.asarray(c), v.aval.dtype)
            except Exception:
                pass
    key = (tuple(rs), tuple(env.get(v) for v in j.constvars), ENABLE)
    if env_out is None:
        memo = _MEMO.get(j)
        hit = None if memo is None else memo.get(key)
        if hit is not None:
            return list(hit)
    for v, r in zip(j.invars, rs):
        if _is_int(v.aval.dtype):
            env[v] = clip(full(r, v.aval.dtype), v.aval.dtype)
    for e in j.eqns:
        ins = [_vrange(v, env) for v in e.invars]
        dts = [_np(getattr(v.aval, "dtype", None)) for v in e.invars]
        try:
            outs = eqn_ranges(e, ins, dts)
        except Exception:
            outs = [None] * len(e.outvars)
        for v, r in zip(e.outvars, outs):
            if not is_dropvar(v) and r is not None:
                env[v] = r
    if env_out is not None:
        env_out.update(env)
    out = [_vrange(v, env) for v in j.outvars]
    if env_out is None:
        memo = _MEMO.setdefault(j, {})
        if len(memo) >= _MEMO_FACTS:
            memo.clear()
        memo[key] = tuple(out)
    return out


def _bound_from_pred(w, ccr, cr):
    """carry index -> an upper bound on that carry inside the body, from a predicate `c < b` / `c <= b` / `b > c`
    / `b >= c` (b's range from the cond's own consts and carries).  The body runs only where the predicate held."""
    cj = w.cond.jaxpr
    if len(cj.outvars) != 1 or cj.outvars[0].aval.shape != () or isinstance(cj.outvars[0], Literal):
        return {}
    o = cj.outvars[0]
    eq = next((e for e in cj.eqns if o in e.outvars), None)
    if eq is None or eq.primitive.name not in ("lt", "le", "gt", "ge"):
        return {}
    env = {}
    jaxpr_ranges(w.cond, list(ccr) + list(cr), env)
    a, b = eq.invars
    name = eq.primitive.name
    if name in ("gt", "ge"):
        a, b = b, a
        name = "lt" if name == "gt" else "le"
    carries = list(cj.invars[w.nc:])
    if isinstance(a, Literal) or a not in carries or not _is_int(a.aval.dtype) or a.aval.shape != ():
        return {}
    rb = _vrange(b, env)
    rb = full(rb, b.aval.dtype) if _is_int(b.aval.dtype) else None
    if rb is None:
        return {}
    return {carries.index(a): rb[1] - (1 if name == "lt" else 0)}


def _widen(old, new, dt):
    if old is None or new is None:
        return None
    b = dtype_range(dt)
    lo = old[0] if new[0] >= old[0] else b[0]
    hi = old[1] if new[1] <= old[1] else b[1]
    return (lo, hi)


def while_ranges(eqn, rs):
    """(carry ranges inside the body, result ranges) of a while equation from its operands' ranges."""
    w = while_parts(eqn)
    init = list(rs[w.nc + w.nb:])
    bconsts = list(rs[w.nc:w.nc + w.nb])
    cav = [v.aval for v in w.body.jaxpr.invars[w.nb:]]
    dts = [_np(a.dtype) for a in cav]
    cr = [clip(full(r, d), d) if _is_int(d) else None for r, d in zip(init, dts)]
    body_r = cr
    for it in range(8):
        ub = _bound_from_pred(w, rs[:w.nc], cr)
        body_r = [meet(full(r, d), (dtype_range(d)[0], ub[k])) if k in ub else r
                  for k, (r, d) in enumerate(zip(cr, dts))]
        if ub:
            body_r = [r if r is None or r[0] <= r[1] else None for r in body_r]
        out = jaxpr_ranges(w.body, bconsts + body_r)
        new = [join(a, b) if _is_int(d) else None for a, b, d in zip(cr, out, dts)]
        if new == cr:
            return body_r, cr
        cr = [_widen(a, b, d) if it >= 1 and _is_int(d) else b for a, b, d in zip(cr, new, dts)]
    return [None] * len(cr), [None] * len(cr)


def scan_ranges(eqn, rs, carries=False):
    """result ranges of a scan (carries by widening; stacked outputs from the body at the carry invariant);
    carries=True: the carry invariant (the body's carry ranges) instead."""
    sc = scan_parts(eqn)
    nc, nk = sc.nconsts, sc.ncarry
    consts, init, xs = list(rs[:nc]), list(rs[nc:nc + nk]), list(rs[nc + nk:])
    dts = [_np(v.aval.dtype) for v in sc.body.jaxpr.invars[nc:nc + nk]]
    cr = [clip(full(r, d), d) if _is_int(d) else None for r, d in zip(init, dts)]
    for it in range(8):
        out = jaxpr_ranges(sc.body, consts + cr + xs)
        new = [join(a, b) if _is_int(d) else None for a, b, d in zip(cr, out[:nk], dts)]
        if new == cr:
            return cr if carries else cr + out[nk:]
        cr = [_widen(a, b, d) if it >= 1 and _is_int(d) else b for a, b, d in zip(cr, new, dts)]
    return [None] * (nk if carries else len(eqn.outvars))


# ----------------------------------------------------------------------------- float facts
# (nonnan, finite): every element is not NaN / is finite.  A min / max of two non-NaN operands needs no NaN test
# (op_table.b_max / b_min; f32 on the device one FMNMX).  Facts flow
# from literals, integer -> float conversions, selects, sign / rounding / abs / neg, min / max, the bounded
# functions, the NaN guards jnp.where(x != x, c, x) / jnp.where(isfinite(x), x, c), and finite + finite (never NaN,
# though it may overflow).  Loop carries carry none (unknown).
FF_NONE = (False, False)


def _is_float(dt):
    dt = _np(dt)
    return dt is not None and dt.kind == "f"


def ff_lit(val, dt):
    if not _is_float(dt):
        return FF_NONE
    a = np.asarray(val, dtype=_np(dt))
    return (bool(not np.isnan(a).any()), bool(np.isfinite(a).all()))


def _and(*fs):
    return (all(f[0] for f in fs), all(f[1] for f in fs))


_FF_SAME = {"neg", "abs", "floor", "ceil", "round", "reduce_max", "reduce_min", "broadcast_in_dim", "reshape",
            "transpose", "slice", "rev", "squeeze", "expand_dims", "dynamic_slice", "copy", "copy_p", "cummax",
            "cummin", "reduce_precision", "real", "imag"}


def float_facts(name, ffs, ins, out_dt):
    """(nonnan, finite) of an equation's (first) float result from its operands' facts `ffs` and values `ins`
    (Val.origin = (primitive, operands) of a computed operand: the NaN-guard patterns; `is` compares SSA values,
    codegen.new_version), or None (not a float)."""
    if not ENABLE or not _is_float(out_dt):
        return None
    try:
        if name == "convert_element_type":
            src = getattr(ins[0], "dtype", None)
            if _is_int(src):
                return (True, True)
            if _is_float(src):
                return (ffs[0][0], ffs[0][1] and _np(out_dt).itemsize >= _np(src).itemsize)
            return FF_NONE
        if name in _FF_SAME:
            return ffs[0]
        if name == "sign":
            return (ffs[0][0], ffs[0][0])
        if name == "iota":
            return (True, True)
        if name in ("max", "min"):
            return _and(ffs[0], ffs[1])
        if name in ("add", "sub", "mul"):
            return (ffs[0][1] and ffs[1][1], False)
        if name in ("exp", "exp2"):
            return (ffs[0][0], False)
        if name in ("tanh", "logistic", "erf"):
            return (ffs[0][0], ffs[0][0])
        if name in ("sin", "cos"):
            return (ffs[0][1], ffs[0][1])
        if name == "select_n":
            out = _and(*ffs[1:])
            org = getattr(ins[0], "origin", None)
            if len(ins) == 3 and org is not None:
                p, ops = org
                # where(x != x, c, x) = select_n(x != x, x, c): x where it equals itself (not NaN), else c
                if p == "ne" and len(ops) == 2 and ops[0] is ops[1] and ops[0] is ins[1]:
                    return (ffs[2][0], ffs[2][1] and ffs[1][1])
                # where(isfinite(x), x, c) = select_n(isfinite(x), c, x): x where finite, else c
                if p == "is_finite" and len(ops) == 1 and ops[0] is ins[2]:
                    return ffs[1]
            return out
        if name == "concatenate" or name == "pad":
            return _and(*ffs)
    except Exception:
        return FF_NONE
    return None                 # no claim: a call / loop result keeps the facts its own body gave it


# ----------------------------------------------------------------------------- magnitude floors (fast mode)
# |x| >= m > 0 for every element (None: unknown).  cuda_math='fast' divides by such a divisor with div.approx
# (MUFU.RCP + FMUL, 2 ulp for 2^-126 <= |b| <= 2^126) instead of div.full's range-safe sequence: the divisor of a
# guarded reciprocal 1 / where(|d| < eps, c, d) or 1 / max(|d|, eps) -- a direction component or a slope kept away
# from zero.  (Above 2^126 div.approx returns 0 where div.full returns a value below 2^-126 |a|; the fast mode is
# validated statistically.)  Sources: nonzero literals, the guards, max with a positive operand, abs / neg,
# products of floored values, selects of floored values.
TINY = 2.0 ** -126


def mag_floor(name, mls, ins, same=None):
    """the magnitude floor of an equation's (first) result from its operands' floors `mls` and values `ins`; same(a,
    b): are two values the same value (jnp indexes d[a] afresh for each use)."""
    same = same or (lambda a, b: a is b)
    try:
        if name in ("neg", "abs", "broadcast_in_dim", "reshape", "squeeze", "expand_dims", "copy", "copy_p",
                    "reduce_precision"):
            return mls[0]
        if name == "mul":
            return mls[0] * mls[1] if mls[0] and mls[1] else None
        if name == "max":
            pos = [float(v.cval) for v in ins if v.cval is not None and np.ndim(v.cval) == 0 and float(v.cval) > 0]
            if pos:
                return max(pos)
            return None
        if name == "select_n" and len(ins) == 3:
            m0, m1 = mls[1], mls[2]
            org = getattr(ins[0], "origin", None)
            # where(|d| < eps, c, d) = select_n(|d| < eps, d, c): d only where |d| >= eps
            if org is not None and org[0] in ("lt", "le") and len(org[1]) == 2:
                a, e = org[1]
                ao = getattr(a, "origin", None)
                if (ao is not None and ao[0] == "abs" and same(ao[1][0], ins[1]) and e.cval is not None
                        and np.ndim(e.cval) == 0 and float(e.cval) > 0):
                    m0 = float(e.cval)              # (lt: |d| >= eps there; le: |d| > eps)
            return min(m0, m1) if m0 and m1 else None
    except Exception:
        return None
    return None
