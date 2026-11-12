"""Per-primitive emission rules for the cuda emitter (see codegen.py).

Every rule has the signature ``rule(em, eqn, ins, **params) -> Val | [Val]``.
IEEE semantics only: round-to-nearest +,-,*,/,sqrt with no FMA contraction; transcendentals from the CUDA math
library (libdevice).  Sugar primitives (square, clamp, integer_pow, squeeze, split, the total-order compares, ...)
have no rule: canon.py rewrites them into the canonical primitives before emission.  No attempt is made to reproduce
XLA's own algebraic rewrites or approximations.
"""
import jax
import numpy as np

from ..._compat import key_impl, random_bits, random_fold_in, random_seed, random_split
from ..._jaxpr import call_jaxpr, cond_branches, scan_parts, while_parts
from .codegen import (
    _ST,
    _UT,
    LAZY_MAX_COST,
    CodegenError,
    Lazy,
    Local,
    Scalar,
    Table,
    View,
    VTally,
    _isf,
    _isint,
    _size,
    ctype,
    elem,
    fill_decl,
    ia,
    im,
    is_key_dtype,
    lazy_cost,
    lit,
    ravel,
    unravel,
)

RULES = {}

def rule(*names):
    def deco(f):
        for n in names:
            RULES[n] = f
        return f
    return deco


def _aval(eqn, i=0):
    return eqn.outvars[i].aval


def phys(aval):
    """(shape, dtype, key_impl) of the physical representation of aval."""
    if is_key_dtype(aval.dtype):
        impl = key_impl(aval.dtype)
        return tuple(aval.shape) + tuple(impl.key_shape), np.dtype(np.uint32), impl
    return tuple(aval.shape), np.dtype(aval.dtype), None


def as_key(v, impl):
    w = View(v, v.shape, lambda idx: idx)
    w.key_impl = impl
    return w


# =============================================================================== elementwise
def _w(t):
    return "u64" if t in ("i64", "u64") else "u32"


def _iar(t, a, b, op):
    return f"(({t})(({_w(t)})({a}) {op} ({_w(t)})({b})))"


def b_add(em, a, ts, t, p, vals):
    if t in ("sj_c64", "sj_c128"):
        return f"sj_cadd({a[0]}, {a[1]})"
    if t == "float":
        return f"__fadd_rn({a[0]}, {a[1]})"
    if t == "double":
        return f"__dadd_rn({a[0]}, {a[1]})"
    if t == "bool":
        return f"({a[0]} || {a[1]})"
    return _iar(t, a[0], a[1], "+")


def b_sub(em, a, ts, t, p, vals):
    if t in ("sj_c64", "sj_c128"):
        return f"sj_csub({a[0]}, {a[1]})"
    if t == "float":
        return f"__fsub_rn({a[0]}, {a[1]})"
    if t == "double":
        return f"__dsub_rn({a[0]}, {a[1]})"
    return _iar(t, a[0], a[1], "-")


def b_mul(em, a, ts, t, p, vals):
    if t in ("sj_c64", "sj_c128"):
        return f"sj_cmul({a[0]}, {a[1]})"
    if t == "float":
        return f"__fmul_rn({a[0]}, {a[1]})"
    if t == "double":
        return f"__dmul_rn({a[0]}, {a[1]})"
    if t == "bool":
        return f"({a[0]} && {a[1]})"
    return _iar(t, a[0], a[1], "*")


def b_div(em, a, ts, t, p, vals):
    if t in ("sj_c64", "sj_c128"):
        return f"sj_cdiv({a[0]}, {a[1]})"
    if t == "float":
        return f"SJ_FDIV({a[0]}, {a[1]})"
    if t == "double":
        return f"__ddiv_rn({a[0]}, {a[1]})"
    return f"sj_idiv({a[0]}, {a[1]})"


def b_rem(em, a, ts, t, p, vals):
    if t == "float":
        return f"fmodf({a[0]}, {a[1]})"
    if t == "double":
        return f"fmod({a[0]}, {a[1]})"
    return f"sj_irem({a[0]}, {a[1]})"


def b_max(em, a, ts, t, p, vals):
    if t in ("float", "double"):
        return f"sj_fmax({a[0]}, {a[1]})"
    if t == "bool":
        return f"({a[0]} || {a[1]})"
    return f"sj_imax<{t}>({a[0]}, {a[1]})"


def b_min(em, a, ts, t, p, vals):
    if t in ("float", "double"):
        return f"sj_fmin({a[0]}, {a[1]})"
    if t == "bool":
        return f"({a[0]} && {a[1]})"
    return f"sj_imin<{t}>({a[0]}, {a[1]})"


def b_neg(em, a, ts, t, p, vals):
    if t in ("sj_c64", "sj_c128"):
        return f"sj_cneg({a[0]})"
    if t in ("float", "double"):
        return f"(-({a[0]}))"
    return f"(({t})(({_w(t)})0 - ({_w(t)})({a[0]})))"


def b_abs(em, a, ts, t, p, vals):
    if ts[0] in ("sj_c64", "sj_c128"):
        return f"sj_cabs({a[0]})"
    if t == "float":
        return f"fabsf({a[0]})"
    if t == "double":
        return f"fabs({a[0]})"
    if t.startswith("u"):
        return a[0]
    return f"(({a[0]}) < 0 ? (({t})(({_w(t)})0 - ({_w(t)})({a[0]}))) : ({a[0]}))"


def b_sign(em, a, ts, t, p, vals):
    if t in ("float", "double"):
        return f"sj_fsign({a[0]})"
    if t.startswith("u"):
        return f"(({t})(({a[0]}) != 0))"
    return f"(({t})((({a[0]}) > 0) - (({a[0]}) < 0)))"


def _f1(f32, f64=None):
    f64 = f64 or f32[:-1]

    def b(em, a, ts, t, p, vals):
        if t in ("sj_c64", "sj_c128") and f32 in ("expf", "logf"):
            return f"sj_c{f32[:3]}({a[0]})"
        if t == "float":
            return f"{f32}({a[0]})"
        if t == "double":
            return f"{f64}({a[0]})"
        raise CodegenError(f"{f32}: dtype {t}")
    return b


def _f2(f32, f64=None):
    f64 = f64 or f32[:-1]

    def b(em, a, ts, t, p, vals):
        if t == "float":
            return f"{f32}({a[0]}, {a[1]})"
        if t == "double":
            return f"{f64}({a[0]}, {a[1]})"
        raise CodegenError(f"{f32}: dtype {t}")
    return b


def b_sqrt(em, a, ts, t, p, vals):
    if t in ("sj_c64", "sj_c128"):
        return f"sj_csqrt({a[0]})"
    if t == "float":
        return f"SJ_FSQRT({a[0]})"
    return f"__dsqrt_rn({a[0]})"


def b_round(em, a, ts, t, p, vals):
    m = int(p.get("rounding_method", 0))
    f = ("roundf", "round") if m == 0 else ("rintf", "rint")
    return f"{f[0] if t == 'float' else f[1]}({a[0]})"


def b_isfinite(em, a, ts, t, p, vals):
    return f"isfinite({a[0]})"


def b_logistic(em, a, ts, t, p, vals):
    if t == "float":
        d = f"__fadd_rn(1.0f, expf(-({a[0]})))"
        return f"SJ_FDIV(1.0f, {d})"
    return f"__ddiv_rn(1.0, __dadd_rn(1.0, exp(-({a[0]}))))"


def b_rsqrt(em, a, ts, t, p, vals):
    if t == "float":
        return f"SJ_FRSQRT({a[0]})"
    return f"rsqrt({a[0]})"


def b_erf(em, a, ts, t, p, vals):
    if t == "float":
        return f"erff({a[0]})"
    return f"erf({a[0]})"


def b_erfinv(em, a, ts, t, p, vals):
    if t == "float":
        return f"erfinvf({a[0]})"
    return f"erfinv({a[0]})"


def b_asin(em, a, ts, t, p, vals):
    return f"{'asinf' if t == 'float' else 'asin'}({a[0]})"


def b_atan(em, a, ts, t, p, vals):
    return f"{'atanf' if t == 'float' else 'atan'}({a[0]})"


def b_exp2(em, a, ts, t, p, vals):
    return f"{'exp2f' if t == 'float' else 'exp2'}({a[0]})"


def b_tanh(em, a, ts, t, p, vals):
    if t == "float":
        return f"tanhf({a[0]})"
    return f"tanh({a[0]})"


def b_and(em, a, ts, t, p, vals):
    return f"({a[0]} && {a[1]})" if t == "bool" else f"(({t})(({a[0]}) & ({a[1]})))"


def b_or(em, a, ts, t, p, vals):
    return f"({a[0]} || {a[1]})" if t == "bool" else f"(({t})(({a[0]}) | ({a[1]})))"


def b_xor(em, a, ts, t, p, vals):
    return f"(({a[0]}) != ({a[1]}))" if t == "bool" else f"(({t})(({a[0]}) ^ ({a[1]})))"


def b_not(em, a, ts, t, p, vals):
    return f"(!({a[0]}))" if t == "bool" else f"(({t})(~({a[0]})))"


def _cmp(op):
    def b(em, a, ts, t, p, vals):
        return f"(({a[0]}) {op} ({a[1]}))"
    return b


def b_log2(em, a, ts, t, p, vals):
    return f"{'log2f' if t == 'float' else 'log2'}({a[0]})"


def b_mulhi(em, a, ts, t, p, vals):
    if t in ("i32",):
        return f"__mulhi({a[0]}, {a[1]})"
    if t == "u32":
        return f"__umulhi({a[0]}, {a[1]})"
    if t == "i64":
        return f"__mul64hi({a[0]}, {a[1]})"
    if t == "u64":
        return f"__umul64hi({a[0]}, {a[1]})"
    return f"(({t})(((i32)({a[0]}) * (i32)({a[1]})) >> {8 if t in ('i8', 'u8') else 16}))"


def b_real(em, a, ts, t, p, vals):
    return f"({a[0]}).re"


def b_imag(em, a, ts, t, p, vals):
    return f"({a[0]}).im"


def b_complex(em, a, ts, t, p, vals):
    return f"sj_mk{t[3:]}({a[0]}, {a[1]})"


def b_conj(em, a, ts, t, p, vals):
    if ts[0] in ("sj_c64", "sj_c128"):
        return f"sj_cconj({a[0]})"
    return conv(a[0], ts[0], t)


def b_shl(em, a, ts, t, p, vals):
    return f"sj_shl<{t},{_UT[t]}>({a[0]}, {a[1]})"


def b_shrl(em, a, ts, t, p, vals):
    return f"sj_shrl<{t},{_UT[t]}>({a[0]}, {a[1]})"


def b_shra(em, a, ts, t, p, vals):
    return f"sj_shra<{t},{_UT[t]},{_ST[t]}>({a[0]}, {a[1]})"


def b_popc(em, a, ts, t, p, vals):
    if t in ("i64", "u64"):
        return f"(({t})__popcll((u64)({a[0]})))"
    return f"(({t})__popc((u32)({_UT[t]})({a[0]})))"


def b_clz(em, a, ts, t, p, vals):
    if t in ("i64", "u64"):
        return f"(({t})__clzll((i64)({a[0]})))"
    sub = 32 - {"i8": 8, "u8": 8, "i16": 16, "u16": 16}.get(t, 32)
    return f"(({t})(__clz((i32)(u32)({_UT[t]})({a[0]})) - {sub}))"


def b_select_n(em, a, ts, t, p, vals):
    w, cases = a[0], a[1:]
    if ts[0] == "bool":
        return f"(({w}) ? ({cases[1]}) : ({cases[0]}))" if len(cases) == 2 else cases[0]
    e = cases[-1]
    for i in range(len(cases) - 2, -1, -1):
        e = f"(({w}) == {i} ? ({cases[i]}) : {e})"
    return e


def b_nextafter(em, a, ts, t, p, vals):
    return f"{'nextafterf' if t == 'float' else 'nextafter'}({a[0]}, {a[1]})"


def b_pow(em, a, ts, t, p, vals):
    if t == "float":
        y = a[1] if ts[1] == "float" else f"((float)({a[1]}))"
        return f"powf({a[0]}, {y})"
    y = a[1] if ts[1] == "double" else f"((double)({a[1]}))"
    return f"pow({a[0]}, {y})"


_F2I = {("float", "i32"): "__float2int_rz", ("float", "u32"): "__float2uint_rz",
        ("float", "i64"): "__float2ll_rz", ("float", "u64"): "__float2ull_rz",
        ("double", "i32"): "__double2int_rz", ("double", "u32"): "__double2uint_rz",
        ("double", "i64"): "__double2ll_rz", ("double", "u64"): "__double2ull_rz"}
_I2F = {("i32", "float"): "__int2float_rn", ("u32", "float"): "__uint2float_rn",
        ("i64", "float"): "__ll2float_rn", ("u64", "float"): "__ull2float_rn",
        ("i32", "double"): "__int2double_rn", ("u32", "double"): "__uint2double_rn",
        ("i64", "double"): "__ll2double_rn", ("u64", "double"): "__ull2double_rn"}


def conv(x, f, t):
    if f == t:
        return x
    if t in ("sj_c64", "sj_c128"):
        ft = "float" if t == "sj_c64" else "double"
        if f in ("sj_c64", "sj_c128"):
            return f"sj_mk{t[3:]}(({ft})({x}).re, ({ft})({x}).im)"
        zero = "0.0f" if ft == "float" else "0.0"
        return f"sj_mk{t[3:]}({conv(x, f, ft)}, {zero})"
    if f in ("sj_c64", "sj_c128"):
        return conv(f"({x}).re", "float" if f == "sj_c64" else "double", t)
    if t == "bool":
        return f"(({x}) != 0)"
    if f == "bool":
        return f"(({t})(({x}) ? 1 : 0))"
    fl = f in ("float", "double")
    tl = t in ("float", "double")
    if fl and tl:
        return f"((double)({x}))" if t == "double" else f"__double2float_rn({x})"
    if not fl and tl:
        g = {"i8": "i32", "i16": "i32", "u8": "u32", "u16": "u32"}.get(f, f)
        return f"{_I2F[(g, t)]}(({g})({x}))"
    if fl and not tl:
        if t in ("i8", "i16", "u8", "u16"):
            lo, hi = {"i8": (-128, 127), "i16": (-32768, 32767), "u8": (0, 255), "u16": (0, 65535)}[t]
            c = "f" if f == "float" else ""
            return f"(({t}){_F2I[(f, 'i32')]}(sj_fmin(({f}){hi}.0{c}, sj_fmax(({f}){lo}.0{c}, {x}))))"
        return f"{_F2I[(f, t)]}({x})"
    return f"(({t})({x}))"


def b_convert(em, a, ts, t, p, vals):
    return conv(a[0], ts[0], t)


def b_bitcast(em, a, ts, t, p, vals):
    return f"sj_bitcast<{t},{ts[0]}>({a[0]})"


def b_reduce_precision(em, a, ts, t, p, vals):
    eb, mb = p["exponent_bits"], p["mantissa_bits"]
    if (t, eb, mb) in (("float", 8, 23), ("double", 11, 52)):
        return a[0]
    raise CodegenError(f"reduce_precision to e{eb}m{mb} is not supported")


EW = {
    "add": b_add, "add_any": b_add,       # (add_any of float0 / bool only: arithmetic add_any is canonicalised)
    "sub": b_sub, "mul": b_mul, "div": b_div, "rem": b_rem,
    "max": b_max, "min": b_min, "neg": b_neg, "abs": b_abs, "sign": b_sign,
    "floor": _f1("floorf"), "ceil": _f1("ceilf"), "round": b_round, "is_finite": b_isfinite,
    "exp": _f1("expf"), "exp2": b_exp2, "log": _f1("logf"), "log1p": _f1("log1pf"),
    "expm1": _f1("expm1f"), "sqrt": b_sqrt, "rsqrt": b_rsqrt, "cbrt": _f1("cbrtf"),
    "sin": _f1("sinf"), "cos": _f1("cosf"), "tan": _f1("tanf"), "asin": b_asin,
    "acos": _f1("acosf"), "atan": b_atan, "sinh": _f1("sinhf"), "cosh": _f1("coshf"),
    "tanh": b_tanh, "asinh": _f1("asinhf"), "acosh": _f1("acoshf"), "atanh": _f1("atanhf"),
    "atan2": _f2("atan2f"), "logistic": b_logistic, "pow": b_pow, "nextafter": b_nextafter,
    "erf": b_erf, "erfc": _f1("erfcf"), "erf_inv": b_erfinv, "lgamma": _f1("lgammaf"),
    "and": b_and, "or": b_or, "xor": b_xor, "not": b_not,
    "eq": _cmp("=="), "ne": _cmp("!="), "lt": _cmp("<"), "le": _cmp("<="), "gt": _cmp(">"),
    "ge": _cmp(">="), "log2": b_log2,
    "real": b_real, "imag": b_imag, "complex": b_complex, "conj": b_conj, "mulhi": b_mulhi, "shift_left": b_shl,
    "shift_right_logical": b_shrl,
    "shift_right_arithmetic": b_shra, "population_count": b_popc, "clz": b_clz,
    "select_n": b_select_n, "convert_element_type": b_convert,
    "bitcast_convert_type": b_bitcast, "reduce_precision": b_reduce_precision,
}

_FOLD = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "neg": np.negative,
         "max": np.maximum, "min": np.minimum, "abs": np.abs}


_LAZY_EW = ("add", "sub", "mul", "neg", "max", "min", "convert_element_type")


def _ew_rule(em, eqn, ins, **p):
    name = eqn.primitive.name
    av = _aval(eqn)
    shape, dt, impl = phys(av)
    ct = ctype(dt)
    ts = [ctype(v.dtype) for v in ins]
    b = EW[name]
    # compile-time folding of exact scalar ops (what XLA's constant folder does on the host)
    if shape == () and all(v.cval is not None for v in ins):
        if name in _FOLD or name == "convert_element_type" or (name == "div" and _isf(dt)):
            with np.errstate(all="ignore"):
                if name == "convert_element_type":
                    r = np.asarray(ins[0].cval).astype(dt)
                elif name == "div":
                    r = np.asarray(np.divide(ins[0].cval, ins[1].cval)).astype(dt)
                else:
                    r = np.asarray(_FOLD[name](*[np.asarray(v.cval) for v in ins])).astype(dt)
            if not (_isint(dt) and name in ("add", "sub", "mul", "neg", "abs")):
                return Scalar(lit(r[()], dt), dt, cval=r[()])
    nd = len(shape)

    def fn(idx):
        args = []
        for v in ins:
            if v.shape == ():
                args.append(elem(v, []))
            elif len(v.shape) == nd:     # rank-equal, with implicit broadcasting of size-1 dims
                args.append(elem(v, [0 if (sv == 1 and so != 1) else i for i, sv, so in zip(idx, v.shape, shape)]))
            else:  # scalar predicate over a key array (physical trailing dims)
                args.append(elem(v, idx[:len(v.shape)]))
        return b(em, args, ts, ct, p, ins)
    ou = getattr(em, "out_uses", None) or [0]
    if (LAZY_GATHER and impl is None and _size(shape) > LAZY_GATHER
            and all(isinstance(v, (Lazy, Table, Scalar, Local, View)) for v in ins)
            and ((name in _LAZY_EW and any(isinstance(v, (Lazy, View, Table)) and v.size > LAZY_GATHER for v in ins))
                 or ou[0] == 1)):
        # (a) one-use large elementwise results are FUSED into their single consumer (XLA loop fusion: a
        #     128-point per-step quadrature chain is evaluated inside the reduction instead of through ~20
        #     128-element per-thread local arrays); (b) cheap ops on a large lazy table gather stay lazy.
        # elementwise op on a large lazy table gather (e.g. row interpolation a*row_i + b*row_j of which the step
        # reads a few entries): stays lazy too, so only the consumed elements are computed (same expression per
        # element => bit-identical)
        cost = 1 + sum(lazy_cost(v) for v in ins)
        if cost <= LAZY_MAX_COST:
            out = Lazy(shape, dt, fn, cost=cost)
            out.origin = (name, ins)
            return out
    out = em.computed(shape, dt, fn)
    out.origin = (name, ins)
    if impl is not None:
        out = as_key(out, impl)
    return out


for _n in EW:
    RULES[_n] = _ew_rule


_UINT = {1: "u8", 2: "u16", 4: "u32", 8: "u64"}


@rule("bitcast_convert_type")
def _bitcast(em, eqn, ins, new_dtype, **p):
    """lax.bitcast_convert_type.  Same width: elementwise.  Different widths change the shape (little-endian, as XLA):
    to a NARROWER type out[..., j] = chunk j of in[...]; to a WIDER type out[...] = the chunks in[..., 0..k-1]."""
    x = ins[0]
    src, dst = np.dtype(x.dtype), np.dtype(new_dtype)
    if src.itemsize == dst.itemsize or is_key_dtype(getattr(_aval(eqn), "dtype", None)):
        return _ew_rule(em, eqn, ins, new_dtype=new_dtype, **p)
    av = _aval(eqn)
    ts, td = ctype(src), ctype(dst)
    us, ud = _UINT[src.itemsize], _UINT[dst.itemsize]
    if src.itemsize > dst.itemsize:                     # [...] -> [..., k]
        bits = dst.itemsize * 8

        def fn(idx):
            w = f"sj_bitcast<{us},{ts}>({elem(x, list(idx[:-1]))})"
            return f"sj_bitcast<{td},{ud}>(({ud})({w} >> (({us})({idx[-1]}) * {bits})))"
    else:                                               # [..., k] -> [...]
        k, bits = dst.itemsize // src.itemsize, src.itemsize * 8

        def fn(idx):
            parts = [f"((({ud})sj_bitcast<{us},{ts}>({elem(x, list(idx) + [j])})) << {j * bits})" for j in range(k)]
            return f"sj_bitcast<{td},{ud}>(({ud})({' | '.join(parts)}))"
    return em.computed(tuple(av.shape), dst, fn)


# =============================================================================== structure
@rule("broadcast_in_dim")
def _bcast(em, eqn, ins, shape, broadcast_dimensions, **p):
    src = ins[0]
    shape = tuple(shape)
    bd = tuple(broadcast_dimensions)
    av = _aval(eqn)
    pshape, _, impl = phys(av)
    extra = len(pshape) - len(shape)          # trailing key dims
    sshape = src.shape
    nsl = len(sshape) - extra

    def fn(idx):
        out = []
        for i in range(nsl):
            out.append(0 if sshape[i] == 1 and shape[bd[i]] != 1 else idx[bd[i]])
        return out + list(idx[len(shape):])
    if src.shape == () and src.cval is not None:
        return Lazy(pshape, src.dtype, lambda idx: src.text, cval=src.cval)
    return View(src, pshape, fn)


def _alias(src, shape):
    if isinstance(src, Local) and shape == () and src.shape != ():     # 1-element array -> scalar
        out = View(src, (), lambda idx, r=len(src.shape): [0] * r)
        out.key_impl = src.key_impl
        return out
    if isinstance(src, Local) and src.shape == () and shape != ():     # scalar variable -> [1]-shaped view
        out = View(src, shape, lambda idx: [])
        out.key_impl = src.key_impl
        return out
    if isinstance(src, Local):
        out = Local(src.name, shape, src.dtype)
        out.key_impl = src.key_impl
        return out
    if isinstance(src, Table) and src.strides == Table(src.ptr, src.shape, src.dtype).strides:
        return Table(src.ptr, shape, src.dtype, src.off)
    if isinstance(src, Scalar) and _size(shape) == 1:
        return View(src, shape, lambda idx: [])
    sshape = src.shape
    return View(src, shape, lambda idx: unravel(ravel(idx, shape), sshape))


@rule("reshape")
def _reshape(em, eqn, ins, new_sizes, dimensions=None, **p):
    if dimensions is not None and tuple(dimensions) != tuple(range(len(ins[0].shape))):
        raise CodegenError("reshape with dimensions (implicit transpose) is not supported")
    pshape, _, _ = phys(_aval(eqn))
    return _alias(ins[0], pshape)


@rule("transpose")
def _transpose(em, eqn, ins, permutation):
    src = ins[0]
    perm = tuple(permutation)
    pshape = phys(_aval(eqn))[0]

    def fn(idx):
        out = [None] * len(src.shape)
        for i, pp in enumerate(perm):
            out[pp] = idx[i]
        for i in range(len(perm), len(src.shape)):
            out[i] = idx[i]
        return out
    return View(src, pshape, fn)


@rule("rev")
def _rev(em, eqn, ins, dimensions):
    src = ins[0]
    dims = set(dimensions)
    sh = src.shape

    def fn(idx):
        return [ia(sh[d] - 1, f"-({i})") if d in dims and not isinstance(i, int) else
                (sh[d] - 1 - i if d in dims else i) for d, i in enumerate(idx)]
    return View(src, sh, fn)


@rule("slice")
def _slice(em, eqn, ins, start_indices, limit_indices, strides=None):
    src = ins[0]
    pshape = phys(_aval(eqn))[0]
    st = list(start_indices) + [0] * (len(src.shape) - len(start_indices))
    sd = list(strides or [1] * len(start_indices)) + [1] * (len(src.shape) - len(start_indices))
    if isinstance(src, Table):
        off = ia(src.off, *[s * k for s, k in zip(st, src.strides)])
        return Table(src.ptr, pshape, src.dtype, off, [k * d for k, d in zip(src.strides, sd)])
    return View(src, pshape, lambda idx: [ia(s, im(i, d)) for s, i, d in zip(st, idx, sd)])


def _clamp_start(em, v, hi):
    """int C variable = clamp(start, 0, hi) with jax/XLA dynamic_slice semantics."""
    if v.cval is not None:
        return int(min(max(int(v.cval), 0), hi))
    n = em.fresh("st")
    t = ctype(v.dtype)
    if t.startswith("u"):
        em.line(f"const int {n} = (int)((u64)({elem(v, [])}) < (u64){hi} ? (u64)({elem(v, [])}) : (u64){hi});")
    else:
        em.line(f"const int {n} = (int)sj_clampi((i64)({elem(v, [])}), {hi});")
    return n


@rule("dynamic_slice")
def _dynamic_slice(em, eqn, ins, slice_sizes):
    src, starts = ins[0], ins[1:]
    pshape = phys(_aval(eqn))[0]
    ss = list(slice_sizes) + list(src.shape[len(slice_sizes):])
    st = [_clamp_start(em, s, src.shape[d] - ss[d]) for d, s in enumerate(starts)]
    st += [0] * (len(src.shape) - len(st))
    if isinstance(src, Table):
        off = ia(src.off, *[im(s, k) for s, k in zip(st, src.strides)])
        return Table(src.ptr, pshape, src.dtype, off, src.strides)
    return View(src, pshape, lambda idx: [ia(s, i) for s, i in zip(st, idx)])


def _in_place(em, op, readers):
    """may a dynamic_update_slice / scatter write straight into its operand's storage (no copy)?  Only when the
    operand is dead after this equation (Emitter.can_mutate) and no other input of the equation -- the update,
    the indices -- reads that storage: the update is written element by element, so such an input would see
    elements this same equation already overwrote (x = dus(x, x[k-1:k+2], k))."""
    return em.can_mutate(em.cur_invars[0], op) and not any(_refs(em, r, op.name) for r in readers)


@rule("dynamic_update_slice")
def _dus(em, eqn, ins, **p):
    src, upd, starts = ins[0], ins[1], ins[2:]
    st = [_clamp_start(em, s, src.shape[d] - upd.shape[d]) for d, s in enumerate(starts)]
    st += [0] * (len(src.shape) - len(st))
    out = src if _in_place(em, src, [upd] + list(starts)) else em.mat_copy(src)

    def body(idx):
        em.line(f"{out.name}[{ravel([ia(s, i) for s, i in zip(st, idx)], src.shape)}] = {elem(upd, idx)};")
    em.each(upd.shape, body)
    out.key_impl = src.key_impl
    out.undef = False                  # (written: no longer an undefined value, codegen.Emitter.assign)
    return out


def _static_pick(idx_d, bounds):
    for j, (lo, hi) in enumerate(bounds):
        if lo <= idx_d < hi:
            return j
    raise IndexError


@rule("concatenate")
def _concat(em, eqn, ins, dimension):
    pshape = phys(_aval(eqn))[0]
    dim = dimension
    bounds, o = [], 0
    for v in ins:
        bounds.append((o, o + v.shape[dim]))
        o += v.shape[dim]

    def fn(idx):
        i = idx[dim]
        if isinstance(i, int):
            j = _static_pick(i, bounds)
            ii = list(idx)
            ii[dim] = i - bounds[j][0]
            return elem(ins[j], ii)
        e = None
        for j in range(len(ins) - 1, -1, -1):
            ii = list(idx)
            ii[dim] = ia(i, -bounds[j][0])
            x = elem(ins[j], ii)
            e = x if e is None else f"(({i}) < {bounds[j][1]} ? ({x}) : {e})"
        return e
    cost = 1 + sum(lazy_cost(v) for v in ins)
    out = Lazy(pshape, ins[0].dtype, fn, cost=cost) if cost <= LAZY_MAX_COST else em.computed(pshape, ins[0].dtype, fn)
    out.key_impl = ins[0].key_impl
    return out


@rule("pad")
def _pad(em, eqn, ins, padding_config):
    src, pv = ins
    av = _aval(eqn)
    sh = src.shape
    pval = elem(pv, [])

    def fn(idx):
        conds, sidx = [], []
        for d, (i, (lo, hi, it)) in enumerate(zip(idx, padding_config)):
            if isinstance(i, int):
                j = i - lo
                if j < 0 or j % (it + 1) or j // (it + 1) >= sh[d]:
                    return pval
                sidx.append(j // (it + 1))
            else:
                j = f"(({i}) - {lo})"
                if it:
                    conds.append(f"({j} >= 0 && {j} % {it + 1} == 0 && {j} / {it + 1} < {sh[d]})")
                    sidx.append(f"({j} / {it + 1})")
                else:
                    conds.append(f"({j} >= 0 && {j} < {sh[d]})")
                    sidx.append(j)
        x = elem(src, sidx)
        return f"(({' && '.join(conds)}) ? ({x}) : ({pval}))" if conds else x
    cost = 1 + lazy_cost(src)
    return Lazy(av.shape, av.dtype, fn, cost=cost) if cost <= LAZY_MAX_COST else em.computed(av.shape, av.dtype, fn)


@rule("iota")
def _iota(em, eqn, ins, dtype, shape, dimension, **p):
    dt = np.dtype(dtype)
    t = ctype(dt)
    return Lazy(tuple(shape), dt, lambda idx: f"(({t})({idx[dimension]}))")


@rule("tallymap_uninit")
def _uninit(em, eqn, ins, **p):
    """tallymap's own uninitialised per-item buffer (_reverse.uninit_p: reverse-mode checkpoint buffers, every slot
    written before it is read): a declaration with no fill."""
    shape, dt, impl = phys(_aval(eqn))
    v = em.declare(shape, dt, "u")
    v.key_impl = impl
    v.undef = True                     # copies of it are free (Emitter.assign) until it is written
    return v


@rule("empty", "empty2")
def _empty(em, eqn, ins, **p):
    """lax.empty: uninitialized in XLA; zeros here (any value is a valid realization)."""
    shape, dt, impl = phys(_aval(eqn))
    return Lazy(shape, dt, lambda idx: lit(0, dt))


# =============================================================================== reductions
_RED = {"reduce_sum": "add", "reduce_prod": "mul", "reduce_max": "max", "reduce_min": "min",
        "reduce_and": "and", "reduce_or": "or", "reduce_xor": "xor"}
_RED_INIT = {"add": 0, "mul": 1}


def _red_init(op, dt):
    dt = np.dtype(dt)
    if op in _RED_INIT:
        return lit(_RED_INIT[op], dt)
    if op in ("and",):
        if dt == np.bool_ or dt.kind == "i":
            return lit(True if dt == np.bool_ else -1, dt)
        return lit(np.iinfo(dt).max, dt)
    if op in ("or", "xor"):
        return lit(0, dt)
    if op == "max":
        if dt.kind == "f":
            return lit(-np.inf, dt)
        if dt == np.bool_:
            return "false"
        return lit(np.iinfo(dt).min, dt)
    if op == "min":
        if dt.kind == "f":
            return lit(np.inf, dt)
        if dt == np.bool_:
            return "true"
        return lit(np.iinfo(dt).max, dt)
    raise CodegenError(op)


def _bin(em, op, a, b, t):
    return EW[op](em, [a, b], [t, t], t, {}, [Scalar(a, None), Scalar(b, None)])


@rule(*_RED)
def _reduce(em, eqn, ins, axes, **p):
    x = ins[0]
    op = _RED[eqn.primitive.name]
    av = _aval(eqn)
    dt = np.dtype(av.dtype)
    t = ctype(dt)
    axes = tuple(sorted(a % len(x.shape) for a in axes))
    keep = [d for d in range(len(x.shape)) if d not in axes]
    rshape = tuple(x.shape[d] for d in axes)
    out = em.declare(tuple(av.shape), dt, "red")

    def body(oidx):
        acc = em.fresh("acc")
        first = [True]
        em.line(f"{t} {acc} = {_red_init(op, dt)};")

        def inner(ridx):
            full = [None] * len(x.shape)
            for j, d in enumerate(keep):
                full[d] = oidx[j]
            for j, d in enumerate(axes):
                full[d] = ridx[j]
            if first[0] and all(isinstance(r, int) and r == 0 for r in ridx):
                first[0] = False
                em.line(f"{acc} = {elem(x, full)};")
                return
            em.line(f"{acc} = {_bin(em, op, acc, elem(x, full), t)};")
        em.each(rshape, inner)
        em.line(f"{out.name}{'' if not av.shape else '[' + str(ravel(oidx, av.shape)) + ']'} = {acc};")
    em.each(tuple(av.shape), body)
    return _local_to_val(out)


def _local_to_val(out):
    if out.shape == ():
        return Scalar(out.name, out.dtype)
    return out


@rule("argmax", "argmin")
def _argminmax(em, eqn, ins, axes, index_dtype):
    x = ins[0]
    av = _aval(eqn)
    ax = axes[0] % len(x.shape)
    keep = [d for d in range(len(x.shape)) if d != ax]
    it = ctype(index_dtype)
    xt = ctype(x.dtype)
    cmp = ">" if eqn.primitive.name == "argmax" else "<"
    out = em.declare(tuple(av.shape), index_dtype, "arg")

    def body(oidx):
        best, bi = em.fresh("best"), em.fresh("bi")

        def at(i):
            full = [None] * len(x.shape)
            for j, d in enumerate(keep):
                full[d] = oidx[j]
            full[ax] = i
            return elem(x, full)
        em.line(f"{xt} {best} = {at(0)}; {it} {bi} = 0;")
        n = x.shape[ax]
        i = em.fresh("i")
        em.open(f"for (int {i} = 1; {i} < {n}; ++{i})")
        v = em.fresh("x")
        em.line(f"const {xt} {v} = {at(i)};")
        if _isf(x.dtype):
            em.line(f"if (({v} {cmp} {best}) || ({v} != {v} && {best} == {best})) "
                    f"{{ {best} = {v}; {bi} = ({it}){i}; }}")
        else:
            em.line(f"if ({v} {cmp} {best}) {{ {best} = {v}; {bi} = ({it}){i}; }}")
        em.close()
        em.line(f"{out.name}{'' if not av.shape else '[' + str(ravel(oidx, av.shape)) + ']'} = {bi};")
    em.each(tuple(av.shape), body)
    return _local_to_val(out)


_CUM = {"cumsum": "add", "cumprod": "mul", "cummax": "max", "cummin": "min"}


@rule(*_CUM)
def _cum(em, eqn, ins, axis, reverse):
    x = ins[0]
    op = _CUM[eqn.primitive.name]
    av = _aval(eqn)
    t = ctype(av.dtype)
    out = em.declare(tuple(av.shape), av.dtype, "cum")
    ax = axis % len(x.shape)
    n = x.shape[ax]
    oshape = tuple(s for d, s in enumerate(x.shape) if d != ax)

    def body(oidx):
        acc = em.fresh("acc")
        i = em.fresh("i")

        def full(j):
            f = list(oidx)
            f.insert(ax, j)
            return f
        j0 = n - 1 if reverse else 0
        em.line(f"{t} {acc} = {elem(x, full(j0))};")
        em.line(f"{out.name}[{ravel(full(j0), x.shape)}] = {acc};")
        em.open(f"for (int {i} = 1; {i} < {n}; ++{i})")
        j = f"({n - 1} - {i})" if reverse else i
        em.line(f"{acc} = {_bin(em, op, acc, elem(x, full(j)), t)};")
        em.line(f"{out.name}[{ravel(full(j), x.shape)}] = {acc};")
        em.close()
    em.each(oshape, body)
    return out


@rule("dot_general")
def _dot(em, eqn, ins, dimension_numbers, precision=None, preferred_element_type=None, **p):
    lhs, rhs = ins
    (lc, rc), (lb, rb) = dimension_numbers
    av = _aval(eqn)
    dt = np.dtype(av.dtype)
    t = ctype(dt)
    lf = [d for d in range(len(lhs.shape)) if d not in lc and d not in lb]
    rf = [d for d in range(len(rhs.shape)) if d not in rc and d not in rb]
    cshape = tuple(lhs.shape[d] for d in lc)
    out = em.declare(tuple(av.shape), dt, "dot")
    lt, rt = ctype(lhs.dtype), ctype(rhs.dtype)
    nb = len(lb)

    def body(oidx):
        acc = em.fresh("acc")
        em.line(f"{t} {acc} = {lit(0, dt)};")
        bidx, lidx, ridx = oidx[:nb], oidx[nb:nb + len(lf)], oidx[nb + len(lf):]

        def inner(cidx):
            li = [None] * len(lhs.shape)
            ri = [None] * len(rhs.shape)
            for j, d in enumerate(lb):
                li[d] = bidx[j]
            for j, d in enumerate(rb):
                ri[d] = bidx[j]
            for j, d in enumerate(lf):
                li[d] = lidx[j]
            for j, d in enumerate(rf):
                ri[d] = ridx[j]
            for j, (dl, dr) in enumerate(zip(lc, rc)):
                li[dl] = cidx[j]
                ri[dr] = cidx[j]
            a = conv(elem(lhs, li), lt, t)
            b = conv(elem(rhs, ri), rt, t)
            em.line(f"{acc} = {_bin(em, 'add', acc, _bin(em, 'mul', a, b, t), t)};")
        em.each(cshape, inner)
        em.line(f"{out.name}{'' if not av.shape else '[' + str(ravel(oidx, av.shape)) + ']'} = {acc};")
    em.each(tuple(av.shape), body)
    return _local_to_val(out)


@rule("sort")
def _sort(em, eqn, ins, dimension, is_stable=True, num_keys=1):
    """Stable insertion sort (small per-item arrays) with lax.sort's comparator: floats compare in total order
    after canonicalising -0 -> +0 and NaN -> +NaN (NaNs last); lexicographic over the first num_keys operands."""
    shape = ins[0].shape
    nd = len(shape)
    dim = dimension % nd
    n = shape[dim]
    arrs = [em.mat_copy(v, "srt") for v in ins]
    outer = tuple(sz for d, sz in enumerate(shape) if d != dim)

    def tkey(x, dt):
        t = ctype(dt)
        if t == "float":
            return f"sj_tkey_f(sj_canon_f({x}))"
        if t == "double":
            return f"sj_tkey_d(sj_canon_d({x}))"
        return f"({x})"

    def body(oidx):
        def at(i):
            full = list(oidx)
            full.insert(dim, i)
            return ravel(full, shape)
        i, j = em.fresh("i"), em.fresh("j")
        em.open(f"for (int {i} = 1; {i} < {n}; ++{i})")
        ks = []
        for q, a in enumerate(arrs):
            kq = em.fresh("k")
            em.line(f"const {ctype(a.dtype)} {kq} = {a.name}[{at(i)}];")
            ks.append(kq)
        em.line(f"int {j} = {i} - 1;")
        less = "false"
        for q in range(num_keys - 1, -1, -1):
            dt = arrs[q].dtype
            kq, aq = tkey(ks[q], dt), tkey(f"{arrs[q].name}[{at(j)}]", dt)
            less = f"(({kq}) < ({aq}) || (!(({aq}) < ({kq})) && {less}))"
        em.open(f"while ({j} >= 0 && {less})")
        for a in arrs:
            em.line(f"{a.name}[{at(ia(j, 1))}] = {a.name}[{at(j)}];")
        em.line(f"--{j};")
        em.close()
        for a, kq in zip(arrs, ks):
            em.line(f"{a.name}[{at(ia(j, 1))}] = {kq};")
        em.close()
    em.each(outer, body)
    return arrs


# =============================================================================== gather / scatter
def _mode(mode):
    s = str(mode).upper()
    if "FILL" in s or "DROP" in s:
        return "fill"
    if "CLIP" in s:
        return "clip"
    if "PROMISE" in s:
        return "promise"
    if "ONE_HOT" in s:
        return "one_hot"
    if mode is None:
        return "promise"
    raise CodegenError(f"gather/scatter mode {mode}")


def _fill_default(dt):
    dt = np.dtype(dt)
    if dt.kind == "f":
        return np.nan
    if dt == np.bool_:
        return True
    if dt.kind == "i":
        return np.iinfo(dt).min
    return np.iinfo(dt).max


LAZY_GATHER = 16   # 0 = always materialise gathers


@rule("gather")
def _gather(em, eqn, ins, dimension_numbers, slice_sizes, mode=None, fill_value=None, **p):
    op, ind = ins
    dn = dimension_numbers
    obd = tuple(getattr(dn, "operand_batching_dims", ()))
    sibd = tuple(getattr(dn, "start_indices_batching_dims", ()))
    av = _aval(eqn)
    oshape, odt, impl = phys(av)
    if impl is not None:
        raise CodegenError("gather of typed PRNG key arrays is not supported")
    offset_dims = tuple(dn.offset_dims)
    collapsed = set(dn.collapsed_slice_dims)
    sim = tuple(dn.start_index_map)
    m = _mode(mode)
    if m == "one_hot":
        raise CodegenError("gather mode ONE_HOT is not supported")
    batch_pos = [d for d in range(len(oshape)) if d not in offset_dims]
    noncol = [d for d in range(len(op.shape)) if d not in collapsed and d not in obd]
    fv = lit(_fill_default(odt) if fill_value is None else fill_value, odt)
    nidx = ind.shape[-1] if ind.shape else 1
    ind_scalar_vec = len(ind.shape) == 0

    # hoist the clamped start indices when there is no batch dimension (the common case)
    pre = None
    if not batch_pos and not obd:
        pre = []
        for k in range(nidx):
            d = sim[k]
            raw = ind if ind_scalar_vec else View(ind, (), lambda idx, k=k: [k])
            hi = op.shape[d] - slice_sizes[d]
            s = _clamp_start(em, raw, hi)
            okv = None
            if m == "fill":
                okv = em.fresh("ok")
                r = elem(raw, [])
                em.line(f"const bool {okv} = sj_inb((i64)({r}), {hi});")
            pre.append((d, s, okv))

    def fn(idx):
        bidx = [idx[pp] for pp in batch_pos]
        offs = [idx[pp] for pp in offset_dims]
        src = [0] * len(op.shape)
        for j, d in enumerate(noncol):
            src[d] = offs[j]
        # vmapped gather: operand batching dim obd[i] is pinned to the index batch coordinate sibd[i]
        # (batch output dims enumerate the index dims except the trailing index-vector dim, in order)
        for od, sd in zip(obd, sibd):
            src[od] = bidx[sd]
        conds = []
        if pre is not None:
            for d, s, okv in pre:
                src[d] = ia(s, src[d])
                if okv:
                    conds.append(okv)
        else:
            for k in range(nidx):
                d = sim[k]
                raw = elem(ind, bidx + [k])
                hi = op.shape[d] - slice_sizes[d]
                src[d] = ia(f"(int)sj_clampi((i64)({raw}), {hi})", src[d])
                if m == "fill":
                    conds.append(f"sj_inb((i64)({raw}), {hi})")
        x = elem(op, src)
        return f"(({' && '.join(conds)}) ? ({x}) : ({fv}))" if conds else x
    if LAZY_GATHER and _size(oshape) > LAZY_GATHER and isinstance(op, (Table, Lazy, View)):
        # large gathers from read-only tables (e.g. a whole table ROW selected by material/energy bin, of which
        # the step then reads 2 elements) stay lazy: elements are loaded where consumed instead of being copied
        # into a per-thread local array (994-float local copies per step on a shower item).  Pure load placement:
        # values are bit-identical.
        cost = 1 + lazy_cost(op) + lazy_cost(ind)
        if cost <= LAZY_MAX_COST:
            return Lazy(oshape, odt, fn, cost=cost)
    return em.computed(oshape, odt, fn, "g")


_SCAT = {"scatter": None, "scatter-add": "add", "scatter_add": "add", "scatter-mul": "mul",
         "scatter_mul": "mul", "scatter-min": "min", "scatter_min": "min", "scatter-max": "max",
         "scatter_max": "max"}


@rule(*_SCAT)
def _scatter(em, eqn, ins, dimension_numbers, mode=None, **p):
    op, ind, upd = ins
    comb = _SCAT[eqn.primitive.name]
    dn = dimension_numbers
    obd = tuple(getattr(dn, "operand_batching_dims", ()))
    sibd = tuple(getattr(dn, "scatter_indices_batching_dims", ()))
    uwd = tuple(dn.update_window_dims)
    iwd = set(dn.inserted_window_dims) | set(obd)
    sdod = tuple(dn.scatter_dims_to_operand_dims)
    m = _mode(mode)
    # in place when the operand is dead (the dynamic_update_slice rule).  x.at[i].set(v) with a traced i is a
    # scatter, not a dynamic_update_slice; copying it every time cost DTRAX's g0102 item 17 copies of its
    # float64[34,23] step carry (106 KB of its 178 KB frame; the private engine line has done this since 6f5d0cf)
    out = op if _in_place(em, op, [ind, upd]) else em.mat_copy(op)
    t = ctype(op.dtype)
    usd = [d for d in range(len(upd.shape)) if d not in uwd]
    wdims = [d for d in range(len(op.shape)) if d not in iwd]
    wsize = {d: 1 for d in iwd}
    for j, d in enumerate(wdims):
        wsize[d] = upd.shape[uwd[j]]

    def body(idx):
        bidx = [idx[d] for d in usd]
        widx = [idx[d] for d in uwd]
        dst = [0] * len(op.shape)
        for j, d in enumerate(wdims):
            dst[d] = widx[j]
        for od, sd in zip(obd, sibd):      # vmapped scatter: batching dim pinned to the index batch coordinate
            dst[od] = bidx[sd]
        conds = []
        for k, d in enumerate(sdod):
            raw = elem(ind, bidx + [k]) if ind.shape else elem(ind, [])
            hi = op.shape[d] - wsize[d]
            if m == "clip":
                dst[d] = ia(f"(int)sj_clampi((i64)({raw}), {hi})", dst[d])
            else:
                dst[d] = ia(f"(int)({raw})", dst[d])
                conds.append(f"sj_inb((i64)({raw}), {hi})")
        tgt = f"{out.name}[{ravel(dst, op.shape)}]" if op.shape else out.name
        val = elem(upd, idx)
        rhs = val if comb is None else _bin(em, comb, tgt, val, t)
        st = f"{tgt} = {rhs};"
        em.line(f"if ({' && '.join(conds)}) {st}" if conds else st)
    em.each(upd.shape, body)
    return _local_to_val(out) if out.shape == () else out


# =============================================================================== control flow
@rule("pjit", "jit", "closed_call", "core_call", "custom_jvp_call", "custom_vjp_call",
      "checkpoint", "remat", "remat2", "custom_lin")
def _primitive(em, eqn, ins, **params):
    if eqn.primitive.name == "custom_lin":
        from ..._errors import CUSTOM_VJP
        raise CodegenError(f"custom_lin (a linearized custom_vjp) cannot run inside a cuda kernel.{CUSTOM_VJP}")
    sub = call_jaxpr(eqn)
    if sub is None:
        raise CodegenError(f"no sub-jaxpr in {sorted(params)}")
    return em.run_closed(sub, list(ins))


TALLY_AWARE = {"while", "cond", "scan", "pjit", "jit", "closed_call", "core_call", "custom_jvp_call",
               "custom_vjp_call", "checkpoint", "remat", "remat2"}


# ------------------------------------------------------------------ buffer assignment at loop / cond boundaries
# (wavefront-state 82d6ee2 "v3" + fbf2c5b "v5", one fixed rule; DTRAX on an L40S runs 1.59x slower without both.)
# v3: a while loop's initial carry array that is dead after the loop (the in-place liveness rule, Emitter.can_mutate)
#     and that no other operand of the loop reads IS the carry storage (no init copy); a cond output array is written
#     into the storage of a dead operand of the same shape and dtype (destination passing: no fresh array, and a
#     branch that passes the operand through or updates it in place copies nothing).
# v5: the same when the value is LAZY and reads that storage only at its own index (Emitter.elementwise_self, e.g. a
#     whole-array jnp.where(p, a, x)): it is evaluated into the storage, element by element; and a loop body result of
#     that kind is written straight back into its carry.  Placement only: every element's expression is unchanged.
def _refs(em, x, name):
    """does the C text value x reads mention storage `name`?  (Local: by name; anything else: rendered)"""
    import re
    if isinstance(x, VTally) or x is None:
        return False
    if isinstance(x, Local):
        return x.name == name
    saved = (em.lines, em.ind, em.uid)
    try:
        em.lines = []
        t = elem(x, [f"_q{k}" for k in range(len(x.shape))])
    except Exception:
        return True
    finally:
        em.lines, em.ind, em.uid = saved
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", t) is not None


def _src_locals(x, out=None, d=0):
    """the per-item Local arrays a lazy value's elements are built from (through View.src / Lazy.origin)."""
    out = {} if out is None else out
    if x is None or d > 12:
        return out
    if isinstance(x, Local):
        out[x.name] = x
    elif isinstance(x, View):
        _src_locals(x.src, out, d + 1)
    elif isinstance(x, Lazy):
        for v in (getattr(x, "origin", None) or ("", ()))[1]:
            _src_locals(v, out, d + 1)
    return out


def _lazy_dest(em, v, others, used):
    """(v5) a dead Local the lazy initial carry v reads only at its own index, and nothing else live or another
    operand of the loop (`others`: every operand but v's own slot) reads: v is evaluated INTO it.  -> that Local or
    None."""
    for X in _src_locals(v).values():
        if X.name in used or tuple(X.shape) != tuple(v.shape) or X.dtype != v.dtype:
            continue
        if not em.elementwise_self(X, v) or not em.can_mutate(None, X):
            continue
        if any(_refs(em, o, X.name) for o in others):
            continue
        return X
    return None


def _init_alias(em, inits, ivars, pt, others):
    """(v3, v5) per initial carry of a while loop: None (copy it into a fresh carry), or the Local that IS the carry
    storage (the init itself, dead after the loop; or for a lazy init, the dead storage it is evaluated into).
    others = the loop's operands, inits last.  The conflict check excludes the init's own SLOT, not every operand
    that is the same value: one value can be an initial carry AND a loop constant or another carry (the single-loop
    adjoint restarts from its initial state), and the body must then read it unchanged."""
    dest, used = [], set()
    n0 = len(others) - len(inits)
    for k, (v, ivar, p_) in enumerate(zip(inits, ivars, pt)):
        X = None
        rest = others[:n0 + k] + others[n0 + k + 1:]
        if not p_ and not isinstance(v, VTally) and v.shape != ():
            if isinstance(v, Local):
                if (v.name not in used and em.can_mutate(ivar, v)
                        and not any(_refs(em, x, v.name) for x in rest)):
                    X = v
            elif not isinstance(v, Table):
                X = _lazy_dest(em, v, rest, used)
        if X is not None:
            used.add(X.name)
        dest.append(X)
    return dest


def _carry_vars(em, inits, pass_through, dest=None):
    cars = []
    for i, (v, pt) in enumerate(zip(inits, pass_through)):
        if pt or isinstance(v, VTally):          # loop invariant, or a virtual tally (no data to carry)
            cars.append(v)
            continue
        X = dest[i] if dest is not None else None
        if X is not None:                       # (v3 / v5) buffer assignment: this storage IS the carry
            if X is not v:
                em.assign(X, v)
                X.key_impl = v.key_impl
            X.undef = False                     # (the loop update writes it: inside the body it is defined)
            cars.append(X)
            continue
        if v.shape == ():
            n = em.fresh("c")
            em.line(f"{ctype(v.dtype)} {n} = {elem(v, [])};")
            c = Local(n, (), v.dtype)
        else:
            c = em.declare(v.shape, v.dtype, "c")
            em.assign(c, v)
        c.key_impl = v.key_impl
        cars.append(c)
    return cars


def _carry_val(c):
    if isinstance(c, Local) and c.shape == ():
        s = Scalar(c.name, c.dtype)
        s.key_impl = c.key_impl
        return s
    return c


def _update_carries(em, cars, outs, pass_through, start=None):
    """write the body results `outs` back into the loop carries.  start: the index in em.lines of the loop's
    opening line (the loop's own text runs from there): an array carry whose new value must be computed into a
    fresh array (it reads the old carry at other indices) then gets a second buffer and a pointer swap instead of a
    copy back (ping-pong) -> [(carry, pointer name)] for _pingpong_exit after the loop closes."""
    pass_through = [pt or isinstance(c, VTally) for c, pt in zip(cars, pass_through)]
    names = {c.name for c, pt in zip(cars, pass_through) if not pt}
    tmps = []
    pp = []
    for c, o, pt in zip(cars, outs, pass_through):
        if pt:
            tmps.append(None)
            continue
        if isinstance(o, Scalar) and o.text not in names:
            tmps.append(o)
        elif isinstance(o, Scalar):
            n = em.fresh("t")
            em.line(f"const {ctype(o.dtype)} {n} = {o.text};")
            tmps.append(Scalar(n, o.dtype))
        elif isinstance(o, Local) and o.name not in names:
            tmps.append(o)
        elif isinstance(o, Local) and o.name == c.name and o.shape == c.shape:
            tmps.append(None)
        elif (not isinstance(o, Local) and em.elementwise_self(c, o)
              and not any(nm != c.name and _refs(em, o, nm) for nm in names)):
            tmps.append(("self", o))             # (v5) written straight back into the carry, element by element
        elif start is not None and em.inplace and isinstance(c, Local) and c.size * c.dtype.itemsize > 4 * 255:
            # ping-pong: the new value is computed straight into the carry's second buffer and the two buffers
            # swap at the end of the iteration, instead of computing it into a fresh array and copying that back
            # (a Gaussian elimination in a fori loop over a 3x12x14 carry: one 504-float copy per pivot).  Only for
            # a carry larger than a thread's 255 registers, which is in local memory whatever the compiler does: a
            # smaller one may live in registers, where the copy is moves and a pointer would demote it to local
            # memory (a 12x14 elimination: 4x slower on an L40S through a pointer)
            q = Local(em.fresh("pq"), c.shape, c.dtype)
            em.assign(q, o)
            pp.append((c, q.name, em.fresh("pp")))
            tmps.append(None)
        else:
            tmps.append(em.mat_copy(o))
    for c, t, pt in zip(cars, tmps, pass_through):
        if pt or t is None:
            continue
        em.assign(c, t[1] if isinstance(t, tuple) else t)
    if not pp:
        return []
    import re
    ct = {c.name: ctype(c.dtype) for c, _, _ in pp}
    for c, q, p_ in pp:                          # inside the loop the carry is reached through its pointer
        em.line(f"{{ {ct[c.name]}* const sj_t = {p_}; {p_} = {q}; {q} = sj_t; }}")
    pat = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(c.name) for c, _, _ in pp) + r")(?![A-Za-z0-9_])")
    ptr = {c.name: p_ for c, _, p_ in pp}
    em.lines[start:] = [pat.sub(lambda m: ptr[m.group(1)], ln) for ln in em.lines[start:]]
    ind = em.lines[start][:len(em.lines[start]) - len(em.lines[start].lstrip())]
    decl = []
    for c, q, p_ in pp:
        b = em.fresh("pb")
        decl.append(f"{ind}{ct[c.name]} {b}[{c.size}];{fill_decl(b)} {ct[c.name]}* {p_} = {c.name}; "
                    f"{ct[c.name]}* {q} = {b};")
    em.lines[start:start] = decl
    return [(c, p_) for c, _, p_ in pp]


def _pingpong_exit(em, pp, swaps=None):
    """after a ping-pong loop: the last value is in the buffer the pointer holds; copy it into the carry array when
    that is the second buffer (swaps: the static trip count, else a run-time check)."""
    for c, p_ in pp:
        if swaps is not None and swaps % 2 == 0:
            continue
        em.open(f"if ({p_} != {c.name})" if swaps is None else "")
        em.fill(c, lambda idx, c=c, p_=p_: f"{p_}[{ravel(idx, c.shape)}]")
        em.close()


def _pass_through(jaxpr, nconsts, ncarry, n_out_offset=0):
    """carry i is loop invariant if the body returns its own input var."""
    ivs = jaxpr.invars[nconsts:nconsts + ncarry]
    ovs = jaxpr.outvars[n_out_offset:n_out_offset + ncarry]
    return [ov is iv for iv, ov in zip(ivs, ovs)]


@rule("while")
def _while(em, eqn, ins, **p):
    w = while_parts(eqn)
    cc = list(ins[:w.nc])
    bc = list(ins[w.nc:w.nc + w.nb])
    init = list(ins[w.nc + w.nb:])
    pt = _pass_through(w.body.jaxpr, w.nb, len(init))
    dest = _init_alias(em, init, list(em.cur_invars)[w.nc + w.nb:], pt, cc + bc + init)
    cars = _carry_vars(em, init, pt, dest)
    vals = [_carry_val(c) if not p_ else c for c, p_ in zip(cars, pt)]
    em.freeze(cc + bc + [c for c, p_ in zip(cars, pt) if p_])     # re-read every iteration: never in place
    start = len(em.lines)
    em.open("while (true)")
    em.cf_depth += 1
    em.freeze(vals)                                                 # the predicate must not consume a carry
    pred = em.run_closed(w.cond, cc + vals)[0]
    em.unfreeze()
    if pred.shape == ():
        em.line(f"if (!({elem(pred, [])})) break;")
        outs = em.run_closed(w.body, bc + vals)
    else:
        outs = _batched_body(em, w, pred, bc, cars, vals, pt)
    pp = _update_carries(em, cars, outs, pt, start)
    em.cf_depth -= 1
    em.close()
    em.unfreeze()
    _pingpong_exit(em, pp)
    return vals


def _batched_body(em, w, pred, bc, cars, vals, pt):
    """JAX's while contract for a BATCHED predicate (jax.vmap of a loop whose condition reads a batched value, e.g.
    dual lanes of a loop whose condition reads the tangent): the predicate has shape B, a prefix of every carry's
    shape; the loop runs while ANY of its B entries holds, and an entry whose predicate is false keeps its carries
    (per-entry select of the body's results) and its emits are dropped (every emit inside the body is masked by
    it: a tally carry is "selected" by not emitting).  -> the selected body results."""
    B = tuple(pred.shape)
    pm = em.declare(B, np.bool_, "pm")                  # the predicate of THIS iteration (the body reads carries)
    em.assign(pm, pred)
    anyv = em.fresh("pa")
    em.line(f"bool {anyv} = false;")
    em.each(B, lambda idx: em.line(f"{anyv} |= {elem(pm, idx)};"))
    em.line(f"if (!{anyv}) break;")
    for c, p_ in zip(cars, pt):
        if not p_ and not isinstance(c, VTally) and tuple(c.shape[:len(B)]) != B:
            raise CodegenError(f"a while loop with a predicate of shape {B} has a carry of shape {tuple(c.shape)} "
                               "(internal error: JAX batches every carry of such a loop)")
    em.freeze(vals)                                     # the select reads the OLD carries: never updated in place
    em.lane_masks.append((pm, B))
    try:
        outs = em.run_closed(w.body, bc + vals)
    finally:
        em.lane_masks.pop()
        em.unfreeze()
    res = []
    for o, c, p_ in zip(outs, vals, pt):
        if p_ or isinstance(c, VTally):
            res.append(o)
            continue
        fn = (lambda idx, o=o, c=c: f"({elem(pm, list(idx[:len(B)]))} ? ({elem(o, idx)}) : ({elem(c, idx)}))")
        sel = Lazy(c.shape, c.dtype, fn, cost=lazy_cost(o) + 1)
        sel.key_impl = c.key_impl
        res.append(sel)
    return res


def _branch(em, br, ops):
    """emit one branch into a separate line buffer -> (lines, results)."""
    saved, ind = em.lines, em.ind
    em.lines = []
    em.ind = ind + 1
    em.cf_depth += 1
    res = em.run_closed(br, ops)
    em.cf_depth -= 1
    lines = em.lines
    em.lines, em.ind = saved, ind
    return lines, res


@rule("cond")
def _cond(em, eqn, ins, **p):
    branches = cond_branches(eqn)
    idx, ops = ins[0], list(ins[1:])
    n = len(branches)
    if idx.cval is not None or n == 1:
        i = 0 if n == 1 else int(idx.cval)
        i = n - 1 if (i < 0 or i >= n) else i
        return em.run_closed(branches[i], ops)
    ie = elem(idx, [])
    iv = em.fresh("bi")
    em.line(f"const int {iv} = (int)({ie});")
    bodies = [_branch(em, br, ops) for br in branches]
    dest = _cond_dest(em, eqn, ops, bodies)
    outs = []
    for j, v in enumerate(eqn.outvars):
        if dest[j] is not None:
            outs.append(("dest", dest[j]))
            continue
        # virtual in ANY branch -> virtual (an inert discard tally may meet a filler value in another branch)
        r0 = next((b[1][j] for b in bodies if isinstance(b[1][j], VTally)), None)
        if r0 is not None:
            outs.append(r0)
            continue
        shape, dt, impl = phys(v.aval)
        o = em.declare(shape, dt, "co")
        o.key_impl = impl
        outs.append(o)
    for i, (lines, res) in enumerate(bodies):
        if i == 0:
            em.open(f"if ({iv} <= 0)")
        elif i < n - 1:
            em.close(f"}} else if ({iv} == {i}) {{")
            em.ind += 1
        else:
            em.close("} else {")
            em.ind += 1
        em.lines += lines
        for o, r in zip(outs, res):
            if not isinstance(o, (VTally, tuple)):
                em.assign(o, r)
        for o, r in zip(outs, res):            # destination outputs last: they overwrite an operand
            if isinstance(o, tuple) and not (isinstance(r, Local) and r.name == o[1].name):
                em.assign(o[1], r)
    em.close()
    return [o[1] if isinstance(o, tuple) else (o if isinstance(o, VTally) else _carry_val(o)) for o in outs]


def _cond_dest(em, eqn, ops, bodies):
    """(v3, v5) per cond output: the dead operand Local it is written into, or None (a fresh array).  Exactly one
    branch runs; in it the other outputs are assigned first, so only another DESTINATION reading this storage could
    see it overwritten, which the second loop excludes."""
    civ = list(eqn.invars[1:])
    n = len(eqn.outvars)
    dest, used = [None] * n, set()
    for j, v in enumerate(eqn.outvars):
        if any(isinstance(bd[1][j], VTally) for bd in bodies):
            continue
        shape, dt, _ = phys(v.aval)
        if shape == ():
            continue
        for k in ([j] if j < len(ops) else []) + [k for k in range(len(ops)) if k != j]:
            op = ops[k]
            if k in used or not isinstance(op, Local) or op.shape != tuple(shape) or op.dtype != dt:
                continue
            if not em.can_mutate(civ[k], op):
                continue
            # every branch result is this storage (passed through / updated in place: nothing to write), another
            # array, or a lazy value that reads this storage only at its own index (v5)
            if all((isinstance(r, Local) and r.name == op.name) or not _refs(em, r, op.name)
                   or em.elementwise_self(op, r) for r in (bd[1][j] for bd in bodies)):
                dest[j] = op
                used.add(k)
                break
    changed = True
    while changed:                     # a destination whose branch result reads ANOTHER written destination: drop it
        changed = False
        for j in range(n):
            if dest[j] is None:
                continue
            for bd in bodies:
                r = bd[1][j]
                if isinstance(r, Local) and r.name == dest[j].name:
                    continue
                if any(dest[m] is not None and m != j and not (isinstance(bd[1][m], Local) and
                                                               bd[1][m].name == dest[m].name)
                       and _refs(em, r, dest[m].name) for m in range(n)):
                    dest[j] = None
                    changed = True
                    break
    return dest


@rule("scan")
def _scan(em, eqn, ins, **p):
    sc = scan_parts(eqn)
    jaxpr, num_consts, num_carry, length, reverse = sc.body, sc.nconsts, sc.ncarry, sc.length, sc.reverse
    consts = list(ins[:num_consts])
    init = list(ins[num_consts:num_consts + num_carry])
    xs = list(ins[num_consts + num_carry:])
    pt = _pass_through(jaxpr.jaxpr, num_consts, num_carry)
    cars = _carry_vars(em, init, pt)
    vals = [_carry_val(c) if not p_ else c for c, p_ in zip(cars, pt)]
    # an inert discard tally may be a scanned input / stacked output (a reverse-mode residual): it stays virtual
    from ..._rewrite import tally_flags_out
    yflag = tally_flags_out(jaxpr.jaxpr, [isinstance(v, VTally) for v in consts + init + xs])[num_carry:]
    ys = []
    for v, f in zip(eqn.outvars[num_carry:], yflag):
        if f:
            ys.append(None)
            continue
        shape, dt, impl = phys(v.aval)
        y = em.declare(shape, dt, "ys")
        y.key_impl = impl
        ys.append(y)
    t = em.fresh("it")
    start = len(em.lines)
    em.open(f"for (int {t} = 0; {t} < {length}; ++{t})")
    ti = f"({length - 1} - {t})" if reverse else t
    xs_t = []
    for x in xs:
        if isinstance(x, VTally):
            xs_t.append(x)
        elif isinstance(x, Table):
            xs_t.append(Table(x.ptr, x.shape[1:], x.dtype, ia(x.off, im(ti, x.strides[0])), x.strides[1:]))
        else:
            v = View(x, x.shape[1:], lambda idx, ti=ti: [ti] + list(idx))
            xs_t.append(v)
    em.cf_depth += 1
    em.freeze(consts + xs + [c for c, p_ in zip(cars, pt) if p_])
    outs = em.run_closed(jaxpr, consts + vals + xs_t)
    em.unfreeze()
    em.cf_depth -= 1
    for j, (y, yv) in enumerate(zip(ys, outs[num_carry:])):
        if y is None or isinstance(yv, VTally):
            if not (y is None and isinstance(yv, VTally)):
                raise CodegenError("internal: a stacked scan output's tally status was mispredicted")
            ys[j] = yv
            continue
        em.each(yv.shape, lambda idx, y=y, yv=yv: em.line(
            f"{y.name}[{ravel([ti] + list(idx), y.shape)}] = {elem(yv, idx)};"))
    pp = _update_carries(em, cars, outs[:num_carry], pt, start)
    em.close()
    _pingpong_exit(em, pp, length)
    return vals + ys


@rule("platform_index")
def _platform_index(em, eqn, ins, platforms, **p):
    # the default branch is the None entry in `platforms` (JAX >= 0.6; 0.5.3 had has_default)
    plats = list(platforms)
    for i, ps in enumerate(plats):
        if ps is not None and any(x in ("cuda", "gpu", "rocm") for x in ps):
            return Scalar(lit(i, np.int32), np.int32, cval=np.int32(i))
    i = next((j for j, ps in enumerate(plats) if ps is None), None)
    if i is not None:
        return Scalar(lit(i, np.int32), np.int32, cval=np.int32(i))
    raise CodegenError("lax.platform_dependent without a cuda/gpu branch and without a default")


# =============================================================================== RNG
@rule("threefry2x32")
def _threefry(em, eqn, ins, **p):
    shape = tuple(_aval(eqn).shape)
    o1 = em.declare(shape, np.uint32, "tf")
    o2 = em.declare(shape, np.uint32, "tf")

    def body(idx):
        a = [elem(v, [0 if sv == 1 else i for i, sv in zip(idx, v.shape)] if v.shape else []) for v in ins]
        r1 = f"{o1.name}[{ravel(idx, shape)}]" if shape else o1.name
        r2 = f"{o2.name}[{ravel(idx, shape)}]" if shape else o2.name
        em.line(f"sj_threefry2x32({a[0]}, {a[1]}, {a[2]}, {a[3]}, {r1}, {r2});")
    em.each(shape, body)
    return [_local_to_val(o1), _local_to_val(o2)]


@rule("iota_2x32_shape")
def _iota2x32(em, eqn, ins, shape):
    shape = tuple(shape)
    hi = Lazy(shape, np.uint32,
              lambda idx: "0u" if _size(shape) < 2 ** 32 else f"(u32)((u64)({ravel(idx, shape)}) >> 32)")
    lo = Lazy(shape, np.uint32, lambda idx: f"((u32)({ravel(idx, shape)}))")
    return [hi, lo]


# The random_* primitives on typed keys are expanded through the key's PRNGImpl methods (impl.random_bits / split /
# fold_in / seed: the public jax.extend.random impls), vmapped over the key array exactly as JAX's own impl rules do
# (tallymap._compat).  The expansion is traced to plain primitives (threefry2x32, shifts, ...) that the table emits;
# tests/ops.py checks the RNG rows bitwise against XLA.
def _key_impl(v, aval):
    impl = v.key_impl
    if impl is None and is_key_dtype(aval.dtype):
        impl = key_impl(aval.dtype)
    if impl is None:
        raise CodegenError("random_* primitive on a value that is not a typed key")
    return impl


def _expand(em, f, vals):
    from .canon import canonical
    avs = [jax.ShapeDtypeStruct(v.shape, v.dtype) for v in vals]
    cj = canonical(jax.make_jaxpr(f)(*avs))
    return em.run_closed(cj, list(vals))


@rule("random_wrap")
def _rwrap(em, eqn, ins, impl):
    return as_key(ins[0], impl)


@rule("random_unwrap", "random_key_data")
def _runwrap(em, eqn, ins, **p):
    v = View(ins[0], ins[0].shape, lambda idx: idx)
    v.key_impl = None
    return v


@rule("random_clone")
def _rclone(em, eqn, ins, **p):
    return ins[0]


@rule("random_bits")
def _rbits(em, eqn, ins, bit_width, shape):
    impl = _key_impl(ins[0], eqn.invars[0].aval)
    kndim = len(eqn.invars[0].aval.shape)
    base = View(ins[0], ins[0].shape, lambda idx: idx)
    base.key_impl = None
    return _expand(em, lambda k: random_bits(impl, k, kndim, bit_width, shape), [base])[0]


@rule("random_split")
def _rsplit(em, eqn, ins, shape):
    impl = _key_impl(ins[0], eqn.invars[0].aval)
    kndim = len(eqn.invars[0].aval.shape)
    base = View(ins[0], ins[0].shape, lambda idx: idx)
    base.key_impl = None
    out = _expand(em, lambda k: random_split(impl, k, kndim, shape), [base])[0]
    return as_key(out, impl)


@rule("random_fold_in")
def _rfold(em, eqn, ins, **p):
    impl = _key_impl(ins[0], eqn.invars[0].aval)
    kshape = tuple(eqn.invars[0].aval.shape)
    base = View(ins[0], ins[0].shape, lambda idx: idx)
    base.key_impl = None
    out = _expand(em, lambda k, m: random_fold_in(impl, k, m, kshape), [base, ins[1]])[0]
    return as_key(out, impl)


@rule("random_seed")
def _rseed(em, eqn, ins, impl):
    out = _expand(em, lambda s: random_seed(impl, s), [ins[0]])[0]
    return as_key(out, impl)
