"""Sugar canonicalisation: a jaxpr -> jaxpr pre-pass of the cuda emitter.

Some JAX primitives are shorthand for a few others: square = mul(x, x), clamp = min(hi, max(lo, x)), integer_pow = a
chain of muls, expand_dims / squeeze = reshape, split / unstack = slices, stack = concatenate, the total-order
compares = integer compares of the sign-folded bit patterns, copy / stop_gradient / ... = identity, a hijax
primitive (JAX >= 0.12) = its expansion. Instead of one emission rule each, the item jaxpr is re-traced once with
every such equation replaced by its canonical form, so the op table only needs the canonical primitives.

No numerics change: each canonical form performs the SAME IEEE operations, in the same order and with the same
operands, as the rule it replaces (checked bitwise by tests/ops.py: square, clip, ipow3 and the rest keep their
recorded numbers).  A program without sugar is returned unchanged (same object).

The re-traced jaxpr keeps the original constants first, in order (the kernel's hoisted-constant inputs are indexed
by position), and nested jaxprs (while / cond / scan / call bodies) are canonicalised recursively.  One pass is
enough: no canonical form contains sugar.
"""
import jax
import numpy as np
from jax import lax
from jax.extend.core import ClosedJaxpr, Literal

from ..._jaxpr import bind, closed, contains, hi_jaxpr, is_dropvar, lit, subjaxprs, with_source


def _float_kind(dt):
    return np.dtype(dt) in (np.dtype(np.float32), np.dtype(np.float64))


def _arith(dt):
    return np.dtype(dt).kind in "fiuc"


def _c(v, dt):
    return np.array(v, dtype=dt)


def _integer_pow(x, *, y):
    if y == 0:
        return lax.full_like(x, 1)
    neg, y = y < 0, abs(y)
    acc = None
    while y > 0:                         # binary exponentiation (the order the former integer_pow rule used)
        if y & 1:
            acc = x if acc is None else lax.mul(acc, x)
        y >>= 1
        if y > 0:
            x = lax.mul(x, x)
    return lax.div(lax.full_like(acc, 1), acc) if neg else acc


def _tkey(x):
    """sign-folded bit pattern: a signed-integer key whose order is XLA's float total order (sj_tkey_f / _d)."""
    w, mask = (np.int32, 0x7FFFFFFF) if np.dtype(x.dtype) == np.float32 else (np.int64, 0x7FFFFFFFFFFFFFFF)
    i = lax.bitcast_convert_type(x, w)
    return lax.bitwise_xor(i, lax.bitwise_and(lax.shift_right_arithmetic(i, _c(np.iinfo(w).bits - 1, w)), _c(mask, w)))


def _total_order(cmp):
    def f(a, b):
        if _float_kind(a.dtype):
            a, b = _tkey(a), _tkey(b)
        return cmp(a, b)
    return f


def _bcast_to(v, shape):
    return v if v.shape == tuple(shape) else lax.broadcast_in_dim(v, tuple(shape), ())


def _clamp(lo, x, hi):
    return lax.min(_bcast_to(hi, x.shape), lax.max(_bcast_to(lo, x.shape), x))


def _split(x, *, sizes, axis):
    out, o = [], 0
    for s in sizes:
        out.append(lax.slice_in_dim(x, o, o + s, axis=axis))
        o += s
    return out


def _stack(*xs, axis):
    shape = list(xs[0].shape)
    shape.insert(axis, 1)
    dims = tuple(d for d in range(len(shape)) if d != axis)
    return lax.concatenate([lax.broadcast_in_dim(x, tuple(shape), dims) for x in xs], axis)


def _unstack(x, *, axis):
    shape = tuple(s for d, s in enumerate(x.shape) if d != axis)
    return [lax.reshape(lax.slice_in_dim(x, i, i + 1, axis=axis), shape) for i in range(x.shape[axis])]


def _identity(*xs, **_):
    return list(xs)


# name -> (applies(eqn) | None, canonical(*ins, **params) -> out | [outs])
SUGAR = {
    "square": (None, lambda x: lax.mul(x, x)),
    "one_minus_square": (None, lambda x: lax.mul(lax.add(_c(1, x.dtype), x), lax.sub(_c(1, x.dtype), x))),
    "integer_pow": (None, _integer_pow),
    "clamp": (None, _clamp),
    "add_any": (lambda e: _arith(e.outvars[0].aval.dtype), lambda a, b: lax.add(a, b)),
    "eq_to": (None, _total_order(lax.eq)),
    "le_to": (None, _total_order(lax.le)),
    "lt_to": (None, _total_order(lax.lt)),
    "expand_dims": (None, None),          # -> reshape to the output shape (below)
    "squeeze": (None, None),
    "split": (None, _split),
    "stack": (None, _stack),
    "unstack": (None, _unstack),
    "copy": (None, _identity), "copy_p": (None, _identity), "device_put": (None, _identity),
    "stop_gradient": (None, _identity), "optimization_barrier": (None, _identity),
    "call_hi_primitive": (None, lambda *ins, _prim: _eval(hi_jaxpr(_prim).jaxpr, hi_jaxpr(_prim).consts, ins)),
}
_RESHAPE = ("expand_dims", "squeeze")


def _applies(eqn):
    s = SUGAR.get(eqn.primitive.name)
    return s is not None and (s[0] is None or s[0](eqn))


def _expand(eqn, ins):
    name = eqn.primitive.name
    if name in _RESHAPE:
        return [lax.reshape(ins[0], tuple(eqn.outvars[0].aval.shape))]
    fn = SUGAR[name][1]
    if name in ("split", "unstack", "integer_pow", "stack", "call_hi_primitive"):
        out = fn(*ins, **eqn.params)
    else:
        out = fn(*ins)
    return list(out) if isinstance(out, (list, tuple)) else [out]


_NAMES = tuple(SUGAR)
_CACHE = {}


def _param(v):
    if isinstance(v, ClosedJaxpr):
        return canonical(v)
    if type(v).__name__ == "Jaxpr":
        c = canonical(closed(v))
        if c.consts:                                     # pragma: no cover (canonical forms create no constants)
            raise NotImplementedError("tallymap: canonicalising an open sub-jaxpr created constants")
        return c.jaxpr
    if isinstance(v, (tuple, list)) and any(isinstance(b, ClosedJaxpr) or type(b).__name__ == "Jaxpr" for b in v):
        out = [_param(b) for b in v]
        return type(v)(*out) if hasattr(v, "_fields") else type(v)(out)     # a namedtuple: custom_linear_solve
    return v


def _eval(jaxpr, consts, args):
    env = {}

    def read(v):
        return lit(v) if isinstance(v, Literal) else env[v]
    for v, x in zip(jaxpr.constvars, consts):
        env[v] = x
    for v, x in zip(jaxpr.invars, args):
        env[v] = x
    for e in jaxpr.eqns:
        ins = [read(v) for v in e.invars]
        if _applies(e):
            with with_source(e):
                outs = _expand(e, ins)
        else:
            if any(contains(s.jaxpr, _NAMES) for s in subjaxprs(e)):
                e = e.replace(params={k: _param(v) for k, v in e.params.items()})
            outs = bind(e, ins)
        for v, x in zip(e.outvars, outs):
            if not is_dropvar(v):
                env[v] = x
    return [read(v) for v in jaxpr.outvars]


def canonical(cj):
    """the canonical form of a ClosedJaxpr (the same object when it contains no sugar)."""
    if not contains(cj.jaxpr, _NAMES):
        return cj
    k = id(cj)
    hit = _CACHE.get(k)
    if hit is not None and hit[0] is cj:
        return hit[1]
    nc = len(cj.consts)
    sds = lambda v: jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype, weak_type=getattr(v.aval, "weak_type", False))
    new = jax.make_jaxpr(lambda *a: _eval(cj.jaxpr, a[:nc], a[nc:]))(
        *[sds(v) for v in list(cj.jaxpr.constvars) + list(cj.jaxpr.invars)])
    j = new.jaxpr
    j = j.replace(constvars=list(j.invars[:nc]) + list(j.constvars), invars=list(j.invars[nc:]))
    out = ClosedJaxpr(j, list(cj.consts) + list(new.consts))
    _CACHE[k] = (cj, out)
    return out
