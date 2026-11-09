"""Jaxpr structure: the ONE module that knows how JAX stores control flow and call-like equations.

Every walker (the tally checker and the re-tracer in _rewrite, the lockstep interpreter, the cuda emitter and its
step split, the dual-lanes size guard in _primitive) reads while / cond / scan / call equations through these
accessors, never through eqn.params directly.  The parameter names are not a JAX API (they changed at least once:
scan's num_consts / num_carry became a flat-tree `ft_in` on JAX 0.10), so a JAX release that renames them breaks here.
"""
import hashlib
import os
import weakref
from typing import NamedTuple

import numpy as np
from jax.extend import source_info_util as _siu
from jax.extend.core import ClosedJaxpr, Literal

# frames inside tallymap are not "user" frames: source summaries of item equations point at the user's code
_siu.register_exclusion(os.path.dirname(os.path.abspath(__file__)) + os.sep)

# call-like primitives: ONE sub-jaxpr whose invars / outvars map 1:1 onto the equation's
CALL_PARAM = {"pjit": "jaxpr", "jit": "jaxpr", "closed_call": "call_jaxpr", "core_call": "call_jaxpr",
              "custom_jvp_call": "call_jaxpr", "custom_vjp_call": "call_jaxpr",
              "custom_vjp_call_jaxpr": "fun_jaxpr", "checkpoint": "jaxpr", "remat": "jaxpr", "remat2": "jaxpr"}


def closed(j):
    return j if isinstance(j, ClosedJaxpr) else ClosedJaxpr(j, [])


def call_jaxpr(eqn):
    """the sub-jaxpr (closed) of a call-like equation, or None."""
    p = CALL_PARAM.get(eqn.primitive.name)
    if p is None or p not in eqn.params:
        return None
    return closed(eqn.params[p])


class While(NamedTuple):
    cond: ClosedJaxpr       # (cond consts, carry) -> bool
    body: ClosedJaxpr       # (body consts, carry) -> carry
    nc: int                 # operands = [cond consts (nc)] [body consts (nb)] [initial carry]
    nb: int


def while_parts(eqn):
    p = eqn.params
    return While(closed(p["cond_jaxpr"]), closed(p["body_jaxpr"]), int(p["cond_nconsts"]), int(p["body_nconsts"]))


def cond_branches(eqn):
    """the branches (closed jaxprs) of a cond / switch; operands = [index] + branch operands."""
    return [closed(b) for b in eqn.params["branches"]]


class Scan(NamedTuple):
    body: ClosedJaxpr       # (consts, carry, x) -> (carry, y)
    nconsts: int            # operands = [consts] [initial carry] [xs]
    ncarry: int
    length: int
    reverse: bool
    unroll: object


def scan_counts(params):
    """(num_consts, num_carry) of scan params (JAX >= 0.10 stores a flat-tree `ft_in` = (consts, carry, xs))."""
    if params.get("num_consts") is not None:
        return params["num_consts"], params["num_carry"]
    c, k, _ = params["ft_in"].unpack()
    return len(c), len(k)


def scan_parts(eqn):
    p = eqn.params
    nc, nk = scan_counts(p)
    return Scan(closed(p["jaxpr"]), nc, nk, int(p["length"]), bool(p["reverse"]), p.get("unroll", 1))


def subjaxprs(eqn):
    """every sub-jaxpr of an equation (closed), whatever its kind."""
    out = []
    for v in eqn.params.values():
        if isinstance(v, ClosedJaxpr) or type(v).__name__ == "Jaxpr":
            out.append(closed(v))
        elif isinstance(v, (tuple, list)):
            out += [closed(b) for b in v if isinstance(b, ClosedJaxpr) or type(b).__name__ == "Jaxpr"]
    return out


def contains(jaxpr, names, _memo=None):
    """does `jaxpr` (or any sub-jaxpr) contain a primitive named in `names`?"""
    _memo = {} if _memo is None else _memo
    k = id(jaxpr)
    if k in _memo:
        return _memo[k]
    r = False
    for e in jaxpr.eqns:
        if e.primitive.name in names or any(contains(s.jaxpr, names, _memo) for s in subjaxprs(e)):
            r = True
            break
    _memo[k] = r
    return r


def host_callbacks(jaxpr, _out=None):
    """the host callables of every equation (`callback` params of pure_callback / io_callback / debug_callback),
    recursively.  The printed jaxpr does not identify them (JAX 0.11 prints only their names)."""
    out = [] if _out is None else _out
    for e in jaxpr.eqns:
        cb = e.params.get("callback")
        if cb is not None:
            out.append(cb)
        for s in subjaxprs(e):
            host_callbacks(s.jaxpr, out)
    return out


def is_dropvar(v):
    return type(v).__name__ == "DropVar"


def lit(v):
    """value of a jaxpr Literal as a typed numpy scalar (0.5.3 stores raw python scalars)."""
    return np.asarray(v.val, dtype=v.aval.dtype)


def with_source(eqn):
    """context: equations traced inside carry `eqn`'s source location (a re-traced program keeps pointing at the
    user's code, like jax.core.eval_jaxpr)."""
    return _siu.user_context(eqn.source_info.traceback)


def bind(eqn, ins):
    """re-bind an equation (the eval_jaxpr way: call-like primitives need their sub-functions rebuilt)."""
    r = eqn.primitive.get_bind_params(eqn.params)
    subfuns, params = r if isinstance(r, tuple) else ((), r)     # jax < 0.7: (subfuns, params); later: params
    with with_source(eqn):
        out = eqn.primitive.bind(*subfuns, *ins, **params)
    return out if eqn.primitive.multiple_results else [out]


def source_summary(eqn):
    """'file:line (function)' of the user code that produced `eqn` (jax.extend.source_info_util), or '?'."""
    try:
        return _siu.summarize(eqn.source_info)
    except Exception:                                            # pragma: no cover
        return "?"


def item_name(fn):
    """'qualname (file:line)' of an item function, for error messages."""
    f = getattr(fn, "func", fn)                                  # functools.partial
    name = getattr(f, "__qualname__", None) or repr(f)
    code = getattr(f, "__code__", None)
    return name + (f" ({os.path.basename(code.co_filename)}:{code.co_firstlineno})" if code is not None else "")


def located(err, eqn, where=""):
    """`err` (an exception) with the item function and the source location of `eqn` appended once (innermost
    equation wins).  Everything else the original carries (its attributes) comes across: a caller may act on them."""
    if getattr(err, "_tm_located", False):
        return err
    msg = f"{err} [{where + ', ' if where else ''}at {source_summary(eqn)}]"
    new = type(err)(msg)
    new.__dict__.update(err.__dict__)
    new._tm_located = True
    return new


# ------------------------------------------------------------------------------------------- structural digest
# The cache keys (the cuda kernel cache, the eager-call cache) need a structural identity of a jaxpr.  str(jaxpr)
# is one, but JAX's pretty-printer builds a document tree per equation: on DTRAX's mode-2 item it took 95 s (A100)
# and 134 s (2080 Ti) per build, for a dict key.  digest() hashes the same information into a sha256 without
# materialising anything (4.2 s / 5.3 s there): primitive names, operands numbered by first appearance (so two
# alpha-equivalent jaxprs hash equal, as their printed forms are equal), output avals, every equation parameter
# by value (sub-jaxprs recursively, anything else by repr, as the printer prints it), and the exact BYTES of every
# literal and of the consts of nested closed jaxprs (the printer rounds floats and does not show nested consts,
# both of which the cuda emitter bakes into the kernel).  A sub-jaxpr is hashed once (memoised by identity on the
# Jaxpr object, weakly, so nothing is kept alive).
_DIGEST = weakref.WeakKeyDictionary()


def _d_aval(h, a):
    h.update(b"|a" + repr(tuple(getattr(a, "shape", ()))).encode() + str(getattr(a, "dtype", None)).encode())
    if getattr(a, "weak_type", False):
        h.update(b"w")


def _d_const(h, c):
    try:
        a = np.asarray(c)
    except Exception:                      # a tracer or an opaque object: identity (never equal to anything else)
        h.update(b"|?" + type(c).__name__.encode() + repr(id(c)).encode())
        return
    if a.dtype == np.dtype(object):
        h.update(b"|?" + repr(c).encode())
        return
    h.update(b"|c" + str(a.dtype).encode() + repr(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())


def _d_param(h, x):
    if isinstance(x, ClosedJaxpr):
        h.update(b"|CJ" + _d_jaxpr(x.jaxpr) + b"%d" % len(x.consts))
        for c in x.consts:
            _d_const(h, c)
    elif type(x).__name__ == "Jaxpr":
        h.update(b"|J" + _d_jaxpr(x))
    elif isinstance(x, (tuple, list)):
        h.update(b"|(%d" % len(x))
        for y in x:
            _d_param(h, y)
        h.update(b")")
    elif isinstance(x, dict):
        h.update(b"|{%d" % len(x))
        for k in sorted(x, key=repr):
            h.update(repr(k).encode())
            _d_param(h, x[k])
        h.update(b"}")
    elif isinstance(x, np.ndarray):
        _d_const(h, x)
    else:
        h.update(b"|r" + repr(x).encode())


def _d_jaxpr(jaxpr):
    """-> the hex digest (bytes) of one Jaxpr, memoised."""
    d = _DIGEST.get(jaxpr)
    if d is not None:
        return d
    g = hashlib.sha256()
    num = {}
    for v in list(jaxpr.constvars) + list(jaxpr.invars):
        num[v] = len(num)
        _d_aval(g, v.aval)
    g.update(b"|%d|%d" % (len(jaxpr.constvars), len(jaxpr.invars)))
    for e in jaxpr.eqns:
        g.update(b"\n" + e.primitive.name.encode())
        for a in e.invars:
            if isinstance(a, Literal):
                g.update(b"|L")
                _d_const(g, a.val)
                _d_aval(g, a.aval)
            else:
                g.update(b"|v%d" % num.get(a, -1))
        for v in e.outvars:
            if is_dropvar(v):
                g.update(b"|_")
            else:
                num.setdefault(v, len(num))
                g.update(b"|o%d" % num[v])
            _d_aval(g, v.aval)
        for k in sorted(e.params):
            g.update(b"|p" + k.encode())
            _d_param(g, e.params[k])
    g.update(b"\nout")
    for v in jaxpr.outvars:
        if isinstance(v, Literal):
            g.update(b"|L")
            _d_const(g, v.val)
            _d_aval(g, v.aval)
        else:
            g.update(b"|v%d" % num.get(v, -1))
    d = g.hexdigest().encode()
    _DIGEST[jaxpr] = d
    return d


def digest(jaxpr):
    """-> hex sha256 of everything str(jaxpr) shows, plus the exact bytes of every literal and of every nested closed
    jaxpr's consts: the cache-key replacement for str(jaxpr).  A ClosedJaxpr's own consts are hashed by value too;
    the cache keys pass the bare `p.cj.jaxpr` and key the top-level consts themselves (by value in the cuda kernel
    key, by identity in the eager-call key)."""
    if isinstance(jaxpr, ClosedJaxpr):
        h = hashlib.sha256()
        _d_param(h, jaxpr)
        return h.hexdigest()
    return _d_jaxpr(jaxpr).decode()


__all__ = ["CALL_PARAM", "Literal", "closed", "call_jaxpr", "While", "while_parts", "cond_branches", "Scan",
           "scan_counts", "scan_parts", "subjaxprs", "contains", "host_callbacks", "is_dropvar", "lit", "with_source",
           "bind", "source_summary", "item_name", "located", "digest"]
