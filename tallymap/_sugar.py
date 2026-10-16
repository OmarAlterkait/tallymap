"""The sugar re-tracer: replace chosen equations of a jaxpr by their expansion, by re-tracing the jaxpr once.

Two rewrite tables use this engine:

| table                       | what it expands                             | when it runs |
|---|---|---|
| `_foreign.NAMES`            | Equinox / Lineax primitives -> their `impl` | at the item trace, before anything else |
| `backends/cuda/canon.SUGAR` | JAX sugar -> what the op table emits        | in the cuda emitter |

`rewrite(cj, table)` returns the SAME object when no equation applies.  The re-traced jaxpr keeps the original
constants first, in order (the cuda kernel's hoisted-constant inputs are indexed by position), and nested jaxprs
(while / cond / scan / call bodies) are rewritten recursively.  An expansion that itself produces expandable
equations is expanded again, to a fixed point (`maybe_set` expands into `select_if_vmap`, which expands again).
"""
import jax
from jax.extend.core import ClosedJaxpr, Literal

from ._jaxpr import bind, closed, contains, is_dropvar, lit, subjaxprs, with_source

MAX_PASSES = 8          # expansions nest a few deep (linear_solve -> nontraceable, maybe_set -> select_if_vmap)


class Sugar:
    """One rewrite table.

    names:   the primitive names it may expand (a cheap pre-filter: a jaxpr without any of them is returned
             unchanged, with no re-trace).
    applies: applies(eqn) -> bool, for an equation whose primitive is in `names`.
    expand:  expand(eqn, ins) -> out | [out], the equations to trace in that equation's place.
    """

    def __init__(self, label, names, applies, expand):
        self.label = label
        self.names = tuple(names)
        self._applies = applies
        self._expand = expand
        self._cache = {}

    def applies(self, eqn):
        return eqn.primitive.name in self.names and self._applies(eqn)

    def expand(self, eqn, ins):
        out = self._expand(eqn, ins)
        return list(out) if isinstance(out, (list, tuple)) else [out]

    def maybe(self, jaxpr):
        """could this jaxpr (or a sub-jaxpr) hold an expandable equation?  Name-based, so it may say yes
        wrongly; `_pending` is the exact answer."""
        return contains(jaxpr, self.names)


def _pending(jaxpr, sug, _memo=None):
    """does `jaxpr` (or a sub-jaxpr) hold an equation this table actually expands?"""
    _memo = {} if _memo is None else _memo
    k = id(jaxpr)
    if k in _memo:
        return _memo[k]
    r = False
    for e in jaxpr.eqns:
        if sug.applies(e) or any(_pending(s.jaxpr, sug, _memo) for s in subjaxprs(e)):
            r = True
            break
    _memo[k] = r
    return r


def _param(v, sug):
    if isinstance(v, ClosedJaxpr):
        return _one(v, sug)
    if type(v).__name__ == "Jaxpr":
        c = _one(closed(v), sug)
        if c.consts:                                     # pragma: no cover (expansions create no constants)
            raise NotImplementedError(f"tallymap: rewriting an open sub-jaxpr ({sug.label}) created constants")
        return c.jaxpr
    if isinstance(v, (tuple, list)) and any(isinstance(b, ClosedJaxpr) or type(b).__name__ == "Jaxpr" for b in v):
        return type(v)(_param(b, sug) for b in v)
    return v


def _eval(jaxpr, consts, args, sug):
    env = {}

    def read(v):
        return lit(v) if isinstance(v, Literal) else env[v]
    for v, x in zip(jaxpr.constvars, consts):
        env[v] = x
    for v, x in zip(jaxpr.invars, args):
        env[v] = x
    for e in jaxpr.eqns:
        ins = [read(v) for v in e.invars]
        if sug.applies(e):
            with with_source(e):
                outs = sug.expand(e, ins)
        else:
            if any(sug.maybe(s.jaxpr) for s in subjaxprs(e)):
                e = e.replace(params={k: _param(v, sug) for k, v in e.params.items()})
            outs = bind(e, ins)
        for v, x in zip(e.outvars, outs):
            if not is_dropvar(v):
                env[v] = x
    return [read(v) for v in jaxpr.outvars]


def _one(cj, sug):
    """one re-tracing pass over cj (the same object when the table's names are absent)."""
    if not sug.maybe(cj.jaxpr):
        return cj
    k = id(cj)
    hit = sug._cache.get(k)
    if hit is not None and hit[0] is cj:
        return hit[1]
    nc = len(cj.consts)
    sds = lambda v: jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype, weak_type=getattr(v.aval, "weak_type", False))
    new = jax.make_jaxpr(lambda *a: _eval(cj.jaxpr, a[:nc], a[nc:], sug))(
        *[sds(v) for v in list(cj.jaxpr.constvars) + list(cj.jaxpr.invars)])
    j = new.jaxpr
    j = j.replace(constvars=list(j.invars[:nc]) + list(j.constvars), invars=list(j.invars[nc:]))
    out = ClosedJaxpr(j, list(cj.consts) + list(new.consts))
    sug._cache[k] = (cj, out)
    return out


def rewrite(cj, sug):
    """-> cj with every equation this table expands replaced by its expansion, to a fixed point.

    The FIRST pass is decided by the name pre-filter alone (`maybe`), so a program whose rewrite is a no-op is
    still re-traced exactly as a single-pass rewrite would have re-traced it; later passes need `_pending`."""
    for i in range(MAX_PASSES):
        if not sug.maybe(cj.jaxpr):
            return cj
        if i and not _pending(cj.jaxpr, sug):
            return cj
        new = _one(cj, sug)
        if new is cj:
            return cj
        cj = new
    raise NotImplementedError(
        f"tallymap: the {sug.label} rewrite did not settle in {MAX_PASSES} passes; one of "
        f"{', '.join(sug.names)} expands into itself")


__all__ = ["Sugar", "rewrite"]
