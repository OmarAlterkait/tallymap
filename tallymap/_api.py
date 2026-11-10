"""The public API (vmap, Sum, Queue, Result) and the ItemProgram: trace the item once, hand ONE primitive to JAX.

    user fn(tally, *args) -> tally | (tally, y)
      |  make_jaxpr + tally linearity check + hoist closed-over tracers as explicit operands
      |  retrace: cap top-level while loops at max_steps (a per-item `truncated` output)
      v
    ItemProgram (item jaxpr + layout)  ->  bind_call(*per_item, *broadcast, node=Node(prog))      (_primitive.py)

Every transformation is a rule of that primitive that builds a new ItemProgram by applying jax.jvp / jax.vjp /
jax.vmap to an ITEM function (jvp_prog here, vjp_prog in _reverse.py, the rest in _primitive.py), so the rules
compose to any order.
"""
import dataclasses
import os
import sys
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.core import Tracer
from jax.extend.core import ClosedJaxpr, jaxpr_as_fun

from . import _errors
from ._rewrite import check_tallies, isfloat, retrace
from ._tally import Tally

FLOAT0 = jax.dtypes.float0
MAX_ITEMS = 2 ** 31 - 1
_HERE = os.path.dirname(os.path.abspath(__file__))
_SKIP = []


def _call_site(fallback=None):
    """(filename, lineno) of the user's call into tallymap: the innermost frame that is neither tallymap's nor
    JAX's own (under jax.jit the mapped function is called by JAX's tracing machinery, not by the user).  A
    backend warning is raised when the program is LOWERED, which is nowhere near the user's code, so the call
    site travels with the program (Node.site) and the warning is attributed to it."""
    if not _SKIP:
        _SKIP.extend((_HERE, os.path.dirname(os.path.abspath(jax.__file__))))
    f = sys._getframe(1)
    while f is not None:
        path = os.path.abspath(f.f_code.co_filename)
        if not any(path == d or path.startswith(d + os.sep) for d in _SKIP):
            return (f.f_code.co_filename, f.f_lineno)
        f = f.f_back
    return fallback


# ------------------------------------------------------------------------------------------- user specs
@dataclasses.dataclass(frozen=True)
class Sum:
    """A commutative (sum) output.  fixed_point=True accumulates round(v * scale) in int64 fixed point: bitwise
    order-independent, each value rounded to a multiple of 1/scale (needs jax_enable_x64 on the xla backend).
    |v * scale| >= 9.2e18 cannot be represented: such an emit is dropped, counted in
    Result.stats['fixed_point_overflow'], and the BIN it was emitted to becomes NaN (every kernel flags its own bins: a
    primal bin by the primal's emits, a jvp / jacfwd tangent bin by that tangent's emits, on every path).  A jvp /
    vjp tangent tally of this output uses the same scale."""
    shape: Any
    dtype: Any = jnp.float32
    fixed_point: bool = False
    scale: float = 2.0 ** 32


@dataclasses.dataclass(frozen=True)
class Queue:
    """Spawn queue of `capacity` pending children (ring buffer).  A child that does not fit is dropped and counted
    in stats['children_dropped'].  The capacity is a tallymap resource limit, so a dropped child makes every float
    tally of the call NaN (never a silently truncated shower; integer tallies cannot be NaN, so they are left as they
    are: check stats['children_dropped']).  Reverse mode (the tree adjoint) records every child of the call, so it
    needs capacity >= the TOTAL number of children; a larger tree gives a NaN gradient (never a wrong one).
    Sizing: README.md, "Sizing the Queue"."""
    capacity: int


class Result(NamedTuple):
    """What a tallymap.vmap function returns.  When a tallymap resource limit intervened (max_steps, the spawn
    Queue capacity) every float output is NaN; a fixed_point=True bin an unrepresentable emit was dropped from is
    NaN.
    The counters in `stats` say which limit it was."""
    outputs: dict        # name -> summed output (float: NaN when a resource limit intervened)
    per_item: Any        # pytree of [N, ...] per-item results (None if the item returns only its tally)
    truncated: Any       # bool[N]: the item hit max_steps in a top-level while loop
    stats: dict          # n_truncated, dropped_emits, fixed_point_overflow; with spawn also spawned /
                         # children_dropped / children_truncated


class UnsupportedError(NotImplementedError):
    """A backend cannot run this program.  backend='auto' on a GPU raises it too (it never falls back to the xla
    backend: pass backend='xla' for that)."""


# ------------------------------------------------------------------------------------------- programs
@dataclasses.dataclass(frozen=True, eq=False)
class TSpec:
    name: str
    S: int                 # flat size
    dtype: Any
    lead: tuple = ()       # extra leading dims (dual tangent lanes)
    fixed_point: bool = False
    kind: str = "sum"      # 'sum' | 'spawn' | 'discard'
    scale: float = 2.0 ** 32


@dataclasses.dataclass(frozen=True, eq=False)
class ItemProgram:
    """invars: [tallies] + [per-item leaves (per-item avals)] + [broadcast leaves];
    outvars: [tallies (same slots)] + [per-item results]."""
    cj: ClosedJaxpr
    tallies: tuple
    nP: int
    nB: int
    trunc: Any = None      # index of the `truncated` flag among the per-item results
    spawn: Any = None      # Queue or None
    tag: str = "primal"
    where: str = ""        # the user's item function ('qualname (file:line)'), for error messages
    nact: Any = None       # n_active: index among the broadcast operands of the int32 ACTIVE ITEM COUNT.  It is the
                           # launch's item count: items i >= it do not run (no emits, per-item results zero).  The
                           # item jaxpr does not read it; every derived program keeps it at the same index.

    @property
    def nT(self):
        return len(self.tallies)

    @property
    def nY(self):
        return len(self.cj.jaxpr.outvars) - self.nT

    def y_avals(self):
        return [v.aval for v in self.cj.jaxpr.outvars[self.nT:]]

    def in_avals(self):
        return [v.aval for v in self.cj.jaxpr.invars]


def _sds(a):
    return jax.ShapeDtypeStruct(a.shape, a.dtype)


def _tally_aval(s):
    if s.kind == "spawn":
        return jax.ShapeDtypeStruct((), s.dtype)
    return jax.ShapeDtypeStruct(tuple(s.lead) + (s.S,), s.dtype)


def _mk(fn, avals):
    """trace fn to a ClosedJaxpr.  Every ItemProgram is built through here, so a failure gets
    _errors.ad_advice, which explains the failures an item makes likely (a custom_vjp, which JAX cannot
    forward-differentiate; a leaked tracer)."""
    try:
        return jax.make_jaxpr(fn)(*avals)
    except Exception as e:
        raise (_errors.ad_advice(e, "item program") or e) from None


# ------------------------------------------------------------------------------------------- transforms
def jvp_prog(p, pert_pi, pert_bc):
    f = jaxpr_as_fun(p.cj)
    nT, nP, nB = p.nT, p.nP, p.nB
    # discarded float tallies get (discarded) tangent tallies too: the jvp of an emit into them is an emit, and it
    # must land in a tally slot (derived programs -- lin / T -- discard their primal tallies)
    tf = [i for i, s in enumerate(p.tallies) if s.kind in ("sum", "discard") and isfloat(s.dtype)]
    ya = p.y_avals()
    yf = [j for j, a in enumerate(ya) if isfloat(a.dtype)]
    ip = [i for i in range(nP) if pert_pi[i]]
    ib = [i for i in range(nB) if pert_bc[i]]

    def J(*a):
        t, td = list(a[:nT]), list(a[nT:nT + len(tf)])
        o = nT + len(tf)                 # layout: [t, t'] [pi, pi'] [bc, bc']
        pi, pid = list(a[o:o + nP]), list(a[o + nP:o + nP + len(ip)])
        o += nP + len(ip)
        bc, bcd = list(a[o:o + nB]), list(a[o + nB:])

        def g(ts, ps, bs):
            tt, pp, bb = list(t), list(pi), list(bc)
            for i, v in zip(tf, ts):
                tt[i] = v
            for i, v in zip(ip, ps):
                pp[i] = v
            for i, v in zip(ib, bs):
                bb[i] = v
            return f(*tt, *pp, *bb)
        out, dout = jax.jvp(g, ([t[i] for i in tf], [pi[i] for i in ip], [bc[i] for i in ib]), (td, pid, bcd))
        dt = [dout[i] for i in tf]
        if p.trunc is not None:      # (max_steps) a truncated item poisons the tangent sums itself: _poison_tally
            dt = [_poison_tally(x, out[nT + p.trunc]) if p.tallies[i].kind == "sum" else x for i, x in zip(tf, dt)]
        return list(out[:nT]) + dt + list(out[nT:]) + [dout[nT + j] for j in yf]
    av = p.in_avals()
    tav, pav, bav = av[:nT], av[nT:nT + nP], av[nT + nP:]
    avals = tav + [tav[i] for i in tf] + pav + [pav[i] for i in ip] + bav + [bav[i] for i in ib]
    from ._reverse import forward_tangents
    with forward_tangents():        # (sparse table reads of an adjoint program: see _reverse.tread)
        cj = _mk(J, [_sds(a) for a in avals])
    tl = list(p.tallies) + [dataclasses.replace(p.tallies[i], name=p.tallies[i].name + "'") for i in tf]
    return ItemProgram(cj, tuple(tl), nP + len(ip), nB + len(ib), p.trunc, p.spawn, p.tag + "+jvp", where=p.where,
                       nact=p.nact)


def _poison_tally(t, bad):
    """the tally t with NaN emitted into every bin when the item's `bad` (its truncation flag) is set.  The primal
    outputs of a truncated call are poisoned by _poison_primal, whose tangent is the identity (so that jax.grad needs
    no forward launch); the tangent sums are poisoned here, inside the tangent / fused / dual-lane kernel, by the
    items that were truncated (per item).  A fixed-point sum flags the
    NaN emits as overflows (stats['fixed_point_overflow'] then counts them too)."""
    from ._rewrite import emit_p_bind
    nan = jnp.full(t.shape[:-1], np.nan, t.dtype)
    return lax.cond(bad, lambda t: lax.fori_loop(0, t.shape[-1], lambda i, t: emit_p_bind(t, i, nan), t),
                    lambda t: t, t)


# (the item VJP program, vjp_prog, lives with the reverse-mode internals in _reverse.py)


# ------------------------------------------------------------------------------------------- dispatch
_WARNED = set()


def _out_tallies(p):
    """backend outputs: [these sum tallies] + [stats int32[5]: spawned, children_dropped, children_truncated,
    fixed_point_overflow, dropped_emits] + [per-item results]."""
    return [i for i, s in enumerate(p.tallies) if s.kind == "sum"]


# ------------------------------------------------------------------------------------------- vmap
_OPTS = {                  # the public options; measurement switches and negative controls are module constants
    "cuda_math": "cuda backend, f32 div/sqrt: 'approx' (default) | 'ieee'",
    "schedule": "cuda backend: 'item' (default: a thread runs whole items) | 'step' (refill at loop-step granularity)",
    "xla_batch_size": "xla backend: items per batch",
    "rev_checkpoints": "reverse mode: checkpoint buffer size C (default 'auto': from the loop state's size)",
    "rev_stats": "reverse through spawn: report the tree adjoint's stats to the host (costs jit's fast path)",
}
BACKENDS = ("auto", "cuda", "xla")


def vmap(fn=None, *, outputs, in_axes=0, max_steps=None, spawn=None, backend="auto", **opts):
    """Map `fn(tally, *args) -> tally | (tally, y)` over the leading axis of the mapped args.

    Meaning: jax.vmap(fn) over the items followed by a sum of their tallies.  Every JAX transformation works.

    outputs:   {name: tallymap.Sum(shape, dtype=float32, fixed_point=False)}
    in_axes:   0 (mapped) or None (broadcast, e.g. tables) per positional arg, like jax.vmap
    max_steps: cap on every top-level while_loop.  None (the default) is exactly jax.vmap: no cap, and an item
               that does not terminate does not terminate.  Setting it is the recommended safety net; an item that
               hits the cap is flagged in Result.truncated and POISONS every float tally of the call (see below)
    spawn:     tallymap.Queue(capacity) to allow tally.spawn(child)
    backend:   'auto': cuda on a GPU (a program it cannot compile raises UnsupportedError; backend='xla' runs
               it), the xla backend elsewhere (with a one-time warning) | 'cuda' | 'xla'
    opts:      cuda_math          cuda f32 div/sqrt: 'approx' (default; XLA:GPU's own PTX) | 'ieee' (correctly
                                  rounded and portable).
               schedule           cuda: 'item' (default: a thread runs one whole item, then claims the next) |
                                  'step' (the step schedule: a thread is refilled with a new item at loop-STEP
                                  granularity, so a thread whose item ended does not idle until its warp's longest
                                  item ends).  'step' needs an item with exactly one top-level while loop and
                                  raises UnsupportedError otherwise; derived kernels (jvp, lanes, an adjoint) use it
                                  wherever they split.  Which is faster depends on the workload: measure.
               xla_batch_size     xla backend: items per batch (default 2^18; bounds its memory).
               rev_checkpoints    reverse mode: states per checkpoint buffer C (default 'auto': 16 for a loop step
                                  of fewer than 1000 equations, else the largest of 4, 8, ..., 64 whose 3C+1
                                  (single-loop adjoint: 2C+1) states fit 64 KiB; _reverse.auto_checkpoints).  A loop
                                  of K <= C steps needs no replay; O(K) work for K <= C^2.
               rev_stats          reverse through spawn: report the tree adjoint's stats to the host
                                  (tallymap.diagnostics()['reverse']).  Off by default: a host callback takes jax.jit
                                  off its C++ fast path.  A failed check makes the gradient NaN either way.
               Any other name raises (there are no private options: measurement switches and negative
               controls are test-only module constants, e.g. codegen.FILL, _primitive.LANES).
    The returned function takes the mapped/broadcast args, plus optionally n_active=<traced int>: only items
    i < n_active run (fixed capacity N, varying active count: no retrace / recompile).  n_active is the launch's
    item count, not part of the item program: items i >= n_active do not run, so they emit nothing, never count
    as truncated, and their per-item results are zero.

    Resource limits.  The result is exactly jax.vmap(fn) over the items followed by a sum, and deviates only where
    a tallymap resource limit intervenes: max_steps, or the spawn Queue capacity.  When one of them does, every
    float tally of the call is NaN, so a truncated or dropped item is never silently missing from a sum.  Which
    limit it was is in Result.stats: n_truncated (and children_truncated) for max_steps, children_dropped for the
    Queue.
    Integer tallies cannot be NaN, so they are left as they are and their counter is the only signal.  Per-item
    outputs are not poisoned: they carry their own per-item flag (Result.truncated) in the same Result.  Gradients
    of a poisoned tally are NaN, and nothing else about them changes.  There is no opt-out: a deliberate step
    budget belongs in the loop condition ((alive) & (step < K)), which is exact JAX semantics.

    An out-of-range emit index (negative or >= the output size) is DROPPED, exactly as JAX's own scatter drops
    out-of-bounds updates, so this is not a deviation and does not poison.  Dropped emits are counted in
    stats['dropped_emits']: every dropped index of the call, over every tally, so an off-by-one binning shows up
    there.  It counts the EMITS the launch executed, so a transformation that emits more than once per index (a
    fused jvp also emits into the tangent tally) counts each of them.
    """
    if fn is None:
        return lambda f: vmap(f, outputs=outputs, in_axes=in_axes, max_steps=max_steps, spawn=spawn,
                              backend=backend, **opts)
    bad = sorted(set(opts) - set(_OPTS))
    if bad:
        raise ValueError(f"tallymap: unknown option(s) {bad}; options are "
                         + ", ".join(_OPTS))
    if opts.get("schedule", "item") not in ("item", "step"):
        raise ValueError(f"tallymap: schedule must be 'item' or 'step', got {opts['schedule']!r}")
    if backend not in BACKENDS:
        raise ValueError(f"tallymap: unknown backend {backend!r}; backends are {', '.join(BACKENDS)}")
    if spawn is not None and not isinstance(spawn, Queue):
        raise TypeError("tallymap: spawn must be tallymap.Queue(capacity)")
    outs_spec = {k: (v if isinstance(v, Sum) else Sum(v)) for k, v in outputs.items()}
    names = sorted(outs_spec)
    from ._jaxpr import item_name
    where = item_name(fn)
    decl = _call_site()          # where tm.vmap was written: the fallback when the call itself is all JAX frames

    def mapped(*args, n_active=None):
        axes = in_axes if isinstance(in_axes, (tuple, list)) else (in_axes,) * len(args)
        if len(axes) != len(args) or any(a not in (0, None) for a in axes):
            raise ValueError("tallymap: in_axes must be 0/None or a tuple of 0/None per positional argument")
        margs = tuple(a for a, x in zip(args, axes) if x == 0)
        bargs = tuple(a for a, x in zip(args, axes) if x is None)
        pi, ptree = jax.tree_util.tree_flatten(margs)
        ctree = jax.tree_util.tree_structure(margs[0] if len(margs) == 1 else margs)    # a spawned child
        bc, btree = jax.tree_util.tree_flatten(bargs)
        pi = [jnp.asarray(x) for x in pi]
        bc = [jnp.asarray(x) for x in bc]
        nmp, nbu = len(pi), len(bc)            # user leaves
        if n_active is not None:               # fixed (bucketed) capacity N, traced active count: no retrace.  The
            if spawn is not None:              # count is a broadcast operand the item does not read (ItemProgram.nact)
                raise ValueError("tallymap: n_active with spawn is not supported")
            bc = bc + [jnp.asarray(n_active, jnp.int32)]
        if not pi:
            raise ValueError("tallymap: at least one argument must be mapped (in_axes=0)")
        Ns = {int(x.shape[0]) if x.ndim else -1 for x in pi}
        if len(Ns) != 1 or -1 in Ns:
            raise ValueError(f"tallymap: mapped arguments must share one leading axis length, got {sorted(Ns)}")
        N = Ns.pop()
        if N > MAX_ITEMS:
            raise ValueError(f"tallymap: N={N} items exceeds 2^31-1; split the call (and fold the chunk index into "
                             "your keys)")
        shapes = {n: (outs_spec[n].shape,) if isinstance(outs_spec[n].shape, int) else tuple(outs_spec[n].shape)
                  for n in names}
        tspecs = [TSpec(n, int(np.prod(shapes[n], dtype=np.int64)), np.dtype(outs_spec[n].dtype), (),
                        bool(outs_spec[n].fixed_point), "sum", float(outs_spec[n].scale)) for n in names]
        if spawn is not None:
            tspecs.append(TSpec("__spawn__", 1, np.dtype(np.int32), (), False, "spawn"))
        nT = len(tspecs)
        meta = {}

        def item(*flat):
            ta = {n: flat[i] for i, n in enumerate(names)}
            t = Tally(ta, shapes, flat[nT - 1] if spawn is not None else None, ctree if spawn is not None else None)
            m = jax.tree_util.tree_unflatten(ptree, list(flat[nT:nT + nmp]))
            b = jax.tree_util.tree_unflatten(btree, list(flat[nT + len(pi):nT + len(pi) + nbu]))
            mi, bi = iter(m), iter(b)
            uargs = [next(mi) if x == 0 else next(bi) for x in axes]
            r = fn(t, *uargs)
            if isinstance(r, Tally):
                t2, y = r, None
            elif isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], Tally):
                t2, y = r
            else:
                raise TypeError("tallymap: the item function must return its Tally or (Tally, per_item_result)")
            tl = jax.tree_util.tree_leaves(t2)
            yl, meta["ytree"] = jax.tree_util.tree_flatten(y)
            return tl + [jnp.asarray(v) for v in yl]
        tav = [_tally_aval(s) for s in tspecs]
        pav = [jax.ShapeDtypeStruct(x.shape[1:], x.dtype) for x in pi]
        bav = [_sds(x) for x in bc]
        cj = _mk(item, tav + pav + bav)
        kinds = [s.kind for s in tspecs]
        check_tallies(cj, kinds, name=f"item function {where}")
        # hoist closed-over tracers (outer jit/grad/jvp/vmap values) as explicit broadcast operands
        hidx = [k for k, c in enumerate(cj.consts) if isinstance(c, Tracer)]
        hoisted = [cj.consts[k] for k in hidx]
        nb0 = len(bc)

        def item2(*a):
            consts = list(cj.consts)
            for k, x in zip(hidx, a[nT + len(pi) + nb0:]):
                consts[k] = x
            return retrace(ClosedJaxpr(cj.jaxpr, consts), cap=max_steps, n_tally=nT)(*a[:nT + len(pi) + nb0])
        cj2 = _mk(item2, tav + pav + bav + [_sds(h) for h in hoisted])
        if spawn is not None:
            _check_spawn(cj2, pav, where)
        nY = len(cj2.jaxpr.outvars) - nT
        prog = ItemProgram(cj2, tuple(tspecs), len(pi), len(bc) + len(hoisted),
                           nY - 1 if max_steps is not None else None, spawn, where=where,
                           nact=nb0 - 1 if n_active is not None else None)
        bc_all = bc + hoisted
        from ._diagnostics import record_program
        record_program(prog)
        from ._primitive import bind_call, make_node
        flat = bind_call(*pi, *bc_all, node=make_node(prog, backend, opts, _call_site(decl)))
        nt = len(_out_tallies(prog))
        tvals, st, ys = flat[:nt], flat[nt], flat[nt + 1:]
        res_out = {n: tvals[i].reshape(shapes[n]) for i, n in enumerate(names)}
        stats = {"fixed_point_overflow": st[3], "dropped_emits": st[4]}
        if spawn is not None:
            stats.update(spawned=st[0], children_dropped=st[1], children_truncated=st[2])
        if max_steps is not None:
            trunc = ys[-1]
            ys = ys[:-1]
        else:
            trunc = jnp.zeros((N,), bool)
        stats["n_truncated"] = jnp.sum(trunc.astype(jnp.int32))
        # (a truncated call: the derived kernels poison their own tangents / cotangents, so the primal's poison
        # has an identity tangent; the tree adjoint (spawn) keeps the poison that differentiates)
        res_out = (_poison if spawn is not None else _poison_primal)(res_out, _limits(stats, st, spawn, max_steps))
        per_item = jax.tree_util.tree_unflatten(meta["ytree"], ys)
        return Result(res_out, per_item, trunc, stats)
    mapped.__wrapped__ = fn
    return mapped


def _limits(stats, st, spawn, max_steps):
    """-> a traced bool: did a tallymap RESOURCE LIMIT intervene in this call?  Either the spawn Queue was too small
    for a child, or an item (or a spawned child) hit max_steps."""
    flags = []
    if spawn is not None:
        flags.append(st[1] > 0)                            # a child that did not fit the Queue
    if max_steps is not None:
        flags.append(stats["n_truncated"] > 0)             # an item hit max_steps
        if spawn is not None:
            flags.append(st[2] > 0)                        # a spawned child hit max_steps
    if not flags:
        return None
    bad = flags[0]
    for f in flags[1:]:
        bad = bad | f
    return bad


def _poison(outs, bad):
    """Every float output becomes NaN when a resource limit intervened: a truncated or dropped item is never
    silently missing from a sum.  The counters in Result.stats stay the signal; integer tallies cannot be NaN.
    x * 1.0 is exact, so a call that hit no limit is bitwise unchanged, and the multiply carries the poison into
    tangents and cotangents (a NaN gradient, never a wrong one)."""
    if bad is None:
        return outs
    poison = jnp.where(bad, jnp.float32(np.nan), jnp.float32(1.0))
    return {n: v * poison.astype(v.dtype) if isfloat(v.dtype) else v for n, v in outs.items()}


@jax.custom_jvp
def _pmul(x, f):
    return x * f


_pmul.defjvp(lambda pr, tg: (_pmul(*pr), tg[0]))


def _poison_primal(outs, bad):
    """_poison whose tangent is the identity: the tangent / adjoint kernels poison themselves (jvp_prog's
    _poison_tally; _primitive._vjp_call with the adjoint's own truncation flags), so neither jax.jvp's tangent nor
    jax.grad's cotangent reads this call's primal outputs, and a gradient is ONE launch (lumen T6b)."""
    if bad is None:
        return outs
    poison = jnp.where(bad, jnp.float32(np.nan), jnp.float32(1.0))
    return {n: _pmul(v, poison.astype(v.dtype)) if isfloat(v.dtype) else v for n, v in outs.items()}


def _check_spawn(cj, pav, where=""):
    from ._jaxpr import located, subjaxprs
    want = [(tuple(a.shape), np.dtype(a.dtype)) for a in pav]

    def walk(j):
        for e in j.eqns:
            if e.primitive.name == "tallymap_spawn":
                got = [(tuple(v.aval.shape), np.dtype(v.aval.dtype)) for v in e.invars[2:]]
                if got != want:
                    raise located(TypeError(f"tallymap: spawned child leaves {got} do not match the mapped inputs "
                                            f"{want}"), e, f"item function {where}")
            for s in subjaxprs(e):
                walk(s.jaxpr)
    walk(cj.jaxpr)
