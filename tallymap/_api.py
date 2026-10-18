"""The public API (vmap, Sum, Queue, Result) and the ItemProgram: trace the item once, hand ONE primitive to JAX.

    user fn(tally, *args) -> tally | (tally, y)
      |  make_jaxpr + tally linearity check + hoist closed-over tracers as explicit operands
      |  retrace: cap top-level while loops at max_steps (a per-item `truncated` output) + the replay fingerprint
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

from . import _foreign
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
    """A commutative (sum) output.  exact=True accumulates round(v * scale) in int64: bitwise order-independent
    (needs jax_enable_x64 on the lockstep backend).  |v * scale| >= 9.2e18 cannot be represented: such emits are
    DROPPED and counted in Result.stats['exact_overflow'] (check it!).  tangent_scale: scale of the jvp tangent
    of this output (default: scale)."""
    shape: Any
    dtype: Any = jnp.float32
    exact: bool = False
    scale: float = 2.0 ** 32
    tangent_scale: Any = None


@dataclasses.dataclass(frozen=True)
class Queue:
    """Spawn queue of `capacity` pending children (ring buffer).  A child that does not fit is dropped and counted
    in stats['dropped'].  The capacity is a tallymap resource limit, so a dropped child makes every float tally of
    the call NaN (never a silently truncated shower; integer tallies cannot be NaN, so they are left as they are:
    check stats['dropped']).  Reverse mode (the tree adjoint) records every child of the call, so it needs
    capacity >= the TOTAL number of children; a larger tree gives a NaN gradient (never a wrong one).
    Sizing: README.md, "Sizing the Queue"."""
    capacity: int


class Result(NamedTuple):
    """What a tallymap.vmap function returns.  When a tallymap resource limit intervened (max_steps, the spawn
    Queue capacity) every float output is NaN; the counters in `stats` say which limit it was."""
    outputs: dict        # name -> summed output (float: NaN when a resource limit intervened)
    per_item: Any        # pytree of [N, ...] per-item results (None if the item returns only its tally)
    truncated: Any       # bool[N]: the item hit max_steps in a top-level while loop
    stats: dict          # n_truncated, dropped_emits, exact_overflow; with spawn also spawned / dropped /
                         # children_truncated


class UnsupportedError(NotImplementedError):
    """A backend cannot run this program.  backend='auto' on a GPU raises it too, unless allow_fallback=True (then it
    falls back to the lockstep backend, with a warning)."""


# ------------------------------------------------------------------------------------------- programs
@dataclasses.dataclass(frozen=True, eq=False)
class TSpec:
    name: str
    S: int                 # flat size
    dtype: Any
    lead: tuple = ()       # extra leading dims (dual tangent lanes)
    exact: bool = False
    kind: str = "sum"      # 'sum' | 'spawn' | 'discard'
    scale: float = 2.0 ** 32
    tscale: Any = None


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
    fp: Any = None         # (RV) index of the replay fingerprint (uint32) among the per-item results
    where: str = ""        # the user's item function ('qualname (file:line)'), for error messages

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
    """trace fn to a ClosedJaxpr, with the Equinox ecosystem's own primitives expanded into JAX primitives
    (_foreign.py; a no-op for a program that has none).  Every ItemProgram is built through here, so the
    expansion happens once, before tallymap's jvp / reverse programs and before either backend - and so does
    _foreign.ad_advice, which explains the one failure a foreign library makes likely (a custom_vjp in the item,
    which JAX cannot forward-differentiate; diffrax's default adjoint is one)."""
    try:
        cj = jax.make_jaxpr(fn)(*avals)
    except Exception as e:
        raise (_foreign.ad_advice(e, "item program") or e) from None
    return _foreign.expand(cj)


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
        return list(out[:nT]) + [dout[i] for i in tf] + list(out[nT:]) + [dout[nT + j] for j in yf]
    av = p.in_avals()
    tav, pav, bav = av[:nT], av[nT:nT + nP], av[nT + nP:]
    avals = tav + [tav[i] for i in tf] + pav + [pav[i] for i in ip] + bav + [bav[i] for i in ib]
    from ._reverse import forward_tangents
    with forward_tangents():        # (sparse table reads of an adjoint program: see _reverse.tread)
        cj = _mk(J, [_sds(a) for a in avals])
    tl = list(p.tallies) + [dataclasses.replace(p.tallies[i], name=p.tallies[i].name + "'",
                                                scale=p.tallies[i].tscale or p.tallies[i].scale) for i in tf]
    return ItemProgram(cj, tuple(tl), nP + len(ip), nB + len(ib), p.trunc, p.spawn, p.tag + "+jvp", where=p.where)


# (the item VJP program, vjp_prog, lives with the reverse-mode internals in _reverse.py)


# ------------------------------------------------------------------------------------------- dispatch
_WARNED = set()
_SQUARE = [False]


class _square_mode:
    """tallymap.compare only: trace tallymap.vmap functions with every emitted value squared (per-bin sum of w^2)."""

    def __enter__(self):
        _SQUARE[0] = True

    def __exit__(self, *a):
        _SQUARE[0] = False


def _out_tallies(p):
    """backend outputs: [these sum tallies] + [stats int32[5]: spawned, dropped, children_truncated,
    exact_overflow, dropped_emits] + [per-item results]."""
    return [i for i, s in enumerate(p.tallies) if s.kind == "sum"]


# ------------------------------------------------------------------------------------------- vmap
_OPTS = {
    # public
    "allow_fallback": "backend='auto' on a GPU: run the lockstep backend (with a warning) instead of raising "
                      "UnsupportedError when the cuda backend cannot compile the program (default False; it is "
                      "only ever suggested for programs the lockstep backend can actually run)",
    "cuda_math": "cuda backend, f32 div/sqrt: 'approx' (default) | 'ieee'",
    "cuda_min_blocks": "cuda backend: __launch_bounds__ minimum blocks per SM",
    "lockstep_block": "lockstep backend: lanes per block",
    "rev_checkpoints": "reverse mode: checkpoint buffer size C",
    "rev_step_segments": "reverse mode: recompute each loop step in this many checkpointed segments (0 = off)",
    "rev_stats": "reverse mode: report replay mismatches / tree stats to the host (costs jit's fast path)",
    # PRIVATE (measurements / tests / negative controls; not API)
    "_schedule": "cuda: 'step' | 'item' | 'events' overrides the rule (step unless the program spawns)",
    "_cuda_inplace": "cuda: False never updates a per-item array in place (A/B control of the aliasing analysis)",
    "_lanes": "'off' never dual lanes | 'force' ignore the local-memory guard | int: chunks of at most this many lanes",
    "_fold": "False: never fold a batch into the item axis",
    "_rev_sparse_min": "smallest table (elements) that gets sparse emitted cotangents (default 32)",
    "_adj_math": "NEGATIVE CONTROL: compile the adjoint with this cuda_math (replay check must fire)",
    "_rev_nodense": "NEGATIVE CONTROL: drop the dense detection of mixed-use tables",
    "_rev_gen_capacity": "tree adjoint: largest generation (default Queue capacity; memory lever)",
    "_tree_neg": "NEGATIVE CONTROL of the tree adjoint: 'detach' | 'replay'",
}
BACKENDS = ("auto", "cuda", "lockstep")


def vmap(fn=None, *, outputs, in_axes=0, max_steps=None, spawn=None, backend="auto", **opts):
    """Map `fn(tally, *args) -> tally | (tally, y)` over the leading axis of the mapped args.

    Meaning: jax.vmap(fn) over the items followed by a sum of their tallies.  Every JAX transformation works.

    outputs:   {name: tallymap.Sum(shape, dtype=float32, exact=False)}
    in_axes:   0 (mapped) or None (broadcast, e.g. tables) per positional arg, like jax.vmap
    max_steps: cap on every top-level while_loop.  None (the default) is exactly jax.vmap: no cap, and an item
               that does not terminate does not terminate.  Setting it is the recommended safety net; an item that
               hits the cap is flagged in Result.truncated and POISONS every float tally of the call (see below)
    spawn:     tallymap.Queue(capacity) to allow tally.spawn(child)
    backend:   'auto': cuda on a GPU (a program it cannot compile raises UnsupportedError unless allow_fallback=True),
               the lockstep backend elsewhere (with a one-time warning) | 'cuda' | 'lockstep'
    opts:      allow_fallback     backend='auto' on a GPU: fall back to lockstep (warning) instead of raising.
               cuda_math          cuda f32 div/sqrt: 'approx' (default; XLA:GPU's own PTX) | 'ieee' (correctly
                                  rounded: the portable reference).
               cuda_min_blocks    cuda __launch_bounds__ minimum blocks per SM (register cap; performance only;
                                  0 = compiler).  The best value is per kernel.
               lockstep_block     lockstep backend: lanes per block (default 2^18; bounds its memory).
               rev_checkpoints    reverse mode: states per checkpoint buffer C (default 16).  O(K) work for loops
                                  of K <= C^2 steps, 2C+1 states of per-item memory.
               rev_step_segments  reverse mode: recompute each loop step in S checkpointed segments inside its VJP
                                  (default 0 = off; helps when a step's residuals spill, costs recomputation).
               rev_stats          reverse mode: report replay mismatches and tree-adjoint stats to the host
                                  (tallymap.diagnostics()['reverse']).  Off by default: a host callback takes jax.jit
                                  off its C++ fast path.  A failed check makes the gradient NaN either way.
               Underscored options are PRIVATE (measurements and negative controls; listed in _OPTS).  Any other
               name raises.
    The returned function takes the mapped/broadcast args, plus optionally n_active=<traced int>: only items
    i < n_active run (fixed capacity N, varying active count: no retrace / recompile).  Items i >= n_active do
    nothing, so they never count as truncated.

    Resource limits.  The result is exactly jax.vmap(fn) over the items followed by a sum, and deviates only where
    a tallymap resource limit intervenes: max_steps, or the spawn Queue capacity.  When one of them does, every
    float tally of the call is NaN, so a truncated or dropped item is never silently missing from a sum.  Which
    limit it was is in Result.stats: n_truncated (and children_truncated) for max_steps, dropped for the Queue.
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
                         + ", ".join(k for k in _OPTS if not k.startswith("_")))
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
        if n_active is not None:               # F4: fixed (bucketed) capacity N, traced active count: no retrace
            if spawn is not None:
                raise ValueError("tallymap: n_active with spawn is not supported")
            pi = pi + [jnp.arange(pi[0].shape[0], dtype=jnp.int32)]
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
                        bool(outs_spec[n].exact), "sum", float(outs_spec[n].scale), outs_spec[n].tangent_scale)
                  for n in names]
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
            if n_active is None:
                r = fn(t, *uargs)
            else:                              # items >= n_active do nothing (outputs zero)
                iid, nact = flat[nT + nmp], flat[-1]
                shp = jax.eval_shape(lambda t_, a: fn(t_, *a), t, uargs)
                skip = lambda t_, a: (t_ if isinstance(shp, Tally) else
                                      (t_, jax.tree_util.tree_map(lambda s_: jnp.zeros(s_.shape, s_.dtype), shp[1])))
                r = lax.cond(iid < nact, lambda t_, a: fn(t_, *a), skip, t, uargs)
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
            fr = retrace(ClosedJaxpr(cj.jaxpr, consts), cap=max_steps, n_tally=nT, square=_SQUARE[0],
                         fingerprint=True)          # (RV) per-item replay fingerprint, last per-item result
            return fr(*a[:nT + len(pi) + nb0])
        cj2 = _mk(item2, tav + pav + bav + [_sds(h) for h in hoisted])
        if spawn is not None:
            _check_spawn(cj2, pav, where)
        nY = len(cj2.jaxpr.outvars) - nT
        prog = ItemProgram(cj2, tuple(tspecs), len(pi), len(bc) + len(hoisted),
                       nY - 2 if max_steps is not None else None, spawn, fp=nY - 1, where=where)
        bc_all = bc + hoisted
        from ._diagnostics import record_program
        record_program(prog)
        from ._primitive import bind_call, make_node
        flat = bind_call(*pi, *bc_all, node=make_node(prog, backend, opts, _call_site(decl)))
        nt = len(_out_tallies(prog))
        tvals, st, ys = flat[:nt], flat[nt], flat[nt + 1:]
        res_out = {n: tvals[i].reshape(shapes[n]) for i, n in enumerate(names)}
        stats = {"exact_overflow": st[3], "dropped_emits": st[4]}
        if spawn is not None:
            stats.update(spawned=st[0], dropped=st[1], children_truncated=st[2])
        ys = ys[:-1]                           # the replay fingerprint (ItemProgram.fp): internal
        if max_steps is not None:
            trunc = ys[-1]
            ys = ys[:-1]
        else:
            trunc = jnp.zeros((N,), bool)
        stats["n_truncated"] = jnp.sum(trunc.astype(jnp.int32))
        res_out = _poison(res_out, _limits(stats, st, spawn, max_steps))
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
