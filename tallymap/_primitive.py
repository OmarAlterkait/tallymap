"""The primitive: ONE JAX primitive for every backend call, so that every JAX transformation composes.

    tallymap_call(*per_item_operands, *broadcast_operands, node=Node)
        -> [sum tallies] + [stats i32[5]] + [per-item results]

A Node is an ItemProgram (the item jaxpr + layout, _api.ItemProgram) plus how it was derived.  The rules only ever
build new ItemPrograms by applying jax.jvp / jax.vjp / jax.vmap to an ITEM function and bind the primitive again, so
they recurse to any order.  Node kinds:

  kind          operands                    meaning                                         linear in
  primal        pi, bc                      the user's program
  tangent       pi, dpi | bc, dbc           tangent part of jax.jvp(base item)              dpi, dbc (trailing operands)
  adjoint       pi, cy  | bc, ct            jax.vjp(base item) (replay adjoint)             cy, ct   (trailing operands)
  dual_lanes    pi, bc (some with T lanes)  jax.vmap of a linear node's item over T dual lanes (batched linear operands)
  folded_batch  pi, b | bc (some [B, ...])  a batch folded into the item axis: B*N items, member b's emits go to
                                            [b*S, (b+1)*S)
  fused_jvp     pi, dpi | bc, dbc           jax.jvp(item): primal + tangent in one kernel (primitive tallymap_jvp;
                                            split into primal + tangent wherever JAX partially evaluates it)

  jvp rule     (any node): primal = the same node; tangent = a `tangent` node of node.prog that takes the PRIMAL
               operands as inputs (residuals = operands, O(1) memory per item; linearize/grad need no partial-eval
               rule: JAX's default partial evaluation keeps the primal node known and stages the tangent node).
  transpose    tangent -> adjoint of the base (vjp_prog: replay adjoint); adjoint -> tangent of the base; dual_lanes
               (or anything else linear) -> the generic vjp of its own item.
  batching     only linear operands batched (jacfwd, jacrev, vmap of jvp/vjp tangents; for a fused_jvp node: only its
               tangents) -> `dual_lanes`: T dual lanes per item, the primal computed once; on cuda in chunks of at
               most MAX_LANES (16) lanes whose lane-dependent loop arrays stay < LANES_MAX_LOCAL (512) elements, one
               launch per chunk.  Otherwise, when no spawn / exact outputs, every batched broadcast operand is small
               (<= FOLD_MAX_BC elements per member) and (cuda) no tally leaves shared memory -> `folded_batch`: ONE
               launch over B*N items; anything else -> lax.map over the batch (one launch per batch member).
  lowering     per PLATFORM: on 'cuda' the cuda backend (backend='auto'|'cuda'), elsewhere the lockstep backend.
  impl         eager calls go through a jax.jit cached on the node's structure.

Registries used (the same ones emit_p uses): Primitive, ad.primitive_jvps, ad.primitive_transposes,
batching.fancy_primitive_batchers (through _compat.register_batcher), mlir.register_lowering + mlir.lower_fun; the
fused jvp alone also uses the two legacy partial-eval registries (feature-detected, see below).  No remat / dce rules.
"""
import collections
import dataclasses
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.core import ShapedArray
from jax.extend.core import Primitive, jaxpr_as_fun
from jax.interpreters import ad, mlir
from jax.interpreters import partial_eval as _pe

from . import _foreign
from ._api import _WARNED, ItemProgram, UnsupportedError, _mk, _sds, jvp_prog
from ._compat import NM, drop_var, register_batcher
from ._reverse import vjp_prog
from ._rewrite import emit_p_bind, isfloat, retrace

tallymap_call_p = Primitive("tallymap_call")
tallymap_call_p.multiple_results = True
# the fused jvp (primal + tangent outputs from ONE kernel).  A separate primitive so that only it carries the
# two partial-eval rules that split it into a known primal call + a staged linear call (see _fused_pe below).
tallymap_jvp_p = Primitive("tallymap_jvp")
tallymap_jvp_p.multiple_results = True


# ------------------------------------------------------------------------------------------- nodes
@dataclasses.dataclass(frozen=True, eq=False)
class Node:
    prog: ItemProgram
    backend: str
    opts: tuple                 # sorted (name, value) backend options
    kind: str = "primal"        # 'primal' | 'tangent' | 'adjoint' | 'dual_lanes' | 'folded_batch' | 'fused_jvp'
    base: Any = None            # tangent / adjoint: the ItemProgram they were derived from
    pp: tuple = ()              # tangent / adjoint: which per-item operands of `base` carry tangents / get cotangents
    pb: tuple = ()              # tangent / adjoint: same for the broadcast operands
    lin_pi: tuple = ()          # which of this node's per-item operands it is linear in (empty: not linear)
    lin_bc: tuple = ()
    site: Any = None            # (file, line) of the user's call, for warnings raised when the program is lowered

    def __repr__(self):         # shows up in jaxprs (make_jaxpr): keep it short
        return f"tallymap:{self.kind}[{self.prog.tag}|{self.backend}]"

    def derived(self, prog, kind, **kw):
        opts = kw.pop("opts", self.opts)
        kw.setdefault("site", self.site)
        return Node(prog, self.backend, opts, kind, **kw)

    @property
    def linear(self):
        return any(self.lin_pi) or any(self.lin_bc)

    def key(self):
        """structural identity: equal keys -> the same compiled program (used for the eager-call cache).  Host
        callbacks enter by identity (the jaxpr text only names them; a cached entry keeps its objects alive)."""
        k = self.__dict__.get("_key")
        if k is None:
            from ._jaxpr import host_callbacks
            p = self.prog
            k = (str(p.cj.jaxpr), tuple(id(c) for c in p.cj.consts), tuple(id(c) for c in host_callbacks(p.cj.jaxpr)),
                 tuple((s.S, str(np.dtype(s.dtype)), tuple(s.lead), s.exact, s.scale, s.kind) for s in p.tallies),
                 p.nP, p.nB, p.trunc, None if p.spawn is None else int(p.spawn.capacity), self.backend, self.opts)
            object.__setattr__(self, "_key", k)
        return k


def bind_call(*args, node):
    """bind tallymap_call_p.  Inside shard_map (jax >= 0.7, vma checking on) operands that are invariant over a manual
    axis some other operand varies over are first cast to varying (lax.pvary / lax.pcast), exactly as JAX does for its
    own n-ary primitives; the cast's transpose (psum) then sums the per-device cotangents of those operands."""
    vs = [_varying(_aval(a)) for a in args]
    if any(vs):
        u = frozenset().union(*vs)
        args = [_cast_varying(a, tuple(sorted(u - v, key=str))) if u - v else a for a, v in zip(args, vs)]
    return _prim_of(node).bind(*args, node=node)


def _prim_of(node):
    return tallymap_jvp_p if node.kind == "fused_jvp" else tallymap_call_p


def _cast_varying(x, axes):
    if hasattr(lax, "pcast"):                 # jax >= 0.9
        return lax.pcast(x, axes, to="varying")
    return lax.pvary(x, axes)                 # jax 0.7 - 0.8


def make_node(prog, backend, opts, site=None):
    return Node(prog, backend, tuple(sorted(opts.items())), site=site)


def _split(node, args):
    n = node.prog.nP
    return list(args[:n]), list(args[n:])


def _out_specs(p):
    return [k for k, s in enumerate(p.tallies) if s.kind == "sum"]


_aval = jax.typeof


def _zero(aval):
    return ad.Zero(aval.to_tangent_aval())


def _is_zero(t):
    return t is None or type(t) is ad.Zero


# ------------------------------------------------------------------------------------------- abstract eval
def _abstract(*args, node):
    p = node.prog
    if p.nP and len(args[0].shape) == 0:
        raise TypeError("tallymap_call: per-item operands need a leading item axis")
    N = args[0].shape[0]
    outs = [ShapedArray(tuple(p.tallies[k].lead) + (p.tallies[k].S,), p.tallies[k].dtype) for k in _out_specs(p)]
    outs.append(ShapedArray((5,), jnp.int32))
    outs += [ShapedArray((N,) + tuple(a.shape), a.dtype) for a in p.y_avals()]
    vs = [_varying(a) for a in args]
    if any(vs):                  # inside shard_map (jax >= 0.7): the result varies over every axis an operand does
        vset = frozenset().union(*vs)
        src = args[[len(v) for v in vs].index(max(len(v) for v in vs))]
        outs = [_with_varying(o, vset, src) for o in outs]
    return outs


tallymap_call_p.def_abstract_eval(_abstract)
tallymap_jvp_p.def_abstract_eval(_abstract)


# ------------------------------------------------------------------------------------------- lowering / impl
def _run(node, args, platform):
    from .backends import lockstep as LB
    p = node.prog
    pi, bc = _split(node, args)
    opts = dict(node.opts)
    fallback = bool(opts.pop("allow_fallback", False))
    b = node.backend
    if b == "auto" and platform != "cuda":
        _warn_once("tallymap: no CUDA GPU on this platform; running the lockstep backend (plain JAX)", node.site)
    if b == "lockstep" or (b == "auto" and platform != "cuda"):
        return list(LB.run(p, pi, bc, **opts))
    if b not in ("cuda", "auto"):
        raise ValueError(f"tallymap: unknown backend {b!r}")
    if platform != "cuda":
        raise UnsupportedError("tallymap: backend='cuda' needs a CUDA GPU, and there is none on this platform "
                               f"(this computation is lowered for {platform!r}); pass backend='lockstep' to run "
                               "the same program in plain JAX, or run it on a CUDA GPU")
    from .backends import cuda as CB
    try:
        return list(CB.run(p, pi, bc, **opts))
    except UnsupportedError as e:
        if b == "cuda":
            raise
        if not fallback:
            why = _lockstep_cannot_run(p)
            if why is not None:          # suggesting a fallback that cannot run either would only waste a run
                raise UnsupportedError(f"{e}\ntallymap: the lockstep backend cannot run this program either "
                                       f"({why}), so allow_fallback=True would not help") from None
            # (C1) never a SILENT ~100x slowdown on a GPU: the lockstep backend is the reference interpreter, not a
            # fast path (measured: a 7 ms cuda program took 678 ms on lockstep)
            raise UnsupportedError(f"{e}\ntallymap: backend='auto' on a GPU does not fall back to the lockstep backend "
                                   "(~100x slower): pass allow_fallback=True to accept the fallback, or "
                                   "backend='lockstep'") from None
        _warn_once(f"tallymap: cuda backend cannot run this program ({e}); falling back to the lockstep backend "
                   "(allow_fallback=True)", node.site)
        return list(LB.run(p, pi, bc, **opts))


def _lockstep_cannot_run(p):
    """-> why the lockstep backend cannot run this item program either, or None.  The lockstep interpreter runs
    every JAX primitive, with two exceptions: side effects, and exact=True outputs without jax_enable_x64."""
    from ._jaxpr import subjaxprs

    def walk(jaxpr):
        for e in jaxpr.eqns:
            if e.effects:
                return (f"the item function has side effects ({', '.join(sorted(str(x) for x in e.effects))}); "
                        "effects are not supported inside tallymap.vmap on any backend")
            for sj in subjaxprs(e):
                why = walk(sj.jaxpr)
                if why is not None:
                    return why
        return None
    why = walk(p.cj.jaxpr)
    if why is not None:
        return why
    if any(s.exact for s in p.tallies) and not jax.config.jax_enable_x64:
        return "exact=True outputs need jax_enable_x64 on the lockstep backend"
    return None


def _warn_once(msg, site=None):
    """Warn once per message, attributed to the user's call site (Node.site) when it is known: the warning is
    raised while the program is LOWERED, where the python stack is all JAX internals."""
    if msg in _WARNED:
        return
    _WARNED.add(msg)
    if site is None:
        warnings.warn(msg, stacklevel=4)
    else:
        warnings.warn_explicit(msg, UserWarning, site[0], site[1])


# shard_map's "varying manual axes" type (jax >= 0.7: aval.vma; >= 0.11: aval.manual_axis_type.varying).  The
# result of an tallymap call varies over every manual axis any operand varies over; the backends' internal ops are not
# vma-aware, so the lowering traces them with plain avals (the lowered program does not depend on the type).
def _varying(a):
    v = getattr(a, "vma", None)
    if v is None:
        v = getattr(getattr(a, "manual_axis_type", None), "varying", None)
    return frozenset(v or ())


def _with_varying(o, vset, src):
    if hasattr(src, "manual_axis_type"):                         # jax >= 0.11
        from jax.sharding import NamedSharding, PartitionSpec
        sh = NamedSharding(src.sharding.mesh, PartitionSpec(*([None] * len(o.shape))))
        return ShapedArray(o.shape, o.dtype, sharding=sh, manual_axis_type=type(src.manual_axis_type)(varying=vset))
    return o.update(vma=vset)                                    # jax 0.7 - 0.10


def _plain(a):
    if not _varying(a):
        return a
    if hasattr(a, "manual_axis_type"):
        return a.update(sharding=None, manual_axis_type=type(a.manual_axis_type)())
    return a.update(vma=frozenset())


def _lowering(platform):
    def f(*args, node):
        return _run(node, args, platform)
    low = mlir.lower_fun(f, multiple_results=True)

    def rule(ctx, *args, node):
        if any(_varying(a) for a in list(ctx.avals_in) + list(ctx.avals_out)):
            ctx = ctx.replace(avals_in=[_plain(a) for a in ctx.avals_in], avals_out=[_plain(a) for a in ctx.avals_out])
        return low(ctx, *args, node=node)
    return rule


for _p in (tallymap_call_p, tallymap_jvp_p):
    mlir.register_lowering(_p, _lowering("other"))
    mlir.register_lowering(_p, _lowering("cuda"), platform="cuda")

_EAGER = collections.OrderedDict()
EAGER_CACHE = 64


def _impl(*args, node):
    if jax.config.jax_disable_jit or jax.config.jax_debug_nans or jax.config.jax_debug_infs:
        # disable_jit: jit would re-enter this impl; debug_nans/infs: JAX re-runs a failing jit op by op, and that
        # re-run must not call the same cached jit again (RecursionError) -> run the backend op by op
        plat = "cuda" if jax.default_backend() == "gpu" else jax.default_backend()
        return _run(node, args, plat)
    k = node.key()
    f = _EAGER.get(k)
    if f is None:
        f = jax.jit(lambda *a: _prim_of(node).bind(*a, node=node))
        _EAGER[k] = f
        while len(_EAGER) > EAGER_CACHE:
            _EAGER.popitem(last=False)
    else:
        _EAGER.move_to_end(k)
    return f(*args)


tallymap_call_p.def_impl(_impl)
tallymap_jvp_p.def_impl(_impl)


# ------------------------------------------------------------------------------------------- derived programs
def tangent_prog(p, pp, pb):
    """tangent part of jax.jvp(item) w.r.t. the per-item / broadcast operands flagged in pp / pb.
    invars [t (primal tallies, discarded), t'] [pi, dpi] [bc, dbc] -> outvars [t, t'] + [dy (float results)]."""
    q = jvp_prog(p, pp, pb)
    nT = p.nT
    ntf = q.nT - nT
    nY = p.nY
    f = jaxpr_as_fun(q.cj)

    def L(*a):
        o = f(*a)
        return list(o[:nT + ntf]) + list(o[nT + ntf + nY:])
    cj = _mk(L, [_sds(a) for a in q.in_avals()])
    tl = tuple(dataclasses.replace(s, kind="discard") if (k < nT and s.kind == "sum") else s
               for k, s in enumerate(q.tallies))
    return ItemProgram(cj, tl, q.nP, q.nB, None, p.spawn, p.tag + "+tangent", where=p.where)


def dual_lanes_prog(p, T, bpi, bbc, shared=(0, 0)):
    """jax.vmap the item of a LINEAR program over T lanes of its batched (linear) operands: every tally gets a
    leading lane dim, every per-item result a lane axis after the item axis.  Control flow is unbatched (the
    lanes are tangents / cotangents), so this is one kernel with T dual lanes per item.
    shared = (nT0, nY0): the first nT0 tallies / nY0 results are lane-INVARIANT (a fused jvp program, jvp_prog of a
    program with nT0 tallies and nY0 results: its primal part) and computed once per item in the same kernel
    (jacfwd = ONE launch, primal included)."""
    f = jaxpr_as_fun(p.cj)
    lt = [s.kind != "spawn" and k >= shared[0] for k, s in enumerate(p.tallies)]
    ly = [j >= shared[1] for j in range(p.nY)]
    in_axes = [0 if l else None for l in lt] + [0 if b else None for b in bpi] + [0 if b else None for b in bbc]
    av = p.in_avals()
    avals = [jax.ShapeDtypeStruct((T,) + tuple(a.shape), a.dtype) if ax == 0 else _sds(a) for a, ax in zip(av, in_axes)]
    out_axes = [0 if l else None for l in lt] + [0 if l else None for l in ly]
    vf = jax.vmap(lambda *a: list(f(*a)), in_axes=tuple(in_axes), out_axes=out_axes, axis_size=T)
    cj = _mk(vf, avals)
    tl = tuple(dataclasses.replace(s, lead=(T,) + tuple(s.lead)) if l else s for s, l in zip(p.tallies, lt))
    return dataclasses.replace(p, cj=cj, tallies=tl, tag=p.tag + f"+dual_lanes{T}"), lt, ly


# ------------------------------------------------------------------------------------------- jvp
def _jvp(primals, tangents, *, node):
    p = node.prog
    pi, bc = _split(node, primals)
    dpi, dbc = _split(node, tangents)
    pp = [isfloat(x.dtype) and not _is_zero(t) for x, t in zip(pi, dpi)]
    pb = [isfloat(x.dtype) and not _is_zero(t) for x, t in zip(bc, dbc)]
    if not any(pp) and not any(pb):
        outs = bind_call(*primals, node=node)
        return outs, [_zero(_aval(o)) for o in outs]
    if p.spawn is not None:            # spawned children carry tangents for every float per-item leaf
        pp = [isfloat(x.dtype) for x in pi]
    pp, pb = tuple(pp), tuple(pb)
    dl = [ad.instantiate_zeros(t) if not _is_zero(t) else jnp.zeros(x.shape, x.dtype)
          for x, t, f in zip(pi, dpi, pp) if f]
    db = [ad.instantiate_zeros(t) for t, f in zip(dbc, pb) if f]
    if FUSE_JVP:
        jn = _fused_node(node, pp, pb)
        o = bind_call(*pi, *dl, *bc, *db, node=jn)
        prim_idx, tan_idx = _fused_layout(p)
        outs = [o[j] for j in prim_idx]
        touts = [o[j] for j in tan_idx]
        return outs, _tangents_out(p, outs, touts[:_ntf(p)] + [None] + touts[_ntf(p):])
    outs = bind_call(*primals, node=node)
    touts = bind_call(*pi, *dl, *bc, *db, node=_tangent_node(node, pp, pb))
    return outs, _tangents_out(p, outs, touts)


def _tangent_node(node, pp, pb):
    p = node.prog
    q = tangent_prog(p, pp, pb)
    return node.derived(q, "tangent", base=p, pp=pp, pb=pb,
                        lin_pi=(False,) * p.nP + (True,) * sum(pp), lin_bc=(False,) * p.nB + (True,) * sum(pb))


def _tangents_out(p, outs, touts):
    """map a tangent node's outputs ([float sum tallies'] + [stats] + [float results']) onto the primal outputs."""
    ot = _out_specs(p)
    nt = len(ot)
    ntf = sum(1 for k in ot if isfloat(p.tallies[k].dtype))
    dt_it, dy_it = iter(touts[:ntf]), iter(touts[ntf + 1:])
    res = []
    for j, k in enumerate(ot):
        res.append(next(dt_it) if isfloat(p.tallies[k].dtype) else _zero(_aval(outs[j])))
    res.append(_zero(_aval(outs[nt])))
    for o, a in zip(outs[nt + 1:], p.y_avals()):
        res.append(next(dy_it) if isfloat(a.dtype) else _zero(_aval(o)))
    return res


ad.primitive_jvps[tallymap_call_p] = _jvp


# ------------------------------------------------------------------------------------------- transpose
def _inst(c, shape, dtype):
    return jnp.zeros(shape, dtype) if _is_zero(c) else c


def _float_slots(p):
    ot = _out_specs(p)
    tf = [j for j, k in enumerate(ot) if isfloat(p.tallies[k].dtype)]       # positions among the tally outputs
    yf = [j for j, a in enumerate(p.y_avals()) if isfloat(a.dtype)]
    return ot, tf, yf


def _val0(x):
    return jnp.zeros(x.aval.shape, x.aval.dtype) if ad.is_undefined_primal(x) else x


def _adj_opts(opts):
    """the adjoint kernel's options = the forward's (identical math / contraction policy).  Private _adj_math
    overrides the adjoint's cuda_math -- NEGATIVE CONTROL of the replay check only."""
    m = dict(opts).get("_adj_math")
    return opts if not m else tuple(sorted(dict(opts, cuda_math=m).items()))


def _vjp_call(node, base, pi, bc, U_pi, U_bc, cts, N):
    """cotangents of base's operands flagged U_pi / U_bc, given cotangents `cts` of ALL base outputs."""
    if base.spawn is not None:          # reverse through spawn: the tree adjoint (one adjoint launch per generation)
        from ._reverse_spawn import tree_vjp
        return tree_vjp(node, base, pi, bc, U_pi, U_bc, cts, N)
    ot, tf, yf = _float_slots(base)
    nt = len(ot)
    ya = base.y_avals()
    ct_t = []
    for j in tf:
        s = base.tallies[ot[j]]
        ct_t.append(_inst(cts[j], tuple(s.lead) + (s.S,), s.dtype))
    ct_y = [_inst(cts[nt + 1 + j], (N,) + tuple(ya[j].shape), ya[j].dtype) for j in yf]
    q = vjp_prog(base, tuple(U_pi), tuple(U_bc), dict(node.opts))   # (RV: rev_* options)
    fp = []
    if q.nP > base.nP + len(yf):
        # (RV) replay consistency: the FORWARD kernel's per-item fingerprint (the same base program and options as
        # the primal call, so XLA may CSE the two) is compared inside the adjoint with its own replay
        fo = bind_call(*[_val0(x) for x in pi], *[_val0(x) for x in bc], node=node.derived(base, "primal"))
        fp = [jax.lax.stop_gradient(fo[len(ot) + 1 + base.fp])]
    tn = node.derived(q, "adjoint", base=base, pp=tuple(U_pi), pb=tuple(U_bc),
                      lin_pi=(False,) * base.nP + (True,) * len(yf) + (False,) * len(fp),
                      lin_bc=(False,) * base.nB + (True,) * len(tf), opts=_adj_opts(node.opts))
    outs = bind_call(*pi, *ct_y, *fp, *bc, *ct_t, node=tn)
    if fp:
        from . import _reverse
        outs = _reverse.check_replay(outs[-1], outs[:-1], bool(dict(node.opts).get("rev_stats")))
    nib = sum(U_bc)
    g_bc_it, g_pi_it = iter(outs[:nib]), iter(outs[nib + 1:])
    g_pi = [next(g_pi_it) if u else None for u in U_pi]
    g_bc = [next(g_bc_it).reshape(x.aval.shape if ad.is_undefined_primal(x) else jnp.shape(x)) if u else None
            for x, u in zip(bc, U_bc)]
    return g_pi, g_bc


def _transpose(cts, *args, node):
    p = node.prog
    U = [ad.is_undefined_primal(a) for a in args]
    lin = list(node.lin_pi) + list(node.lin_bc)
    if not node.linear or any(u and not l for u, l in zip(U, lin)):
        raise NotImplementedError("tallymap: transposing a tallymap call w.r.t. an operand it is not linear in "
                                  "(this is an internal error: please report it)")
    if all(_is_zero(c) for c in cts) or not any(U):
        return [None] * len(args)
    pi, bc = _split(node, args)
    N = (pi[0].aval.shape if ad.is_undefined_primal(pi[0]) else jnp.shape(pi[0]))[0]
    if node.kind == "tangent":
        # tangent map of `base` -> its transpose is the replay adjoint (vjp) of base w.r.t. the undefined tangents
        base = node.base
        nPb, nBb = base.nP, base.nB
        ip = [i for i, f in enumerate(node.pp) if f]
        ib = [i for i, f in enumerate(node.pb) if f]
        U_pi = [False] * nPb
        U_bc = [False] * nBb
        for k, i in enumerate(ip):
            U_pi[i] = ad.is_undefined_primal(pi[nPb + k])
        for k, i in enumerate(ib):
            U_bc[i] = ad.is_undefined_primal(bc[nBb + k])
        # tangent outputs = [float tallies'] + [stats] + [float results']: re-expand to all base outputs
        ot, tf, yf = _float_slots(base)
        full = [None] * len(ot) + [None] + [None] * base.nY
        for c, j in zip(cts[:len(tf)], tf):
            full[j] = c
        for c, j in zip(cts[len(tf) + 1:], yf):
            full[len(ot) + 1 + j] = c
        g_pi, g_bc = _vjp_call(node, base, pi[:nPb], bc[:nBb], U_pi, U_bc, full, N)
        res_pi = [None] * nPb + [g_pi[i] for i in ip]
        res_bc = [None] * nBb + [g_bc[i] for i in ib]
        return res_pi + res_bc
    if node.kind == "adjoint":
        # adjoint of `base` -> its transpose is the tangent map of base, fed with the cotangents of the adjoint's
        # outputs
        base = node.base
        nPb, nBb = base.nP, base.nB
        ip = [i for i, f in enumerate(node.pp) if f]
        ib = [i for i, f in enumerate(node.pb) if f]
        nib = len(ib)
        dpi = [_inst(c, (N,) + tuple(base.in_avals()[base.nT + i].shape), base.in_avals()[base.nT + i].dtype)
               for c, i in zip(cts[nib + 1:], ip)]
        dbc = []
        for c, i in zip(cts[:nib], ib):
            a = base.in_avals()[base.nT + nPb + i]
            dbc.append(_inst(c, (int(np.prod(a.shape, dtype=np.int64)),), a.dtype).reshape(a.shape))
        q = tangent_prog(base, node.pp, node.pb)
        ln = node.derived(q, "tangent", base=base, pp=node.pp, pb=node.pb,
                          lin_pi=(False,) * nPb + (True,) * len(ip), lin_bc=(False,) * nBb + (True,) * nib)
        touts = bind_call(*pi[:nPb], *dpi, *bc[:nBb], *dbc, node=ln)
        ot, tf, yf = _float_slots(base)
        g_ct = list(touts[:len(tf)])                      # -> cotangents of the adjoint node's tally-ct operands
        g_cy = list(touts[len(tf) + 1:])                  # -> cotangents of its per-item cy operands
        res_pi = [None] * nPb + [g if ad.is_undefined_primal(x) else None for g, x in zip(g_cy, pi[nPb:])]
        res_pi += [None] * (len(pi) - len(res_pi))               # (RV) the forward fingerprint operand
        res_bc = [None] * nBb + [g if ad.is_undefined_primal(x) else None for g, x in zip(g_ct, bc[nBb:])]
        return res_pi + res_bc
    # any other linear node (lanes): the generic replay adjoint of its OWN item (linear in the undefined operands,
    # so the point at which they are evaluated does not matter: zeros)
    val = lambda x: jnp.zeros(x.aval.shape, x.aval.dtype) if ad.is_undefined_primal(x) else x
    U_pi, U_bc = U[:p.nP], U[p.nP:]
    g_pi, g_bc = _vjp_call(node, p, [val(x) for x in pi], [val(x) for x in bc], U_pi, U_bc, list(cts), N)
    return g_pi + g_bc


ad.primitive_transposes[tallymap_call_p] = _transpose


# ------------------------------------------------------------------------------------------- batching
def _batch(args, dims, *, node):
    r = _dual_lanes(args, dims, node, fused=False)
    if r is not None:
        return r
    bat = [d is not NM for d in dims]
    size = next(a.shape[d] for a, d in zip(args, dims) if d is not NM)
    if _can_fold(node, args, dims, bat, size):
        return _fold_batch(node, args, dims, bat, size)
    # general batching: one launch per batch member (full SIMT divergence per member, no per-lane tally selects)
    idx = [k for k, b in enumerate(bat) if b]
    xs = [jnp.moveaxis(args[k], dims[k], 0) for k in idx]

    def one(sl):
        a = list(args)
        for k, v in zip(idx, sl):
            a[k] = v
        return bind_call(*a, node=node)
    outs = lax.map(one, xs)
    return list(outs), [0] * len(outs)


def _dual_lanes(args, dims, node, fused):
    """dual lanes: when only operands the node is LINEAR in are batched (tangent / cotangent lanes: jacfwd, jacrev,
    vmap of jvp / vjp), ONE kernel with T lanes per item.  fused: a fused jvp node whose primal operands are
    unbatched -> one kernel computing the primal once and T tangent lanes.  -> (outs, out dims) or None."""
    p = node.prog
    nP = p.nP
    if fused:
        fwd, pp, pb, chain = node.base
        lin_pi = (False,) * fwd.prog.nP + (True,) * sum(pp)
        lin_bc = (False,) * fwd.prog.nB + (True,) * sum(pb)
    else:
        lin_pi, lin_bc = node.lin_pi, node.lin_bc
    bat = [d is not NM for d in dims]
    lin = list(lin_pi) + list(lin_bc)
    mode = dict(node.opts).get("_lanes", "auto")
    if not any(lin) or any(b and not l for b, l in zip(bat, lin)) or mode == "off":
        return None
    size = next(a.shape[d] for a, d in zip(args, dims) if d is not NM)
    bpi, bbc = tuple(bat[:nP]), tuple(bat[nP:])
    if p.spawn is not None:     # spawned children carry per-lane tangents: every linear per-item leaf is laned
        bpi = tuple(b or l for b, l in zip(bpi, lin_pi))
    shared = (fwd.prog.nT, fwd.prog.nY) if fused else (0, 0)
    gpu = mode == "auto" and _gpu(node)
    T = min(size, mode) if isinstance(mode, int) else min(size, MAX_LANES) if gpu else size
    q, lt, ly = dual_lanes_prog(p, T, bpi, bbc, shared)
    if _foreign.lanes_unsafe(q.cj):     # a batched foreign primitive: one lane at a time (see lanes_unsafe)
        return None
    if gpu:
        # see LANES_MAX_LOCAL: the cuda kernel keeps T copies of every lane-dependent loop array in local memory.
        # Too big -> chunks of Tc < T lanes (the lane state is linear in T), one launch per chunk.
        m = _taint_max(q.cj.jaxpr, list(lt) + list(bpi) + list(bbc), False)[0]
        if m >= LANES_MAX_LOCAL:
            T = min(T - 1, (LANES_MAX_LOCAL - 1) * T // m)
            if T < 2:
                return None
            q, lt, ly = dual_lanes_prog(p, T, bpi, bbc, shared)
            if _taint_max(q.cj.jaxpr, list(lt) + list(bpi) + list(bbc), False)[0] >= LANES_MAX_LOCAL:
                return None
    if fused:
        ln = node.derived(q, "fused_jvp", base=(fwd, pp, pb, chain + ((T, bpi, bbc),)), pp=pp, pb=pb)
    else:
        ln = node.derived(q, "dual_lanes", lin_pi=lin_pi, lin_bc=lin_bc)
    nC = -(-size // T)                                   # chunks of T lanes (the last one zero-padded)

    def lanes_of(x, ax):                                 # [.., size, ..] (lane axis ax) -> [nC, .., T, ..]
        if nC == 1:
            return x
        x = jnp.pad(x, [(0, nC * T - size) if i == ax else (0, 0) for i in range(x.ndim)])
        x = x.reshape(x.shape[:ax] + (nC, T) + x.shape[ax + 1:])
        return jnp.moveaxis(x, ax, 0)
    a2 = [lanes_of(jnp.moveaxis(x, d, 1), 1) if b and k < nP else lanes_of(jnp.moveaxis(x, d, 0), 0) if b else
          (jnp.broadcast_to(jnp.expand_dims(x, 1), (x.shape[0], T) + tuple(x.shape[1:])) if k < nP and bpi[k] else x)
          for k, (x, d, b) in enumerate(zip(args, dims, bat))]
    ot = _out_specs(p)
    od = [0 if lt[k] else NM for k in ot] + [NM] + [1 if l else NM for l in ly]
    if nC == 1:
        return bind_call(*a2, node=ln), od
    ch = [k for k, b in enumerate(bat) if b]

    def one(xs):
        a3 = list(a2)
        for k, x in zip(ch, xs):
            a3[k] = x
        return bind_call(*a3, node=ln)
    outs = lax.map(one, [a2[k] for k in ch])
    res = []                                             # [nC, ..] -> the T-lane layout, lanes [:size]
    for o, d in zip(outs, od):
        if d is NM:
            res.append(o[0])                             # lane-invariant (primal, stats): identical in every chunk
        elif d == 0:
            res.append(o.reshape((nC * T,) + o.shape[2:])[:size])
        else:
            o = jnp.moveaxis(o, 0, 1)
            res.append(o.reshape((o.shape[0], nC * T) + o.shape[3:])[:, :size])
    return res, od


register_batcher(tallymap_call_p, _batch)


# Lanes vmap the item: a dense intermediate of S elements inside a loop becomes [T, S].  The cuda backend keeps such
# arrays in per-thread local memory; measured (L40S, 65k items): jacfwd(grad) of a walk gathering a 256-entry table
# (its replay adjoint scatters the table cotangent into a dense [256] array every step) -> lanes kernel with [2, 256]
# arrays: 6 KB stack frame, 95 ms vs 2.3 ms for one lane.  A particle-shower item with 129 lane-dependent elements
# per lane (L40S): T lanes in one kernel cost 1.34 / 1.99 / 2.27 / 2.57 x fwd for T = 1..4 but 14 / 23 / 32 x for
# T = 6 / 8 / 10
# (one launch per column: 2.2 + 1.2 (T-1) x).  On cuda the lane-dependent loop state is kept below LANES_MAX_LOCAL
# elements by splitting the lanes into chunks (one launch each); a single lane over the limit -> fold / lax.map.
LANES_MAX_LOCAL = 512
# ... and at most MAX_LANES lanes per kernel (the guard sees the largest lane array, not the sum of many small lane
# values).  Measured (L40S, toy walk 1M items, jacrev of 256 outputs = 256 cotangent lanes of the adjoint): one
# kernel per chunk of 4 / 8 / 16 / 32 / 64 lanes 229 / 191 / 175 / 176 / 325 ms, all 256 at once 4.3 s, fold 533 ms;
# hessian 8x8 (8 lanes) 83 ms vs fold 102 ms.  A photon-transport jacfwd of 10 parameters stays one kernel.
MAX_LANES = 16


def _gpu(node):
    return node.backend == "cuda" or (node.backend == "auto" and jax.default_backend() == "gpu")


def _taint_max(jaxpr, tin, in_loop):
    """(max size of a tainted first-order loop intermediate, taint of the outvars)."""
    from jax.extend.core import Literal

    from ._jaxpr import call_jaxpr, cond_branches, scan_parts, subjaxprs, while_parts
    tv = {v: t for v, t in zip(jaxpr.invars, tin)}
    m = 0
    for e in jaxpr.eqns:
        ti = [False if isinstance(v, Literal) else tv.get(v, False) for v in e.invars]
        name = e.primitive.name
        to = None
        if name == "while":
            w = while_parts(e)
            nc, nb = w.nc, w.nb
            car = ti[nc + nb:]
            for _ in range(len(car) + 1):
                mb, ob = _taint_max(w.body.jaxpr, ti[nc:nc + nb] + car, True)
                new = [a or b for a, b in zip(car, ob)]
                if new == car:
                    break
                car = new
            m = max(m, mb)
            to = car
        elif name == "scan":
            sc = scan_parts(e)
            nc, nk = sc.nconsts, sc.ncarry
            car = ti[nc:nc + nk]
            for _ in range(len(car) + 1):
                mb, ob = _taint_max(sc.body.jaxpr, ti[:nc] + car + ti[nc + nk:], True)
                new = [a or b for a, b in zip(car, ob[:nk])]
                if new == car:
                    break
                car = new
            m = max(m, mb)
            to = car + ob[nk:]
        elif name == "cond":
            outs = [_taint_max(b.jaxpr, ti[1:], in_loop) for b in cond_branches(e)]
            m = max([m] + [o[0] for o in outs])
            to = [any(o[1][j] for o in outs) or ti[0] for j in range(len(e.outvars))]
        elif call_jaxpr(e) is not None:
            mb, to = _taint_max(call_jaxpr(e).jaxpr, ti, in_loop)
            m = max(m, mb)
        elif subjaxprs(e):
            to = [any(ti)] * len(e.outvars)
            if any(ti):
                m = max([m] + [_largest_loop_array(sj.jaxpr, in_loop) for sj in subjaxprs(e)])
        else:
            to = [any(ti)] * len(e.outvars)
            if in_loop and any(ti) and name != "tallymap_emit":
                for v in e.outvars:
                    sh = getattr(v.aval, "shape", ())
                    m = max(m, int(np.prod(sh, dtype=np.int64)) if sh else 1)
        for v, t in zip(e.outvars, to):
            tv[v] = t
    return m, [False if isinstance(v, Literal) else tv.get(v, False) for v in jaxpr.outvars]


def _largest_loop_array(jaxpr, in_loop=False):
    from ._jaxpr import subjaxprs
    m = 0
    for e in jaxpr.eqns:
        subs = subjaxprs(e)
        if in_loop and not subs and e.primitive.name != "tallymap_emit":     # first-order, non-tally intermediates
            for v in e.outvars:
                sh = getattr(v.aval, "shape", ())
                m = max(m, int(np.prod(sh, dtype=np.int64)) if sh else 1)
        loop = in_loop or e.primitive.name in ("while", "scan")
        for sj in subs:
            m = max(m, _largest_loop_array(sj.jaxpr, loop))
    return m


# ------------------------------------------------------------------------------------------- fold a batch into items
FOLD_MAX_BC = 4096      # largest batched broadcast operand (elements per batch member) indexed per item when folding


def _can_fold(node, args, dims, bat, size):
    p = node.prog
    if dict(node.opts).get("_fold", True) is False or p.spawn is not None:
        return False            # spawn: one shared queue per launch would change capacity semantics
    if any(s.exact for s in p.tallies if s.kind == "sum"):
        return False            # stats['exact_overflow'] is per launch, not per batch member
    for k in range(p.nP, len(args)):
        if bat[k] and int(np.prod(args[k].shape, dtype=np.int64)) // max(size, 1) > FOLD_MAX_BC:
            return False        # a large batched table would be indexed per item (lockstep backend: gathered per lane)
    if _gpu(node):
        # the cuda kernel privatises small tallies in shared memory; a fold that pushes a tally out of shared memory
        # turns block-local atomics into contended global ones (measured, L40S, 1M items, 256 bins, B=8: fold 21 ms
        # vs lax.map 4.6 ms).  Fold only when every privatised tally stays privatised (the kernel's own rule).
        from .backends.cuda.kernel import privatised
        if privatised(p.tallies) != privatised(p.tallies, B=size):
            return False
    return True


def fold_prog(p, B, bbc):
    """the batch folded into the item axis: per-item operands + member index b; batched broadcast operands keep
    their [B, ...] axis and are indexed with b inside the item; tallies are B times longer and every emit of member b
    lands in [b*S, (b+1)*S) (out-of-range indices stay dropped)."""
    nT, nP = p.nT, p.nP

    def em(t, idx, val):
        S = t.shape[-1] // B
        b = em.b
        i = jnp.asarray(idx).astype(jnp.int32)
        return emit_p_bind(t, jnp.where((i >= 0) & (i < S), i + b * S, -1), val)

    def item(*a):
        t, pi, b, bc = a[:nT], a[nT:nT + nP], a[nT + nP], list(a[nT + nP + 1:])
        bc = [lax.dynamic_index_in_dim(x, b, 0, keepdims=False) if f else x for x, f in zip(bc, bbc)]
        em.b = b
        return retrace(p.cj, n_tally=nT, emit=em)(*t, *pi, *bc)
    av = p.in_avals()
    tav = [jax.ShapeDtypeStruct(tuple(s.lead) + (B * s.S,), a.dtype) if s.kind != "spawn" else _sds(a)
           for s, a in zip(p.tallies, av[:nT])]
    pav = [_sds(a) for a in av[nT:nT + nP]]
    bav = [jax.ShapeDtypeStruct((B,) + tuple(a.shape), a.dtype) if f else _sds(a) for a, f in zip(av[nT + nP:], bbc)]
    cj = _mk(item, tav + pav + [jax.ShapeDtypeStruct((), jnp.int32)] + bav)
    tl = tuple(dataclasses.replace(s, S=B * s.S) for s in p.tallies)
    return dataclasses.replace(p, cj=cj, tallies=tl, nP=nP + 1, tag=p.tag + f"+folded_batch{B}")


def _fold_batch(node, args, dims, bat, B):
    p = node.prog
    nP = p.nP
    a0 = jnp.moveaxis(args[0], dims[0], 0) if bat[0] else args[0][None]
    N = a0.shape[1]
    pi2 = []
    for k in range(nP):
        x = jnp.moveaxis(args[k], dims[k], 0) if bat[k] else jnp.broadcast_to(args[k], (B,) + args[k].shape)
        pi2.append(x.reshape((B * N,) + x.shape[2:]))
    bid = jnp.repeat(jnp.arange(B, dtype=jnp.int32), N, total_repeat_length=B * N)
    bc2 = [jnp.moveaxis(args[k], dims[k], 0) if bat[k] else args[k] for k in range(nP, len(args))]
    q = fold_prog(p, B, tuple(bat[nP:]))
    fn = node.derived(q, "folded_batch", lin_pi=(node.lin_pi + (False,)) if node.linear else (),
                      lin_bc=node.lin_bc if node.linear else ())
    outs = bind_call(*pi2, bid, *bc2, node=fn)
    ot = _out_specs(p)
    nt = len(ot)
    res, od = [], []
    for j, k in enumerate(ot):
        o = outs[j]
        res.append(o.reshape(o.shape[:-1] + (B, p.tallies[k].S)))
        od.append(len(p.tallies[k].lead))
    res.append(outs[nt])
    od.append(NM)
    for o in outs[nt + 1:]:
        res.append(o.reshape((B, N) + o.shape[1:]))
        od.append(0)
    return res, od


# ------------------------------------------------------------------------------------------- fused jvp
# jax.jvp of a tallymap call that returns the primal too would otherwise run two kernels (primal node + tangent node;
# the tangent kernel recomputes the primal internally).  The fused_jvp node is jax.jvp(item) itself: ONE kernel.  It is
# not linear, so wherever JAX partially evaluates it (linearize / grad / vjp, jax.checkpoint, scan/cond/pjit partial
# eval) the two rules below split it back into a KNOWN primal node and a STAGED tangent node (the unfused design), and
# its batching rule unfuses too (jacfwd etc. are unchanged).  These are the two legacy registries
# pe.custom_partial_eval_rules and pe.partial_eval_jaxpr_custom_rules: if a JAX release drops them, FUSE_JVP turns
# off and everything keeps working (two kernels per jvp).
_HAVE_PE = hasattr(_pe, "custom_partial_eval_rules") and hasattr(_pe, "partial_eval_jaxpr_custom_rules")
FUSE_JVP = _HAVE_PE           # (tests may set tallymap._primitive.FUSE_JVP = False to exercise the two-node design)


def _ntf(p):
    return sum(1 for k in _out_specs(p) if isfloat(p.tallies[k].dtype))


def _fused_layout(p):
    """fused_jvp outputs = [primal sums (nt)] [tangent sums (ntf)] [stats] [primal results (nY)]
    [tangent results (nyf)];
    -> (indices of the primal outputs, indices of the tangent outputs) in the fused output list."""
    nt, ntf, nY = len(_out_specs(p)), _ntf(p), p.nY
    nyf = sum(1 for a in p.y_avals() if isfloat(a.dtype))
    prim = list(range(nt)) + [nt + ntf] + list(range(nt + ntf + 1, nt + ntf + 1 + nY))
    tan = list(range(nt, nt + ntf)) + list(range(nt + ntf + 1 + nY, nt + ntf + 1 + nY + nyf))
    return prim, tan


def _fused_node(node, pp, pb):
    p = node.prog
    q = jvp_prog(p, pp, pb)
    return node.derived(q, "fused_jvp", base=(node, pp, pb, ()), pp=pp, pb=pb)


def _fused_parts(jn):
    """-> (primal node, tangent node, primal operand positions).  base = (primal node, pp, pb, lanes chain): a fused_jvp
    node batched into dual lanes (_dual_lanes) keeps the chain of (T, bpi, bbc) so that its tangent part is the same
    dual_lanes node."""
    fwd, pp, pb, chain = jn.base
    p = fwd.prog
    ln = jn.__dict__.get("_lin")
    if ln is None:
        ln = _tangent_node(fwd, pp, pb)
        for T, bpi, bbc in chain:
            ln = ln.derived(dual_lanes_prog(ln.prog, T, bpi, bbc)[0], "dual_lanes", lin_pi=ln.lin_pi, lin_bc=ln.lin_bc)
        object.__setattr__(jn, "_lin", ln)
    nPj = p.nP + sum(pp)
    pos = list(range(p.nP)) + list(range(nPj, nPj + p.nB))
    return fwd, ln, pos


def _assemble(p, outs, touts):
    prim_idx, tan_idx = _fused_layout(p)
    ntf = _ntf(p)
    res = [None] * (len(prim_idx) + len(tan_idx))
    for j, o in zip(prim_idx, outs):
        res[j] = o
    for j, o in zip(tan_idx, list(touts[:ntf]) + list(touts[ntf + 1:])):
        res[j] = o
    return res


def _fused_jvp(primals, tangents, *, node):
    # second order through a fused_jvp node: the generic rule on its own prog (jvp of jvp(item))
    return _jvp(primals, tangents, node=node)


def _fused_batch(args, dims, *, node):
    fwd, pp, pb, _ = node.base
    nPj = fwd.prog.nP + sum(pp)
    if all(dims[i] is NM for i in list(range(fwd.prog.nP)) + list(range(nPj, nPj + fwd.prog.nB))):
        r = _dual_lanes(args, dims, node, fused=True)        # jacfwd / vmap of jvp: ONE kernel, primal + T lanes
        if r is not None:
            return r
    fwd, ln, pos = _fused_parts(node)
    p = fwd.prog
    pd = [dims[i] for i in pos]
    if all(d is NM for d in pd):
        outs, od = bind_call(*[args[i] for i in pos], node=fwd), [NM] * (len(_out_specs(p)) + 1 + p.nY)
    else:
        outs, od = (_fused_batch if fwd.kind == "fused_jvp" else _batch)([args[i] for i in pos], pd, node=fwd)
    touts, td = _batch(list(args), list(dims), node=ln)
    return _assemble(p, outs, touts), _assemble(p, od, td)


def _fused_transpose(cts, *args, node):
    raise NotImplementedError("tallymap: a fused jvp node reached transposition without partial evaluation (internal "
                              "error: please report it)")


def _fused_pe(trace, *tracers, node):
    """JaxprTrace (linearize / vjp / partial_eval_jaxpr_nounits): primal operands known, some tangent unknown ->
    known primal node + staged tangent node (residuals = the operands); otherwise the default."""
    fwd, ln, pos = _fused_parts(node)
    kn = [t.pval.get_known() for t in tracers]
    if all(k is not None for k in kn) or any(kn[i] is None for i in pos):
        return trace.default_process_primitive(tallymap_jvp_p, tracers, dict(node=node))
    outs = bind_call(*[kn[i] for i in pos], node=fwd)
    touts = trace.default_process_primitive(tallymap_call_p, tracers, dict(node=ln))
    return _assemble(fwd.prog, [trace.new_const(o) for o in outs], touts)


def _fused_pe_custom(saveable, unks_in, inst_in, eqn):
    """partial_eval_jaxpr_custom (jax.checkpoint / remat, scan & cond inside remat): same split at the eqn level."""
    from jax.extend.core import Literal
    node = eqn.params["node"]
    fwd, ln, pos = _fused_parts(node)
    n = len(eqn.outvars)
    res_of = lambda vs, insts: [v for v, i in zip(vs, insts) if not isinstance(v, Literal) and not i]
    if not any(unks_in):          # all known: keep it known, its outputs saved (correct for every policy)
        return eqn, None, [False] * n, [False] * n, []
    if any(unks_in[i] for i in pos):                 # primal unknown: everything staged
        return None, eqn, [True] * n, [True] * n, res_of(eqn.invars, inst_in)
    prim_idx, tan_idx = _fused_layout(fwd.prog)
    ntf = _ntf(fwd.prog)
    known = eqn.replace(primitive=tallymap_call_p, invars=[eqn.invars[i] for i in pos],
                        outvars=[eqn.outvars[j] for j in prim_idx], params=dict(node=fwd))
    # the tangent node has its own stats output: give it a fresh (dropped) var (reusing the fused node's stats var is
    # wrong),
    # so build its outvars as [tangent sums] + [a dropped var] + [tangent results]
    drop = drop_var(eqn.outvars[prim_idx[len(_out_specs(fwd.prog))]].aval)
    tvars = [eqn.outvars[j] for j in tan_idx]
    staged = eqn.replace(primitive=tallymap_call_p, outvars=tvars[:ntf] + [drop] + tvars[ntf:], params=dict(node=ln),
                         invars=list(eqn.invars))
    unk = [False] * n
    for j in tan_idx:
        unk[j] = True
    return known, staged, unk, list(unk), res_of([eqn.invars[i] for i in pos], [inst_in[i] for i in pos])


ad.primitive_jvps[tallymap_jvp_p] = _fused_jvp
ad.primitive_transposes[tallymap_jvp_p] = _fused_transpose
register_batcher(tallymap_jvp_p, _fused_batch)
if _HAVE_PE:
    _pe.custom_partial_eval_rules[tallymap_jvp_p] = _fused_pe
    _pe.partial_eval_jaxpr_custom_rules[tallymap_jvp_p] = _fused_pe_custom
