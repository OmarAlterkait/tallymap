"""xla backend: the portable backend.  Plain JAX (XLA runs it), any platform.

A masked lockstep interpreter of the item jaxpr over a batch of P items: every value is (array, batched?), control
flow runs while ANY active item of the batch needs it, and emits / spawns / per-item writes are masked by the
active items.  Unlike jax.vmap, tallies are never per-item: an emit is a masked scatter-add into ONE shared
accumulator, so memory is O(P * state + outputs), not O(P * outputs).  Items are processed in batches of P
(xla_batch_size); with spawn, children are queued in a ring buffer and run in later batches (children first).
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend.core import Literal, jaxpr_as_fun

from .._errors import ad_advice
from .._jaxpr import bind, call_jaxpr, cond_branches, lit, located, scan_parts, while_parts

BLOCK = 1 << 18
# f32 sum tallies: R partial accumulators (lane p adds into row p % R), summed by jnp.sum at the end.  One
# accumulator takes every item's every emit in series (XLA:CPU's scatter-add is a serial loop), so its rounding
# error grows with the number of adds per bin: 85x plain JAX's on a 3.6e3-per-bin tally (round4/xla, Q1).  R rows
# cut the adds per bin R-fold; the rows cost R x the tally memory, so R shrinks to stay within ROWS_ELEMS.
ROWS, ROWS_ELEMS = 64, 1 << 24


class VT:
    """a virtual tally value (no data): slot index."""
    __slots__ = ("slot",)

    def __init__(self, slot):
        self.slot = slot


class B:
    """an interpreter value: array x (with a leading batch axis iff b)."""
    __slots__ = ("x", "b")

    def __init__(self, x, b):
        self.x, self.b = x, b


def _bc(v, P):
    return v.x if v.b else jnp.broadcast_to(v.x, (P,) + jnp.shape(v.x))


class _St:
    def __init__(self, p, P, accs, queue):
        self.p, self.P = p, P
        self.accs = accs                # slot -> array
        self.queue = queue              # None | dict(leaves, head, tail, spawned, dropped)
        self.ovf = jnp.int32(0)         # fixed-point overflow count
        self.bad = {k: jnp.zeros(a.shape[:-1] + (a.shape[-1] // 2,), bool)                         # flagged bins
                    for k, a in accs.items() if p.tallies[k].fixed_point}
        self.drops = jnp.int32(0)       # out-of-range (dropped) emit indices
        self.mask = None
        self.lanes = []                 # enclosing while loops with a batched predicate: their [P, *B] masks

    # ------------------------------------------------------------- state threaded through lax control flow
    def pack(self):
        q = self.queue
        return (tuple(self.accs[k] for k in sorted(self.accs)), tuple(self.bad[k] for k in sorted(self.bad)), self.ovf,
                None if q is None else (tuple(q["leaves"]), q["head"], q["tail"], q["spawned"], q["dropped"]),
                self.drops)

    def unpack(self, s):
        a, bad, self.ovf, q, self.drops = s
        self.accs = dict(zip(sorted(self.accs), a))
        self.bad = dict(zip(sorted(self.bad), bad))
        if q is not None:
            self.queue = dict(leaves=list(q[0]), head=q[1], tail=q[2], spawned=q[3], dropped=q[4])


def _fixed(v, scale):
    """-> (int64 fixed-point values, representable mask); round-half-even like __double2ll_rn."""
    if not jax.config.jax_enable_x64:
        raise NotImplementedError("tallymap: fixed_point=True outputs need jax_enable_x64 on the xla backend")
    x = v.astype(jnp.float64) * scale
    ok = jnp.abs(x) < 9.2e18
    return jnp.where(ok, jnp.round(x), 0).astype(jnp.int64), ok


def _wide(a, s):
    """a fixed-point tally's accumulator [*lead, 2S] (the sums of the values' low 32 bits, unsigned, and of their
    high 32 bits, signed) -> (its output: the exact total / scale, in s.dtype; the bins that do not fit s.dtype).
    The kernel backend's carry word made explicit: no int64 bin wraps (the low sums hold 2^31 values per bin).
    A total that fits int64 converts as before; a larger one is hi * 2^32 + lo rounded once (|hi| < 2^53)."""
    S = s.S
    lo, hi = a[..., :S], a[..., S:]
    hi = hi + (lo >> 32)
    lo = lo & jnp.int64(0xFFFFFFFF)
    fits = (hi >= -(1 << 31)) & (hi < (1 << 31))
    f64 = jnp.float64
    t = jnp.where(fits, ((hi << 32) + lo).astype(f64), hi.astype(f64) * 2.0 ** 32 + lo.astype(f64))
    v = t / s.scale
    if jnp.issubdtype(s.dtype, jnp.floating):
        o = v.astype(s.dtype)
        return o, ~jnp.isfinite(o)
    if jnp.issubdtype(s.dtype, jnp.integer):
        ii = jnp.iinfo(s.dtype)
        return v.astype(s.dtype), ~((v >= float(ii.min)) & (v < float(ii.max) + 1.0))
    return v.astype(s.dtype), jnp.zeros(v.shape, bool)


def _rows(spec):
    """partial rows of a float, non-fixed-point sum tally (see ROWS); 1 for every other tally."""
    if spec.kind != "sum" or spec.fixed_point or not jnp.issubdtype(spec.dtype, jnp.floating):
        return 1
    n = int(np.prod(spec.lead, dtype=np.int64)) * spec.S
    return int(max(1, min(ROWS, ROWS_ELEMS // max(n, 1))))


def _emit(st, t, idx, val):
    spec = st.p.tallies[t.slot]
    if spec.kind == "discard":
        return t
    P = st.P
    S = spec.S
    L = tuple(spec.lead)
    i = _bc(idx, P)                                   # [P, *I]
    v = _bc(val, P)                                   # [P, *L, *I]
    for lm in st.lanes:     # inside a loop with a batched predicate (_while): a lane whose predicate is false
        nb = lm.ndim - 1    # emits nothing (adding zero), the "select" of a tally carry
        if tuple(L[:nb]) != tuple(lm.shape[1:]):
            raise NotImplementedError(f"tallymap: an emit into a tally with lead dims {L} inside a while loop whose "
                                      f"predicate has shape {lm.shape[1:]} (internal error)")
        v = jnp.where(lm.reshape(lm.shape + (1,) * (v.ndim - lm.ndim)), v, jnp.zeros((), v.dtype))
    m = jnp.broadcast_to(st.mask.reshape((P,) + (1,) * (i.ndim - 1)), i.shape)
    live = m & (i >= 0) & (i < S)
    st.drops = st.drops + jnp.sum((m & ~live).astype(jnp.int32), dtype=jnp.int32)   # out of range: dropped
    R = _rows(spec)
    if R > 1:                                         # lane p -> partial row p % R; dropped stays out of range
        row = jnp.broadcast_to((jnp.arange(P, dtype=jnp.int32) % R).reshape((P,) + (1,) * (i.ndim - 1)), i.shape)
        f = jnp.where(live, row * S + i, R * S).reshape(-1).astype(jnp.int32)
    else:
        f = jnp.where(live, i, S).reshape(-1).astype(jnp.int32)
    nl = len(L)
    v = jnp.moveaxis(v, 0, nl).reshape(L + (-1,))      # [*L, P*I]
    a = st.accs[t.slot]
    if spec.fixed_point:
        v, ok = _fixed(v, spec.scale)
        live = jnp.broadcast_to(f < S, ok.shape)
        st.ovf = st.ovf + jnp.sum((~ok & live).astype(jnp.int32), dtype=jnp.int32)
        st.bad[t.slot] = st.bad[t.slot].at[..., f].max(~ok, mode="drop")        # a bin that dropped an emit
        # the wide accumulator (_wide): the low 32 bits of each value into [0, S), the signed high 32 into [S, 2S)
        fl, fh = jnp.where(f < S, f, 2 * S), jnp.where(f < S, f + S, 2 * S)
        st.accs[t.slot] = a.at[..., jnp.concatenate([fl, fh])].add(
            jnp.concatenate([v & jnp.int64(0xFFFFFFFF), v >> 32], axis=-1), mode="drop")
        return t
    v = v.astype(a.dtype)
    st.accs[t.slot] = a.at[..., f].add(v, mode="drop")
    return t


def _check_tail(qleaves, leaves):
    """a spawn in a jvp program may carry only its PRIMAL leaves: JAX does not call spawn_p's jvp rule (which appends
    the tangents of the float leaves) when every tangent reaching it is a symbolic zero.  The missing trailing leaves
    are those tangents, i.e. zero, and must be written as zero (a queue slot is reused)."""
    if len(leaves) > len(qleaves) or any(not jnp.issubdtype(ql.dtype, jnp.inexact) for ql in qleaves[len(leaves):]):
        raise TypeError(f"tallymap: a spawn with {len(leaves)} child leaves in a program whose queue has "
                        f"{len(qleaves)} (internal error)")


def _spawn(st, t, when, leaves):
    if st.lanes:
        raise NotImplementedError("tallymap: a spawn inside a while loop with a batched predicate is not supported")
    q = st.queue
    P = st.P
    Q = st.p.spawn.capacity
    w = _bc(when, P) & st.mask
    pos = q["tail"] + jnp.cumsum(w.astype(jnp.int32), dtype=jnp.int32) - 1
    ok = w & (pos - q["head"] < Q)
    slot = jnp.where(ok, pos % Q, Q)
    _check_tail(q["leaves"], leaves)
    q["leaves"] = [ql.at[slot].set(_bc(leaves[i], P).astype(ql.dtype) if i < len(leaves) else
                                   jnp.zeros((P,) + ql.shape[1:], ql.dtype), mode="drop")
                   for i, ql in enumerate(q["leaves"])]
    n_ok = jnp.sum(ok.astype(jnp.int32), dtype=jnp.int32)
    q["tail"] = q["tail"] + n_ok
    q["spawned"] = q["spawned"] + n_ok
    q["dropped"] = q["dropped"] + jnp.sum((w & ~ok).astype(jnp.int32), dtype=jnp.int32)
    return t


def _ev(st, jaxpr, consts, args):
    env = {}

    def read(v):
        return B(lit(v), False) if isinstance(v, Literal) else env[v]

    for v, c in zip(jaxpr.constvars, consts):
        env[v] = B(c, False)
    for v, a in zip(jaxpr.invars, args):
        env[v] = a
    for e in jaxpr.eqns:
        ins = [read(v) for v in e.invars]
        name = e.primitive.name
        if name == "tallymap_emit":
            outs = [_emit(st, ins[0], ins[1], ins[2])]
        elif name == "tallymap_spawn":
            outs = [_spawn(st, ins[0], ins[1], ins[2:])]
        elif call_jaxpr(e) is not None:
            s = call_jaxpr(e)
            outs = _ev(st, s.jaxpr, s.consts, ins)
        elif name == "while":
            outs = _while(st, e, ins)
        elif name == "cond":
            outs = _cond(st, e, ins)
        elif name == "scan":
            outs = _scan(st, e, ins)
        else:
            if e.effects:
                raise located(NotImplementedError(
                    f"tallymap: primitive {name!r} has side effects {e.effects}; effects (debug.print, callbacks "
                    "with effects) are not supported inside tallymap.vmap."), e,
                    f"item function {st.p.where}")
            if any(isinstance(a, VT) for a in ins):
                raise located(TypeError(f"tallymap: {name} reads a tally"), e, f"item function {st.p.where}")
            try:
                if not any(a.b for a in ins):
                    outs = [B(x, False) for x in bind(e, [a.x for a in ins])]
                else:
                    r = jax.vmap(lambda *xs: bind(e, xs),
                                 in_axes=[0 if a.b else None for a in ins])(*[a.x for a in ins])
                    outs = [B(x, True) for x in r]
            except Exception as ex:
                better = ad_advice(ex, f"item function {st.p.where}")
                if better is None:
                    raise
                raise located(better, e, f"item function {st.p.where}") from None

        for v, o in zip(e.outvars, outs):
            env[v] = o
    return [read(v) for v in jaxpr.outvars]


def _sel(m, new, old):
    """per-item select for batched values (VT passes through)."""
    if isinstance(old, VT):
        return old
    mm = m.reshape(m.shape + (1,) * (new.ndim - m.ndim))
    return jnp.where(mm, new, old)


def _while(st, e, ins):
    """A predicate of shape B (batched: jax.vmap of a loop whose condition reads a batched value) follows JAX's while
    contract: the loop runs while ANY entry holds; an entry whose predicate is false keeps its carries and emits
    nothing.  The predicate is then [P, *B] and selects every carry ([P, *B, ...]) per entry."""
    w = while_parts(e)
    nc, nb = w.nc, w.nb
    cc, bc, init = ins[:nc], ins[nc:nc + nb], ins[nc + nb:]
    P = st.P
    isvt = [isinstance(a, VT) for a in init]
    arr0 = [_bc(a, P) for a, v in zip(init, isvt) if not v]
    outer = st.mask

    def carries(arr):
        it = iter(arr)
        return [a if v else B(next(it), True) for a, v in zip(init, isvt)]

    def pred(arr):
        r = _bc(_ev(st, w.cond.jaxpr, w.cond.consts, cc + carries(arr))[0], P)
        return r & outer.reshape(outer.shape + (1,) * (r.ndim - 1))

    def cond_f(s):
        return jnp.any(s[1])

    def body_f(s):
        arr, m, packed = s
        st.unpack(packed)
        saved = st.mask
        st.mask = m if m.ndim == 1 else jnp.any(m.reshape(P, -1), axis=1)
        if m.ndim > 1:
            st.lanes.append(m)
        out = _ev(st, w.body.jaxpr, w.body.consts, bc + carries(arr))
        if m.ndim > 1:
            st.lanes.pop()
        st.mask = saved
        new = [_sel(m, _bc(o, P), a) for o, a in zip([o for o, v in zip(out, isvt) if not v], arr)]
        return new, pred(new), st.pack()
    arr, _, packed = lax.while_loop(cond_f, body_f, (arr0, pred(arr0), st.pack()))
    st.unpack(packed)
    return carries(arr)


def _cond(st, e, ins):
    """Each branch is interpreted ONCE, into a jaxpr (jax.make_jaxpr); lax.cond / lax.switch then bind that jaxpr,
    so a cond nested d deep costs d interpretations, not 3^d (an interpretation per tally probe, per shape probe and
    per lax trace).  Which outputs are virtual tallies, and the shapes of the others, come from the same trace."""
    idx, ops = ins[0], ins[1:]
    brs = cond_branches(e)
    n = len(brs)
    P = st.P
    ii = jnp.clip(idx.x, 0, n - 1)

    def trace(k, mask):
        vt = {}

        def f(packed):
            st.unpack(packed)
            saved = st.mask
            st.mask = mask
            out = _ev(st, brs[k].jaxpr, brs[k].consts, ops)
            st.mask = saved
            vt.update({j: o for j, o in enumerate(out) if isinstance(o, VT)})
            return [None if isinstance(o, VT) else _bc(o, P) for o in out], st.pack()
        saved = st.pack()
        cj, shapes = jax.make_jaxpr(f, return_shape=True)(saved)
        st.unpack(saved)
        tree = jax.tree_util.tree_structure(shapes)
        run = lambda packed: jax.tree_util.tree_unflatten(
            tree, jaxpr_as_fun(cj)(*jax.tree_util.tree_leaves(packed)))
        return run, vt, shapes[0]
    masks = [st.mask if not idx.b else st.mask & (ii == k) for k in range(n)]
    traced = [trace(k, masks[k]) for k in range(n)]
    # outputs that are a virtual tally in ANY branch stay virtual: an inert (discard) tally may meet a filler value
    # in another branch (JAX's cond partial eval pads a residual that only one branch produces with zeros)
    vts = [None] * len(e.outvars)
    for _, vt, _ in traced:
        for j, o in vt.items():
            vts[j] = vts[j] or o
    keep = [j for j in range(len(vts)) if vts[j] is None]

    def branch(k):
        run = traced[k][0]

        def g(packed):
            out, packed = run(packed)
            return [out[j] for j in keep], packed
        return g
    if not idx.b:
        res, packed = lax.switch(ii, [branch(k) for k in range(n)], st.pack())
        st.unpack(packed)
    else:
        outs = []
        for k in range(n):
            r0 = [traced[k][2][j] for j in keep]
            skip = lambda packed, r0=r0: ([jnp.zeros(a.shape, a.dtype) for a in r0], packed)
            r, packed = lax.cond(jnp.any(masks[k]), branch(k), skip, st.pack())     # rare branches cost nothing
            st.unpack(packed)
            outs.append(r)
        res = [_select_n(ii, [o[j] for o in outs]) for j in range(len(outs[0]))]
    it = iter(res)
    return [vt if vt is not None else B(next(it), True) for vt in vts]


def _select_n(ii, cases):
    shp = cases[0].shape
    w = ii.reshape(ii.shape + (1,) * (len(shp) - 1))
    w = jnp.broadcast_to(w, shp)
    return lax.select_n(w, *cases) if len(cases) > 1 else cases[0]


def _scan(st, e, ins):
    sc = scan_parts(e)
    nc, nk = sc.nconsts, sc.ncarry
    consts, init, xs = ins[:nc], ins[nc:nc + nk], ins[nc + nk:]
    P = st.P
    isvt = [isinstance(a, VT) for a in init]
    arr0 = [_bc(a, P) for a, v in zip(init, isvt) if not v]
    # (an inert discard tally may be a scanned input / stacked output: a residual of reverse mode; it stays virtual)
    xvt = [isinstance(a, VT) for a in xs]
    xr = [a for a in xs if not isinstance(a, VT)]
    xsb = [a.b for a in xr]
    xarr = [jnp.moveaxis(a.x, 1, 0) if a.b else a.x for a in xr]     # scanned axis first
    yvt = {}

    def body(s, x):
        arr, packed = s
        st.unpack(packed)
        it = iter(arr)
        car = [a if v else B(next(it), True) for a, v in zip(init, isvt)]
        xi = iter(zip(x, xsb))
        xin = [a if v else B(*next(xi)) for a, v in zip(xs, xvt)]
        out = _ev(st, sc.body.jaxpr, sc.body.consts, consts + car + xin)
        new = [_bc(o, P) for o, v in zip(out[:nk], isvt) if not v]
        yvt.update((j, o) for j, o in enumerate(out[nk:]) if isinstance(o, VT))
        return (new, st.pack()), [_bc(o, P) for o in out[nk:] if not isinstance(o, VT)]
    (arr, packed), ys = lax.scan(body, (arr0, st.pack()), xarr, length=sc.length, reverse=sc.reverse)
    st.unpack(packed)
    it = iter(arr)
    car = [a if v else B(next(it), True) for a, v in zip(init, isvt)]
    yi = iter(ys)
    return car + [yvt[j] if j in yvt else B(jnp.moveaxis(next(yi), 0, 1), True) for j in range(len(e.outvars) - nk)]


# ------------------------------------------------------------------------------------------- driver
def run(p, pi, bc, xla_batch_size=None, **_):
    """-> [sum tally arrays] + [stats int32[5]] + [per-item results [N, ...]].  p.nact (n_active): items
    i >= bc[p.nact] are masked off like the padding of the last batch (no emits, per-item results zero)."""
    N = int(pi[0].shape[0])
    nlim = N if p.nact is None else jnp.clip(jnp.asarray(bc[p.nact], jnp.int32), 0, N)
    nT = p.nT
    ya = p.y_avals()
    accs = {}
    for k, s in enumerate(p.tallies):
        if s.kind == "sum":
            dt = jnp.int64 if s.fixed_point else s.dtype
            if s.fixed_point:
                _fixed(jnp.zeros(()), 1.0)
            accs[k] = jnp.zeros(tuple(s.lead) + ((2 if s.fixed_point else _rows(s)) * s.S,), dt)
    ys0 = [jnp.zeros((N,) + tuple(a.shape), a.dtype) for a in ya]
    P = int(min(max(N, 1), xla_batch_size or BLOCK))
    cj = p.cj
    queue = None
    if p.spawn is not None:
        Q = int(p.spawn.capacity)
        pav = [v.aval for v in cj.jaxpr.invars[nT:nT + p.nP]]
        queue = dict(leaves=[lax.full((Q,) + tuple(a.shape), 0, a.dtype) for a in pav], head=jnp.int32(0),
                     tail=jnp.int32(0), spawned=jnp.int32(0), dropped=jnp.int32(0))
    st = _St(p, P, accs, queue)
    lanes = jnp.arange(P, dtype=jnp.int32)

    def block_fn(ids, mask, lane_pi, ys, primary):
        st.mask = mask
        args = [VT(k) for k in range(nT)] + [B(x, True) for x in lane_pi] + [B(x, False) for x in bc]
        out = _ev(st, cj.jaxpr, cj.consts, args)
        yv = [_bc(o, P) for o in out[nT:]]
        wid = jnp.where(primary, ids, N)
        ys = [y.at[wid].set(v.astype(y.dtype), mode="drop") for y, v in zip(ys, yv)]
        ctr = jnp.int32(0)
        if p.trunc is not None:
            tr = yv[p.trunc].astype(bool).reshape(P, -1).any(axis=1)     # a laned program (jacrev) flags per lane
            ctr = jnp.sum((tr & mask & ~primary).astype(jnp.int32), dtype=jnp.int32)
        return ys, ctr

    if N == 0 and queue is None:
        return _finish(p, accs, None, ys0, st)
    if queue is None:
        nblk = -(-N // P)

        def body(c, s):
            packed, ys = s
            st.unpack(packed)
            ids = c * P + lanes
            mask = ids < nlim
            lane_pi = [jnp.take(x, jnp.clip(ids, 0, N - 1), axis=0, mode="clip") for x in pi]
            ys, _ = block_fn(ids, mask, lane_pi, ys, mask)
            return st.pack(), ys
        if nblk == 1:
            packed, ys = body(0, (st.pack(), ys0))
        else:
            packed, ys = lax.fori_loop(0, nblk, body, (st.pack(), ys0))
        st.unpack(packed)
        return _finish(p, st.accs, None, ys, st)

    # spawn: children (queued) first, then primaries
    def cond_s(s):
        nxt, packed, ys, ctr = s
        return (nxt < N) | (packed[3][2] > packed[3][1])

    def body_s(s):
        nxt, packed, ys, ctr = s
        st.unpack(packed)
        q = st.queue
        kq = jnp.minimum(q["tail"] - q["head"], P)
        is_child = lanes < kq
        pid = nxt + lanes - kq
        primary = ~is_child & (pid < N)
        mask = is_child | primary
        qs = (q["head"] + lanes) % Q
        take = lambda a, i: jnp.take(a, i, axis=0, mode="clip")
        if N > 0:
            lane_pi = [jnp.where(_lm(is_child, x), take(ql, qs), take(x, jnp.clip(pid, 0, max(N - 1, 0))))
                       for ql, x in zip(q["leaves"], pi)]
        else:
            lane_pi = [take(ql, qs) for ql in q["leaves"]]
        if p.rec is not None:           # the recording primal (_reverse_spawn.rec_primal): `me` <- slot / -1 - index
            lane_pi[p.rec] = jnp.where(is_child, qs, -1 - pid).astype(jnp.int32)
        q["head"] = q["head"] + kq
        ys, c = block_fn(pid, mask, lane_pi, ys, primary)
        return jnp.minimum(nxt + P - kq, N), st.pack(), ys, ctr + c
    _, packed, ys, ctr = lax.while_loop(cond_s, body_s, (jnp.int32(0), st.pack(), ys0, jnp.int32(0)))
    st.unpack(packed)
    q = st.queue
    return _finish(p, st.accs, (q["spawned"], q["dropped"], ctr), ys, st) + (q["leaves"] if p.rec is not None else [])


def _lm(m, x):
    return m.reshape(m.shape + (1,) * (x.ndim - 1))


def _finish(p, accs, sp, ys, st):
    out = []
    for k, s in enumerate(p.tallies):
        if s.kind == "sum":
            a = accs[k]
            if _rows(s) > 1:
                a = jnp.sum(a.reshape(tuple(s.lead) + (_rows(s), s.S)), axis=-2)
            if s.fixed_point:                       # a bin that dropped an emit is NaN (as on cuda)
                a, big = _wide(a, s)
                st.ovf = st.ovf + jnp.sum((big & ~st.bad[k]).astype(jnp.int32), dtype=jnp.int32)
                if jnp.issubdtype(s.dtype, jnp.floating):
                    a = jnp.where(st.bad[k] | big, jnp.asarray(np.nan, s.dtype), a)
            out.append(a)
    sp = sp if sp is not None else (jnp.int32(0),) * 3
    return out + [jnp.stack([jnp.asarray(x, jnp.int32) for x in (*sp, st.ovf, st.drops)])] + list(ys)
