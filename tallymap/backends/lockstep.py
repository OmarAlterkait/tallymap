"""lockstep backend: the reference and the fallback.  Plain JAX, any platform.

A masked lockstep interpreter of the item jaxpr over P lanes: every value is (array, batched?), control flow
runs while ANY active lane needs it, and emits / spawns / per-item writes are masked by the active lanes.
Unlike jax.vmap, tallies are never per-lane: an emit is a masked scatter-add into ONE shared accumulator, so
memory is O(P * state + outputs), not O(P * outputs).  Items are processed in blocks of P lanes; with spawn,
children are queued in a ring buffer and run in later blocks (children first).
"""
import jax
import jax.numpy as jnp
from jax import lax
from jax.extend.core import Literal

from .._foreign import ad_advice, advice
from .._jaxpr import bind, call_jaxpr, cond_branches, lit, located, scan_parts, while_parts

BLOCK = 1 << 18


class VT:
    """a virtual tally value (no data): slot index."""
    __slots__ = ("slot",)

    def __init__(self, slot):
        self.slot = slot


class B:
    """an interpreter value: array x (with a leading lane axis iff b)."""
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
        self.ovf = jnp.int32(0)         # exact-output overflow count
        self.drops = jnp.int32(0)       # out-of-range (dropped) emit indices
        self.mask = None

    # ------------------------------------------------------------- state threaded through lax control flow
    def pack(self):
        q = self.queue
        return (tuple(self.accs[k] for k in sorted(self.accs)), self.ovf,
                None if q is None else (tuple(q["leaves"]), q["head"], q["tail"], q["spawned"], q["dropped"]),
                self.drops)

    def unpack(self, s):
        a, self.ovf, q, self.drops = s
        self.accs = dict(zip(sorted(self.accs), a))
        if q is not None:
            self.queue = dict(leaves=list(q[0]), head=q[1], tail=q[2], spawned=q[3], dropped=q[4])


def _fixed(v, scale):
    """-> (int64 fixed-point values, representable mask); round-half-even like __double2ll_rn."""
    if not jax.config.jax_enable_x64:
        raise NotImplementedError("tallymap: exact=True outputs need jax_enable_x64 on the lockstep backend")
    x = v.astype(jnp.float64) * scale
    ok = jnp.abs(x) < 9.2e18
    return jnp.where(ok, jnp.round(x), 0).astype(jnp.int64), ok


def _emit(st, t, idx, val):
    spec = st.p.tallies[t.slot]
    if spec.kind == "discard":
        return t
    P = st.P
    S = spec.S
    L = tuple(spec.lead)
    i = _bc(idx, P)                                   # [P, *I]
    v = _bc(val, P)                                   # [P, *L, *I]
    m = jnp.broadcast_to(st.mask.reshape((P,) + (1,) * (i.ndim - 1)), i.shape)
    live = m & (i >= 0) & (i < S)
    st.drops = st.drops + jnp.sum((m & ~live).astype(jnp.int32), dtype=jnp.int32)   # out of range: dropped
    f = jnp.where(live, i, S).reshape(-1).astype(jnp.int32)
    nl = len(L)
    v = jnp.moveaxis(v, 0, nl).reshape(L + (-1,))      # [*L, P*I]
    a = st.accs[t.slot]
    if spec.exact:
        v, ok = _fixed(v, spec.scale)
        live = jnp.broadcast_to(f < S, ok.shape)
        st.ovf = st.ovf + jnp.sum((~ok & live).astype(jnp.int32), dtype=jnp.int32)
    else:
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
                    f"with effects) are not supported inside tallymap.vmap.{advice(e)}"), e,
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
    """per-lane select for batched values (VT passes through)."""
    if isinstance(old, VT):
        return old
    mm = m.reshape(m.shape + (1,) * (new.ndim - 1))
    return jnp.where(mm, new, old)


def _while(st, e, ins):
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
        r = _ev(st, w.cond.jaxpr, w.cond.consts, cc + carries(arr))[0]
        return _bc(r, P) & outer

    def cond_f(s):
        return jnp.any(s[1])

    def body_f(s):
        arr, m, packed = s
        st.unpack(packed)
        saved = st.mask
        st.mask = m
        out = _ev(st, w.body.jaxpr, w.body.consts, bc + carries(arr))
        st.mask = saved
        new = [_sel(m, _bc(o, P), a) for o, a in zip([o for o, v in zip(out, isvt) if not v], arr)]
        return new, pred(new), st.pack()
    arr, _, packed = lax.while_loop(cond_f, body_f, (arr0, pred(arr0), st.pack()))
    st.unpack(packed)
    return carries(arr)


def _cond(st, e, ins):
    idx, ops = ins[0], ins[1:]
    brs = cond_branches(e)
    n = len(brs)
    P = st.P
    # outputs that are a virtual tally in ANY branch stay virtual: an inert (discard) tally may meet a filler value
    # in another branch (JAX's cond partial eval pads a residual that only one branch produces with zeros)
    vts = [None] * len(e.outvars)

    def probe(k):
        def f(packed):
            st.unpack(packed)
            for j, o in enumerate(_ev(st, brs[k].jaxpr, brs[k].consts, ops)):
                if isinstance(o, VT) and vts[j] is None:
                    vts[j] = o
            return 0
        return f
    saved = st.pack()
    for k in range(n):
        jax.eval_shape(probe(k), saved)
        st.unpack(saved)

    def run_branch(k, mask):
        def f(packed):
            st.unpack(packed)
            saved = st.mask
            st.mask = mask
            out = _ev(st, brs[k].jaxpr, brs[k].consts, ops)
            st.mask = saved
            return [_bc(o, P) for o, vt in zip(out, vts) if vt is None], st.pack()
        return f
    if not idx.b:
        res, packed = lax.switch(jnp.clip(idx.x, 0, n - 1), [run_branch(k, st.mask) for k in range(n)], st.pack())
        st.unpack(packed)
    else:
        ii = jnp.clip(idx.x, 0, n - 1)
        outs = []
        for k in range(n):
            mk = st.mask & (ii == k)
            f = run_branch(k, mk)
            saved = st.pack()
            r0 = jax.eval_shape(f, saved)[0]
            st.unpack(saved)
            skip = lambda packed, r0=r0: ([jnp.zeros(a.shape, a.dtype) for a in r0], packed)
            r, packed = lax.cond(jnp.any(mk), f, skip, st.pack())     # rare branches cost nothing
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
def run(p, pi, bc, lockstep_block=None, **_):
    """-> [sum tally arrays] + [stats int32[5]] + [per-item results [N, ...]]"""
    N = int(pi[0].shape[0])
    nT = p.nT
    ya = p.y_avals()
    accs = {}
    for k, s in enumerate(p.tallies):
        if s.kind == "sum":
            dt = jnp.int64 if s.exact else s.dtype
            if s.exact:
                _fixed(jnp.zeros(()), 1.0)
            accs[k] = jnp.zeros(tuple(s.lead) + (s.S,), dt)
    ys0 = [jnp.zeros((N,) + tuple(a.shape), a.dtype) for a in ya]
    P = int(min(max(N, 1), lockstep_block or BLOCK))
    cj = p.cj
    queue = None
    if p.spawn is not None:
        Q = int(p.spawn.capacity)
        pav = [v.aval for v in cj.jaxpr.invars[nT:nT + p.nP]]
        queue = dict(leaves=[jnp.zeros((Q,) + tuple(a.shape), a.dtype) for a in pav], head=jnp.int32(0),
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
            ctr = jnp.sum((yv[p.trunc].astype(bool) & mask & ~primary).astype(jnp.int32), dtype=jnp.int32)
        return ys, ctr

    if N == 0 and queue is None:
        return _finish(p, accs, None, ys0, st.ovf, st.drops)
    if queue is None:
        nblk = -(-N // P)

        def body(c, s):
            packed, ys = s
            st.unpack(packed)
            ids = c * P + lanes
            mask = ids < N
            lane_pi = [jnp.take(x, jnp.clip(ids, 0, N - 1), axis=0, mode="clip") for x in pi]
            ys, _ = block_fn(ids, mask, lane_pi, ys, mask)
            return st.pack(), ys
        if nblk == 1:
            packed, ys = body(0, (st.pack(), ys0))
        else:
            packed, ys = lax.fori_loop(0, nblk, body, (st.pack(), ys0))
        st.unpack(packed)
        return _finish(p, st.accs, None, ys, st.ovf, st.drops)

    # spawn: children (queued) first, then primaries
    def cond_s(s):
        nxt, packed, ys, ctr = s
        return (nxt < N) | (packed[2][2] > packed[2][1])

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
        q["head"] = q["head"] + kq
        ys, c = block_fn(pid, mask, lane_pi, ys, primary)
        return jnp.minimum(nxt + P - kq, N), st.pack(), ys, ctr + c
    _, packed, ys, ctr = lax.while_loop(cond_s, body_s, (jnp.int32(0), st.pack(), ys0, jnp.int32(0)))
    st.unpack(packed)
    q = st.queue
    return _finish(p, st.accs, (q["spawned"], q["dropped"], ctr), ys, st.ovf, st.drops)


def _lm(m, x):
    return m.reshape(m.shape + (1,) * (x.ndim - 1))


def _finish(p, accs, sp, ys, ovf, drops):
    out = []
    for k, s in enumerate(p.tallies):
        if s.kind == "sum":
            a = accs[k]
            out.append((a.astype(jnp.float64) / s.scale).astype(s.dtype) if s.exact else a)
    sp = sp if sp is not None else (jnp.int32(0),) * 3
    return out + [jnp.stack([jnp.asarray(x, jnp.int32) for x in (*sp, ovf, drops)])] + list(ys)
