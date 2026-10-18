"""lockstep backend: the reference and the fallback.  Plain JAX, any platform.

A masked lockstep interpreter of the item jaxpr over P lanes: every value is (array, batched?), control flow
runs while ANY active lane needs it, and emits / spawns / per-item writes are masked by the active lanes.
Unlike jax.vmap, tallies are never per-lane: an emit is a masked scatter-add into ONE shared accumulator, so
memory is O(P * state + outputs), not O(P * outputs).  Items are processed in blocks of P lanes; with spawn,
children are queued in a ring buffer and run in later blocks (children first).

COMPACTION (`_while`, `_cond_pool`): masking alone makes a while loop cost P * max(trip count), and a cond with
a per-lane index cost the SUM of the branches on every lane.  So divergent control flow runs on the lanes that
need it instead, gathered into a dense prefix of a NARROWER lane axis: a while loop hands its survivors to the
same loop over P >> WSHIFT lanes once at most W of them are left, and a cond runs each branch on chunks of the
lanes that take it.  Both are measured in REPORT.md; `docs/design.md` section 9 has the reasoning and the two
designs that were measured and rejected.  The one number behind all of it, on a CPU (ns per element, 2^16 lanes,
f32[4] rows, one milano node): `jnp.where` 0.20, dynamic_slice 0.17, dynamic_update_slice 0.13, gather 0.45,
scatter 2.9, a permutation 6.0.  Moving a lane costs 3 to 30 times touching it, so lanes are moved once per
level, never per round.
"""
import jax
import jax.numpy as jnp
from jax import lax
from jax.extend.core import Literal

from .._foreign import ad_advice, advice
from .._jaxpr import bind, call_jaxpr, cond_branches, lit, located, scan_parts, subjaxprs, while_parts

BLOCK = 1 << 18
WSHIFT = 2        # each narrowing divides the lane count by 2^WSHIFT
WMIN = 1 << 10    # a lane count at or below this is never narrowed: too few lanes left to pay for the gather
NARROW = 1        # how many times ONE loop narrows: each costs one more instantiation of the loop body
# per-element cost of a gather and of a scatter, in units of one element of an elementwise operation (measured
# above).  They decide ONE thing: whether compacting a cond's branches pays for moving its operands (_pays).
GATHER, SCATTER = 5, 12


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
    def __init__(self, p, P, accs, queue, compact=None, cond=None):
        self.p, self.P = p, P
        self.accs = accs                # slot -> array
        self.queue = queue              # None | dict(leaves, head, tail, spawned, dropped)
        self.ovf = jnp.int32(0)         # exact-output overflow count
        self.drops = jnp.int32(0)       # out-of-range (dropped) emit indices
        self.mask = None
        self.compact = compact          # None: the rule | False: never compact | int: that chunk width
        self.cond = cond                # None: the _pays rule | True: compact every batched cond | False: none
        self.loop = 0                   # loop bodies the interpreter is inside (narrowing is top-level only)

    def width(self, P):
        """-> the narrower lane count to compact into, or None: stay at P lanes and mask."""
        if self.compact is False:
            return None
        if self.compact is None:
            return None if P <= WMIN else max(P >> WSHIFT, WMIN)
        w = int(self.compact)
        return None if w <= 0 or w >= P else w

    def narrow(self, P, level):
        """the same, for the survivors of a while loop.  A narrowing costs one more instantiation of that loop's
        body, so only a loop the interpreter is not already inside a loop for narrows (`self.loop`), and only
        NARROW times: the compiled program then grows with NARROW instead of exponentially with the loop
        nesting (measured: the shower's reverse-mode program took 899 s to compile when every nested loop
        narrowed, against 56 s masked and 0.87x the run time)."""
        return None if self.loop or level >= NARROW else self.width(P)

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


def _spawn(st, t, when, leaves):
    q = st.queue
    P = st.P
    Q = st.p.spawn.capacity
    w = _bc(when, P) & st.mask
    pos = q["tail"] + jnp.cumsum(w.astype(jnp.int32), dtype=jnp.int32) - 1
    ok = w & (pos - q["head"] < Q)
    slot = jnp.where(ok, pos % Q, Q)
    q["leaves"] = [ql.at[slot].set(_bc(x, P).astype(ql.dtype), mode="drop") for ql, x in zip(q["leaves"], leaves)]
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


def _order(m, n, P):
    """a permutation of the lane axis that puts the lanes of `m` first, in lane order, and the rest after it.

    The stopped lanes follow in the suffix, so it is a full permutation and a prefix of it of any length is a
    valid set of lanes to gather."""
    i = jnp.arange(P, dtype=jnp.int32)
    pos = jnp.where(m, jnp.cumsum(m, dtype=jnp.int32) - 1, n + jnp.cumsum(~m, dtype=jnp.int32) - 1)
    return jnp.zeros((P,), jnp.int32).at[pos].set(i, mode="drop")


def _while(st, e, ins, level=0):
    """a while loop over the current lanes, then the SAME loop over its survivors on a NARROWER lane axis.

    Masking alone costs P * max(trip count): every round runs on every lane, including the lanes that stopped
    long ago.  So the masked loop stops as soon as at most W = P >> WSHIFT lanes are still running, gathers
    those lanes into a W-wide lane axis and goes on there, where a round is 1 / 2^WSHIFT of the cost -- and
    recursively, NARROW times.  A lane that stops never restarts (its carry is frozen, so its predicate stays
    false), which is why nothing is ever gathered back: one gather per carry per level, and one scatter at the
    end to put the survivors back where they came from.

    Chunking EVERY round into ceil(active / W) chunks instead -- which is what the per-element cost model says
    to do, and what the cuda backend does per thread -- was implemented and measured first, and rejected: on a
    CPU an HLO operation costs a thread fork-join whatever its width, so chunked rounds cost 1.1 to 4.1x the
    masked loop on the problems whose lanes stay active (barrier, uniform, particle_filter, interacting_md) for
    2x on the divergent ones.  Narrowing instead pays nothing per round.  REPORT.md has both tables.
    """
    w = while_parts(e)
    nc, nb = w.nc, w.nb
    init = ins[nc + nb:]
    P = st.P
    isvt = [isinstance(a, VT) for a in init]

    def carries(arr):
        it = iter(arr)
        return [a if v else B(next(it), True) for a, v in zip(init, isvt)]
    W = st.narrow(P, level)
    arr, m = _while_masked(st, e, ins, 0 if W is None else W)
    if W is None:
        return carries(arr)
    n = jnp.sum(m, dtype=jnp.int32)
    g = _order(m, n, P)[:W]                        # the lanes still running (at most W of them) come first
    ops = [B(jnp.take(a.x, g, axis=0, mode="clip"), True) if not isinstance(a, VT) and a.b else a
           for a in ins[:nc + nb]]
    saved = st.P, st.mask
    st.P, st.mask = W, jnp.arange(W, dtype=jnp.int32) < n
    out = _while(st, e, ops + carries([jnp.take(a, g, axis=0, mode="clip") for a in arr]), level + 1)
    st.P, st.mask = saved
    it = iter([o for o, v in zip(out, isvt) if not v])
    return carries([a.at[g].set(_bc(next(it), W).astype(a.dtype), mode="drop") for a in arr])


def _while_masked(st, e, ins, floor=0):
    """the masked loop: every round runs on all P lanes.  -> (carry arrays, the still-running mask) when at most
    `floor` lanes are still running (floor = 0: when none is)."""
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
        return jnp.any(s[1]) if floor == 0 else jnp.sum(s[1], dtype=jnp.int32) > floor

    def body_f(s):
        arr, m, packed = s
        st.unpack(packed)
        saved = st.mask
        st.mask, st.loop = m, st.loop + 1
        out = _ev(st, w.body.jaxpr, w.body.consts, bc + carries(arr))
        st.mask, st.loop = saved, st.loop - 1
        new = [_sel(m, _bc(o, P), a) for o, a in zip([o for o, v in zip(out, isvt) if not v], arr)]
        return new, pred(new), st.pack()
    arr, m, packed = lax.while_loop(cond_f, body_f, (arr0, pred(arr0), st.pack()))
    st.unpack(packed)
    return arr, m


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
        it = iter(res)
        return [vt if vt is not None else B(next(it), True) for vt in vts]
    ii = jnp.clip(idx.x, 0, n - 1)
    W = st.width(P) if (_pays(brs, ops, e, vts) if st.cond is None else st.cond) else None
    if W is not None:
        res = _cond_pool(st, brs, ops, e, vts, ii, W)
    else:
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


def _size(aval, skip=0):
    """elements of one lane's value: the aval of an item-level value, or a batched array minus its lane axis."""
    n = 1
    for d in getattr(aval, "shape", ())[skip:]:
        n *= int(d)
    return n


def _cost(jaxpr, _memo=None):
    """a per-lane cost for ONE evaluation of `jaxpr`, in units of one element of an elementwise operation: the
    elements its equations produce, with the moves weighted as measured (GATHER / SCATTER) and a while loop
    counted as a single trip, which is the conservative count (it can only UNDER-state what compaction saves)."""
    _memo = {} if _memo is None else _memo
    if id(jaxpr) in _memo:
        return _memo[id(jaxpr)]
    total = 0
    for e in jaxpr.eqns:
        sz = sum(_size(v.aval) for v in e.outvars) or 1
        name = e.primitive.name
        if name == "tallymap_emit":
            total += SCATTER * _size(e.invars[1].aval)
        elif name.startswith("scatter"):
            total += SCATTER * sz
        elif name in ("gather", "dynamic_slice", "dynamic_update_slice"):
            total += GATHER * sz
        elif name == "cond":
            total += max(_cost(b.jaxpr, _memo) for b in cond_branches(e))
        elif name == "scan":
            sc = scan_parts(e)
            total += sc.length * _cost(sc.body.jaxpr, _memo)
        elif (s := subjaxprs(e)):
            total += sum(_cost(x.jaxpr, _memo) for x in s)
        else:
            total += sz
    _memo[id(jaxpr)] = total
    return total


def _pays(brs, ops, e, vts):
    """-> does compacting this cond pay?  The masked interpreter runs EVERY branch that any lane takes on every
    lane, so of the branch work it does, all but ONE lane's worth is thrown away; compacting runs each branch on
    the lanes that take it, which costs the average branch instead of their sum, plus a gather of the
    lane-varying operands and a scatter of the results.  So it pays when

        sum(branches) - sum(branches) / nbranches  >  GATHER * operands + SCATTER * results

    with everything estimated from the jaxpr (_cost), per lane.  The decision is static: it cannot depend on how
    the lanes actually split, and an even split is the only assumption available.  Measured on a switch over K
    phases (REPORT.md): compacting is 1.3 to 11.7x faster forward and 0.9 to 8.4x in reverse mode once a branch
    does more than a handful of flops, and 1.5 to 1.7x SLOWER when the branches are two flops and no emit, which
    is the case this rule declines."""
    if len(brs) < 2:
        return False
    c = [_cost(b.jaxpr) for b in brs]
    move = GATHER * sum(_size(o.x, 1) for o in ops if not isinstance(o, VT) and o.b) \
        + SCATTER * sum(_size(v.aval) for v, vt in zip(e.outvars, vts) if vt is None)
    return sum(c) - sum(c) // len(c) > move


def _cond_pool(st, brs, ops, e, vts, ii, W):
    """every branch on the lanes that TAKE it, gathered into chunks of W lanes.  -> the batched outputs."""
    P = st.P
    idx = jnp.arange(W, dtype=jnp.int32)
    res = [jnp.zeros((P,) + tuple(v.aval.shape), v.aval.dtype) for v, vt in zip(e.outvars, vts) if vt is None]
    outer = st.mask
    for k in range(len(brs)):
        mk = outer & (ii == k)
        nk = jnp.sum(mk, dtype=jnp.int32)
        g = _order(mk, nk, P)

        def chunk(s, k=k):
            cur, res, packed = s
            st.unpack(packed)
            start = jnp.minimum(cur, P - W)
            sel = lax.dynamic_slice_in_dim(g, start, W, 0)
            live = (idx >= cur - start) & (start + idx < nk)
            st.P, st.mask = W, live
            out = _ev(st, brs[k].jaxpr, brs[k].consts,
                      [o if isinstance(o, VT) or not o.b else B(jnp.take(o.x, sel, axis=0, mode="clip"), True)
                       for o in ops])
            st.P, st.mask = P, outer
            at = jnp.where(live, sel, P)                             # a lane outside the chunk is dropped
            res = [r.at[at].set(_bc(o, W).astype(r.dtype), mode="drop")
                   for r, o in zip(res, [o for o, vt in zip(out, vts) if vt is None])]
            return cur + W, res, st.pack()
        _, res, packed = lax.while_loop(lambda s: s[0] < nk, chunk, (jnp.int32(0), res, st.pack()))
        st.unpack(packed)
    return res


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
        st.loop += 1
        out = _ev(st, sc.body.jaxpr, sc.body.consts, consts + car + xin)
        st.loop -= 1
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
def run(p, pi, bc, lockstep_block=None, _lockstep_compact=None, _lockstep_cond=None, **_):
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
    st = _St(p, P, accs, queue, _lockstep_compact, _lockstep_cond)
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
