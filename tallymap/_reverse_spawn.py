"""Reverse mode THROUGH spawn: the tree adjoint.

The idea in one line: a spawn is an EMIT of the child's initial state into a "child slot" output, so the
cotangent of the child's initial state reaches the parent exactly where the transpose of an emit puts it -- as the
cotangent of the emitted value, at the spawn step, inside the parent's ordinary replay adjoint.  No new adjoint
machinery: the per-item VJP is the existing vjp_prog (replay / RV checkpointing) of a rewritten item program.

Rewritten item program (the "rec" program, built once per ItemProgram by re-tracing its jaxpr with a spawn hook):
    spawn(q, when, *child)  ->  emit float leaves of `child` into SPf_l[q*L_l + (0..L_l-1)]  (cotangent carrier)
                                emit the uint32 words of every leaf into BIT_l[q*W_l + ...]   (exact payload record)
                                q += when;  h = fnv(h, words)                                 (fingerprint)
    every emit also folds the bits of its value into h.
    q starts at the item's per-item operand `off` (its first child slot); the item returns n = q_end - off
    (its number of children) and h (a hash of every value it emitted and spawned, in program order).
    Items i >= nact (a broadcast int) do nothing (fixed capacity, traced count): their top-level loops are gated off
    and their emits / spawns masked -- NOT a cond around the item (JAX's partial eval of such a cond forwards the
    item's big constant tables as residual OUTPUTS of the cond, which the cuda emitter cannot hold per item).  The
    caller feeds them a copy of a valid item (row 0) so their straight-line code stays finite.

Forward (recording), generation by generation (roots = generation 0):
    count pass   (q from 0)              -> n_k, h_k
    write pass   (off = exclusive cumsum n_k)  -> BIT payloads of the next generation (in (parent, ordinal) order)
    The two passes are different kernels: their (n, h) must agree (stats fwd_mismatch).
Reverse, deepest generation first, ONE adjoint launch per generation:
    adjoint of generation g = vjp_prog(rec) with output cotangents: the user's output cotangents, SPf <- the
    cotangents of generation g+1's initial states (so the parent's replay injects them at the spawn step through
    the emit transpose), FP <- 1.  It returns the cotangent of its own initial states (-> the SPf cotangent of
    generation g-1) and its broadcast (theta) cotangents (summed over generations).  Roots' initial-state
    cotangents are the cotangents of the per-item operands xs.
Replay consistency: the adjoint program re-computes (n, h) in its own forward replay and compares them with the
recorded (n, h) of the write pass; a mismatch is counted through the derivative of a zero-valued emit
(FP += z * mismatch, z = 0, d/dz = mismatch), so it costs nothing and needs no extra output.  Counted in
stats['replay_mismatch'].

Dropped / overflowing children: the recorded tree never drops.  It is consistent with the forward iff the
forward dropped nothing; that is GUARANTEED when the total number of children <= the Queue capacity (pending <=
total), so a larger tree poisons the gradient (NaN) and is reported (stats 'drop_risk').  Replay or forward
mismatches poison too.  Never silently wrong.

Capacities are derived from the Queue capacity Q (the user sets nothing): the tree holds up to Q non-root
particles (exactly the bound under which the forward provably dropped nothing), a generation up to Q (private
option _rev_gen_capacity lowers it to save memory; a larger generation then poisons: stats 'overflow'), depth up to
Q.  Generation launches use capacity buckets Cg, Cg/2, Cg/4, ... (>= 256) so a small generation does not pay for Cg
idle items.  Private _tree_neg ('detach' | 'replay') are the negative controls.
Stats of the last tree adjoint (host side, via jax.debug.callback, OPT-IN: rev_stats=True -- a host callback takes
jax.jit off its C++ dispatch path): tallymap.diagnostics()['reverse']['tree'] (= _reverse_spawn.LAST[-1]).
Second order through this adjoint raises (the recorded payloads are bit-cast, they carry no tangent).
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ._compat import key_impl
from ._rewrite import check_tallies, emit_p_bind, isfloat, retrace

U32 = jnp.uint32
FNV = 16777619
LAST = []            # host-side stats of executed tree adjoints (appended by a debug callback)
MIN_BUCKET = 256


# ------------------------------------------------------------------------------------------- words / hash
def _is_key(dt):
    return jax.dtypes.issubdtype(dt, jax.dtypes.prng_key)


def _words(x):
    """-> flat uint32 words of x (bit-exact; keys -> key data)."""
    x = jnp.asarray(x)
    if _is_key(x.dtype):
        x = jax.random.key_data(x)
    dt = np.dtype(x.dtype)
    if dt == np.bool_:
        x = x.astype(U32)
    elif dt.itemsize < 4:
        x = (lax.bitcast_convert_type(x, jnp.uint16 if dt.itemsize == 2 else jnp.uint8) if isfloat(dt)
             else x.astype(jnp.int32)).astype(U32)
    return lax.bitcast_convert_type(x, U32).reshape(-1) if np.dtype(x.dtype) != np.uint32 else x.reshape(-1)


def _unwords(w, aval):
    """inverse of _words for one item of aval (w: uint32[nw])."""
    dt, shape = aval.dtype, tuple(aval.shape)
    if _is_key(dt):
        kd = jax.eval_shape(jax.random.key_data, jax.ShapeDtypeStruct(shape, dt))
        return jax.random.wrap_key_data(w.reshape(kd.shape), impl=key_impl(dt))
    dt = np.dtype(dt)
    if dt == np.bool_:
        return w.reshape(shape) != 0
    if dt.itemsize == 4:
        return lax.bitcast_convert_type(w.reshape(shape), dt)
    if dt.itemsize == 8:
        return lax.bitcast_convert_type(w.reshape(shape + (2,)), dt)
    if isfloat(dt):
        return lax.bitcast_convert_type(w.astype(jnp.uint16 if dt.itemsize == 2 else jnp.uint8).reshape(shape), dt)
    return lax.bitcast_convert_type(w, jnp.int32).astype(dt).reshape(shape)


def _nwords(aval):
    return int(jax.eval_shape(_words, jax.ShapeDtypeStruct(aval.shape, aval.dtype)).shape[0])


def _mix(h, w):
    return (h ^ w) * U32(FNV)


def _wsum(x):
    w = _words(x)
    return jnp.sum(w, dtype=U32) if w.size else U32(0)


# ------------------------------------------------------------------------------------------- the rec program
@dataclasses.dataclass(frozen=True, eq=False)
class Rec:
    cj: object           # rec item jaxpr (forward kernels: count / write)
    cj_adj: object       # the same, for the adjoint (differs only under the negative control _tree_neg='replay')
    us: tuple            # user tally slots of the base prog
    fl: tuple            # float per-item leaves
    ls: tuple            # elements per leaf
    nw: tuple            # uint32 words per leaf
    Cg: int


def _size(shape):
    return int(np.prod(shape, dtype=np.int64))


def build_rec(p, Cg, neg=""):
    """the rewritten item program of base ItemProgram p (see module docstring).
    invars : [user tallies][SPf_l (float leaves)][BIT_l (all leaves)][FP] + [pi][off, iid, cnt, H, z] + [bc][nact]
    outvars: same tallies + [user per-item results][n, h]"""
    from ._api import _mk, _sds
    nT, nP, nB = p.nT, p.nP, p.nB
    ks = [k for k, s in enumerate(p.tallies) if s.kind == "spawn"]
    if len(ks) != 1:
        raise NotImplementedError("tallymap tree adjoint: expected exactly one spawn queue")
    ks = ks[0]
    us = tuple(k for k in range(nT) if k != ks)
    av = p.in_avals()
    tav, pav, bav = av[:nT], av[nT:nT + nP], av[nT + nP:]
    fl = tuple(i for i, a in enumerate(pav) if isfloat(a.dtype))
    ls = tuple(_size(a.shape) for a in pav)
    nw = tuple(_nwords(a) for a in pav)
    if Cg * max(ls + nw) >= 2 ** 31:
        raise ValueError(f"tallymap tree adjoint: rev_gen_capacity {Cg} x leaf size overflows int32 slot indices")
    nU, nF = len(us), len(fl)

    def make(neg_replay):
        def item(*a):
            sizes = (nU, nF, nP, 1, nP, 5, nB, 1)
            cuts = np.cumsum((0,) + sizes)
            tu, tf_, tb, tfp, pi, scal, bc, nact = (list(a[i:j]) for i, j in zip(cuts[:-1], cuts[1:]))
            tfp, nact = tfp[0], nact[0]
            off, iid, cnt, H, z = scal
            if neg_replay:        # NEGATIVE CONTROL ONLY: the adjoint replays a 1-ulp different item
                x0 = pi[fl[0]]
                xs = lax.stop_gradient(x0)
                pi[fl[0]] = x0 + (jnp.nextafter(xs, jnp.asarray(np.inf, x0.dtype)) - xs)

            act = iid < nact          # items >= nact: loops skipped, effects masked (no cond: see build_rec doc)

            def em(t, idx, val):
                arr, h = t
                i = jnp.asarray(idx)
                return emit_p_bind(arr, jnp.where(act, i, -1), val), _mix(h, _wsum(val))

            def sp(tok, when, *child):
                q, h, spf, bits = tok
                w = jnp.asarray(when) & act
                spf2, bits2, hw = [], [], U32(0)
                for j, i in enumerate(fl):
                    L = ls[i]
                    idx = jnp.where(w, q * L + jnp.arange(L, dtype=jnp.int32), -1)
                    spf2.append(emit_p_bind(spf[j], idx, jnp.reshape(child[i], (L,)).astype(spf[j].dtype)))
                for i in range(nP):
                    W = nw[i]
                    wd = _words(child[i])
                    idx = jnp.where(w, q * W + jnp.arange(W, dtype=jnp.int32), -1)
                    bits2.append(emit_p_bind(bits[i], idx, lax.bitcast_convert_type(wd, jnp.int32)))
                    hw = _mix(hw, jnp.sum(wd, dtype=U32))
                return (q + w.astype(jnp.int32), jnp.where(w, _mix(h, hw), h), tuple(spf2), tuple(bits2))

            targs = [None] * nT
            for kk, t in zip(us, tu):
                targs[kk] = (t, U32(0))
            targs[ks] = (off, U32(0), tuple(tf_), tuple(tb))
            outs = retrace(p.cj, n_tally=nT, emit=em, spawn=sp, gate=act)(*targs, *pi, *bc)
            tv, y = outs[:nT], outs[nT:]
            q, h, spf, bits = tv[ks]
            for kk in us:
                h = _mix(h, tv[kk][1])
            n = jnp.where(act, q - off, 0).astype(jnp.int32)
            h = jnp.where(act, h, U32(0))
            mm = (act & ((n != cnt) | (h != H))).astype(z.dtype)
            fp = emit_p_bind(tfp, jnp.int32(0), z * mm)
            y = [jnp.where(act, jnp.asarray(v), jnp.zeros_like(jnp.asarray(v))) for v in y]
            return [tv[kk][0] for kk in us] + list(spf) + list(bits) + [fp] + y + [n, h]

        avals = ([_sds(tav[k]) for k in us]
                 + [jax.ShapeDtypeStruct((Cg * ls[i],), pav[i].dtype) for i in fl]
                 + [jax.ShapeDtypeStruct((Cg * nw[i],), jnp.int32) for i in range(nP)]
                 + [jax.ShapeDtypeStruct((1,), jnp.float32)]
                 + [_sds(a) for a in pav]
                 + [jax.ShapeDtypeStruct((), d) for d in (jnp.int32, jnp.int32, jnp.int32, U32, jnp.float32)]
                 + [_sds(a) for a in bav] + [jax.ShapeDtypeStruct((), jnp.int32)])
        return _mk(item, avals)
    cj = make(False)
    cj_adj = make(True) if neg == "replay" else cj
    return Rec(cj, cj_adj, us, fl, ls, nw, Cg)


def _rec_specs(p, R, user, spf, bit, fp):
    """tally specs of the rec program; user / spf / bit / fp: kind of each group (None -> keep the user's kind)."""
    from ._api import TSpec
    pav = p.in_avals()[p.nT:p.nT + p.nP]
    ts = [dataclasses.replace(p.tallies[k], kind=user or p.tallies[k].kind) for k in R.us]
    ts += [TSpec(f"__spf{i}", R.Cg * R.ls[i], np.dtype(pav[i].dtype), kind=spf) for i in R.fl]
    ts += [TSpec(f"__bit{i}", R.Cg * R.nw[i], np.dtype(np.int32), kind=bit) for i in range(p.nP)]
    ts += [TSpec("__fp", 1, np.dtype(np.float32), kind=fp)]
    return tuple(ts)


def _rec_prog(p, R, cj, specs, tag):
    from ._api import ItemProgram
    return ItemProgram(cj, specs, p.nP + 5, p.nB + 1, None, None, p.tag + tag, where=p.where)


def _progs(p, Cg, U_bc, o):
    """(Rec, count ItemProgram, write ItemProgram, adjoint-base ItemProgram, T ItemProgram) for base p; cached on p."""
    from ._reverse import vjp_prog
    cache = p.__dict__.setdefault("_tree_cache", {})
    key = (Cg, tuple(U_bc), tuple(sorted(o.items())))
    if key in cache:
        return cache[key]
    R = build_rec(p, Cg, o.get("_tree_neg", ""))
    us_kinds = [p.tallies[k].kind for k in R.us]
    if any(k not in ("sum", "discard") for k in us_kinds):
        raise NotImplementedError("tallymap tree adjoint: unsupported tally kinds " + str(us_kinds))
    count = _rec_prog(p, R, R.cj, _rec_specs(p, R, "discard", "discard", "discard", "discard"), "+rec_count")
    write = _rec_prog(p, R, R.cj, _rec_specs(p, R, None, "discard", "sum", "discard"), "+rec_write")
    adj_specs = _rec_specs(p, R, None, "sum", "discard", "sum")
    check_tallies(R.cj_adj, [s.kind for s in adj_specs])
    adj = _rec_prog(p, R, R.cj_adj, adj_specs, "+rec_adj")
    pert_pi = tuple(i in R.fl for i in range(p.nP)) + (False, False, False, False, True)
    pert_bc = tuple(U_bc) + (False,)
    T = vjp_prog(adj, pert_pi, pert_bc, o)          # (rev_checkpoints etc. as for any reverse call)
    cache[key] = (R, count, write, adj, T, pert_pi, pert_bc)
    return cache[key]


# ------------------------------------------------------------------------------------------- orchestration
def _buckets(Cg):
    b = [int(Cg)]
    while b[-1] // 2 >= MIN_BUCKET:
        b.append(b[-1] // 2)
    return b


def _bucketed(n, Cg, f):
    """f(b) for the smallest capacity bucket b >= n (static shapes: f pads its per-item outputs to Cg)."""
    bs = _buckets(Cg)
    if len(bs) == 1:
        return f(bs[0])
    idx = jnp.sum(jnp.asarray([n <= b for b in bs[1:]]).astype(jnp.int32))
    return lax.switch(idx, [lambda b=b: f(b) for b in bs])


def _pad(x, Cg):
    return jnp.pad(x, [(0, Cg - x.shape[0])] + [(0, 0)] * (x.ndim - 1)) if x.shape[0] < Cg else x


def _excl_cumsum(c):
    return (jnp.cumsum(c, dtype=jnp.int32) - c).astype(jnp.int32)


@jax.custom_jvp
def _no_second_order(xs):
    return xs


@_no_second_order.defjvp
def _nso_jvp(primals, tangents):
    raise NotImplementedError("tallymap: second-order differentiation through the spawn tree adjoint is not supported "
                              "(the recorded child payloads carry no tangent)")


def tree_vjp(node, base, pi, bc, U_pi, U_bc, cts, N):
    """cotangents (g_pi, g_bc) of base's per-item / broadcast operands flagged U_pi / U_bc, given the cotangents
    `cts` of ALL base outputs ([sum tallies] + [stats] + [per-item results]); base spawns."""
    from ._primitive import _float_slots, _inst, bind_call
    o = dict(node.opts)
    Q = int(base.spawn.capacity)
    Cg = min(int(o.get("_rev_gen_capacity", Q)), Q)
    TOT = Q                     # all non-root particles: more than Q -> the forward may have dropped (drop_risk)
    GMAX = Q + 1                # depth <= particles + 1: never the binding limit
    neg = o.get("_tree_neg", "")
    R, count, write, adj, T, pert_pi, pert_bc = _progs(base, Cg, U_bc, o)
    nP, nB = base.nP, base.nB
    av = base.in_avals()
    pav, bav = av[base.nT:base.nT + nP], av[base.nT + nP:]
    ya = base.y_avals()
    ot, tf, yf = _float_slots(base)
    nut = sum(1 for k in R.us if base.tallies[k].kind == "sum")     # user sum outputs of the write pass
    fwd_node = lambda q: node.derived(q, "primal")
    n_count, n_write = fwd_node(count), fwd_node(write)
    ib = [i for i, f in enumerate(pert_bc) if f]
    t_node = node.derived(T, "adjoint", base=adj, pp=pert_pi, pb=pert_bc,
                          lin_pi=(False,) * adj.nP + (True,) * sum(1 for a in adj.y_avals() if isfloat(a.dtype)),
                          lin_bc=(False,) * adj.nB
                          + (True,) * sum(1 for s in adj.tallies if s.kind == "sum" and isfloat(s.dtype)))
    # output cotangents of the adjoint program's float sum tallies, in tally order: user, SPf, FP
    ct_user = []
    for j, k in enumerate(ot):
        s = base.tallies[k]
        if isfloat(s.dtype):
            ct_user.append(_inst(cts[j], tuple(s.lead) + (s.S,), s.dtype))
    i32 = lambda n: jnp.zeros((n,), jnp.int32)

    def call(nd, per_item, bcast):
        return bind_call(*per_item, *bcast, node=nd)

    def fwd_gen(X, b, nact):
        """count + write passes over b items -> (user sums, BIT payloads, n, h, mismatch count)"""
        iid = jnp.arange(b, dtype=jnp.int32)
        z = jnp.zeros((b,), jnp.float32)
        oc = call(n_count, list(X) + [i32(b), iid, i32(b), jnp.zeros((b,), U32), z], list(bc) + [nact])
        nc, hc = oc[-2], oc[-1]
        ow = call(n_write, list(X) + [_excl_cumsum(nc), iid, nc, hc, z], list(bc) + [nact])
        sums, bits = ow[:nut], ow[nut:nut + nP]
        nw_, hw_ = ow[-2], ow[-1]
        fmm = jnp.sum(((nw_ != nc) | (hw_ != hc)).astype(jnp.int32))
        return list(sums), list(bits), nc, hw_, fmm

    # ---- forward: generation 0 (roots)
    sums0, bits0, nc0, h0, fmm0 = fwd_gen(pi, N, jnp.int32(N))
    n1 = jnp.sum(nc0, dtype=jnp.int32)
    z0 = (jnp.int32(0), jnp.int32(0))
    FB = [lax.dynamic_update_slice(jnp.zeros((TOT + Cg, R.nw[i]), jnp.int32), bits0[i].reshape(Cg, R.nw[i]), z0)
          for i in range(nP)]
    FC = jnp.zeros((TOT + Cg,), jnp.int32)
    FH = jnp.zeros((TOT + Cg,), U32)
    goff = jnp.zeros((GMAX + 2,), jnp.int32)
    ng = jnp.zeros((GMAX + 2,), jnp.int32).at[1].set(n1)
    ovf = (n1 > Cg) | (n1 > TOT)

    def decode(FB, st, n):
        """generation rows [st, st+Cg); rows >= n (inactive) get a copy of row 0 (a valid item: finite code)"""
        live = (jnp.arange(Cg, dtype=jnp.int32) < n)[:, None]
        rows = ((lax.dynamic_slice_in_dim(F, st, Cg), a) for F, a in zip(FB, pav))
        return [jax.vmap(lambda w, a=a: _unwords(lax.bitcast_convert_type(w, U32), a))(jnp.where(live, r, r[:1]))
                for r, a in rows]

    def f_cond(s):
        g, FB, FC, FH, goff, ng, ovf, fmm, sums = s
        return (g <= GMAX) & (ng[g] > 0) & ~ovf

    def f_body(s):
        g, FB, FC, FH, goff, ng, ovf, fmm, sums = s
        st, n = goff[g], ng[g]
        X = decode(FB, st, n)

        def one(b):
            sm, bt, nc, hw, fm = fwd_gen([x[:b] for x in X], b, n)
            return sm, bt, _pad(nc, Cg), _pad(hw, Cg), fm
        sm, bt, nc, hw, fm = _bucketed(n, Cg, one)
        nn = jnp.sum(nc, dtype=jnp.int32)
        FC = lax.dynamic_update_slice(FC, nc, (st,))
        FH = lax.dynamic_update_slice(FH, hw, (st,))
        st2 = st + n
        FB = [lax.dynamic_update_slice(F, b_.reshape(Cg, w), (st2, jnp.int32(0))) for F, b_, w in zip(FB, bt, R.nw)]
        goff = goff.at[g + 1].set(st2)
        ng = ng.at[g + 1].set(nn)
        ovf = ovf | (nn > Cg) | (st2 + nn > TOT)
        return g + 1, FB, FC, FH, goff, ng, ovf, fmm + fm, [a + b for a, b in zip(sums, sm)]
    s = lax.while_loop(f_cond, f_body, (jnp.int32(1), FB, FC, FH, goff, ng, ovf, fmm0, sums0))
    G, FB, FC, FH, goff, ng, ovf, fmm, sums = s
    ovf = ovf | (ng[jnp.minimum(G, GMAX + 1)] > 0)          # stopped by rev_max_generations (or overflow)
    total = goff[G]                                          # all non-root particles (recorded)
    drop_risk = total + ng[jnp.minimum(G, GMAX + 1)] > Q      # (+ a generation not recorded: tree > Queue capacity)

    # ---- reverse: generations G-1 .. 1, then the roots
    def adj_gen(X, cnt, H, b, nact, Gc, cy):
        iid = jnp.arange(b, dtype=jnp.int32)
        cnt = jnp.where(iid < nact, cnt, 0)
        if neg == "detach":          # NEGATIVE CONTROL ONLY: children's cotangents are not injected
            Gc = [jnp.zeros_like(g_) for g_ in Gc]
        ct = ct_user + [g_.reshape(-1) for g_ in Gc] + [jnp.ones((1,), jnp.float32)]
        outs = call(t_node, list(X) + [_excl_cumsum(cnt), iid, cnt, H, jnp.zeros((b,), jnp.float32)] + list(cy),
                    list(bc) + [nact] + ct)
        gt = list(outs[:len(ib)])
        dp = list(outs[len(ib) + 1:])
        return gt, dp[:-1], jnp.sum(dp[-1])

    Gc0 = [jnp.zeros((Cg,) + tuple(pav[i].shape), pav[i].dtype) for i in R.fl]
    gt0 = [jnp.zeros((_size(bav[i].shape),), bav[i].dtype) for i in ib]
    cy_children = lambda b: [jnp.zeros((b,) + tuple(ya[j].shape), ya[j].dtype) for j in yf]

    def r_cond(s):
        return s[0] >= 1

    def r_body(s):
        g, Gc, gt, mm = s
        st, n = goff[g], ng[g]
        X = decode(FB, st, n)
        cnt = lax.dynamic_slice_in_dim(FC, st, Cg)
        H = lax.dynamic_slice_in_dim(FH, st, Cg)

        def one(b):
            gtb, dx, m = adj_gen([x[:b] for x in X], cnt[:b], H[:b], b, n, Gc, cy_children(b))
            return gtb, [_pad(d, Cg) for d in dx], m
        gtg, dx, m = _bucketed(n, Cg, one)
        return g - 1, dx, [a + b_ for a, b_ in zip(gt, gtg)], mm + m
    cy_root = [_inst(cts[len(ot) + 1 + j], (N,) + tuple(ya[j].shape), ya[j].dtype) for j in yf]
    _, Gc, gt, mm = lax.while_loop(r_cond, r_body, (G - 1, Gc0, gt0, jnp.float32(0)))
    gtr, dxr, mr = adj_gen(list(pi), nc0, h0, N, jnp.int32(N), Gc, cy_root)
    gt = [a + b_ for a, b_ in zip(gt, gtr)]
    mm = mm + mr

    bad = ovf | drop_risk | (mm > 0) | (fmm > 0)
    poison = lambda x: jnp.where(bad, jnp.asarray(np.nan, x.dtype), x)
    g_pi = [None] * nP
    for k, i in enumerate(R.fl):
        if U_pi[i]:
            g_pi[i] = poison(dxr[k])
    g_bc = [None] * nB
    for k, i in enumerate(ib):
        g_bc[i] = poison(gt[k]).reshape(bav[i].shape)
    st_ = dict(generations=G - 1, particles=total, max_generation=jnp.max(ng), overflow=ovf, drop_risk=drop_risk,
               replay_mismatch=mm.astype(jnp.int32), fwd_mismatch=fmm, n_roots=jnp.int32(N),
               launches=2 * G + G)       # (count + write) per forward generation + one adjoint per generation
    if o.get("rev_stats"):
        # opt-in: a host callback takes jax.jit off its C++ dispatch path (+1.3 s per call measured on a shower);
        # the gradient is NaN-poisoned on any failure without it
        host = lambda v: [np.asarray(x) for x in v] if isinstance(v, list) else np.asarray(v)
        jax.debug.callback(lambda d: LAST.append({k: host(v) for k, v in d.items()}), st_)
    outs = _no_second_order((g_pi, g_bc))
    return outs
