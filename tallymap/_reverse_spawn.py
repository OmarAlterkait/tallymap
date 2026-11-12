"""Reverse mode THROUGH spawn: the tree adjoint.

The idea in one line: a spawn is an EMIT of the child's initial state into a "child slot" output, so the
cotangent of the child's initial state reaches the parent exactly where the transpose of an emit puts it -- as the
cotangent of the emitted value, at the spawn step, inside the parent's ordinary replay adjoint.  No new adjoint
machinery: the per-item VJP is the existing vjp_prog (replay / RV checkpointing) of a rewritten item program.

The record is the forward launch.  Under reverse mode the primal call runs rec_primal(p) (_primitive: the known
half of a spawning call's jvp): every item carries four int32 leaves (me, par, ord, gen) after its own; a spawn
appends (0, me, k, gen + 1) to the child (k = the item's spawn count so far), and the backend sets `me` to each
item's queue slot (a child) or -1 - its index (a root) when it starts (ItemProgram.rec).  The queue ring is returned
with the outputs: with at most Q children no slot is reused, so it holds every child's initial state, parent,
ordinal and generation.  It reaches the transpose as residual operands of the staged tangent call.

Adjoint item program (the "rec" program, built once per ItemProgram by re-tracing its jaxpr with a spawn hook):
    spawn(q, when, *child)  ->  emit the float leaves of `child` into SPf_l[q*L_l + (0..L_l-1)];  q += when
    q starts at the item's per-item operand `off` (its first child slot); n = q_end - off is its child count.
    A generation of n items is a launch with n_active (ItemProgram.nact) = n: items i >= n do not run.  The caller
    feeds them a copy of a valid item (row 0): the xla backend computes masked lanes.
Reverse, deepest generation first, ONE adjoint launch per generation (its items in (parent, ordinal) order):
    adjoint of generation g = vjp_prog(rec) with output cotangents: the user's output cotangents, SPf <- the
    cotangents of generation g+1's initial states (its item j at SPf slot j, so the parent's replay
    injects it at the spawn step through the emit transpose), FP <- 1.  It returns the cotangent of its own initial
    states and its broadcast (theta) cotangents (summed over generations).  Roots' initial-state cotangents are the
    cotangents of the per-item operands xs.  Launches: 1 (the recording primal) + one per generation.
Replay consistency: the adjoint re-counts each item's children and compares with the recorded count; a mismatch
is counted through the derivative of a zero-valued emit (FP += z * mismatch, z = 0, d/dz = mismatch), so it costs
nothing and needs no extra output.  Counted in stats['replay_mismatch'].  (Bitwise replay itself is gated per item
in tests/reverse.py, "REPLAY GATE".)

Dropped / overflowing children: the record is complete iff the forward pushed at most Q children and dropped none
(no queue slot was reused); otherwise the gradient is NaN (stats 'drop_risk').  A generation larger than its
capacity (stats 'generation_overflow') or a replay mismatch poisons too.  Never silently wrong.

Capacities are derived from the Queue capacity Q (the user sets nothing): the tree holds up to Q children, a
generation up to Q.  Generation launches use capacity buckets Cg, Cg/2, Cg/4, ... (>= 256) so a small generation
does not pay for Cg idle items, nor for Cg-sized XLA work around the launch.  The test-only module constants
GEN_CAPACITY and TREE_NEG (below) lower the generation capacity and select the negative controls.
Stats of the last tree adjoint (host side, via jax.debug.callback, OPT-IN: rev_stats=True -- a host callback takes
jax.jit off its C++ dispatch path): tallymap.diagnostics()['reverse']['tree'] (= _reverse_spawn.LAST[-1]).
Second order through this adjoint raises (the recorded payloads carry no tangent).
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ._rewrite import emit_p_bind, isfloat, retrace

LAST = []            # host-side stats of executed tree adjoints (appended by a debug callback)
MIN_BUCKET = 256
# TEST-ONLY (tests/reverse_spawn.py patches them, like codegen.FILL; never set by the library):
GEN_CAPACITY = None  # largest generation the tree adjoint takes (None: the Queue capacity); smaller -> overflow
TREE_NEG = ""        # NEGATIVE CONTROLS: 'detach' (children's cotangents not injected) | 'replay' (the adjoint
                     # replays a perturbed item, whose child counts the replay check must catch)


def _spawn_slot(p):
    ks = [k for k, s in enumerate(p.tallies) if s.kind == "spawn"]
    if len(ks) != 1:
        raise NotImplementedError("tallymap tree adjoint: expected exactly one spawn queue")
    return ks[0]


# ------------------------------------------------------------------------------------------- the recording primal
def rec_primal(p):
    """p with the int32 per-item leaves (me, par, ord, gen) appended (see the module docstring); cached on p."""
    q = p.__dict__.get("_rec_primal")
    if q is None:
        from ._api import _mk, _sds
        from ._tally import spawn_p
        nT, nP, ks = p.nT, p.nP, _spawn_slot(p)
        av = [_sds(a) for a in p.in_avals()]

        def item(*a):
            me, par, ordn, gen = a[nT + nP:nT + nP + 4]

            def sp(tok, when, *child):
                qt, k = tok
                w = jnp.asarray(when)
                return spawn_p.bind(qt, w, *child, jnp.int32(0), me, k, gen + 1), k + w.astype(jnp.int32)
            targs = list(a[:nT])
            targs[ks] = (targs[ks], jnp.int32(0))
            outs = list(retrace(p.cj, n_tally=nT, spawn=sp)(*targs, *a[nT:nT + nP], *a[nT + nP + 4:]))
            outs[ks] = outs[ks][0]
            return outs
        cj = _mk(item, av[:nT + nP] + [jax.ShapeDtypeStruct((), jnp.int32)] * 4 + av[nT + nP:])
        q = dataclasses.replace(p, cj=cj, nP=nP + 4, tag=p.tag + "+rec", rec=nP)
        p.__dict__["_rec_primal"] = q
    return q


# ------------------------------------------------------------------------------------------- the adjoint program
@dataclasses.dataclass(frozen=True, eq=False)
class Rec:
    cj: object           # rec item jaxpr (differs from the base only under the negative control TREE_NEG='replay')
    us: tuple            # user tally slots of the base prog
    fl: tuple            # float per-item leaves
    ls: tuple            # elements per leaf


def _size(shape):
    return int(np.prod(shape, dtype=np.int64))


def build_rec(p, Cg, neg=""):
    """the rewritten item program of base ItemProgram p (see module docstring).
    invars : [user tallies][SPf_l (float leaves)][FP] + [pi][off, cnt, z] + [bc][nact]
    outvars: same tallies + [user per-item results][n]"""
    from ._api import _mk, _sds
    nT, nP, nB = p.nT, p.nP, p.nB
    ks = _spawn_slot(p)
    us = tuple(k for k in range(nT) if k != ks)
    av = p.in_avals()
    tav, pav, bav = av[:nT], av[nT:nT + nP], av[nT + nP:]
    fl = tuple(i for i, a in enumerate(pav) if isfloat(a.dtype))
    ls = tuple(_size(a.shape) for a in pav)
    if Cg * max(ls) >= 2 ** 31:
        raise ValueError(f"tallymap tree adjoint: generation capacity {Cg} x leaf size overflows int32 slot indices")
    nU, nF = len(us), len(fl)

    def item(*a):
        cuts = np.cumsum((0, nU, nF, 1, nP, 3, nB, 1))
        tu, tf_, tfp, pi, scal, bc, _ = (list(a[i:j]) for i, j in zip(cuts[:-1], cuts[1:]))
        off, cnt, z = scal            # (the last operand, nact, is the launch's n_active: the item does not read it)
        if neg == "replay":           # NEGATIVE CONTROL ONLY: the adjoint replays a different item
            pi[fl[0]] = pi[fl[0]] * 1.5

        def sp(tok, when, *child):
            q, spf = tok
            w = jnp.asarray(when)
            spf2 = []
            for j, i in enumerate(fl):
                L = ls[i]
                idx = jnp.where(w, q * L + jnp.arange(L, dtype=jnp.int32), -1)
                spf2.append(emit_p_bind(spf[j], idx, jnp.reshape(child[i], (L,)).astype(spf[j].dtype)))
            return q + w.astype(jnp.int32), tuple(spf2)

        targs = [None] * nT
        for kk, t in zip(us, tu):
            targs[kk] = t
        targs[ks] = (off, tuple(tf_))
        outs = retrace(p.cj, n_tally=nT, spawn=sp)(*targs, *pi, *bc)
        tv, y = outs[:nT], outs[nT:]
        q, spf = tv[ks]
        n = (q - off).astype(jnp.int32)
        fp = emit_p_bind(tfp[0], jnp.int32(0), z * (n != cnt).astype(z.dtype))
        return [tv[kk] for kk in us] + list(spf) + [fp] + list(y) + [n]

    avals = ([_sds(tav[k]) for k in us]
             + [jax.ShapeDtypeStruct((Cg * ls[i],), pav[i].dtype) for i in fl]
             + [jax.ShapeDtypeStruct((1,), jnp.float32)]
             + [_sds(a) for a in pav]
             + [jax.ShapeDtypeStruct((), d) for d in (jnp.int32, jnp.int32, jnp.float32)]
             + [_sds(a) for a in bav] + [jax.ShapeDtypeStruct((), jnp.int32)])
    return Rec(_mk(item, avals), us, fl, ls)


def _progs(p, Cg, U_bc, o):
    """(Rec, adjoint-base ItemProgram, its vjp ItemProgram, pert_pi, pert_bc) for base p; cached on p."""
    from ._api import ItemProgram, TSpec
    from ._reverse import check_slots, vjp_prog
    cache = p.__dict__.setdefault("_tree_cache", {})
    key = (Cg, tuple(U_bc), tuple(sorted(o.items())), TREE_NEG)
    if key in cache:
        return cache[key]
    R = build_rec(p, Cg, TREE_NEG)
    if any(p.tallies[k].kind not in ("sum", "discard") for k in R.us):
        raise NotImplementedError("tallymap tree adjoint: unsupported tally kinds "
                                  + str([p.tallies[k].kind for k in R.us]))
    pav = p.in_avals()[p.nT:p.nT + p.nP]
    specs = tuple(p.tallies[k] for k in R.us)
    specs += tuple(TSpec(f"__spf{i}", Cg * R.ls[i], np.dtype(pav[i].dtype), kind="sum") for i in R.fl)
    specs += (TSpec("__fp", 1, np.dtype(np.float32), kind="sum"),)
    check_slots(R.cj, specs, p.tag + "+rec_adj")
    adj = ItemProgram(R.cj, specs, p.nP + 3, p.nB + 1, None, None, p.tag + "+rec_adj", where=p.where, nact=p.nB)
    pert_pi = tuple(i in R.fl for i in range(p.nP)) + (False, False, True)
    pert_bc = tuple(U_bc) + (False,)
    T = vjp_prog(adj, pert_pi, pert_bc, o)          # (rev_checkpoints etc. as for any reverse call)
    cache[key] = (R, adj, T, pert_pi, pert_bc)
    return cache[key]


# ------------------------------------------------------------------------------------------- orchestration
def _buckets(Cg):
    b = [int(Cg)]
    while b[-1] // 2 >= MIN_BUCKET:
        b.append(b[-1] // 2)
    return b


def _bucketed(n, Cg, f):
    """f(b) for the smallest capacity bucket b >= n (static shapes: f pads its per-item outputs to Cg).  Under jax.vmap
    a batched switch index would launch every bucket (a select: 4-17x B grads, round5/tree)."""
    bs = _buckets(Cg)
    pick = jax.custom_batching.custom_vmap(lambda n: jnp.sum(jnp.asarray([n <= b for b in bs[1:]]).astype(jnp.int32)))
    pick.def_vmap(lambda axis_size, in_batched, n: (pick(jnp.max(n)), False))  # vmap: ONE bucket, the largest member's
    return lax.switch(pick(n), [lambda b=b: f(b) for b in bs])


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


def tree_vjp(node, base, pi, bc, U_pi, U_bc, cts, N, ring):
    """cotangents (g_pi, g_bc) of base's per-item / broadcast operands flagged U_pi / U_bc, given the cotangents
    `cts` of ALL base outputs ([sum tallies] + [stats] + [per-item results]) and the record `ring` = [stats] +
    [queue leaves] of rec_primal(base) (see the module docstring); base spawns."""
    from ._primitive import _float_slots, _inst, bind_call
    o = dict(node.opts)
    Q = int(base.spawn.capacity)
    Cg = min(int(GEN_CAPACITY or Q), Q)
    R, adj, T, pert_pi, pert_bc = _progs(base, Cg, U_bc, o)
    nP, nB = base.nP, base.nB
    av = base.in_avals()
    pav, bav = av[base.nT:base.nT + nP], av[base.nT + nP:]
    ya = base.y_avals()
    ot, tf, yf = _float_slots(base)
    ib = [i for i, f in enumerate(pert_bc) if f]
    t_node = node.derived(T, "adjoint", base=adj, pp=pert_pi, pb=pert_bc,
                          lin_pi=(False,) * adj.nP + (True,) * sum(1 for a in adj.y_avals() if isfloat(a.dtype)),
                          lin_bc=(False,) * adj.nB
                          + (True,) * sum(1 for s in adj.tallies if s.kind == "sum" and isfloat(s.dtype)))
    # output cotangents of the adjoint program's float sum tallies, in tally order: user, SPf, FP
    ct_user = [_inst(cts[j], tuple(base.tallies[k].lead) + (base.tallies[k].S,), base.tallies[k].dtype)
               for j, k in enumerate(ot) if isfloat(base.tallies[k].dtype)]

    # ---- the record
    stats, leaves = ring[0], ring[1:1 + nP]
    par, ordn, gen = ring[2 + nP:5 + nP]
    total = stats[0]
    drop_risk = (total > Q) | (stats[1] > 0)         # a slot reused or a child dropped: the record is incomplete
    slot = jnp.arange(Q, dtype=jnp.int32)
    valid = slot < total
    gk = jnp.where(valid, gen, Q + 1)
    ng = jnp.zeros((Q + 2,), jnp.int32).at[gk].add(1)             # generation sizes
    goff = _excl_cumsum(ng)                                        # generation g: ORD[goff[g]:goff[g] + ng[g]]
    G = jnp.max(jnp.where(valid, gen, 0))           # the deepest generation (0: no children)
    ovf = jnp.max(ng[1:Q + 1]) > Cg
    root = par < 0
    cnt_root = jnp.zeros((N,), jnp.int32).at[jnp.where(valid & root, -1 - par, N)].add(1, mode="drop")
    cnt_slot = jnp.zeros((Q,), jnp.int32).at[jnp.where(valid & ~root, par, Q)].add(1, mode="drop")
    lane = jnp.arange(Cg, dtype=jnp.int32)

    # each generation in (parent, ordinal) order: a child's place is its parent's first child slot (the exclusive
    # cumsum of the child counts of its generation, in this order) + its ordinal, so the adjoint's SPf slot j of a
    # generation is the next generation's item j
    def place(g, ORD, POS, at):
        mine = gk == g
        return ORD.at[jnp.where(mine, goff[g] + at, Q + Cg)].set(slot, mode="drop"), jnp.where(mine, at, POS)

    def o_body(s):
        g, ORD, POS = s
        off = _excl_cumsum(cnt_slot[lax.dynamic_slice_in_dim(ORD, goff[g], Cg)])
        return (g + 1,) + place(g + 1, ORD, POS, off[POS[jnp.clip(par, 0, Q - 1)]] + ordn)
    off_root = _excl_cumsum(cnt_root)
    ORD, POS = place(1, jnp.zeros((Q + Cg,), jnp.int32), jnp.zeros((Q,), jnp.int32),
                     off_root[jnp.clip(-1 - par, 0, N - 1)] + ordn)
    _, ORD, _ = lax.while_loop(lambda s: s[0] < G, o_body, (jnp.int32(1), ORD, POS))

    def adj_gen(X, cnt, b, nact, dx, cy):
        """the adjoint launch of a generation; dx: its children's initial-state cotangents (its SPf cotangent)"""
        if TREE_NEG == "detach":     # NEGATIVE CONTROL ONLY: children's cotangents are not injected
            dx = [jnp.zeros_like(d) for d in dx]
        ct = ct_user + [d.reshape(-1) for d in dx] + [jnp.ones((1,), jnp.float32)]
        outs = bind_call(*X, _excl_cumsum(cnt), cnt, jnp.zeros((b,), jnp.float32), *cy, *bc, nact, *ct, node=t_node)
        dp = list(outs[len(ib) + 1:])
        return list(outs[:len(ib)]), dp[:-1], jnp.sum(dp[-1])

    cy_children = lambda b: [jnp.zeros((b,) + tuple(ya[j].shape), ya[j].dtype) for j in yf]

    def r_body(s):
        g, dx_c, gt, mm = s
        n = ng[g]
        rows = lax.dynamic_slice_in_dim(ORD, goff[g], Cg)
        rx = jnp.where(lane < n, rows, rows[0])       # (inactive rows: a valid item, see the module docstring)
        X = [lf[rx] for lf in leaves]
        cnt = cnt_slot[rows]                          # (entries >= n: other items; those do not run)

        def one(b):
            gtb, dx, m = adj_gen([x[:b] for x in X], cnt[:b], b, n, dx_c, cy_children(b))
            return gtb, [_pad(d, Cg) for d in dx], m
        gtg, dx, m = _bucketed(n, Cg, one)
        return g - 1, dx, [a + b_ for a, b_ in zip(gt, gtg)], mm + m

    dx0 = [jnp.zeros((Cg,) + tuple(pav[i].shape), pav[i].dtype) for i in R.fl]
    gt0 = [jnp.zeros((_size(bav[i].shape),), bav[i].dtype) for i in ib]
    _, dx_c, gt, mm = lax.while_loop(lambda s: s[0] >= 1, r_body, (G, dx0, gt0, jnp.float32(0)))
    cy_root = [_inst(cts[len(ot) + 1 + j], (N,) + tuple(ya[j].shape), ya[j].dtype) for j in yf]
    gtr, dxr, mr = adj_gen(list(pi), cnt_root, N, jnp.int32(N), dx_c, cy_root)
    gt = [a + b_ for a, b_ in zip(gt, gtr)]
    mm = mm + mr

    bad = ovf | drop_risk | (mm > 0)
    poison = lambda x: jnp.where(bad, jnp.asarray(np.nan, x.dtype), x)
    g_pi = [None] * nP
    for k, i in enumerate(R.fl):
        if U_pi[i]:
            g_pi[i] = poison(dxr[k])
    g_bc = [None] * nB
    for k, i in enumerate(ib):
        g_bc[i] = poison(gt[k]).reshape(bav[i].shape)
    if o.get("rev_stats"):
        # opt-in: a host callback takes jax.jit off its C++ dispatch path (+1.3 s per call measured on a shower);
        # the gradient is NaN-poisoned on any failure without it
        st_ = dict(generations=G, children=total, max_generation=jnp.max(ng[1:Q + 1]),
                   generation_overflow=ovf, drop_risk=drop_risk, replay_mismatch=mm.astype(jnp.int32),
                   n_roots=jnp.int32(N), launches=G + 2)   # the recording primal + one adjoint per generation
        jax.debug.callback(lambda d: LAST.append({k: np.asarray(v) for k, v in d.items()}), st_)
    return _no_second_order((g_pi, g_bc))
