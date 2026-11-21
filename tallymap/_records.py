"""Records outputs, one pass (tallymap.Records, the default wherever it is valid): the pieces the primitive uses.

A one-pass program (ItemProgram.app set; built by _api.vmap) appends with _tally.claim_p: each record takes the next
slot of the launch's RAW buffer (an atomic counter per output) and writes its fields and its key (cid = the item's
index, k = its append ordinal) there; the item's final k is a per-item result (its count).

  canon(p, outs)    after every launch of such a program (_primitive._run): the raw layout -> the canonical one, from
                    the SAME launch's keys and counts: record (cid, k) goes to slot excl_cumsum(counts)[cid] + k, i.e.
                    the layout count-then-fill writes.  Every field tally of the output (primal, tangents, dual lanes)
                    moves with it, and the key tallies too, so every output of the primitive is deterministic and the
                    AD rules see the count-then-fill layout.
  twin(q)           the FILL TWIN of a one-pass program q (or of one derived from it: a fused jvp, its lanes): the
                    same program with each claim replaced by slot = off + k (k < cnt), off / cnt two hidden per-item
                    operands per output appended after q's own (the record: the primal's counts, tables()), and its
                    truncation flag (ItemProgram.trunc) = its final k != cnt, or truncated.  A launch that replays an
                    item in ANOTHER launch (the tangent node under partial evaluation, its replay adjoint) runs the
                    twin, so its records land at the primal's canonical slots and a replay that disagrees with the
                    primal's counts poisons its own outputs.
  tables(p, outs)   the twin's off / cnt operands from a one-pass launch's outputs (its counts).
"""
import jax
import jax.numpy as jnp
from jax import lax

from . import _limits as _lim
from ._rewrite import retrace
from ._tally import KI, KK, rkey


def _counts(p, outs):
    nt = len(p.out_tallies)
    return [outs[nt + 1 + cy] for _, _, cy in p.app[1]]


def canon(p, outs, axes=(), cid0=0):
    """outs: a one-pass launch's backend outputs ([sum tallies] + [stats] + [per-item results] (+ ring)) -> the same
    with every Records tally in the canonical layout.  axes: under a sharded jit, the mesh axes the items are split
    over (each device writes its records at the call's global slots: the devices' records in device order, so the
    psum of the devices' outputs adds disjoint slots).  cid0: the launch's first item index (the cid operand is the
    item index of the call: a device holds a contiguous slice of it, and its counts are indexed locally)."""
    outs = list(outs)
    names = [p.tallies[k].name for k in p.out_tallies]
    for (r, C, _), cnt in zip(p.app[1], _counts(p, outs)):
        pre = rkey(r, "")
        ki, kk = names.index(rkey(r, KI)), names.index(rkey(r, KK))
        cid, k = outs[ki], outs[kk]
        cnt = cnt.astype(jnp.int32)
        n = _lim.clip(_lim.total(cnt), C)             # (the counts' exact sum and prefix sums, clipped at C: a slot
        off = _lim.sub(_lim.prefix(cnt), _lim.count(cnt))     # at or past C holds no record, and nothing wraps)
        if axes:                                     # (this device's records after the earlier devices')
            dev = _lim.offsets(lax.all_gather(n, axes), C)[lax.axis_index(axes)]
            off = _lim.add(off, _lim.count(dev))
        off = _lim.clip(off, C)
        j = jnp.arange(C, dtype=jnp.int32)
        pos = jnp.where(j < n, off[jnp.clip(cid - cid0, 0, cnt.shape[0] - 1)] + k, C)
        for i, nm in enumerate(names):
            if nm.startswith(pre):
                o = outs[i]
                outs[i] = jnp.zeros_like(o).at[..., pos].set(o, mode="drop", unique_indices=True)
    return outs


def tables(p, outs):
    """the fill twin's hidden per-item operands [off, cnt] per Records output, from the counts of a launch of p (a
    one-pass program).  _api._COUNTS_HOOK (test-only) alters the counts handed over (the mismatch negative control)."""
    from . import _api
    res = []
    for (r, C, _), c in zip(p.app[1], _counts(p, outs)):
        c = c.astype(jnp.int32)
        if _api._COUNTS_HOOK is not None:
            c = _api._COUNTS_HOOK(r, c)
        res += [_lim.offsets(c, C), c]           # (clipped at the capacity: no slot past it is written)
    return res


def ntab(p):
    return 2 * len(p.app[1])


def twin(q):
    """the fill twin of the one-pass (or one-pass derived) program q; cached on q."""
    t = q.__dict__.get("_twin")
    if t is not None:
        return t
    from ._api import ItemProgram, _mk, _sds  # (here: _api imports this module's users)
    R = len(q.app[1])
    nT, nP = q.nT, q.nP
    cys = [cy for _, _, cy in q.app[1]]

    def item(*a):
        tabs = a[nT + nP:nT + nP + 2 * R]
        main = list(a[:nT + nP]) + list(a[nT + nP + 2 * R:])

        def claim(tok, when, k, *, r):
            off, cnt = tabs[2 * r], tabs[2 * r + 1]
            return tok, jnp.where(when & (k < cnt), off + k, -1).astype(jnp.int32)
        outs = list(retrace(q.cj, n_tally=nT, claim=claim)(*main))
        ys = outs[nT:]
        bad = jnp.bool_(False)
        for j, cy in enumerate(cys):
            bad = bad | (ys[cy] != tabs[2 * j + 1])
        if q.trunc is not None:
            bad = bad | ys[q.trunc].astype(bool).reshape(-1).any()
        return outs + [bad]
    av = q.in_avals()
    tav = [jax.ShapeDtypeStruct((), jnp.int32)] * (2 * R)
    cj = _mk(item, [_sds(x) for x in av[:nT + nP]] + tav + [_sds(x) for x in av[nT + nP:]])
    t = ItemProgram(cj, q.tallies, nP + 2 * R, q.nB, q.nY, q.spawn, q.tag + "+fill", where=q.where, nact=q.nact)
    object.__setattr__(q, "_twin", t)
    return t


def twin_args(p, pi, bc, tabs):
    """operands of p's fill twin from p's own (pi, bc) and tables(p, ...)."""
    return list(pi) + list(tabs), list(bc)

