"""The shipped contract: the dense reference.

Public name: tallymap.reference.

  reference(fn, outputs=..., in_axes=..., max_steps=...)(*args)
      the DENSE meaning of a tallymap.vmap function: jax.vmap over items with real per-item tally arrays, summed.
      O(N * output size) memory: small problems only.  Differentiable with JAX's own rules (forward modes;
      reverse only if the item has no while_loop).  max_steps caps the item's top-level while loops exactly as
      tallymap.vmap does, including the rule that a truncated item makes every float tally NaN.
      A Records output: each item appends into its OWN buffer (index k = its append ordinal; K slots per item, K =
      the largest per-item count, so the eager call needs concrete counts, or K = the capacity when one is given),
      and the buffers are concatenated in item order: O(N * K) memory.  Overflow poisons as tallymap.vmap does.
"""
import jax
import jax.numpy as jnp
import numpy as np

from ._api import Records, Result, Sum, _poison
from ._rewrite import isfloat, retrace
from ._tally import Tally, rkey


def reference(fn, *, outputs, in_axes=0, max_steps=None):
    spec = {k: (v if isinstance(v, (Sum, Records)) else Sum(v)) for k, v in outputs.items()}
    rspec = {k: v for k, v in spec.items() if isinstance(v, Records)}
    names = sorted(k for k in spec if k not in rspec)
    rnames = sorted(rspec)
    shapes = {n: (spec[n].shape,) if isinstance(spec[n].shape, int) else tuple(spec[n].shape) for n in names}

    def mapped(*args):
        axes = in_axes if isinstance(in_axes, (tuple, list)) else (in_axes,) * len(args)
        meta = {}

        def run(mode, K):
            rinfo = (mode, tuple((r, (K.get(r, 0), rspec[r].fields)) for r in rnames)) if rnames else None
            fk = [(r, rkey(r, f), dt) for r in rnames for f, dt in rspec[r].fields] if mode == "ref" else []

            def one(*a):
                arrs = {n: jnp.zeros((int(np.prod(shapes[n], dtype=np.int64)),), spec[n].dtype) for n in names}
                arrs.update((key, jnp.zeros((K[r],), dt)) for r, key, dt in fk)
                t = Tally(arrs, dict(shapes, **{key: (K[r],) for r, key, _ in fk}), None, None,
                          {r: (jnp.int32(0),) for r in rnames}, rinfo)
                r = fn(t, *a)
                t2, y = (r, None) if isinstance(r, Tally) else r
                yl, meta["tree"] = jax.tree_util.tree_flatten(y)
                return ([t2._a[n] for n in names] + list(yl) + [t2._a[key] for _, key, _ in fk]
                        + [t2._r[r][0] for r in rnames])

            def one_capped(*a):
                if max_steps is None:
                    return one(*a), jnp.bool_(False)
                flat, tree = jax.tree_util.tree_flatten(a)
                cj = jax.make_jaxpr(lambda *l: one(*jax.tree_util.tree_unflatten(tree, list(l))))(*flat)
                o = retrace(cj, cap=max_steps)(*flat)
                return o[:-1], o[-1]
            return jax.vmap(one_capped, in_axes=tuple(axes))(*args), fk

        K = {}
        if rnames:                                  # the counts (one plain pass), then each item's own buffer
            (o, _), _ = run("count", {})
            cnt = dict(zip(rnames, o[len(o) - len(rnames):]))
            for r in rnames:
                K[r] = rspec[r].capacity or max(int(jnp.max(cnt[r])) if cnt[r].size else 0, 1)
        (outs, tr), fk = run("ref" if rnames else None, K)
        res = {n: outs[i].sum(0).reshape(shapes[n]) for i, n in enumerate(names)}
        nr = len(rnames)
        ny = len(outs) - len(names) - len(fk) - nr
        y = jax.tree_util.tree_unflatten(meta["tree"], outs[len(names):len(names) + ny])
        nt = jnp.sum(tr, dtype=jnp.int32)
        stats = {"n_truncated": nt}
        bad = (nt > 0) if max_steps is not None else None
        if rnames:
            bufs = dict(zip([key for _, key, _ in fk], outs[len(names) + ny:len(names) + ny + len(fk)]))
            kk = dict(zip(rnames, outs[len(outs) - nr:]))
            stats["appended"], stats["records_dropped"] = {}, {}
            ovf = jnp.bool_(False)
            for r in rnames:
                c = kk[r]
                tot = jnp.sum(c, dtype=jnp.int32)
                C = rspec[r].capacity or 1 << max(0, (int(tot) - 1).bit_length())
                off = jnp.cumsum(c, dtype=jnp.int32) - c
                # slot of item i's record k: off[i] + k for k < its count (the plain meaning: concatenation)
                kidx = jnp.arange(K[r])[None, :]
                dest = jnp.where(kidx < c[:, None], off[:, None] + kidx, C).reshape(-1)
                for f, dt in rspec[r].fields:
                    res[rkey(r, f)] = jnp.zeros((C,), dt).at[dest].set(bufs[rkey(r, f)].reshape(-1), mode="drop")
                stats["appended"][r] = tot
                stats["records_dropped"][r] = jnp.maximum(tot - C, 0)
                ovf = ovf | (tot > C)
            bad = ovf if bad is None else bad | ovf
        res = _poison(res, bad)                     # the same resource-limit rule as tm.vmap
        for r in rnames:
            res[r] = {f: res.pop(rkey(r, f)) if isfloat(dt) else jnp.where(bad, jnp.asarray(-1, dt),
                                                                           res.pop(rkey(r, f)))
                      for f, dt in rspec[r].fields}
        return Result(res, y, tr, stats)
    return mapped
