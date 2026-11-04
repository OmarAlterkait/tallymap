"""The shipped contract: the dense reference.

Public name: tallymap.reference.

  reference(fn, outputs=..., in_axes=..., max_steps=...)(*args)
      the DENSE meaning of a tallymap.vmap function: jax.vmap over items with real per-item tally arrays, summed.
      O(N * output size) memory: small problems only.  Differentiable with JAX's own rules (forward modes;
      reverse only if the item has no while_loop).  max_steps caps the item's top-level while loops exactly as
      tallymap.vmap does, including the rule that a truncated item makes every float tally NaN.
"""
import jax
import jax.numpy as jnp
import numpy as np

from ._api import Result, Sum, _poison
from ._rewrite import retrace
from ._tally import Tally


def reference(fn, *, outputs, in_axes=0, max_steps=None):
    spec = {k: (v if isinstance(v, Sum) else Sum(v)) for k, v in outputs.items()}
    names = sorted(spec)
    shapes = {n: (spec[n].shape,) if isinstance(spec[n].shape, int) else tuple(spec[n].shape) for n in names}

    def mapped(*args):
        axes = in_axes if isinstance(in_axes, (tuple, list)) else (in_axes,) * len(args)
        meta = {}

        def one(*a):
            t = Tally({n: jnp.zeros((int(np.prod(shapes[n], dtype=np.int64)),), spec[n].dtype) for n in names},
                      shapes)
            r = fn(t, *a)
            t2, y = (r, None) if isinstance(r, Tally) else r
            yl, meta["tree"] = jax.tree_util.tree_flatten(y)
            return [t2._a[n] for n in names] + list(yl)

        def one_capped(*a):
            if max_steps is None:
                return one(*a), jnp.bool_(False)
            flat, tree = jax.tree_util.tree_flatten(a)
            cj = jax.make_jaxpr(lambda *l: one(*jax.tree_util.tree_unflatten(tree, list(l))))(*flat)
            o = retrace(cj, cap=max_steps)(*flat)
            return o[:-1], o[-1]
        outs, tr = jax.vmap(one_capped, in_axes=tuple(axes))(*args)
        res = {n: outs[i].sum(0).reshape(shapes[n]) for i, n in enumerate(names)}
        y = jax.tree_util.tree_unflatten(meta["tree"], outs[len(names):])
        nt = jnp.sum(tr.astype(jnp.int32))
        res = _poison(res, (nt > 0) if max_steps is not None else None)   # the same resource-limit rule as tm.vmap
        return Result(res, y, tr, {"n_truncated": nt})
    return mapped
