"""The shipped contract: the dense reference and the statistical comparison of two implementations.

Public names: tallymap.reference and tallymap.compare.

  reference(fn, outputs=..., in_axes=..., max_steps=...)(*args)
      the DENSE meaning of a tallymap.vmap function: jax.vmap over items with real per-item tally arrays, summed.
      O(N * output size) memory: small problems only.  Differentiable with JAX's own rules (forward modes;
      reverse only if the item has no while_loop).  max_steps caps the item's top-level while loops exactly as
      tallymap.vmap does, including the rule that a truncated item makes every float tally NaN.
  compare(f, *args, other=...)      (public: tallymap.compare)
      distributional comparison of two runs of the same item program with the same per-item keys (CRN),
      e.g. backend='cuda' vs backend='lockstep'.  The criteria (criteria()) were pre-registered and are frozen
      (docs/validation.md).
"""
import jax
import jax.numpy as jnp
import numpy as np

from ._api import Result, Sum, _poison, _square_mode
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


def _flips(a, b, rtol):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "iub":
        bad = a != b
    else:
        bad = ~(np.isclose(a, b, rtol=rtol, atol=0) | (np.isnan(a) & np.isnan(b)))
    return bad.reshape(bad.shape[0], -1).any(1)


def _ulp(a):
    def f(x):
        x = jnp.asarray(x)
        if jnp.issubdtype(x.dtype, jnp.floating):
            return jnp.nextafter(x, jnp.asarray(np.inf, x.dtype))
        return x
    return jax.tree_util.tree_map(f, a)


def compare(f, *args, other=None, perturb=_ulp, rtol=1e-4, k_floor=5.0, verbose=True, floor_fn=None):
    """f, other: callables returning tallymap.Result for the same item program (e.g. backend='cuda' vs 'lockstep').
    NOT jitted (the floor run re-traces `floor_fn` in square mode).  The control is `other` run on 1-ulp perturbed
    float inputs (perturb).  floor_fn (default `other`): a tallymap.vmap function whose square-mode run gives the
    floor.  Criteria (criteria()), all required:
      flip rate(f vs other)                <= 3 * flip rate(control) + 3/N       (per-item results, rtol)
      rms over bins of |f - other| / floor <= 3 * rms(control) + 0.05
      max over bins of |f - other| / floor <= max(k_floor, 2 * the control's max)
    floor = sqrt(per-bin sum of squared emitted values (floor_fn) + its median over hit bins): the MC standard
    error of the bin, regularised so that one item moving into a sparse bin counts as ~1 sigma.  Only bins hit by
    the floor run enter the bin criteria (bins hit only by f are reported as empty_bin_mismatch, not judged).
    -> (ok, report)"""
    if other is None:
        raise ValueError("tallymap.compare: pass other= (e.g. the same function with backend='lockstep')")
    A = f(*args)
    Bm = other(*args)
    with _square_mode():
        B2 = (floor_fn or other)(*args)
    C = other(*perturb(args))
    ok, rep = criteria(A, Bm, B2, C, rtol=rtol, k_floor=k_floor)
    if verbose:
        print("tallymap.compare:", rep)
    return ok, rep


def criteria(A, Bm, B2, C, rtol=1e-4, k_floor=5.0):
    """the compare criteria on precomputed Results: A = f, Bm = other, B2 = square-mode floor run, C = control."""
    rep, ok = {}, True
    ya, yb, yc = ([np.asarray(x) for x in jax.tree_util.tree_leaves(r.per_item)] + [np.asarray(r.truncated)]
                  for r in (A, Bm, C))
    N = int(yb[0].shape[0])
    if N:
        fl, fc = np.zeros(N, bool), np.zeros(N, bool)
        for a, b, c in zip(ya, yb, yc):
            fl |= _flips(a, b, rtol)
            fc |= _flips(c, b, rtol)
        rep["flip_rate"], rep["flip_rate_control"] = float(fl.mean()), float(fc.mean())
        ok &= rep["flip_rate"] <= 3 * rep["flip_rate_control"] + 3.0 / N
    for n in Bm.outputs:
        a, b, c = (np.asarray(r.outputs[n], np.float64) for r in (A, Bm, C))
        w2 = np.asarray(B2.outputs[n], np.float64)
        hit = w2 > 0
        # + one typical bin's sum of w^2: a single item landing in a sparse bin is a ~1-sigma event, not 6
        floor = np.sqrt(np.where(hit, w2, 0.0) + (np.median(w2[hit]) if hit.any() else 1.0))
        z, zc = (np.abs(a - b) / floor)[hit], (np.abs(c - b) / floor)[hit]
        r = dict(rms_z=float(np.sqrt(np.mean(z ** 2))) if z.size else 0.0,
                 rms_z_control=float(np.sqrt(np.mean(zc ** 2))) if zc.size else 0.0,
                 max_z=float(z.max()) if z.size else 0.0, max_z_control=float(zc.max()) if zc.size else 0.0,
                 empty_bin_mismatch=int(((~hit) & (a != b)).sum()),
                 rel_total=float(abs(a.sum() - b.sum()) / max(abs(b.sum()), 1e-300)))
        rep[n] = r
        ok &= r["rms_z"] <= 3 * r["rms_z_control"] + 0.05 and r["max_z"] <= max(k_floor, 2 * r["max_z_control"])
    rep["ok"] = bool(ok)
    return bool(ok), rep
