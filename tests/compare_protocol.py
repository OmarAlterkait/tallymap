"""The pre-registered `compare` protocol (docs/validation.md): the statistical comparison of two implementations of
the same item program on the same per-item keys (common random numbers), e.g. backend='cuda' vs backend='xla'.
A test protocol, not library API.  The criteria (criteria()) were pre-registered and are frozen.

    from compare_protocol import compare
    ok, rep = compare(f, *args, other=g, floor_fn=g_sq)
"""
import jax
import jax.numpy as jnp
import numpy as np


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


def compare(f, *args, other, floor_fn, perturb=_ulp, rtol=1e-4, k_floor=5.0, verbose=True):
    """f, other: callables returning tallymap.Result for the same item program (e.g. backend='cuda' vs 'xla').
    The control is `other` run on 1-ulp perturbed float inputs (perturb).  floor_fn: the same item program with
    every emitted value squared (its tallies are the per-bin sums of w^2), usually on other's backend.
    Criteria (criteria()), all required:
      flip rate(f vs other)                <= 3 * flip rate(control) + 3/N       (per-item results, rtol)
      rms over bins of |f - other| / floor <= 3 * rms(control) + 0.05
      max over bins of |f - other| / floor <= max(k_floor, 2 * the control's max)
    floor = sqrt(per-bin sum of squared emitted values (floor_fn) + its median over hit bins): the MC standard
    error of the bin, regularised so that one item moving into a sparse bin counts as ~1 sigma.  Only bins hit by
    the floor run enter the bin criteria (bins hit only by f are reported as empty_bin_mismatch, not judged).
    -> (ok, report)"""
    A = f(*args)
    Bm = other(*args)
    B2 = floor_fn(*args)
    C = other(*perturb(args))
    ok, rep = criteria(A, Bm, B2, C, rtol=rtol, k_floor=k_floor)
    if verbose:
        print("compare:", rep)
    return ok, rep


def criteria(A, Bm, B2, C, rtol=1e-4, k_floor=5.0):
    """the compare criteria on precomputed Results: A = f, Bm = other, B2 = squared-emit floor run, C = control."""
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
