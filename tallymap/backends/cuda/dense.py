"""Small dense linear algebra as plain JAX: the `lax.linalg` primitives an item can use.

XLA lowers `lu`, `cholesky`, `geqrf`, `ormqr` and `triangular_solve` to cuSOLVER / cuBLAS custom calls, which a
single fused kernel cannot call.  The algorithms themselves are a few dozen floating-point operations for the
matrix sizes an item uses, so the cuda emitter emits them: op_table.py traces one of these functions on the
equation's avals and emits the resulting jaxpr (the same `_expand` route the RNG rules use).

Every loop here is a PYTHON loop over the static matrix dimension, so the emitted code is straight line: no loop
primitive, no dynamic row indexing except the LU pivot swap, and nothing for tallymap's reverse mode to replay.
That is only reasonable while the matrix is small, hence `MAXD`.  Items in this library's intended use solve
2x2 to 8x8 systems (an implicit Runge-Kutta stage, a Newton step, a least-squares fit of a few parameters).

These are textbook unblocked algorithms with LAPACK's conventions, not a reimplementation of cuSOLVER:

| primitive                  | algorithm                        | convention it matches |
|---|---|---|
| `lu`                       | right-looking `getf2`            | `pivots[j]` = the row swapped with j |
| `lu_pivots_to_permutation` | the swaps applied to iota        | - |
| `cholesky`                 | Cholesky-Banachiewicz, by column | the lower factor; the strict upper is ZERO |
| `geqrf`                    | Householder `geqr2` / `larfg`    | R above the diagonal, v below, `taus` as `larfg` |
| `ormqr`                    | the reflectors applied in turn   | Q = H_0 ... H_(k-1), H_j = I - tau_j v_j v_j^T |
| `triangular_solve`         | forward / back substitution      | - |

XLA leaves `cholesky`'s strict upper triangle unspecified, so zero is as good a choice as any; every caller in
JAX masks it.

Results agree with cuSOLVER only to floating-point rounding, like every other tallymap number (the summation
order of a substitution is not XLA's).  A singular or indefinite input gives inf / NaN here as it does there, but
not necessarily the same inf / NaN pattern.  Real float32 / float64 only: complex matrices are not supported.
"""
import jax.numpy as jnp
import numpy as np
from jax import lax

MAXD = 16          # largest matrix dimension emitted; above it the primitive raises (op_table.py)
MAXB = 64          # largest leading batch axis of a vmapped factorisation (op_table.py maps over it)


def _c(v, dt):
    return jnp.asarray(np.array(v, dtype=np.dtype(dt)))


# --------------------------------------------------------------------------------------------------- LU
def lu(a):
    """-> (packed LU, pivots, permutation), as lax.linalg.lu."""
    m, n = a.shape
    k = min(m, n)
    ipiv = []
    perm = jnp.arange(m, dtype=jnp.int32)
    for j in range(k):
        col = jnp.abs(a[j:, j])                                  # pivot: first row of largest magnitude
        p = jnp.argmax(col).astype(jnp.int32) + np.int32(j)
        ipiv.append(p)
        rj = a[j]
        rp = lax.dynamic_index_in_dim(a, p, 0, keepdims=False)
        a = a.at[j].set(rp)
        a = lax.dynamic_update_index_in_dim(a, rj, p, 0)
        pj = perm[j]
        pp = lax.dynamic_index_in_dim(perm, p, 0, keepdims=False)
        perm = perm.at[j].set(pp)
        perm = lax.dynamic_update_index_in_dim(perm, pj, p, 0)
        if j + 1 < m:
            sub = a[j + 1:, j] / a[j, j]
            a = a.at[j + 1:, j].set(sub)
            if j + 1 < n:
                a = a.at[j + 1:, j + 1:].add(-sub[:, None] * a[j, j + 1:][None, :])
    piv = jnp.stack(ipiv) if ipiv else jnp.zeros((0,), jnp.int32)
    return a, piv, perm


def lu_pivots_to_permutation(pivots, *, permutation_size):
    """-> the permutation of `permutation_size` elements the swaps in `pivots` produce."""
    perm = jnp.arange(permutation_size, dtype=pivots.dtype)
    for i in range(pivots.shape[0]):
        p = pivots[i]
        pi = perm[i]
        pp = lax.dynamic_index_in_dim(perm, p, 0, keepdims=False)
        perm = perm.at[i].set(pp)
        perm = lax.dynamic_update_index_in_dim(perm, pi, p, 0)
    return perm


# --------------------------------------------------------------------------------------------------- Cholesky
def cholesky(a):
    """-> the lower Cholesky factor; the strict upper triangle is zero.  NaN for an indefinite input."""
    n = a.shape[0]
    low = jnp.zeros_like(a)
    for j in range(n):
        d = a[j, j]
        if j:
            d = d - jnp.sum(low[j, :j] * low[j, :j])
        ljj = jnp.sqrt(d)
        low = low.at[j, j].set(ljj)
        if j + 1 < n:
            s = a[j + 1:, j]
            if j:
                s = s - jnp.sum(low[j + 1:, :j] * low[j, :j][None, :], axis=1)
            low = low.at[j + 1:, j].set(s / ljj)
    return low


# --------------------------------------------------------------------------------------------------- QR
def geqrf(a):
    """-> (R in the upper triangle with the Householder vectors below it, taus), as lax.linalg.geqrf."""
    m, n = a.shape
    k = min(m, n)
    dt = a.dtype
    one, zero = _c(1, dt), _c(0, dt)
    taus = []
    for j in range(k):
        alpha = a[j, j]
        x = a[j + 1:, j]
        xn2 = jnp.sum(x * x) if x.shape[0] else zero
        safe = xn2 > zero                                        # larfg: a zero tail needs no reflector
        s = jnp.where(alpha >= zero, one, -one)                  # Fortran SIGN(., alpha), so sign(0) = +1
        beta = -s * jnp.sqrt(alpha * alpha + xn2)
        tau = jnp.where(safe, (beta - alpha) / jnp.where(safe, beta, one), zero)
        den = jnp.where(safe, alpha - beta, one)
        v = jnp.where(safe, x / den, jnp.zeros_like(x))
        taus.append(tau)
        a = a.at[j, j].set(jnp.where(safe, beta, alpha))
        a = a.at[j + 1:, j].set(v)
        if j + 1 < n:
            vv = jnp.concatenate([one.reshape((1,)), v])
            blk = a[j:, j + 1:]
            w = jnp.sum(vv[:, None] * blk, axis=0)
            a = a.at[j:, j + 1:].add(-tau * vv[:, None] * w[None, :])
    return a, (jnp.stack(taus) if taus else jnp.zeros((0,), dt))


def ormqr(a, taus, c, *, left, transpose):
    """-> Q @ c, Q.T @ c, c @ Q or c @ Q.T, with Q the reflectors of `geqrf`, as lax.linalg.ormqr."""
    if not left:                                                 # c @ Q = (Q.T @ c.T).T
        return ormqr(a, taus, c.T, left=True, transpose=not transpose).T
    k = taus.shape[0]
    one = _c(1, a.dtype)
    # Q = H_0 ... H_(k-1) and every H is symmetric, so Q.T @ c applies H_0 first and Q @ c applies H_(k-1) first
    for j in (range(k) if transpose else range(k - 1, -1, -1)):
        v = jnp.concatenate([one.reshape((1,)), a[j + 1:, j]])
        blk = c[j:, :]
        w = jnp.sum(v[:, None] * blk, axis=0)
        c = c.at[j:, :].add(-taus[j] * v[:, None] * w[None, :])
    return c


# --------------------------------------------------------------------------------------------------- substitution
def _subst(a, b, lower, unit_diagonal):
    """a x = b by forward (lower) or back (upper) substitution; a is (m, m), b is (m, p)."""
    m = a.shape[0]
    xs = [None] * m
    for i in (range(m) if lower else range(m - 1, -1, -1)):
        s = b[i]
        for j in (range(i) if lower else range(i + 1, m)):
            s = s - a[i, j] * xs[j]
        xs[i] = s if unit_diagonal else s / a[i, i]
    return jnp.stack(xs)


def triangular_solve(a, b, *, left_side, lower, transpose_a, conjugate_a, unit_diagonal):
    """as lax.linalg.triangular_solve, for a real matrix."""
    at = a.T if transpose_a else a
    low = bool(lower) != bool(transpose_a)
    if left_side:
        return _subst(at, b, low, unit_diagonal)
    return _subst(at.T, b.T, not low, unit_diagonal).T           # x a = b  <=>  a.T x.T = b.T


FUNCS = {"lu": lu, "lu_pivots_to_permutation": lu_pivots_to_permutation, "cholesky": cholesky,
         "geqrf": geqrf, "ormqr": ormqr, "triangular_solve": triangular_solve}
# ndim of each operand WITHOUT a batch axis: op_table.py reads one extra leading axis as a batch and maps over it
OPERAND_NDIM = {"lu": (2,), "lu_pivots_to_permutation": (1,), "cholesky": (2,), "geqrf": (2,),
                "ormqr": (2, 1, 2), "triangular_solve": (2, 2)}
