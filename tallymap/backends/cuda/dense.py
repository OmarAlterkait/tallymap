"""Small dense linear algebra as plain JAX: the `lax.linalg` factorisations JAX has no pure-JAX form of.

XLA lowers `cholesky`, `geqrf` and `ormqr` to cuSOLVER custom calls, which a single fused kernel cannot call.  The
algorithms themselves are a few dozen floating-point operations for the matrix sizes an item uses, so the cuda
emitter emits them: op_table.py traces one of these functions on the equation's avals and emits the resulting jaxpr
(the same `_expand` route the RNG rules use).

Every loop here is a PYTHON loop over the static matrix dimension, so the emitted code is straight line: no loop
primitive, no dynamic row indexing, and nothing for tallymap's reverse mode to replay.  That is only reasonable
while the matrix is small, hence `MAXD`.  Items in this library's intended use solve 2x2 to 8x8 systems (an implicit
Runge-Kutta stage, a Newton step, a least-squares fit of a few parameters).

These are textbook unblocked algorithms with LAPACK's conventions, not a reimplementation of cuSOLVER:

| primitive                  | algorithm                        | convention it matches |
|---|---|---|
| `cholesky`                 | Cholesky-Banachiewicz, by column | the lower factor; the strict upper is ZERO |
| `geqrf`                    | Householder `geqr2` / `larfg`    | R above the diagonal, v below, `taus` as `larfg` |
| `ormqr`                    | the reflectors applied in turn   | Q = H_0 ... H_(k-1), H_j = I - tau_j v_j v_j^T |

XLA leaves `cholesky`'s strict upper triangle unspecified, so zero is as good a choice as any; every caller in
JAX masks it.

`lu`, `lu_pivots_to_permutation` and `triangular_solve` are not here: the cuda emitter expands them, after autodiff,
into JAX's own pure-JAX LU and a loop substitution (`_foreign.expand_late`).  That was measured against the
straight-line versions this file used to hold (2080 Ti, 1M items, `jnp.linalg.solve` of n = 3 / 6 / 12): the loop
forms were faster in 8 of 9 cells (forward, grad, jacfwd; up to 1.6x) and 9% slower in one.

Results agree with cuSOLVER only to floating-point rounding, like every other tallymap number.  A singular or
indefinite input gives inf / NaN here as it does there, but not necessarily the same inf / NaN pattern.  Real
float32 / float64 only: complex matrices are not supported.
"""
import jax.numpy as jnp
import numpy as np

MAXD = 16          # largest matrix dimension emitted; above it the primitive raises (op_table.py)
MAXB = 64          # largest leading batch axis of a vmapped factorisation (op_table.py maps over it)


def _c(v, dt):
    return jnp.asarray(np.array(v, dtype=np.dtype(dt)))


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


FUNCS = {"cholesky": cholesky, "geqrf": geqrf, "ormqr": ormqr}
# ndim of each operand WITHOUT a batch axis: op_table.py reads one extra leading axis as a batch and maps over it
OPERAND_NDIM = {"cholesky": (2,), "geqrf": (2,), "ormqr": (2, 1, 2)}
