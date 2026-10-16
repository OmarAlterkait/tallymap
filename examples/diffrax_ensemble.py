"""A diffrax solve inside a tallymap item: one stiff ODE per parameter sample, results summed into a histogram.

    python examples/diffrax_ensemble.py            # CPU: the lockstep backend; GPU: one CUDA kernel, and a timing
    env N=4096  SOLVER=kvaerno5|tsit5  BENCH=0

Needs diffrax (pip install diffrax); tallymap itself does not depend on it.

The item function is the diffeqsolve call a diffrax user already writes.  It runs un-batched, one trajectory per
item, so every step-size decision is that item's own: tallymap's kernel retires an item as soon as its solve
converges, instead of running the whole batch for as many steps as its stiffest member needs.

Two things a diffrax user must know inside an item (docs/equinox.md has the rest):
  * pass adjoint=diffrax.ForwardMode().  That makes the solver's loop a plain lax.while_loop, which tallymap
    differentiates itself (forward AND reverse) by replaying it;
  * set EQX_ON_ERROR before importing equinox, as this file does.  Equinox's default raises from a host callback,
    and an item runs inside one kernel, which cannot call the host.  `off` is what this file sets, because it also
    takes a gradient: with `nan` the gradient of an IMPLICIT solver inside an item comes out NaN (docs/equinox.md,
    "EQX_ON_ERROR").  For a forward-only item prefer `nan`, which marks a failed solve instead of hiding it.
"""
import os
import time

os.environ.setdefault("EQX_ON_ERROR", "off")

import diffrax as dfx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import tallymap as tm  # noqa: E402

NB = 16                                                     # histogram bins
N = int(os.environ.get("N", 4096))
SOLVER = os.environ.get("SOLVER", "kvaerno5")


def vector_field(t, y, mu):
    """Van der Pol: stiff for large mu, and mu varies across the ensemble."""
    return jnp.stack([y[1], mu * ((1.0 - y[0] * y[0]) * y[1] - y[0])])


def solve(mu):
    """ONE solve: the literal diffrax call, written as if there were no tallymap."""
    solver = dfx.Kvaerno5() if SOLVER == "kvaerno5" else dfx.Tsit5()
    sol = dfx.diffeqsolve(
        dfx.ODETerm(vector_field), solver,
        t0=0.0, t1=2.0, dt0=0.01, y0=jnp.array([2.0, 0.0]), args=mu,
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-7),
        adjoint=dfx.ForwardMode(),                           # <- the one line an item needs
        max_steps=16384)
    return sol.ys[0, 0]


def item(t, mu):
    """ONE item: solve, then tally the result.  t is the item's Tally."""
    y = solve(mu)
    # y(2) lands in about [-2, -1] for this mu range, so bin that interval (out-of-range bins are dropped)
    t = t.add("hist", jnp.clip(((y + 2.0) * NB).astype(jnp.int32), 0, NB - 1), 1.0)
    t = t.add("mean_y", 0, y)
    return t, y


f = tm.vmap(item, outputs={"hist": tm.Sum((NB,)), "mean_y": tm.Sum((1,))}, max_steps=16384)

mus = jnp.asarray(5.0 + 15.0 * np.linspace(0.0, 1.0, N), jnp.float32)
res = jax.jit(f)(mus)
print(f"{SOLVER}, N={N}: hist {np.asarray(res.outputs['hist'])}")
print(f"mean y(2) = {float(res.outputs['mean_y'][0]) / N:.6f}   truncated {int(res.stats['n_truncated'])}")

# ------------------------------------------------------------------ the check: jax.vmap of the same function
M = min(N, 256)
want = jax.jit(jax.vmap(solve))(mus[:M])
got = np.asarray(res.per_item[:M], np.float64)
err = float(np.max(np.abs(got - np.asarray(want, np.float64))))
scale = float(np.max(np.abs(np.asarray(want, np.float64))))
print(f"vs jax.vmap(diffeqsolve) on {M} items: max |diff| {err:.3e} (|y| <= {scale:.3g})")
assert err <= 1e-3 * scale + 1e-5, f"disagrees with jax.vmap: {err:.3e}"

# ------------------------------------------------------------------ the gradient JAX cannot take by itself
loss = lambda p: jnp.sum(jax.jit(f)(p).per_item)
g = np.asarray(jax.jit(jax.grad(loss))(mus[:M]), np.float64)
gf = np.asarray(jax.jit(jax.jacfwd(loss))(mus[:M]), np.float64)   # loss is scalar, so this IS the gradient
gerr = float(np.max(np.abs(g - gf)))
print(f"grad vs jacfwd on {M} items: max |diff| {gerr:.3e} (|grad| <= {np.max(np.abs(gf)):.3g}) "
      f"(jax.grad(jax.vmap(solve)) does not exist: ForwardMode's loop is a lax.while_loop)")
assert gerr <= 1e-2 * float(np.max(np.abs(gf))) + 1e-4, f"grad disagrees with jacfwd: {gerr:.3e}"


def timeit(fn, *a):
    fn(*a)
    jax.block_until_ready(fn(*a))
    t0 = time.perf_counter()
    for _ in range(3):
        jax.block_until_ready(fn(*a))
    return (time.perf_counter() - t0) / 3


if jax.default_backend() == "gpu" and os.environ.get("BENCH", "1") == "1":
    tm_t = timeit(jax.jit(lambda p: f(p).per_item), mus)
    try:
        jx_t = timeit(jax.jit(jax.vmap(solve)), mus)
        print(f"forward, N={N}: tallymap {tm_t * 1e3:.1f} ms   jax.vmap(diffeqsolve) {jx_t * 1e3:.1f} ms"
              f"   speedup {jx_t / tm_t:.1f}x")
    except Exception as e:                       # jax.vmap of a big stiff ensemble can run out of memory
        print(f"forward, N={N}: tallymap {tm_t * 1e3:.1f} ms;  jax.vmap failed: {type(e).__name__}")

print("DIFFRAX_ENSEMBLE OK")
