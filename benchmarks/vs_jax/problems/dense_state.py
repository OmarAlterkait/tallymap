"""BAD FIT 2: a per-item DENSE D x D state.  The sweep axis is D, and the point is to find the crossover.

Each item multiplies its own length-D vector by a shared D x D matrix eight times, with a tanh in between, and
tallies the sum of the result.  The trip count is uniform; the per-step work is dense linear algebra.  A
persistent-thread kernel runs that as scalar code in registers and local memory, while `jax.vmap` turns it into
one batched GEMM.  Beyond some D the GEMM wins, and this problem measures where.

ITEM      one vector, 8 steps of x <- tanh(A x).
TALLY     Sum((1,)): the sum of the final vector.
AXIS      D, the state dimension (the N is fixed at 2^14).
THETA     the matrix A itself, D*D entries, which is also why `jacfwd` is not in the mode list here: D*D forward
          lanes is not a sensible program.
"""
import arms as A
import harness as H
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm

NAME = "dense_state"
TITLE = "Per-item dense D x D state (a bad fit): the crossover in D"
GROUP = "bad-fit"
AXIS = "d"
SWEEP = {"full": (4, 8, 16, 24, 32, 48, 64), "quick": (8, 32)}
DIVS = ("none",)
ARMS = ("tm_cuda", "tm_lockstep", "jax_while", "jax_scan")
MODES = ("fwd", "grad")
CHECK_VAL = 8

N = 1 << 14
STEPS = 8


def step(c, mat):
    """ONE dense step of ONE item.  Shared by every arm."""
    k, x = c
    return k + 1, jnp.tanh(mat @ x)


def cond(c):
    return c[0] < STEPS


def setup(d, div):
    rng = np.random.default_rng(0)
    return dict(d=d, n=N, mat=jnp.asarray(rng.normal(0, 1.0 / np.sqrt(d), (d, d)), jnp.float32),
                x0=jnp.asarray(rng.normal(0, 1, (N, d)), jnp.float32))


def telemetry(ctx):
    return H.trip_stats(np.full(ctx["n"], STEPS))


def sim_of(ctx, arm, maxs):
    x0 = ctx["x0"]
    if arm.startswith("tm_"):
        def item(t, x, mat):
            c = lax.while_loop(cond, lambda c: step(c, mat), (jnp.int32(0), x))
            return t.add("s", 0, jnp.sum(c[1])), None
        f = tm.vmap(item, outputs={"s": tm.Sum((1,))}, in_axes=(0, None),
                    backend="cuda" if arm == "tm_cuda" else "lockstep")
        return lambda mat: f(x0, mat).outputs["s"]

    def one(x, mat):
        c0 = (jnp.int32(0), x)
        if arm == "jax_while":
            c = lax.while_loop(cond, lambda c: step(c, mat), c0)
        else:
            c, _ = A.masked_scan(lambda c: step(c, mat), cond, c0, maxs, remat=A.remat_of(arm))
        return jnp.sum(c[1])

    return lambda mat: jnp.sum(jax.vmap(one, in_axes=(0, None))(x0, mat), keepdims=True)


def loss_of(ctx, arm, maxs):
    sim = sim_of(ctx, arm, maxs)
    return lambda mat: jnp.sum(sim(mat) ** 2)


def measured(ctx, arm, mode, maxs):
    extra = dict(d=ctx["d"], n=ctx["n"], steps=STEPS, padded_steps=A.padded_count(arm, maxs))
    return A.mode_fn(mode, sim_of(ctx, arm, maxs), loss_of(ctx, arm, maxs), ctx["mat"]), (ctx["mat"],), extra


def check(ctx, maxs):
    return A.cross_arm_check(lambda a: sim_of(ctx, a, maxs), lambda a: loss_of(ctx, a, maxs), ctx["mat"],
                             arms=("tm_cuda", "jax_while", "jax_scan"), grad_ref="jax_scan")
