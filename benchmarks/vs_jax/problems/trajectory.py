"""BAD FIT 3: the output IS the per-step trajectory.  The sweep axis is L, the trajectory length.

Each item runs an AR(1) recursion for L steps and must return all L samples, not a sum.  tallymap has no
per-step output: an item that wants its whole history has to build it in a per-item buffer, which leaves the
registers and goes to per-thread local memory.  `jax.vmap` + `lax.scan` stacks the per-step outputs for free.
This problem measures the break-even length.

ITEM      one AR(1) path of L steps, returning a float32[L] buffer as its per-item result.
TALLY     Sum((1,)): the final value, present only because the comparison needs one.
AXIS      L, the trajectory length (N is fixed at 2^14).
"""
import arms as A
import harness as H
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm

NAME = "trajectory"
TITLE = "Per-step trajectory as the output (a bad fit): the break-even in L"
GROUP = "bad-fit"
AXIS = "L"
SWEEP = {"full": (8, 16, 32, 64, 128, 256), "quick": (16, 64)}
DIVS = ("none",)
ARMS = ("tm_cuda", "tm_xla", "jax_scan")      # the trip count is fixed: a while_loop arm
                                                   # would be the same program as the scan one
MODES = ("fwd",)
CHECK_VAL = 16

N = 1 << 14


def setup(ell, div):
    return dict(L=ell, n=N, keys=jax.random.split(jax.random.PRNGKey(5), N), th=jnp.float32(0.9))


def telemetry(ctx):
    return H.trip_stats(np.full(ctx["n"], ctx["L"]))


def sim_of(ctx, arm, maxs):
    keys, ell = ctx["keys"], ctx["L"]

    def draw(k, x, key, th):
        return x * th + 0.1 * jax.random.normal(jax.random.fold_in(key, k))

    if arm.startswith("tm_"):
        def item(t, key, th):
            def body(c):
                k, x, buf = c
                x = draw(k, x, key, th)
                return k + 1, x, lax.dynamic_update_slice(buf, jnp.reshape(x, (1,)), (jnp.minimum(k, ell - 1),))
            c = lax.while_loop(lambda c: c[0] < ell, body,
                               (jnp.int32(0), jnp.float32(1.0), jnp.zeros((ell,), jnp.float32)))
            return t.add("s", 0, c[1]), c[2]
        f = tm.vmap(item, outputs={"s": tm.Sum((1,))}, in_axes=(0, None),
                    backend="cuda" if arm == "tm_cuda" else "xla")
        return lambda th: f(keys, th).per_item

    def one(key, th):
        def st(c, _):
            k, x = c
            x = draw(k, x, key, th)
            return (k + 1, x), x
        _, ys = lax.scan(st, (jnp.int32(0), jnp.float32(1.0)), None, length=ell)
        return ys

    return lambda th: jax.vmap(one, in_axes=(0, None))(keys, th)


def measured(ctx, arm, mode, maxs):
    return (sim_of(ctx, arm, maxs), (ctx["th"],),
            dict(L=ctx["L"], n=ctx["n"], padded_steps=A.padded_count(arm, maxs) or ctx["L"]))


def check(ctx, maxs):
    ref = jax.jit(sim_of(ctx, "tm_xla", maxs))(ctx["th"])
    out = {}
    for arm in ("tm_cuda", "jax_scan"):
        try:
            out[f"rel_{arm}"] = H.rel(jax.jit(sim_of(ctx, arm, maxs))(ctx["th"]), ref)
        except Exception as e:                                                  # noqa: BLE001
            out[f"rel_{arm}"] = type(e).__name__
    return out
