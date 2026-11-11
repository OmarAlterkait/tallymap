"""The plain-JAX arms, written once so every problem compares against the same baselines.

The arms (`run.py --arms` lists the ones each problem supports):

| arm             | what it is                                                              | modes        |
|-----------------|-------------------------------------------------------------------------|--------------|
| `tm_cuda`       | tm.vmap, backend='auto' (the kernel: cuda on a GPU, the CPU target else) | all          |
| `tm_xla`        | tm.vmap, backend='xla' (the portable interpreter)                       | all          |
| `jax_while`     | jax.vmap of the same item in a lax.while_loop, then one .at[].add       | forward only |
| `jax_scan`      | jax.vmap of the same item as a masked lax.scan padded to the OBSERVED   | all          |
|                 | maximum trip count (an oracle: a real program cannot know it)           |              |
| `jax_scan2x`    | the same, padded to 2x the observed maximum (a realistic cap)           | all          |
| `jax_ckpt`      | `jax_scan` with a nested jax.checkpoint-ed inner scan                   | all          |
| `jax_ckpt2x`    | `jax_scan2x` with the same recursive checkpointing                      | all          |
| `diffrax`       | diffrax itself (the ODE problem only)                                   | all          |

`jax_while` has no gradient row because JAX cannot reverse-differentiate a `lax.while_loop` whose trip count
depends on the data: `jax.grad` of one raises.  That rewrite into a padded, masked `lax.scan` is what the other
baselines are.

Padding is never hidden.  Every scan row carries `padded_steps` (the number of step bodies the arm actually ran)
next to `steps_max` (the largest number any item needed), so a reader can see the waste.
"""
import math

import jax
import jax.numpy as jnp
from jax import lax

SCAN_ARMS = ("jax_scan", "jax_scan2x", "jax_ckpt", "jax_ckpt2x")
TM_ARMS = ("tm_cuda", "tm_xla")
ALL_ARMS = TM_ARMS + ("jax_while",) + SCAN_ARMS + ("diffrax",)
GRAD_CAPABLE = TM_ARMS + SCAN_ARMS + ("diffrax",)


def pad_for(arm, steps_max):
    """The padded trip count an arm uses, from the OBSERVED maximum.

    `jax_scan` / `jax_ckpt` use the observed maximum exactly: an oracle baseline, the best a padded scan could
    possibly do.  `jax_scan2x` / `jax_ckpt2x` use twice it, which is what a careful user picks when they do not
    know the maximum in advance."""
    m = int(steps_max)
    return max(1, m * (2 if arm.endswith("2x") else 1))


def segments(maxs):
    """(outer, inner) for the recursively checkpointed nested scan: outer*inner >= maxs, both near sqrt(maxs).

    outer*inner may exceed maxs; the extra iterations are no-ops on a finished item, and the arm reports
    `padded_steps = outer*inner`, so the cost is counted."""
    outer = max(1, int(math.isqrt(max(1, maxs))))
    inner = max(1, -(-maxs // outer))
    return outer, inner


def masked_scan(step, cond, c0, maxs, remat=False):
    """`step` applied to the carry while `cond(carry)` holds, as exactly `padded` iterations of a lax.scan.

    An item whose condition has gone false has its carry frozen, so this computes the same function as the
    while_loop version for every item that needs at most `padded` steps.  -> (carry, padded)."""
    def body(c, _):
        alive = cond(c)
        cn = step(c)
        return jax.tree.map(lambda a, b: jnp.where(alive, b, a), c, cn), None
    if not remat:
        c, _ = lax.scan(body, c0, None, length=maxs)
        return c, maxs
    outer, inner = segments(maxs)
    seg = jax.checkpoint(lambda c, _: lax.scan(body, c, None, length=inner))
    c, _ = lax.scan(seg, c0, None, length=outer)
    return c, outer * inner


def while_loop(step, cond, c0):
    """The same item loop with no bound at all: `lax.while_loop`.  Forward only."""
    return lax.while_loop(cond, step, c0)


def padded_loop(body, c0, maxs, remat=False):
    """`body(carry) -> carry` run exactly `padded` times, for a loop whose carry is SHARED by every item.

    Use this when the tally is written every step (a histogram over the whole batch): the loop then has to sit
    outside the `jax.vmap`, carrying the shared tally, and `body` does its own masking of finished items.  The
    `jax_while` version of the same loop runs `jnp.any(alive)` iterations, which is the batch MAXIMUM.
    -> (carry, padded)."""
    def sbody(c, _):
        return body(c), None
    if not remat:
        c, _ = lax.scan(sbody, c0, None, length=maxs)
        return c, maxs
    outer, inner = segments(maxs)
    seg = jax.checkpoint(lambda c, _: lax.scan(sbody, c, None, length=inner))
    c, _ = lax.scan(seg, c0, None, length=outer)
    return c, outer * inner


def remat_of(arm):
    return arm.startswith("jax_ckpt")


def padded_count(arm, maxs):
    """How many step bodies the arm actually runs.  0 = no padding at all (tallymap, or an unbounded while_loop;
    the while_loop arm instead runs the batch maximum, which the row reports as `steps_max`)."""
    if arm.startswith("tm_") or arm in ("jax_while", "diffrax"):
        return 0
    if arm.startswith("jax_ckpt"):
        outer, inner = segments(maxs)
        return outer * inner
    return maxs


def scatter_add(size, idx, val, dtype=jnp.float32):
    """The plain-JAX tally: one dense scatter-add, dropping out-of-range indices as tallymap drops them.

    `mode="drop"` is not enough on its own.  `x.at[i].add(v)` normalises a NEGATIVE index the way NumPy does,
    so an index of -1, which tallymap drops, lands in the LAST bin instead.  A plain-JAX arm written the
    obvious way therefore computes a different function: in the transport problem at its high divergence level
    every absorbed packet was being deposited into bin 31, and the tally differed from tallymap's by 4e-2 while
    every per-item value was bitwise identical.  Mapping anything outside [0, size) to `size` first makes the
    index genuinely out of bounds, which `mode="drop"` then drops."""
    idx = jnp.where((idx < 0) | (idx >= size), size, idx)
    return jnp.zeros((size,), dtype).at[idx].add(val, mode="drop")


def mode_fn(mode, sim, loss, theta):
    """The function a cell measures, for one of the four modes.

    fwd     the simulation itself
    jvp     one jax.jvp at a tangent of all ones (primal and tangent)
    jacfwd  jax.jacfwd of the scalar loss: P forward-mode lanes, where P = theta.size
    grad    jax.grad of the scalar loss: reverse mode
    """
    return {"fwd": sim,
            "jvp": lambda th: jax.jvp(sim, (th,), (jnp.ones_like(th),)),
            "jacfwd": jax.jacfwd(loss),
            "grad": jax.grad(loss)}[mode]


def jacfwd_k(theta):
    return int(jnp.asarray(theta).size)


def cross_arm_check(sim_of, loss_of, theta, arms=("tm_cuda", "jax_while", "jax_scan", "jax_ckpt"),
                    ref="tm_xla", grad_ref="jax_scan", grads=True):
    """Agreement of every arm against one reference arm, all in one process at a small size.

    `sim_of(arm) -> f(theta) -> tally` and `loss_of(arm) -> f(theta) -> scalar`.  Reported:
      rel_<arm>              the tally of that arm vs the reference arm's tally
      rel_grad_vs_jacfwd     reverse vs forward mode of the reference arm (an internal consistency check)
      rel_grad_vs_<grad_ref> the reference arm's gradient vs the padded scan's gradient of the same program
    These are f32 programs summed in a different order, so the contract is reassociation-level agreement, about
    one f32 ulp (1e-7), not bitwise equality."""
    import harness as _H
    out = {}
    base = jax.jit(sim_of(ref))(theta)
    for arm in arms:
        try:
            out[f"rel_{arm}"] = _H.rel(jax.jit(sim_of(arm))(theta), base)
        except Exception as e:                                                  # noqa: BLE001
            out[f"rel_{arm}"] = type(e).__name__
    if not grads:
        return out
    g = jax.jit(jax.grad(loss_of(ref)))(theta)
    out["rel_grad_vs_jacfwd"] = _H.rel(g, jax.jit(jax.jacfwd(loss_of(ref)))(theta))
    try:
        out[f"rel_grad_vs_{grad_ref}"] = _H.rel(g, jax.jit(jax.grad(loss_of(grad_ref)))(theta))
    except Exception as e:                                                      # noqa: BLE001
        out[f"rel_grad_vs_{grad_ref}"] = type(e).__name__
    return out
