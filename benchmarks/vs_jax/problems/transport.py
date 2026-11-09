"""Transport with an absorbing boundary into a sensor tally: the photon_toy item, measured against plain JAX.

ITEM      one photon packet.  It starts inside the unit disk and takes exponential flights of mean length lam
          with isotropic scattering, its weight attenuated by exp(-mu * s) each flight, until it leaves the disk
          or its weight falls below 1e-3 (absorbed).
TALLY     Sum((32,)): where a packet leaves the disk it deposits weight * efficiency into the sensor at that
          angle.  One emit per item, at the end.
DIVERGE   the mean free path lam.  A short lam means many scatters before escape, with a long tail.
THETA     (lam, mu): 2 parameters.  The 32-entry efficiency table is a broadcast argument; reverse mode gives it
          a sparse cotangent, which is why it is a table and not 32 more parameters here.

The plain-JAX arms put the whole item inside the `jax.vmap` (the emit is at the end, so a per-item scalar is
enough) and finish with one `.at[].add` over the batch.
"""
import arms as A
import harness as H
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm

NAME = "transport"
TITLE = "Transport, absorbing boundary, 32 sensors"
GROUP = "natural"
AXIS = "n"
SWEEP = {"full": (1 << 14, 1 << 16, 1 << 19, 1 << 22), "quick": (1 << 10, 1 << 12)}
DIVS = ("mid", "low", "high")
ARMS = ("tm_cuda", "tm_xla", "jax_while", "jax_scan", "jax_scan2x", "jax_ckpt", "jax_ckpt2x")
MODES = ("fwd", "jvp", "jacfwd", "grad")
CHECK_VAL = 1 << 12
DIV_VAL = 1 << 19      # the other divergence levels are measured at this one size

NS = 32
WMIN = 1e-3
LAM = {"low": 0.5, "mid": 0.2, "high": 0.05}
TH0 = {"low": (0.5, 0.5), "mid": (0.2, 0.5), "high": (0.05, 0.5)}


def step(c, key, th):
    """ONE flight of ONE packet -> carry.  carry = (position, direction, weight, k, escaped).  Every arm uses it."""
    p, d, w, k, out = c
    u = jax.random.uniform(jax.random.fold_in(key, k), (2,))
    s = -jnp.log1p(-u[0]) * th[0]
    p2 = p + s * d
    esc = jnp.sum(p2 * p2) >= 1.0
    w2 = w * jnp.exp(-th[1] * s)
    phi = 2.0 * jnp.pi * u[1]
    return p2, jnp.stack([jnp.cos(phi), jnp.sin(phi)]), w2, k + 1, esc


def alive(c):
    return (~c[4]) & (c[2] > WMIN)


def deposit(c, eff):
    """The emit of one finished packet: (sensor index, value).  Index -1 for an absorbed packet (dropped)."""
    p, d, w, k, esc = c
    i = ((jnp.arctan2(p[1], p[0]) + jnp.pi) * (NS / (2.0 * jnp.pi))).astype(jnp.int32) % NS
    return jnp.where(esc, i, -1), w * eff[i]


def c0_of(p0):
    return (p0, jnp.array([1.0, 0.0], jnp.float32), jnp.float32(1.0), jnp.int32(0), jnp.bool_(False))


def setup(n, div):
    rng = np.random.default_rng(0)
    r = np.sqrt(rng.uniform(0, 0.25, n))
    a = rng.uniform(0, 2 * np.pi, n)
    return dict(n=n, p0=jnp.asarray(np.stack([r * np.cos(a), r * np.sin(a)], 1), jnp.float32),
                keys=jax.random.split(jax.random.PRNGKey(0), n),
                th=jnp.asarray(TH0[div], jnp.float32),
                eff=jnp.asarray(0.8 + 0.2 * np.cos(np.arange(NS)), jnp.float32),
                tgt=jnp.asarray(rng.uniform(5.0, 30.0, NS), jnp.float32))


def telemetry(ctx):
    def one(p0, key, th):
        return lax.while_loop(alive, lambda c: step(c, key, th), c0_of(p0))[3]
    k = jax.jit(jax.vmap(one, in_axes=(0, 0, None)))(ctx["p0"], ctx["keys"], ctx["th"])
    return H.trip_stats(k)


def exit_of(ctx, arm, maxs):
    """Per-item (sensor index, weight) at the end of the walk, for the agreement cell.

    The tally alone cannot tell a mechanics difference from a chaotic one.  A packet takes up to a few hundred
    scattering flights, and a 1-ulp difference in an early flight sends it to a different sensor entirely, so
    the right question is what FRACTION of packets land in the same sensor, not what the summed tally does."""
    p0, keys, eff = ctx["p0"], ctx["keys"], ctx["eff"]

    def one(p, key, th):
        if arm.startswith("tm_"):
            raise ValueError(arm)
        if arm == "jax_while":
            c = lax.while_loop(alive, lambda c: step(c, key, th), c0_of(p))
        else:
            c, _ = A.masked_scan(lambda c: step(c, key, th), alive, c0_of(p), maxs, remat=A.remat_of(arm))
        i, v = deposit(c, eff)
        return i, c[3], v, c[2]
    if arm.startswith("tm_"):
        def packet(t, pp, key, th, ef):
            c = lax.while_loop(alive, lambda c: step(c, key, th), c0_of(pp))
            i, v = deposit(c, ef)
            return t.add("Q", i, v), (i, c[3], v, c[2])
        f = tm.vmap(packet, outputs={"Q": tm.Sum((NS,))}, in_axes=(0, 0, None, None),
                    backend="cuda" if arm == "tm_cuda" else "xla")
        return lambda th: f(p0, keys, th, eff).per_item
    return lambda th: jax.vmap(one, in_axes=(0, 0, None))(p0, keys, th)


def sim_of(ctx, arm, maxs):
    p0, keys, eff = ctx["p0"], ctx["keys"], ctx["eff"]
    if arm.startswith("tm_"):
        def packet(t, p, key, th, ef):
            c = lax.while_loop(alive, lambda c: step(c, key, th), c0_of(p))
            i, v = deposit(c, ef)
            return t.add("Q", i, v), c[3]
        f = tm.vmap(packet, outputs={"Q": tm.Sum((NS,))}, in_axes=(0, 0, None, None),
                    backend="cuda" if arm == "tm_cuda" else "xla")
        return lambda th: f(p0, keys, th, eff).outputs["Q"]

    def one(p, key, th, ef):
        if arm == "jax_while":
            c = lax.while_loop(alive, lambda c: step(c, key, th), c0_of(p))
        else:
            c, _ = A.masked_scan(lambda c: step(c, key, th), alive, c0_of(p), maxs, remat=A.remat_of(arm))
        return deposit(c, ef)

    def call(th):
        i, v = jax.vmap(one, in_axes=(0, 0, None, None))(p0, keys, th, eff)
        return A.scatter_add(NS, i, v)
    return call


def measured(ctx, arm, mode, maxs):
    sim = sim_of(ctx, arm, maxs)
    tgt = ctx["tgt"]
    loss = lambda th: jnp.sum((sim(th) - tgt) ** 2) / NS
    extra = dict(nsensors=NS, padded_steps=A.padded_count(arm, maxs), jacfwd_k=A.jacfwd_k(ctx["th"]))
    return A.mode_fn(mode, sim, loss, ctx["th"]), (ctx["th"],), extra


def check(ctx, maxs):
    out = A.cross_arm_check(lambda a: sim_of(ctx, a, maxs),
                            lambda a: (lambda t: jnp.sum((sim_of(ctx, a, maxs)(t) - ctx["tgt"]) ** 2) / NS),
                            ctx["th"])
    ref = [np.asarray(x) for x in jax.jit(exit_of(ctx, "tm_xla", maxs))(ctx["th"])]
    for arm in ("tm_cuda", "jax_while", "jax_scan"):
        try:
            i, k, v, w = (np.asarray(x) for x in jax.jit(exit_of(ctx, arm, maxs))(ctx["th"]))
            out[f"same_sensor_{arm}"] = float(np.mean(i == ref[0]))
            out[f"same_steps_{arm}"] = float(np.mean(k == ref[1]))
            out[f"peritem_rel_{arm}"] = H.rel(v, ref[2])
            out[f"weight_rel_{arm}"] = H.rel(w, ref[3])
            out[f"peritem_bitwise_{arm}"] = float(np.mean(v == ref[2]))
        except Exception as e:                                                  # noqa: BLE001
            out[f"same_sensor_{arm}"] = type(e).__name__
    return out
