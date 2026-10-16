"""Branching shower via `spawn`, measured against a generation-by-generation plain-JAX baseline.

ITEM      one particle.  It steps along z, depositing a fraction of its energy per step, until its energy falls
          below a floor or it has taken 24 steps.  At each step, with a small probability and only while its
          generation is below the depth limit, it SPLITS: the child takes a share of the energy and runs the
          same function.  A particle splits at most ONCE, so the tree is binary and generation g holds at most
          as many particles as generation g-1.
TALLY     Sum((16,)): the deposit profile along z, over every particle of every generation.
DIVERGE   the per-step split probability, which sets how wide the tree gets: the three levels give roughly
          20%, 50% and 85% of particles splitting.
THETA     (deposit fraction, child energy share): 2 parameters.  The gradient goes through every generation
          (tallymap's tree adjoint: one adjoint launch per generation, deepest first).

The plain-JAX baseline is the generation-by-generation program a JAX user has to write, because `jax.vmap` has
no way to grow its batch: run a fixed-capacity batch of particles, collect each one's child and a validity
flag, COMPACT the valid children into a dense fixed-capacity array (a cumsum of the flag gives each child its
slot), and run the next generation on that array.  The compaction is plain `.at[].set`, and it is what makes
the arm work at all; the cost it cannot avoid is that every generation is sized for its capacity, not for the
particles that are actually alive in it.

  `jax_gen`     capacity = the OBSERVED population of each generation, rounded up to a power of two.  An
                oracle: a real program does not know the population of generation 5 before it runs.
  `jax_gen2x`   twice that capacity, which is what a program that must not drop a particle has to reserve.

Both use a masked `lax.scan` padded to the observed maximum step count for the per-particle loop, since a
`while_loop` would make the gradient impossible.
"""
import arms as A
import harness as H
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm

NAME = "shower"
TITLE = "Branching shower (spawn)"
GROUP = "natural"
AXIS = "n"
SWEEP = {"full": (1 << 12, 1 << 14, 1 << 16, 1 << 18), "quick": (1 << 7, 1 << 9)}
DIVS = ("mid", "low", "high")
ARMS = ("tm_cuda", "tm_lockstep", "jax_gen", "jax_gen2x")
MODES = ("fwd", "grad")
CHECK_VAL = 1 << 7
DIV_VAL = 1 << 16      # the other divergence levels are measured at this one size
PROBE_KEYS = ("populations",)

NZ, DEPTH, KMAX = 16, 6, 24
EMIN = 1e-3
PSPLIT = {"low": 0.01, "mid": 0.03, "high": 0.08}      # per STEP, and a particle splits at most once
TH0 = (0.2, 0.6)


def step(s, th, psplit):
    """ONE step of ONE particle -> (state, z bin, deposit, child state, split flag).  Shared by every arm.

    `s` is a dict so the tallymap arm can hand the child state straight to `tally.spawn`."""
    k = jax.random.fold_in(s["key"], s["n"])
    u = jax.random.uniform(k, (3,))
    z = s["z"] - jnp.log1p(-u[0]) * 0.5
    dep = s["E"] * th[0]
    split = (u[1] < psplit) & (s["g"] < DEPTH) & (~s["did"]) & (s["E"] > EMIN)
    Ec = (s["E"] - dep) * th[1] * (0.5 + 0.5 * u[2])
    child = dict(E=Ec, z=z, key=jax.random.fold_in(k, 1), g=s["g"] + 1, n=jnp.int32(0),
                 did=jnp.bool_(False))
    E = s["E"] - dep - jnp.where(split, Ec, 0.0)
    sn = dict(E=E, z=z, key=s["key"], g=s["g"], n=s["n"] + 1, did=s["did"] | split)
    return sn, zbin(z), dep, child, split


def zbin(z):
    return jnp.clip(jnp.floor(z * 2.0).astype(jnp.int32), 0, NZ - 1)


def cond(s):
    return (s["E"] > EMIN) & (s["n"] < KMAX)


def roots_of(n, rng):
    return dict(E=jnp.asarray(rng.uniform(1.0, 2.0, n), jnp.float32), z=jnp.zeros((n,), jnp.float32),
                key=jax.random.split(jax.random.PRNGKey(0), n), g=jnp.zeros((n,), jnp.int32),
                n=jnp.zeros((n,), jnp.int32), did=jnp.zeros((n,), bool))


def setup(n, div):
    rng = np.random.default_rng(0)
    return dict(n=n, psplit=PSPLIT[div], roots=roots_of(n, rng), th=jnp.asarray(TH0, jnp.float32),
                wz=jnp.asarray(rng.uniform(0.5, 1.5, NZ), jnp.float32))


def _one_particle_while(s, th, psplit):
    """One particle's whole loop, unbounded: -> (steps, deposit sum per bin is tallied by the caller)."""
    def body(c):
        s, dep, nb, child, has = c
        sn, b, d, ch, sp = step(s, th, psplit)
        return (sn, dep + d, nb, jax.tree.map(lambda a, b_: jnp.where(sp, b_, a), child, ch), has | sp)
    s0 = (s, jnp.float32(0.0), jnp.int32(0), jax.tree.map(jnp.zeros_like, s), jnp.bool_(False))
    return lax.while_loop(lambda c: cond(c[0]), body, s0)


def telemetry(ctx):
    """Steps per particle and the POPULATION of every generation, measured with the tallymap arm's own tree.

    The generation populations are what the plain-JAX arm's capacities come from, so they have to be measured.
    """
    th, ps, roots = ctx["th"], ctx["psplit"], ctx["roots"]
    pops, ks = [ctx["n"]], []
    cur = roots
    for _ in range(DEPTH + 1):
        c = jax.jit(jax.vmap(lambda s: _one_particle_while(s, th, ps)))(cur)
        ks.append(np.asarray(c[0]["n"]))
        has = np.asarray(c[4])
        if not has.any():
            break
        idx = np.nonzero(has)[0]
        cur = jax.tree.map(lambda a: jnp.asarray(np.asarray(a)[idx]), c[3])
        pops.append(int(idx.size))
    return dict(**H.trip_stats(np.concatenate(ks)), generations=len(pops), populations=pops,
                particles=int(sum(pops)), children=int(sum(pops[1:])))


def capacities(ctx, arm, pops):
    """Per-generation capacity of the plain-JAX arm: the observed population rounded up to a power of two,
    doubled for the `2x` arm."""
    mult = 2 if arm.endswith("2x") else 1
    return [max(1, 1 << int(np.ceil(np.log2(max(1, p * mult))))) for p in pops]


def sim_of(ctx, arm, maxs):
    ps, roots, n = ctx["psplit"], ctx["roots"], ctx["n"]
    if arm.startswith("tm_"):
        def particle(t, s, th):
            def body(c):
                s, t = c
                sn, b, d, ch, sp = step(s, th, ps)
                t = t.add("dep", b, d)
                t = t.spawn(ch, when=sp)
                return sn, t
            s, t = lax.while_loop(lambda c: cond(c[0]), body, (s, t))
            return t
        f = tm.vmap(particle, outputs={"dep": tm.Sum((NZ,))}, in_axes=(0, None), spawn=tm.Queue(queue_capacity(ctx)),
                    backend="cuda" if arm == "tm_cuda" else "lockstep")
        return lambda th: f(roots, th).outputs["dep"]

    caps = capacities(ctx, arm, ctx.get("populations") or [n] * (DEPTH + 1))

    def call(th):
        Q = jnp.zeros((NZ,), jnp.float32)
        cur, live = roots, jnp.ones((n,), bool)
        for cap in caps:
            def one(s, th_=th):
                def body(c):
                    s, dep, child, has = c
                    sn, b, d, ch, sp = step(s, th_, ps)
                    return (sn, dep.at[b].add(d, mode="drop"),
                            jax.tree.map(lambda a, b_: jnp.where(sp, b_, a), child, ch), has | sp)
                c0 = (s, jnp.zeros((NZ,), jnp.float32), jax.tree.map(jnp.zeros_like, s), jnp.bool_(False))
                c, _ = A.masked_scan(lambda c: body(c), lambda c: cond(c[0]), c0, maxs, remat=True)
                return c[1], c[2], c[3]
            dep, child, has = jax.vmap(one)(cur)
            has = has & live
            Q = Q + jnp.sum(dep * live[:, None].astype(jnp.float32), 0)
            cur, live = compact(child, has, cap)
        return Q
    return call


def compact(child, has, cap):
    """Pack the valid children of this generation into a dense array of `cap` slots (plain JAX)."""
    slot = jnp.cumsum(has.astype(jnp.int32)) - 1
    keep = has & (slot < cap)
    idx = jnp.where(keep, slot, cap)                       # out-of-capacity children scatter to a dropped slot
    nxt = jax.tree.map(lambda a: jnp.zeros((cap + 1,) + a.shape[1:], a.dtype).at[idx].set(a)[:cap], child)
    return nxt, jnp.zeros((cap,), bool).at[idx].set(keep)[:cap]


def loss_of(ctx, arm, maxs):
    sim = sim_of(ctx, arm, maxs)
    return lambda th: jnp.sum(ctx["wz"] * sim(th))


def queue_capacity(ctx):
    """The spawn Queue capacity.

    The reverse pass (the tree adjoint) records EVERY child of the call, not just the pending ones, so the
    Queue needs capacity >= the total number of children; a smaller Queue gives a NaN gradient, never a wrong
    one.  The probe measured that total, so this arm is given it, which is the same oracle the plain-JAX arm's
    per-generation capacities are."""
    return 1 << int(np.ceil(np.log2(max(2, 2 * (ctx.get("children") or ctx["n"] * (DEPTH + 1))))))


def measured(ctx, arm, mode, maxs):
    extra = dict(nz=NZ, psplit=ctx["psplit"], depth=DEPTH, padded_steps=A.padded_count(arm, maxs),
                 jacfwd_k=A.jacfwd_k(ctx["th"]), children=ctx.get("children"))
    if arm.startswith("tm_"):
        extra["queue_capacity"] = queue_capacity(ctx)
    if arm.startswith("jax_gen"):
        caps = capacities(ctx, arm, ctx.get("populations") or [ctx["n"]] * (DEPTH + 1))
        extra.update(capacities=caps, capacity_total=int(sum(caps)))
    fn = sim_of(ctx, arm, maxs) if mode == "fwd" else jax.grad(loss_of(ctx, arm, maxs))
    return fn, (ctx["th"],), extra


def check(ctx, maxs):
    """Agreement, plus the two spawn statistics that decide whether the gradient is meaningful at all: a
    dropped child makes every float tally NaN, and a Queue smaller than the total number of children makes the
    gradient NaN (never wrong)."""
    out = A.cross_arm_check(lambda a: sim_of(ctx, a, maxs), lambda a: loss_of(ctx, a, maxs), ctx["th"],
                            arms=("tm_cuda", "jax_gen"), grad_ref="jax_gen")
    import tallymap as _tm
    f = tm.vmap(lambda t, s, th: _particle_for_stats(t, s, th, ctx["psplit"]),
                outputs={"dep": tm.Sum((NZ,))}, in_axes=(0, None), spawn=tm.Queue(queue_capacity(ctx)),
                backend="lockstep", rev_stats=True)
    r = jax.jit(f)(ctx["roots"], ctx["th"])
    jax.jit(jax.grad(lambda th: jnp.sum(ctx["wz"] * f(ctx["roots"], th).outputs["dep"])))(ctx["th"])
    tree = _tm.diagnostics()["reverse"].get("tree", {})
    out.update(queue_capacity=queue_capacity(ctx), spawned=int(r.stats["spawned"]),
               dropped=int(r.stats["dropped"]), drop_risk=bool(tree.get("drop_risk", True)),
               replay_mismatch=int(tree.get("replay_mismatch", -1)))
    return out


def _particle_for_stats(t, s, th, ps):
    def body(c):
        s, t = c
        sn, b, d, ch, sp = step(s, th, ps)
        t = t.add("dep", b, d)
        t = t.spawn(ch, when=sp)
        return sn, t
    s, t = lax.while_loop(lambda c: cond(c[0]), body, (s, t))
    return t
