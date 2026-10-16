"""A Gillespie SSA ensemble: crn-jax's own shipped hot loop against the same loop inside a tallymap item.

The incumbent is crn-jax 0.3.1 (`pip install crn_jax`), a pure-JAX GPU Gillespie library for chemical reaction
networks.  Its shipped batch entry point is `crn_jax.models._common.make_vmap_simulator`
(`models/_common.py:43-89`):

    jax.jit(jax.vmap(simulate_one))               # simulate_one -> gillespie.simulate_trajectory
      -> lax.scan over n_steps fixed intervals   (gillespie.py:309 / 333)
         -> gillespie.simulate_interval          (gillespie.py:194)
            -> gillespie.simulate_until          (gillespie.py:75)
               -> lax.while_loop over reactions  (gillespie.py:185)

so a batch of N replicates pays, for EVERY interval, the maximum reaction count over the N replicates, and the
outer scan stacks the whole (N, n_steps, n_species) trajectory tensor.

This demo calls the library's own `simulate_interval` -- imported from the installed package, unmodified -- as
the body of a tallymap item.  Item = one replicate and its whole 64-interval nest of reactions.  Tally =
`Sum((64, 2))`, the ensemble mean trajectory indexed by (interval, species), plus a terminal 2-D histogram.

Arms:
  tm           the tallymap item: `lax.fori_loop` over intervals calling crn-jax's `simulate_interval`.
  crnjax       the library AS SHIPPED: `jax.vmap(simulate_trajectory)`, stacking (N, n_steps, n_species).
  crnjax_lean  the best plain-JAX program: `jax.vmap` of a scan that calls the same `simulate_interval` and
               accumulates only the running sum, with no stacked trajectory.  Same keys as `tm`, so this is
               the arm the agreement check uses as its reference.
  jax_scan     `crnjax_lean` with the inner `while_loop` replaced by a masked `lax.scan` padded to the
               OBSERVED maximum reactions per interval.  This is what you must write for any gradient, since
               `lax.while_loop` has no transpose.

Model: `crn_jax.models.toggle_switch`, the Gardner-Cantor-Collins / Lugagne-2017 LacI-TetR mutual-repression
circuit.  crn-jax's models close over a frozen dataclass of parameters, so differentiating with respect to them
needs them as traced arrays; `props_toggle` below rebuilds the same propensity expression
(`models/toggle_switch.py:89-103`) with theta as an argument, and `--quick` checks it against the library's own
closure bit for bit.

Divergence knob: reactions per interval.  It varies Poisson-ly within a replicate and across replicates through
a per-replicate production multiplier passed down crn-jax's own per-replicate `input` channel, which is how
cell-to-cell (extrinsic) variability is modelled.  `--div` sets the lognormal spread of that multiplier.

  python run.py --quick
  python run.py --full
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _demo as D  # noqa: E402
from crn_jax.gillespie import simulate_interval, simulate_trajectory  # noqa: E402
from crn_jax.types import State  # noqa: E402

import tallymap as tm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LIBS = ("crn_jax", "jax")
DT = 0.5                      # minutes per observation interval
NSTEP = 64                    # observation intervals per replicate
MAXS = 10_000                 # crn-jax's own per-interval `max_steps` guard (its default, gillespie.py:262)
NHB, HMAX = 16, 1200.0        # terminal histogram bins per species, and the species range they cover
NSP = 2
SPREAD = {"none": 0.0, "low": 0.25, "mid": 0.8, "high": 1.6}
ARMS = ("tm", "crnjax", "crnjax_lean", "jax_scan")
LABEL = {"tm": "tallymap", "crnjax": "crn-jax as shipped", "crnjax_lean": "crn-jax loop, sum only",
         "jax_scan": "padded scan"}

# beta_A1, beta_B1, delta (shared), K_B -- the four parameters the score function differentiates
THETA0 = np.array([58.23, 17.39, 0.0165, 30.00], np.float32)
BETA_A0, BETA_B0, K_A, N_HILL = 0.2246, 1.005, 31.94, 2.0
STOICH = jnp.asarray([[+1., 0.], [-1., 0.], [0., +1.], [0., -1.]], jnp.float32)


def props_toggle(theta):
    """(state, u) -> propensities.  u is the per-replicate production multiplier (crn-jax's `input` channel)."""
    bA1, bB1, dl, KB = theta[0], theta[1], theta[2], theta[3]

    def f(state, u):
        A, B = state.x[0], state.x[1]
        rb = KB ** N_HILL / (KB ** N_HILL + B ** N_HILL)
        ra = K_A ** N_HILL / (K_A ** N_HILL + A ** N_HILL)
        return jnp.stack([u * (BETA_A0 + bA1 * rb), dl * A, u * (BETA_B0 + bB1 * ra), dl * B])
    return f


def apply_reaction(state, j):
    """`crn_jax.models._common.make_apply_reaction(_STOICHIOMETRY)` inlined: one gather, no `lax.switch`."""
    return state._replace(x=jnp.maximum(0.0, state.x + STOICH[j]))


def init_state(x0):
    return State(time=jnp.float32(0.0), x=x0, next_reaction_time=jnp.float32(jnp.inf))


def one_interval(state, key, u, theta, maxs):
    """crn-jax's own `simulate_interval` (gillespie.py:194), called unmodified."""
    return simulate_interval(key=key, state=state, input=u, timestep=DT, max_steps=maxs,
                             compute_propensities_fn=props_toggle(theta), apply_reaction_fn=apply_reaction,
                             previous_input=u)


def hbin(x):
    i = jnp.clip((x[0] * (NHB / HMAX)).astype(jnp.int32), 0, NHB - 1)
    j = jnp.clip((x[1] * (NHB / HMAX)).astype(jnp.int32), 0, NHB - 1)
    return i, j


def obs_weight():
    """A smooth, non-constant weight on (interval, species): the loss is <w, mean trajectory>."""
    w = np.outer(1.0 + 0.5 * np.cos(np.arange(NSTEP) * 0.3), np.array([1.0, -0.7]))
    return jnp.asarray(w, jnp.float32)


# ------------------------------------------------------------------------------------------------- tallymap arm
def make_tm(maxs, **o):
    def replicate(t, x0, key, u, theta):
        def body(i, c):
            s, t = c
            s = one_interval(s, jax.random.fold_in(key, i), u, theta, maxs)
            t = t.add("mean", (jnp.full((NSP,), i), jnp.arange(NSP)), s.x)
            return s, t
        s, t = lax.fori_loop(0, NSTEP, body, (init_state(x0), t))
        t = t.add("hist", hbin(s.x), 1.0)
        return t, s.x
    return tm.vmap(replicate, outputs={"mean": tm.Sum((NSTEP, NSP)), "hist": tm.Sum((NHB, NHB))},
                   in_axes=(0, 0, 0, None), **o)


# ------------------------------------------------------------------------------------------------ plain-JAX arms
def one_rep_lean(x0, key, u, theta, maxs):
    """Accumulate the running sum only: the best plain-JAX program, with no stacked trajectory."""
    def body(c, i):
        s, acc = c
        s = one_interval(s, jax.random.fold_in(key, i), u, theta, maxs)
        return (s, acc.at[i].add(s.x)), None
    (s, acc), _ = lax.scan(body, (init_state(x0), jnp.zeros((NSTEP, NSP), jnp.float32)), jnp.arange(NSTEP))
    return acc, s.x


def interval_padded(state, key, u, theta, maxs):
    """`simulate_interval` with the inner `while_loop` replaced by a masked `lax.scan` of `maxs` steps.

    This mirrors `gillespie.simulate_until` (gillespie.py:138-191) line for line, including the pending-reaction
    carry, so the two arms compute the same function; only the loop lowering differs.
    """
    pf = props_toggle(theta)
    target = state.time + DT
    key, key_init = jax.random.split(key)
    a0 = pf(state, u)
    tau = jax.random.exponential(key_init) / jnp.maximum(jnp.sum(a0), 1e-10)
    nrt = jnp.where(jnp.isinf(state.next_reaction_time), state.time + tau, state.next_reaction_time)

    def step(c, _):
        s, nrt, a, key = c
        alive = nrt < target
        key, kr, kt = jax.random.split(key, 3)
        s2 = s._replace(time=nrt)
        j = jax.random.choice(kr, 4, p=a / jnp.maximum(jnp.sum(a), 1e-10))
        s2 = apply_reaction(s2, j)
        a2 = pf(s2, u)
        nrt2 = s2.time + jax.random.exponential(kt) / jnp.maximum(jnp.sum(a2), 1e-10)
        return (State(time=jnp.where(alive, s2.time, s.time), x=jnp.where(alive, s2.x, s.x),
                      next_reaction_time=s.next_reaction_time),
                jnp.where(alive, nrt2, nrt), jnp.where(alive, a2, a), key), None
    (s, nrt, _, _), _ = lax.scan(step, (state, nrt, a0, key), None, length=maxs)
    return s._replace(time=target, next_reaction_time=nrt)


def one_rep_padded(x0, key, u, theta, maxs):
    def body(c, i):
        s, acc = c
        s = interval_padded(s, jax.random.fold_in(key, i), u, theta, maxs)
        return (s, acc.at[i].add(s.x)), None
    (s, acc), _ = lax.scan(body, (init_state(x0), jnp.zeros((NSTEP, NSP), jnp.float32)), jnp.arange(NSTEP))
    return acc, s.x


def make_jax(arm, maxs):
    if arm == "crnjax":
        # the library AS SHIPPED: simulate_trajectory under jax.vmap, stacking (n_steps, n_species) per replicate
        def one(x0, key, u, theta):
            st = simulate_trajectory(key=key, initial_state=init_state(x0), timestep=DT, n_steps=NSTEP,
                                     compute_propensities_fn=props_toggle(theta),
                                     apply_reaction_fn=apply_reaction,
                                     inputs=jnp.full((NSTEP,), u), max_steps=maxs)
            return st.x, st.x[-1]

        def call(x0, keys, us, theta):
            xs, xT = jax.vmap(one, in_axes=(0, 0, 0, None))(x0, keys, us, theta)
            return jnp.sum(xs, axis=0), xT
        return call
    f = {"crnjax_lean": one_rep_lean, "jax_scan": one_rep_padded}[arm]

    def call(x0, keys, us, theta):
        acc, xT = jax.vmap(lambda a, b, c: f(a, b, c, theta, maxs))(x0, keys, us)
        return jnp.sum(acc, axis=0), xT
    return call


# ------------------------------------------------------ the gradient an SSA actually has: the score function
# The pathwise gradient of an SSA ensemble is exactly zero (the state only ever changes by an integer
# stoichiometry vector, so the trajectory is piecewise constant in the rates).  `--quick` measures that.  The
# estimator that does work is the score function, and crn-jax's loop cannot accumulate it, because the score
# needs the propensity DERIVATIVES at every event and the library's carry holds only the propensities.
# `interval_scored` is gillespie.py:138-191 with one extra 4-vector in the carry; `--quick` confirms it produces
# the bitwise-identical trajectory to the library's own loop.
#
#   log P(path) = sum_events [ log a_r(x) - a_0(x) tau ]  -  a_0(x_last) (T - t_last)
#   d/dtheta    = sum_events [ da_r/a_r - tau da_0 ]      -  (T - t_last) da_0(x_last)
#
# so d E[f] / dtheta = E[ f(path) * d log P / dtheta ] is unbiased and comes out of ONE forward pass.
def dprops_toggle(state, u, theta):
    """d a / d theta for theta = (beta_A1, beta_B1, delta, K_B).  -> (4 parameters, 4 reactions)."""
    bA1, _bB1, _dl, KB = theta[0], theta[1], theta[2], theta[3]
    A, B = state.x[0], state.x[1]
    KBn, Bn = KB ** N_HILL, B ** N_HILL
    rb, ra = KBn / (KBn + Bn), K_A ** N_HILL / (K_A ** N_HILL + A ** N_HILL)
    drb_dKB = N_HILL * KB ** (N_HILL - 1.0) * Bn / (KBn + Bn) ** 2
    z = jnp.zeros(())
    return jnp.stack([jnp.stack([u * rb, z, z, z]),                 # d/d beta_A1
                      jnp.stack([z, z, u * ra, z]),                 # d/d beta_B1
                      jnp.stack([z, A, z, B]),                      # d/d delta
                      jnp.stack([u * bA1 * drb_dKB, z, z, z])])     # d/d K_B


def interval_scored(state, key, u, theta, maxs, sc):
    """One interval of crn-jax's SSA, plus the path-score increment for the four parameters."""
    pf = props_toggle(theta)
    target = state.time + DT
    key, key_init = jax.random.split(key)
    a0 = pf(state, u)
    tau0 = jax.random.exponential(key_init) / jnp.maximum(jnp.sum(a0), 1e-10)
    nrt = jnp.where(jnp.isinf(state.next_reaction_time), state.time + tau0, state.next_reaction_time)

    def body(c):
        s, nrt, a, i, key, sc = c
        key, kr, kt = jax.random.split(key, 3)
        da = dprops_toggle(s, u, theta)                              # (4, 4) at the PRE-reaction state
        tau = nrt - s.time
        s = s._replace(time=nrt)
        r = jax.random.choice(kr, 4, p=a / jnp.maximum(jnp.sum(a), 1e-10))
        sc = sc + da[:, r] / jnp.maximum(a[r], 1e-20) - tau * jnp.sum(da, axis=1)
        s = apply_reaction(s, r)
        a2 = pf(s, u)
        return s, s.time + jax.random.exponential(kt) / jnp.maximum(jnp.sum(a2), 1e-10), a2, i + 1, key, sc
    s, nrt, a, _, _, sc = lax.while_loop(lambda c: (c[1] < target) & (c[3] < maxs), body,
                                         (state, nrt, a0, jnp.int32(0), key, sc))
    sc = sc - (target - s.time) * jnp.sum(dprops_toggle(s, u, theta), axis=1)   # the surviving no-event time
    return s._replace(time=target, next_reaction_time=nrt), sc


def make_tm_score(maxs, **o):
    def replicate(t, x0, key, u, theta):
        def body(i, c):
            s, sc, t = c
            s, sc = interval_scored(s, jax.random.fold_in(key, i), u, theta, maxs, sc)
            t = t.add("mean", (jnp.full((NSP,), i), jnp.arange(NSP)), s.x)
            return s, sc, t
        s, sc, t = lax.fori_loop(0, NSTEP, body, (init_state(x0), jnp.zeros((4,), jnp.float32), t))
        obs = s.x[0] - 0.7 * s.x[1]                                 # the terminal observable whose E[.] we want
        t = t.add("obs", 0, obs)
        t = t.add("dobs", jnp.arange(4), obs * sc)                  # d E[obs] / d theta, one emit per parameter
        t = t.add("dobs2", jnp.arange(4), (obs * sc) ** 2)          # second moment -> the estimator's s.e.
        return t, s.x
    return tm.vmap(replicate, outputs={"mean": tm.Sum((NSTEP, NSP)), "obs": tm.Sum((1,)),
                                       "dobs": tm.Sum((4,)), "dobs2": tm.Sum((4,))},
                   in_axes=(0, 0, 0, None), **o)


def tm_obs_only(maxs, **o):
    """The same item WITHOUT the score carry: the control for "what does the score cost?"."""
    def replicate(t, x0, key, u, theta):
        def body(i, c):
            s, t = c
            s = one_interval(s, jax.random.fold_in(key, i), u, theta, maxs)
            t = t.add("mean", (jnp.full((NSP,), i), jnp.arange(NSP)), s.x)
            return s, t
        s, t = lax.fori_loop(0, NSTEP, body, (init_state(x0), t))
        t = t.add("obs", 0, s.x[0] - 0.7 * s.x[1])
        return t, s.x
    return tm.vmap(replicate, outputs={"mean": tm.Sum((NSTEP, NSP)), "obs": tm.Sum((1,))},
                   in_axes=(0, 0, 0, None), **o)


# --------------------------------------------------------------------------------------------------- the inputs
def inputs(n, div):
    rng = np.random.default_rng(0)
    # half the replicates start in each basin of the bistable toggle (both basins are visited anyway)
    x0 = np.where(rng.random((n, 1)) < 0.5, np.array([[660., 63.]]), np.array([[18., 864.]]))
    x0 = jnp.asarray(x0 + rng.normal(0, 8.0, (n, NSP)).clip(-30, 30), jnp.float32).clip(0.0)
    keys = jax.random.split(jax.random.PRNGKey(3), n)
    sp = SPREAD[div]
    us = jnp.asarray(np.exp(rng.normal(-0.5 * sp * sp, sp, n)), jnp.float32)
    return x0, keys, us


def count_events(x0, keys, us, theta, maxs):
    """(N, NSTEP) reaction counts per replicate per interval: the divergence statistic, measured not assumed.

    Mirrors `gillespie.simulate_until` with a counter in the carry, so what is counted is the trajectory the
    library's own loop runs.
    """
    pf = props_toggle(theta)

    def interval(state, key, u):
        target = state.time + DT
        key, key_init = jax.random.split(key)
        a0 = pf(state, u)
        tau = jax.random.exponential(key_init) / jnp.maximum(jnp.sum(a0), 1e-10)
        nrt = jnp.where(jnp.isinf(state.next_reaction_time), state.time + tau, state.next_reaction_time)

        def body(c):
            s, nrt, a, i, key = c
            key, kr, kt = jax.random.split(key, 3)
            s = s._replace(time=nrt)
            s = apply_reaction(s, jax.random.choice(kr, 4, p=a / jnp.maximum(jnp.sum(a), 1e-10)))
            a2 = pf(s, u)
            return s, s.time + jax.random.exponential(kt) / jnp.maximum(jnp.sum(a2), 1e-10), a2, i + 1, key
        s, nrt, _, i, _ = lax.while_loop(lambda c: (c[1] < target) & (c[3] < maxs), body,
                                         (state, nrt, a0, jnp.int32(0), key))
        return s._replace(time=target, next_reaction_time=nrt), i

    def one(x0i, key, u):
        _, nn = lax.scan(lambda s, i: interval(s, jax.random.fold_in(key, i), u),
                         init_state(x0i), jnp.arange(NSTEP))
        return nn
    return jax.vmap(one)(x0, keys, us)


# ------------------------------------------------------------------------------------------------ one measurement
def one(a):
    D.banner(f"gillespie arm={a.arm} mode={a.mode} n={a.n} div={a.div} smaxs={a.smaxs}", LIBS)
    x0, keys, us = inputs(a.n, a.div)
    theta = jnp.asarray(THETA0)
    w = obs_weight()
    # the padded-scan arm is padded to the OBSERVED maximum reactions per interval (--smaxs); `maxs` is
    # crn-jax's own max_steps guard, which costs the while-loop arms nothing
    smaxs = a.smaxs or MAXS
    base = dict(demo="gillespie_crn", arm=a.arm, mode=a.mode, n=a.n, div=a.div, maxs=MAXS, smaxs=smaxs,
                nstep=NSTEP, dt=DT, spread=SPREAD[a.div])

    if a.mode == "calib":                 # measure the divergence structure that sizes the padded arm
        # Count on as many replicates as the sweep measures (capped so the counter itself stays cheap), because
        # the number the padded arm has to cover is the BATCH maximum and that grows with N.
        m = min(a.n, 262144)
        nev = np.asarray(jax.jit(count_events)(x0[:m], keys[:m], us[:m], theta, MAXS))
        D.row(**base, status="ok", calib_n=m, ev_mean=float(nev.mean()), ev_max_interval=int(nev.max()),
              ev_item_mean=float(nev.sum(1).mean()), ev_item_max=int(nev.sum(1).max()),
              ev_batchmax_sum=int(nev.max(0).sum()))
        return

    if a.mode in ("score", "noscore"):    # the score-function gradient, and what carrying it costs
        f = (make_tm_score if a.mode == "score" else tm_obs_only)(MAXS)

        def go_s():
            out, r = D.measure(lambda th: f(x0, keys, us, th).outputs["obs"], (theta,), reps=a.reps)
            res = jax.jit(lambda th: f(x0, keys, us, th))(theta)
            r.update(base, status="ok", obs=float(res.outputs["obs"][0]) / a.n,
                     n_truncated=int(res.stats["n_truncated"]))
            if a.mode == "score":
                m1 = np.asarray(res.outputs["dobs"]) / a.n
                m2 = np.asarray(res.outputs["dobs2"]) / a.n
                r["dobs_dtheta"] = m1
                r["dobs_se"] = np.sqrt(np.maximum(m2 - m1 ** 2, 0.0) / a.n)
            D.row(**r)
        return D.guard(go_s, base)

    if a.arm == "tm":
        ftm = make_tm(MAXS)
        sim = lambda th: ftm(x0, keys, us, th).outputs["mean"]                           # noqa: E731
    else:
        g = make_jax(a.arm, smaxs if a.arm == "jax_scan" else MAXS)
        sim = lambda th: g(x0, keys, us, th)[0]                                          # noqa: E731
    loss = lambda th: jnp.sum(sim(th) * w) / a.n                                         # noqa: E731
    fn = {"fwd": sim, "grad": jax.grad(loss), "jacfwd": jax.jacfwd(loss)}[a.mode]

    def go():
        out, r = D.measure(fn, (theta,), reps=a.reps)
        r.update(base, status="ok", out_norm=float(jnp.linalg.norm(jnp.asarray(out).ravel())))
        if a.arm == "tm" and a.mode == "fwd":
            res = jax.jit(lambda th: make_tm(MAXS)(x0, keys, us, th))(theta)
            r["n_truncated"] = int(res.stats["n_truncated"])
        D.row(**r)
    D.guard(go, base)


# -------------------------------------------------------------------------------------------------------- --quick
def quick(a):
    n = min(a.n, 256)
    D.banner(f"quick n={n}", LIBS)
    x0, keys, us = inputs(n, "mid")
    theta = jnp.asarray(THETA0)
    w = obs_weight()
    checks = []

    # (1) our theta-ised propensities against the library's own frozen closure
    try:
        from crn_jax.models import toggle_switch as TS
        lib = TS.propensities_fn(TS.Params())
        st = init_state(jnp.asarray([400., 200.], jnp.float32))
        r = D.rel(props_toggle(theta)(st, jnp.float32(1.0)), lib(st, jnp.float32(0.0)))
        checks.append(("propensities match crn_jax.models.toggle_switch's own closure", r == 0.0,
                       f"rel = {r:.3g} (bitwise)" if r == 0.0 else f"rel = {r:.3g}"))
    except Exception as e:                                                               # noqa: BLE001
        checks.append(("propensities match crn_jax.models.toggle_switch's own closure", False,
                       f"could not import the model: {type(e).__name__}: {e}"))

    # (2) the tally and the terminal state against all three plain-JAX arms
    ftm = make_tm(MAXS)
    rt = jax.jit(lambda th: ftm(x0, keys, us, th))(theta)
    nev = np.asarray(jax.jit(count_events)(x0, keys, us, theta, MAXS))
    smaxs = int(nev.max())
    for arm in ("crnjax", "crnjax_lean", "jax_scan"):
        o = jax.jit(lambda th: make_jax(arm, smaxs if arm == "jax_scan" else MAXS)(x0, keys, us, th))(theta)
        rr = D.rel(rt.outputs["mean"], o[0])
        eq = float(np.mean(np.asarray(rt.per_item) == np.asarray(o[1])))
        checks.append((f"mean trajectory vs {LABEL[arm]}", rr < 1e-5 and eq == 1.0,
                       f"tally rel = {rr:.2g}, terminal state bitwise equal for {100 * eq:.0f}% of replicates"))

    # (3) the scored loop reproduces the library's own loop exactly
    fs = make_tm_score(MAXS)
    rs = jax.jit(lambda th: fs(x0, keys, us, th))(theta)
    eq = float(np.mean(np.asarray(rt.per_item) == np.asarray(rs.per_item)))
    checks.append(("the score-carrying loop gives crn-jax's trajectory bit for bit", eq == 1.0,
                   f"terminal state equal for {100 * eq:.0f}% of replicates, "
                   f"tally rel = {D.rel(rs.outputs['mean'], rt.outputs['mean']):.2g}"))

    # (4) the measured negative result: the pathwise gradient of an SSA ensemble is exactly zero
    gt = jax.jit(jax.grad(lambda th: jnp.sum(ftm(x0, keys, us, th).outputs["mean"] * w) / n))(theta)
    gs = jax.jit(jax.grad(lambda th: jnp.sum(
        make_jax("jax_scan", smaxs)(x0, keys, us, th)[0] * w) / n))(theta)
    z = float(np.max(np.abs(np.asarray(gt)))) + float(np.max(np.abs(np.asarray(gs))))
    checks.append(("the pathwise gradient is exactly zero in both arms (so use the score function)", z == 0.0,
                   f"max|grad| = {z:.3g}"))

    # (5) the score estimator against a common-random-number central difference of E[obs]
    g0 = tm_obs_only(MAXS)
    obs = lambda th: float(jax.jit(lambda t_: g0(x0, keys, us, t_).outputs["obs"][0])(th)) / n  # noqa: E731
    m1 = np.asarray(rs.outputs["dobs"]) / n
    se = np.sqrt(np.maximum(np.asarray(rs.outputs["dobs2"]) / n - m1 ** 2, 0.0) / n)
    worst, detail = 0.0, []
    for j, h in ((0, 2.0), (2, 1.0e-3)):
        fd = (obs(theta.at[j].add(h)) - obs(theta.at[j].add(-h))) / (2 * h)
        zz = abs(m1[j] - fd) / max(se[j], 1e-30)
        worst = max(worst, zz)
        detail.append(f"theta[{j}]: score {m1[j]:.4g} +- {se[j]:.3g} vs CRN difference {fd:.4g} (z = {zz:.2f})")
    checks.append(("the score estimator agrees with a common-random-number finite difference", worst < 4.0,
                   "; ".join(detail)))

    checks.append((f"divergence is present at div=mid (n={n})", nev.max() > 2 * nev.mean(),
                   f"{nev.mean():.1f} reactions per interval on average, "
                   f"{nev.max():.0f} at the batch maximum"))
    return D.verdict(checks)


# --------------------------------------------------------------------------------------------------------- --full
NS = (16384, 65536, 262144, 1048576)


def plots(rows, meta):
    """Redraw plots/ from `rows`.  Called by --full and by --plots."""
    P = os.path.join(HERE, "plots")
    mid = [r for r in rows if r.get("mode") == "fwd" and r.get("div") == "mid"]
    D.panel(os.path.join(P, "fwd_time.png"), mid, ARMS, LABEL,
            "Toggle-switch SSA, 64 intervals, forward: time per call", "ms per call")
    D.panel(os.path.join(P, "fwd_mem.png"), mid, ARMS, LABEL,
            "Toggle-switch SSA, forward: peak device memory", "peak device MB", ykey="peak_mb")
    sc = [dict(r, arm={"score": "tm", "noscore": "jax_scan"}[r["mode"]])
          for r in rows if r.get("mode") in ("score", "noscore") and r.get("status") == "ok"]
    D.panel(os.path.join(P, "score_time.png"), sc, ("tm", "jax_scan"),
            {"tm": "item + 4-parameter score", "jax_scan": "item alone"},
            "The score-function gradient of E[obs]: what carrying the path score costs", "ms per call")

def full(a):
    script = os.path.join(HERE, "run.py")
    out = os.path.join(HERE, "results", "gillespie_crn.json")
    # stage 1: measure the reactions per interval, which is what the padded arm has to be sized for
    rows = D.run_jobs(script, [dict(arm="tm", mode="calib", n=max(NS), div=d) for d in SPREAD])
    smax = {r["div"]: int(r["ev_max_interval"]) for r in rows if r.get("status") == "ok"}
    sm = smax.get("mid", MAXS)
    print(f"\npadded-scan length at div=mid: {sm} (the observed maximum reactions per interval)", flush=True)

    jobs = [dict(arm=arm, mode="fwd", n=n, div="mid", smaxs=sm) for n in NS for arm in ARMS]
    jobs += [dict(arm=arm, mode="fwd", n=262144, div=d, smaxs=smax.get(d, MAXS))
             for d in ("none", "low", "high") for arm in ("tm", "crnjax", "jax_scan")]
    jobs += [dict(arm="tm", mode=m, n=n, div="mid") for n in (65536, 262144, 1048576)
             for m in ("noscore", "score")]
    rows += D.run_jobs(script, jobs)
    rows, meta = D.write_results(out, rows, libs=LIBS, extra_meta=dict(ns=list(NS), scan_lengths=smax))

    plots(rows, meta)

    print(f"\n{len(rows)} rows -> {out}")
    return 0


def main():
    p = D.parser(smaxs=0)
    a = p.parse_args()
    if a.one:
        one(a)
        return 0
    if a.plots:
        rows, meta = D.load(os.path.join(HERE, "results", "gillespie_crn.json"))
        plots(rows, meta)
        return 0
    return quick(a) if a.quick else full(a)


if __name__ == "__main__":
    sys.exit(main())
