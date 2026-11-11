"""A Gillespie SSA ensemble: crn-jax's own shipped hot loop against the same loop inside a tallymap item.

The incumbent is crn-jax 0.3.1.  Its batch entry point is `jax.jit(jax.vmap(simulate_trajectory))`: a `lax.scan`
over fixed observation intervals, each a `lax.while_loop` over reactions, so a batch pays, for every interval, the
maximum reaction count over the batch, and it stacks the (N, n_steps, n_species) trajectory tensor.

Model: the LacI-TetR toggle switch (`crn_jax.models.toggle_switch`), 64 intervals of half a minute, with a
lognormal per-replicate production multiplier passed down crn-jax's own per-replicate `input` channel.

  tm      item = one replicate: a `lax.fori_loop` over intervals calling crn-jax's own `simulate_interval`,
          unmodified; tally = the ensemble mean trajectory Sum((64, 2)).
  crnjax  the library as shipped: `jax.vmap(simulate_trajectory)`, summed over replicates.

Same keys in both arms, so the terminal state is bitwise equal per replicate on a GPU; on the CPU the kernel's libm
can differ from XLA in the last bit and send a replicate down another path, so the check allows 2% of them.
`props_toggle` rebuilds the model's propensities with theta as an argument, checked against the library's closure.

  python run.py             # CPU, N = 256
  python run.py --n 262144  # on a GPU
"""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _demo as D  # noqa: E402

import tallymap as tm  # noqa: E402

D.need("crn_jax")
from crn_jax.gillespie import simulate_interval, simulate_trajectory  # noqa: E402
from crn_jax.models import toggle_switch as TS  # noqa: E402
from crn_jax.types import State  # noqa: E402

DT, NSTEP, MAXS, NSP = 0.5, 64, 10_000, 2          # interval, intervals, crn-jax's own max_steps guard, species
THETA0 = np.array([58.23, 17.39, 0.0165, 30.00], np.float32)     # beta_A1, beta_B1, delta, K_B
BETA_A0, BETA_B0, K_A, N_HILL = 0.2246, 1.005, 31.94, 2.0
STOICH = jnp.asarray([[+1., 0.], [-1., 0.], [0., +1.], [0., -1.]], jnp.float32)


def props_toggle(theta):
    bA1, bB1, dl, KB = theta[0], theta[1], theta[2], theta[3]

    def f(state, u):
        A, B = state.x[0], state.x[1]
        rb = KB ** N_HILL / (KB ** N_HILL + B ** N_HILL)
        ra = K_A ** N_HILL / (K_A ** N_HILL + A ** N_HILL)
        return jnp.stack([u * (BETA_A0 + bA1 * rb), dl * A, u * (BETA_B0 + bB1 * ra), dl * B])
    return f


def apply_reaction(state, j):
    return state._replace(x=jnp.maximum(0.0, state.x + STOICH[j]))


def init_state(x0):
    return State(time=jnp.float32(0.0), x=x0, next_reaction_time=jnp.float32(jnp.inf))


def replicate(t, x0, key, u, theta):
    def body(i, c):
        s, t = c
        s = simulate_interval(key=jax.random.fold_in(key, i), state=s, input=u, timestep=DT, max_steps=MAXS,
                              compute_propensities_fn=props_toggle(theta), apply_reaction_fn=apply_reaction,
                              previous_input=u)
        return s, t.add("mean", (jnp.full((NSP,), i), jnp.arange(NSP)), s.x)
    s, t = lax.fori_loop(0, NSTEP, body, (init_state(x0), t))
    return t, s.x


f_tm = tm.vmap(replicate, outputs={"mean": tm.Sum((NSTEP, NSP))}, in_axes=(0, 0, 0, None))


def run_tm(x0, keys, us, theta):
    r = f_tm(x0, keys, us, theta)
    return r.outputs["mean"], r.per_item


def run_crnjax(x0, keys, us, theta):
    def one(x, key, u):
        st = simulate_trajectory(key=key, initial_state=init_state(x), timestep=DT, n_steps=NSTEP,
                                 compute_propensities_fn=props_toggle(theta), apply_reaction_fn=apply_reaction,
                                 inputs=jnp.full((NSTEP,), u), max_steps=MAXS)
        return st.x, st.x[-1]
    xs, xT = jax.vmap(one)(x0, keys, us)
    return jnp.sum(xs, axis=0), xT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=256)
    n = p.parse_args().n
    rng = np.random.default_rng(0)
    x0 = np.where(rng.random((n, 1)) < 0.5, np.array([[660., 63.]]), np.array([[18., 864.]]))
    x0 = jnp.asarray(x0 + rng.normal(0, 8.0, (n, NSP)).clip(-30, 30), jnp.float32).clip(0.0)
    keys = jax.random.split(jax.random.PRNGKey(3), n)
    us = jnp.asarray(np.exp(rng.normal(-0.5 * 0.8 ** 2, 0.8, n)), jnp.float32)
    theta = jnp.asarray(THETA0)

    st = init_state(jnp.asarray([400., 200.], jnp.float32))
    rp = D.rel(props_toggle(theta)(st, jnp.float32(1.0)), TS.propensities_fn(TS.Params())(st, jnp.float32(0.0)))
    mt, xt = jax.jit(run_tm)(x0, keys, us, theta)
    mc, xc = jax.jit(run_crnjax)(x0, keys, us, theta)
    eq = float(np.mean(np.all(np.asarray(xt) == np.asarray(xc), axis=1)))
    checks = [("propensities match crn-jax's own toggle_switch closure", rp == 0.0, f"rel = {rp:.3g}"),
              ("mean trajectory and terminal state vs crn-jax as shipped", D.rel(mt, mc) < 0.05 and eq >= 0.98,
               f"tally rel = {D.rel(mt, mc):.2g}, terminal state bitwise equal for {100 * eq:.0f}% of replicates")]
    D.timings(n, {"tallymap": (run_tm, (x0, keys, us, theta)),
                  "crn-jax as shipped": (run_crnjax, (x0, keys, us, theta))})
    return D.verdict(checks)


if __name__ == "__main__":
    sys.exit(main())
