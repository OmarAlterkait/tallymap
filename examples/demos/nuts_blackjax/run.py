"""BlackJAX samplers over many chains, and a gradient through one.

NUTS (blackjax 1.7.1, unmodified): its transition is nested `lax.while_loop`s with a data-dependent number of
leapfrog steps, so under BlackJAX's documented `jax.vmap`-over-chains pattern every chain pays the slowest chain's
work at every draw.

  bj_vmap  the documented pattern: `lax.scan` over draws of `jax.vmap(alg.step)` across chains.
  tm       one chain = one tallymap item running the `lax.scan` over draws of the same `alg.step`.

The two arms consume different random streams, so the check is statistical: the chain means agree to Monte-Carlo
error and the two arms take the same number of leapfrog steps to within 20%.  Skipped without blackjax.

The gradient part needs only JAX: `jax.grad` through a short elliptical-slice chain (the bracket-shrinking
`while_loop` written out here, the way blackjax writes it).  Plain JAX cannot reverse-differentiate a
`while_loop`, so the reference is the rewrite a user has to write instead, a masked `lax.scan` padded to the
observed maximum shrinkage count.

  python run.py              # CPU, 128 chains
  python run.py --n 16384    # on a GPU
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

DIM, DRAWS = 6, 8                                  # elliptical-slice chain: dimension, draws per chain
PREC = 25.0 * (jnp.arange(1.0, DIM + 1.0) / DIM)


def loglik(x):
    return -0.5 * jnp.sum(PREC * (x - 0.3) ** 2)


def ess_draw(theta, x, lx, key, maxsub=None):
    """One elliptical-slice transition with prior N(0, theta^2 I).  maxsub=None: a `while_loop` (blackjax's
    form); otherwise the same transition as a masked `lax.scan` of `maxsub` steps."""
    k_nu, k_u, k_th, k_sl = jax.random.split(key, 4)
    nu = theta * jax.random.normal(k_nu, (DIM,))
    logy = lx + jnp.log(jax.random.uniform(k_u))
    th0 = 2 * jnp.pi * jax.random.uniform(k_th)
    prop = lambda t: (x * jnp.cos(t) + nu * jnp.sin(t), loglik(x * jnp.cos(t) + nu * jnp.sin(t)))  # noqa: E731
    p, lp = prop(th0)
    if maxsub is None:
        def body(v):
            _, i, tmin, tmax, _ = v
            t = jax.random.uniform(jax.random.fold_in(k_sl, i), minval=tmin, maxval=tmax)
            p, lp = prop(t)
            return lp, i + 1, jnp.where(t < 0, t, tmin), jnp.where(t > 0, t, tmax), p
        lp, i, _, _, p = lax.while_loop(lambda v: v[0] <= logy, body, (lp, 1, th0 - 2 * jnp.pi, th0, p))
        return p, lp, i

    def body(c, i):
        lp, tmin, tmax, p, n = c
        t = jax.random.uniform(jax.random.fold_in(k_sl, i + 1), minval=tmin, maxval=tmax)
        pn, lpn = prop(t)
        upd = lp <= logy
        return (jnp.where(upd, lpn, lp), jnp.where(upd & (t < 0), t, tmin), jnp.where(upd & (t > 0), t, tmax),
                jnp.where(upd, pn, p), n + upd.astype(jnp.int32)), None
    (lp, _, _, p, n), _ = lax.scan(body, (lp, th0 - 2 * jnp.pi, th0, p, 1), jnp.arange(maxsub))
    return p, lp, n


def chain(theta, x0, key, maxsub=None):
    def step(c, i):
        x, lx, acc, ns = c
        x, lx, n = ess_draw(theta, x, lx, jax.random.fold_in(key, i), maxsub)
        return (x, lx, acc + x, ns + n), None
    (_, _, acc, ns), _ = lax.scan(step, (x0, loglik(x0), jnp.zeros((DIM,)), 0), jnp.arange(DRAWS))
    return acc, ns


def make_ess(n):
    """One chain = one item, tallying its sum of draws into row `cid` of Sum((n, DIM))."""
    def item(t, x0, key, cid, theta):
        acc, ns = chain(theta, x0, key)
        return t.add("acc", (cid, jnp.arange(DIM)), acc).add("steps", 0, ns.astype(jnp.float32))
    f = tm.vmap(item, outputs={"acc": tm.Sum((n, DIM)), "steps": tm.Sum((1,))}, in_axes=(0, 0, 0, None),
                max_steps=4096)
    return lambda theta, x0, keys: f(x0, keys, jnp.arange(n, dtype=jnp.int32), theta).outputs["acc"]


def acc_jax(theta, x0, keys, maxsub=None):
    return jax.vmap(lambda a, b: chain(theta, a, b, maxsub)[0])(x0, keys)


def loss(acc_fn, m):
    """sum over chains of m_c |sum of chain c's draws|^2: separable, so a chain can be masked out."""
    return lambda theta, x0, keys: jnp.sum(m[:, None] * acc_fn(theta, x0, keys) ** 2) / (x0.shape[0] * DRAWS) ** 2


def nuts_checks(blackjax, n, checks):
    dim, draws = 4, 8
    sd = 10.0 ** jnp.linspace(-1.5, 1.5, dim).astype(jnp.float32)

    def logdensity(x):
        m = jax.nn.logsumexp(jnp.array([-0.5 * ((x[0] - 2.0) / 0.3) ** 2, -0.5 * ((x[0] + 2.0) / 1.5) ** 2]))
        return -0.5 * jnp.sum((x / sd) ** 2) + 0.3 * m
    (_, prm), _ = blackjax.window_adaptation(blackjax.nuts, logdensity).run(
        jax.random.PRNGKey(3), jnp.zeros((dim,)), num_steps=200)
    alg = blackjax.nuts(logdensity, step_size=float(prm["step_size"]),
                        inverse_mass_matrix=prm["inverse_mass_matrix"], max_num_doublings=6)
    x0 = 0.5 * jax.random.normal(jax.random.PRNGKey(11), (n, dim)) * sd
    keys = jax.random.split(jax.random.PRNGKey(0), n)

    def bj_vmap(x0, k):
        def one(c, kk):
            st, acc, nl = c
            st, info = jax.vmap(alg.step)(jax.random.split(kk, n), st)
            return (st, acc + st.position, nl + info.num_integration_steps.sum()), None
        (_, acc, nl), _ = lax.scan(one, (jax.vmap(alg.init)(x0), jnp.zeros_like(x0), 0), jax.random.split(k, draws))
        return acc / draws, nl

    def item(t, x, key, cid):
        def body(c, i):
            st, t = c
            st, info = alg.step(jax.random.fold_in(key, i), st)
            t = t.add("m", (cid, jnp.arange(dim)), st.position / draws)
            return (st, t.add("count", 0, info.num_integration_steps.astype(jnp.float32))), None
        (_, t), _ = lax.scan(body, (alg.init(x), t), jnp.arange(draws))
        return t
    f = tm.vmap(item, outputs={"m": tm.Sum((n, dim)), "count": tm.Sum((1,))}, max_steps=1 << 16)
    cids = jnp.arange(n, dtype=jnp.int32)

    def run_tm(x0, keys):
        r = f(x0, keys, cids)
        return r.outputs["m"], r.outputs["count"][0]
    mb, cb = jax.jit(bj_vmap)(x0, jax.random.PRNGKey(0))
    mt, ct = jax.jit(run_tm)(x0, keys)
    mb, mt = np.asarray(mb), np.asarray(mt)
    z = float(np.max(np.abs(mb.mean(0) - mt.mean(0)) / np.sqrt(mb.var(0, ddof=1) / n + mt.var(0, ddof=1) / n)))
    checks += [("NUTS: chain means agree with blackjax + vmap", z < 4.0, f"z = {z:.2f}"),
               ("NUTS: both arms take the same number of leapfrog steps", 0.8 < float(ct) / float(cb) < 1.25,
                f"tallymap {float(ct):,.0f} vs blackjax {float(cb):,.0f}")]
    return {"NUTS, tallymap": (run_tm, (x0, keys)), "NUTS, blackjax + jax.vmap": (bj_vmap, (x0, jax.random.PRNGKey(0)))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=128)
    n = p.parse_args().n
    x0 = 0.2 * jax.random.normal(jax.random.PRNGKey(7), (n, DIM), jnp.float32)
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    th = jnp.float32(1.0)
    _, ns = jax.jit(jax.vmap(lambda a, b: chain(th, a, b)))(x0, keys)
    maxsub = 24                     # the observed maximum shrinkage count per draw is below this; checked below
    _, ns_scan = jax.jit(jax.vmap(lambda a, b: chain(th, a, b, maxsub)))(x0, keys)
    acc_tm = make_ess(n)
    at, aw = np.asarray(jax.jit(acc_tm)(th, x0, keys)), np.asarray(jax.jit(acc_jax)(th, x0, keys))
    # A last-bit difference between the CPU kernel's libm and XLA's sin/cos can flip one accept decision, after
    # which that chain follows another path; compare the gradient over the chains whose paths match.
    m = np.all(np.abs(at - aw) <= 1e-5 * (1.0 + np.abs(aw)), axis=1)
    mk = jnp.asarray(m, jnp.float32)
    scan = lambda t, a, b: acc_jax(t, a, b, maxsub)                                        # noqa: E731
    gt = float(jax.jit(jax.grad(loss(acc_tm, mk)))(th, x0, keys))
    gs = float(jax.jit(jax.grad(loss(scan, mk)))(th, x0, keys))
    try:
        jax.jit(jax.grad(loss(acc_jax, mk)))(th, x0, keys)
        raised = "no error"
    except Exception as e:                                                                 # noqa: BLE001
        raised = type(e).__name__
    checks = [("ESS: the padded scan covers every shrinkage step", int(ns_scan.sum()) == int(ns.sum()),
               f"{int(ns_scan.sum())} vs {int(ns.sum())} steps"),
              ("ESS: forward, chains matching the same while_loop program", m.mean() >= 0.9,
               f"{int(m.sum())}/{n} chains to 1e-5"),
              ("ESS: grad, tallymap vs the padded-scan rewrite", abs(gt - gs) / abs(gs) < 1e-4,
               f"{gt:.8g} vs {gs:.8g}"),
              ("ESS: plain JAX cannot take grad of the while_loop", raised != "no error", raised)]
    blackjax = D.have("blackjax")
    nuts_arms = nuts_checks(blackjax, n, checks) if blackjax is not None else None
    one = jnp.ones((n,), jnp.float32)
    D.timings(n, {"ESS grad, tallymap": (jax.grad(loss(acc_tm, one)), (th, x0, keys)),
                  "ESS grad, padded scan": (jax.grad(loss(scan, one)), (th, x0, keys))})
    if nuts_arms:
        D.timings(n, nuts_arms)
    return D.verdict(checks)


if __name__ == "__main__":
    sys.exit(main())
