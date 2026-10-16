"""gymnax episodic rollouts: gymnax's own RolloutWrapper against the same episode as a tallymap item.

The incumbent is `gymnax.experimental.rollout.RolloutWrapper` (gymnax 1.0.0).  Its `single_rollout`
(`gymnax/experimental/rollout.py:72-108`) runs ONE `lax.scan` of `env_params.max_steps_in_episode` steps per
episode whatever happens and multiplies the reward by a `valid_mask` that goes to zero after `done`;
`batch_rollout` and `population_rollout` `jax.vmap` that over keys and over policy parameters.  So on
CartPole-v1 an episode that falls over after 20 steps still costs the full 500-step horizon.

Arms, all stepping the SAME `env.step` and `env.reset` from the installed gymnax:

  gx_scan   gymnax's own structure: `jax.vmap` over episodes of a fixed 500-step `lax.scan` with a valid mask.
  gx_while  the obvious plain-JAX improvement: `jax.vmap` of a `lax.while_loop` that stops on `done`.  Still
            lockstep, so it costs the batch maximum episode length.
  tm        the same `while_loop` as a tallymap item: one episode = one item.

Outputs, identical in all three arms: the summed return, a 64-bin histogram of episode lengths, and the
per-episode (return, length).  That is what an evolution-strategies or CEM fitness evaluation needs.

Divergence knob: the policy.  `--div none` is a policy under which every episode runs the full 500-step horizon,
so it is a genuine zero-divergence control; `low`, `mid` and `high` shorten the episodes.

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
import gymnax  # noqa: E402

import tallymap as tm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LIBS = ("gymnax", "jax")
NB = 64                                          # episode-length histogram bins
ENV = "CartPole-v1"
W_OPT = jnp.array([0.4, 1.0, 4.0, 2.0], jnp.float32)      # a hand-set linear CartPole policy
# scale on W_OPT.  4.0 keeps the pole up for the whole horizon in every episode (the zero-divergence control);
# 0.0 is a uniformly random policy.
DIVS = {"none": 4.0, "low": 1.0, "mid": 0.22, "high": 0.0}
ARMS = ("tm", "gx_scan", "gx_while")
LABEL = {"tm": "tallymap", "gx_scan": "gymnax RolloutWrapper", "gx_while": "vmap + while_loop"}


def make_policy(div):
    w = DIVS[div] * W_OPT

    def policy(obs, key):
        # at div=high the weights are zero, so the logit is zero and the action is a fair coin
        return (jax.nn.sigmoid(jnp.dot(w, obs)) > jax.random.uniform(key)).astype(jnp.int32)
    return policy


def setup():
    env, params = gymnax.make(ENV)
    return env, params, int(params.max_steps_in_episode)


def hist_of(ln, hmax):
    b = jnp.clip((ln * (NB / (hmax + 1.0))).astype(jnp.int32), 0, NB - 1)
    return jnp.zeros((NB,), jnp.float32).at[b].add(1.0)


def arm_gx_scan(env, params, policy, keys, hmax):
    """RolloutWrapper.single_rollout under jax.vmap: a fixed hmax-step scan with a validity mask."""
    def single(key):
        kr, ke = jax.random.split(key)
        obs, st = env.reset(kr, params)

        def step(c, _):
            obs, st, k, cum, mask, ln = c
            k, ks, ka = jax.random.split(k, 3)
            a = policy(obs, ka)
            obs, st, r, term, trunc, _ = env.step(ks, st, a, params)
            done = jnp.logical_or(term, trunc)
            return (obs, st, k, cum + r * mask, mask * (1 - done), ln + mask), None
        c, _ = lax.scan(step, (obs, st, ke, jnp.float32(0.0), jnp.float32(1.0), jnp.float32(0.0)), (), hmax)
        return c[3], c[5]
    ret, ln = jax.vmap(single)(keys)
    return ret.sum(), hist_of(ln, hmax), (ret, ln)


def arm_gx_while(env, params, policy, keys, hmax):
    """Stop on `done`.  jax.vmap makes it lockstep on the batch maximum episode length."""
    def single(key):
        kr, ke = jax.random.split(key)
        obs, st = env.reset(kr, params)

        def body(c):
            obs, st, k, cum, done, ln = c
            k, ks, ka = jax.random.split(k, 3)
            a = policy(obs, ka)
            obs, st, r, term, trunc, _ = env.step(ks, st, a, params)
            return obs, st, k, cum + r, jnp.logical_or(term, trunc), ln + 1
        c = lax.while_loop(lambda c: (~c[4]) & (c[5] < hmax), body,
                           (obs, st, ke, jnp.float32(0.0), jnp.bool_(False), 0))
        return c[3], c[5].astype(jnp.float32)
    ret, ln = jax.vmap(single)(keys)
    return ret.sum(), hist_of(ln, hmax), (ret, ln)


def arm_tm(env, params, policy, hmax):
    """One episode = one item.  The body is identical to gx_while's; the tally replaces the dense arrays."""
    def item(t, key):
        kr, ke = jax.random.split(key)
        obs, st = env.reset(kr, params)

        def body(c):
            obs, st, k, cum, done, ln, t = c
            k, ks, ka = jax.random.split(k, 3)
            a = policy(obs, ka)
            obs, st, r, term, trunc, _ = env.step(ks, st, a, params)
            return obs, st, k, cum + r, jnp.logical_or(term, trunc), ln + 1, t
        obs, st, k, cum, done, ln, t = lax.while_loop(
            lambda c: (~c[4]) & (c[5] < hmax), body,
            (obs, st, ke, jnp.float32(0.0), jnp.bool_(False), 0, t))
        t = t.add("ret", 0, cum)
        t = t.add("len_hist", jnp.clip((ln * NB) // (hmax + 1), 0, NB - 1), 1.0)
        return t, (cum, ln.astype(jnp.float32))

    f = tm.vmap(item, outputs={"ret": tm.Sum((1,)), "len_hist": tm.Sum((NB,))}, max_steps=hmax + 2)

    def call(keys):
        r = f(keys)
        return r.outputs["ret"][0], r.outputs["len_hist"], r.per_item
    return call


def build(arm, env, params, policy, hmax):
    if arm == "tm":
        return arm_tm(env, params, policy, hmax)
    if arm == "gx_scan":
        return lambda k: arm_gx_scan(env, params, policy, k, hmax)
    if arm == "gx_while":
        return lambda k: arm_gx_while(env, params, policy, k, hmax)
    raise SystemExit(f"unknown arm {arm}")


# ------------------------------------------------------------------------------------------------- one measurement
def one(a):
    env, params, hmax = setup()
    D.banner(f"gymnax arm={a.arm} mode={a.mode} n={a.n} div={a.div} env={ENV} hmax={hmax}", LIBS)
    policy = make_policy(a.div)
    keys = jax.random.split(jax.random.PRNGKey(0), a.n)
    base = dict(demo="gymnax_rollouts", arm=a.arm, mode=a.mode, n=a.n, div=a.div, env=ENV, hmax=hmax)
    fn = build(a.arm, env, params, policy, hmax)

    def go():
        out, r = D.measure(fn, (keys,), reps=a.reps)
        ln = np.asarray(out[2][1])
        D.row(**base, **r, ret_total=float(np.asarray(out[0])), len_mean=float(ln.mean()),
              len_max=float(ln.max()), status="ok")
    D.guard(go, base)


# -------------------------------------------------------------------------------------------------------- --quick
def quick(a):
    n = min(a.n, 512)
    env, params, hmax = setup()
    D.banner(f"quick n={n} env={ENV} hmax={hmax}", LIBS)
    checks = []
    for div in ("mid", "none"):
        policy = make_policy(div)
        keys = jax.random.split(jax.random.PRNGKey(0), n)
        o = {arm: jax.jit(build(arm, env, params, policy, hmax))(keys) for arm in ARMS}
        rs, hs, (rr_s, ll_s) = o["gx_scan"]
        for arm in ("gx_while", "tm"):
            R, H, (RR, LL) = o[arm]
            e = dict(ret=D.rel(R, rs), hist=D.rel(H, hs), per_ret=D.rel(RR, rr_s), per_len=D.rel(LL, ll_s))
            exact = float(np.mean(np.asarray(LL) == np.asarray(ll_s)))
            checks.append((f"div={div}: {arm} vs gymnax RolloutWrapper, bitwise",
                           max(e.values()) == 0.0 and exact == 1.0,
                           f"return/hist/per-episode rel = {e['ret']:.3g}/{e['hist']:.3g}/"
                           f"{max(e['per_ret'], e['per_len']):.3g}, lengths equal {exact:.3f}"))
        ln = np.asarray(ll_s)
        checks.append((f"div={div}: episode lengths vary" if div == "mid" else
                       f"div={div}: every episode runs the full horizon",
                       (ln.max() > 2 * ln.mean()) if div == "mid" else bool(np.all(ln == hmax)),
                       f"mean {ln.mean():.1f}, max {ln.max():.0f} of {hmax}"))
    return D.verdict(checks)


# --------------------------------------------------------------------------------------------------------- --full
NS = (4096, 16384, 65536, 262144)
DIVS_FULL = ("none", "low", "mid", "high")


def plots(rows, meta):
    """Redraw plots/ from `rows`.  Called by --full and by --plots."""
    mid = D.sel(rows, div="mid", mode="fwd")
    D.panel(os.path.join(HERE, "plots", "fwd_time.png"), mid, ARMS, LABEL,
            f"{ENV} rollouts, forward: time per call (div=mid)", "ms per call")
    D.panel(os.path.join(HERE, "plots", "fwd_mem.png"), mid, ARMS, LABEL,
            f"{ENV} rollouts, forward: peak device memory (div=mid)", "peak device MB", ykey="peak_mb")
    sw = D.sel(rows, n=65536, mode="fwd")
    for r in sw:
        r["divx"] = {"none": 0, "low": 1, "mid": 2, "high": 3}.get(r.get("div"), -1)
    D.panel(os.path.join(HERE, "plots", "fwd_div.png"), sw, ARMS, LABEL,
            "Forward time vs policy quality at 65,536 episodes", "ms per call",
            xlabel="0 = full horizon every episode, 3 = random policy", xkey="divx", logx=False)

def full(a):
    script = os.path.join(HERE, "run.py")
    jobs = [dict(arm=arm, mode="fwd", n=n, div="mid") for n in NS for arm in ARMS]
    jobs += [dict(arm=arm, mode="fwd", n=65536, div=d) for d in DIVS_FULL for arm in ARMS if d != "mid"]
    out = os.path.join(HERE, "results", "gymnax_rollouts.json")
    rows, meta = D.sweep(script, jobs, out, libs=LIBS, extra_meta=dict(env=ENV, ns=list(NS)))
    plots(rows, meta)

    print(f"\n{len(rows)} rows -> {out}")
    return 0


def main():
    p = D.parser()
    a = p.parse_args()
    if a.one:
        one(a)
        return 0
    if a.plots:
        rows, meta = D.load(os.path.join(HERE, "results", "gymnax_rollouts.json"))
        plots(rows, meta)
        return 0
    return quick(a) if a.quick else full(a)


if __name__ == "__main__":
    sys.exit(main())
