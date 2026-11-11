"""gymnax episodic rollouts: gymnax's own RolloutWrapper pattern against the same episode as a tallymap item.

gymnax 1.0.0's `RolloutWrapper.single_rollout` runs one `lax.scan` of `max_steps_in_episode` steps per episode
whatever happens and masks the reward after `done`, so on CartPole-v1 an episode that falls over after 20 steps
still costs the full 500-step horizon.  All three arms step the SAME `env.step` / `env.reset` from gymnax:

  gx_scan   gymnax's structure: `jax.vmap` over episodes of a fixed 500-step `lax.scan` with a valid mask.
  gx_while  `jax.vmap` of a `lax.while_loop` that stops on `done`: still lockstep on the batch maximum.
  tm        the same `while_loop` as a tallymap item, one episode per item.

Outputs, identical in all three: the summed return, a 64-bin histogram of episode lengths and the per-episode
(return, length).  The policy is a noisy linear policy, so episode lengths vary.

  python run.py             # CPU, N = 512
  python run.py --n 262144  # on a GPU
"""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
from jax import lax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _demo as D  # noqa: E402

import tallymap as tm  # noqa: E402

gymnax = D.need("gymnax")
NB = 64
W = 0.22 * jnp.array([0.4, 1.0, 4.0, 2.0], jnp.float32)      # a hand-set linear CartPole policy, scaled down
env, params = gymnax.make("CartPole-v1")
H = int(params.max_steps_in_episode)


def policy(obs, key):
    return (jax.nn.sigmoid(jnp.dot(W, obs)) > jax.random.uniform(key)).astype(jnp.int32)


def hbin(ln):
    return jnp.clip((ln * NB) // (H + 1), 0, NB - 1)


def step(c):
    obs, st, k, cum, done, ln = c
    k, ks, ka = jax.random.split(k, 3)
    obs, st, r, term, trunc, _ = env.step(ks, st, policy(obs, ka), params)
    return obs, st, k, cum + r, jnp.logical_or(term, trunc), ln + 1


def start(key):
    kr, ke = jax.random.split(key)
    obs, st = env.reset(kr, params)
    return obs, st, ke, jnp.float32(0.0), jnp.bool_(False), jnp.int32(0)


def gx_scan(keys):
    def single(key):
        obs, st, ke, *_ = start(key)

        def body(c, _):
            obs, st, k, cum, mask, ln = c
            k, ks, ka = jax.random.split(k, 3)
            obs, st, r, term, trunc, _ = env.step(ks, st, policy(obs, ka), params)
            done = jnp.logical_or(term, trunc)
            return (obs, st, k, cum + r * mask, mask * (1 - done), ln + mask), None
        c, _ = lax.scan(body, (obs, st, ke, jnp.float32(0.0), jnp.float32(1.0), jnp.float32(0.0)), (), H)
        return c[3], c[5].astype(jnp.int32)
    ret, ln = jax.vmap(single)(keys)
    return ret.sum(), jnp.zeros((NB,), jnp.float32).at[hbin(ln)].add(1.0), ret, ln


def gx_while(keys):
    def single(key):
        c = lax.while_loop(lambda c: (~c[4]) & (c[5] < H), step, start(key))
        return c[3], c[5]
    ret, ln = jax.vmap(single)(keys)
    return ret.sum(), jnp.zeros((NB,), jnp.float32).at[hbin(ln)].add(1.0), ret, ln


def episode(t, key):
    c = lax.while_loop(lambda c: (~c[4]) & (c[5] < H), step, start(key))
    t = t.add("ret", 0, c[3])
    t = t.add("len_hist", hbin(c[5]), 1.0)
    return t, (c[3], c[5])


f_tm = tm.vmap(episode, outputs={"ret": tm.Sum((1,)), "len_hist": tm.Sum((NB,))}, max_steps=H + 2)


def run_tm(keys):
    r = f_tm(keys)
    return r.outputs["ret"][0], r.outputs["len_hist"], r.per_item[0], r.per_item[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=512)
    n = p.parse_args().n
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    ref = jax.jit(gx_scan)(keys)
    checks = []
    for name, f in (("tallymap", run_tm), ("vmap + while_loop", gx_while)):
        o = jax.jit(f)(keys)
        e = max(D.rel(a, b) for a, b in zip(o, ref))
        same = float(jnp.mean(o[3] == ref[3]))
        # bitwise on a GPU; on the CPU the kernel's libm can differ from XLA in the last bit and end an episode
        # one step apart, so the contract there is: (almost) every episode the same length
        checks.append((f"{name} vs gymnax's RolloutWrapper scan", same >= 0.98 and e < 0.05,
                       f"episode lengths equal for {100 * same:.1f}%, max rel over return / histogram / "
                       f"per-episode = {e:.3g}"))
    ln = ref[3]
    checks.append(("episode lengths vary", float(ln.max()) > 2 * float(ln.mean()),
                   f"mean {float(ln.mean()):.1f}, max {int(ln.max())} of {H}"))
    D.timings(n, {"tallymap": (run_tm, (keys,)), "gymnax RolloutWrapper scan": (gx_scan, (keys,)),
                  "vmap + while_loop": (gx_while, (keys,))})
    return D.verdict(checks)


if __name__ == "__main__":
    sys.exit(main())
