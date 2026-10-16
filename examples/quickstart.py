"""tallymap in one page: a random walk per item, results summed into a histogram (a "tally").

    python examples/quickstart.py          # CPU: the lockstep backend; GPU: one CUDA kernel

Each item (a walker) runs its own while_loop with its own number of steps.  tallymap.vmap maps the item over N
walkers; its meaning is exactly jax.vmap(item) followed by a sum over items, and every JAX transformation works.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm
from tallymap import reference

NB = 16                                                    # histogram bins


def walker(t, x, key, theta):
    """ONE item.  t is its Tally: thread it through the loop like a PRNG key (t = t.add(...))."""
    def body(c):
        k, x, t = c
        u = jax.random.uniform(jax.random.fold_in(key, k))
        t = t.add("hist", (x * NB).astype(jnp.int32), theta * (1.0 - x))   # out-of-range bins are dropped
        return k + 1, x * (0.5 + 0.5 * u), t
    k, x, t = lax.while_loop(lambda c: c[1] > 0.05, body, (0, x, t))
    return t, k                                            # the Tally, plus a per-item result (steps taken)


f = tm.vmap(walker, outputs={"hist": tm.Sum((NB,))}, in_axes=(0, 0, None), max_steps=100)

N = 10_000
xs = jnp.linspace(0.1, 0.99, N)
keys = jax.random.split(jax.random.PRNGKey(0), N)
theta = jnp.float32(2.0)

res = jax.jit(f)(xs, keys, theta)
print("hist[:4]", np.asarray(res.outputs["hist"][:4]), " mean steps", float(res.per_item.mean()),
      " truncated", int(res.stats["n_truncated"]))

# derivatives: pass the evaluation point as a jit ARGUMENT
loss = lambda th: jnp.sum(f(xs, keys, th).outputs["hist"] ** 2)
g = jax.jit(jax.grad(loss))(theta)                         # reverse mode (checkpointed replay)
_, t = jax.jit(lambda th: jax.jvp(loss, (th,), (1.0,)))(theta)  # forward mode
print("d loss / d theta: grad", float(g), " jvp", float(t))

# the dense meaning (jax.vmap + sum) on a small N agrees
n = 256
dense = reference(walker, outputs={"hist": tm.Sum((NB,))}, in_axes=(0, 0, None), max_steps=100)
a = f(xs[:n], keys[:n], theta).outputs["hist"]
b = dense(xs[:n], keys[:n], theta).outputs["hist"]
err = float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))
print("max rel. difference to the dense reference", err)
assert err < 1e-5 and abs(float(g) - float(t)) <= 1e-4 * abs(float(t))
print("QUICKSTART OK")
