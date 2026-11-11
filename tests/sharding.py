"""Sharded jax.jit: the call's partitioning rule (_primitive, "sharding").

Items sharded over a 1-D mesh of k devices must give the single-device result (tallies and stats summed over the
devices, per-item results split like the items), for the forward pass, grad, jacfwd, spawn and grad through spawn,
and the compiled program must not all-gather the items (the call runs on N/k items per device).  The devices: the
default backend's when it has two or more, else 4 forced host CPU devices (XLA_FLAGS is set here, before JAX loads).
Negative control: with the rule switched off (_primitive._multi_device patched to False) XLA all-gathers the items
again, and the all-gather check must see it.
"""
import os
import re
import sys

if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P

import tallymap as tm  # noqa: E402
from tallymap import _primitive  # noqa: E402

RES = []


def report(name, ok, msg):
    RES.append((name, ok))
    print(f"{name:34s} {'PASS' if ok else 'FAIL'}  {msg}", flush=True)


devs = jax.devices()
if len(devs) < 2:
    try:
        devs = jax.devices("cpu")
    except RuntimeError:
        devs = devs[:1]
print("jax", jax.__version__, "devices", devs, flush=True)
if len(devs) < 2:
    report("sharding (one device only)", True, "skipped: needs two or more devices")
    print("TOTAL", len(RES), "PASS", sum(ok for _, ok in RES))
    sys.exit(0)

NB, N, k = 16, 8192, len(devs)


def walker(t, x, key, theta):
    def body(c):
        i, x, t = c
        u = jax.random.uniform(jax.random.fold_in(key, i))
        t = t.add("hist", (x * NB).astype(jnp.int32), theta * (1.0 - x))
        return i + 1, x * (0.5 + 0.5 * u), t
    i, x, t = lax.while_loop(lambda c: c[1] > 0.05, body, (0, x, t))
    return t, x


def shower(t, x):
    t = t.add("hist", (x * NB).astype(jnp.int32), x)
    return t.spawn(x * 0.5, when=x > 0.2)


f = tm.vmap(walker, outputs={"hist": tm.Sum((NB,))}, in_axes=(0, 0, None), max_steps=100)
fs = tm.vmap(shower, outputs={"hist": tm.Sum((NB,))}, spawn=tm.Queue(4 * N))   # >= the whole tree: nothing dropped
CASES = {   # name -> (function, all-gathers allowed: grad through spawn compacts its generations in plain JAX)
    "forward": (lambda xs, ks, th: (lambda r: (r.outputs["hist"], r.per_item, r.stats["n_truncated"]))(
        f(xs, ks, th)), 0),
    "grad": (jax.grad(lambda xs, ks, th: f(xs, ks, th).outputs["hist"].sum() + f(xs, ks, th).per_item.sum(),
                      argnums=(0, 2)), 0),
    "jacfwd": (lambda xs, ks, th: jax.jacfwd(lambda t: f(xs, ks, t).outputs["hist"])(th), 0),
    "spawn": (lambda xs, ks, th: (lambda r: (r.outputs["hist"], r.stats["spawned"], r.stats["children_dropped"]))(
        fs(xs)), 0),
    "grad through spawn": (jax.grad(lambda xs, ks, th: fs(xs * th).outputs["hist"].sum(), argnums=(0, 2)), None),
}
xs = np.linspace(0.1, 0.99, N, dtype=np.float32)
ks = np.asarray(jax.random.split(jax.random.PRNGKey(0), N))
th = np.float32(2.0)
one = tuple(jax.device_put(a, devs[0]) for a in (xs, ks, th))
mesh = Mesh(np.array(devs), ("d",))
args = (jax.device_put(xs, NamedSharding(mesh, P("d"))), jax.device_put(ks, NamedSharding(mesh, P("d"))),
        jax.device_put(th, NamedSharding(mesh, P())))
gathers = lambda fn: len(re.findall(r"\ball-gather(-start)?\(", jax.jit(fn).lower(*args).compile().as_text()))

for name, (fn, allowed) in CASES.items():
    try:
        ref = jax.tree_util.tree_leaves(jax.jit(fn)(*one))
        out = jax.tree_util.tree_leaves(jax.jit(fn)(*args))
        rel = max(float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)))
                        / (np.max(np.abs(np.asarray(b, np.float64))) + 1e-30)) for a, b in zip(out, ref))
        ag = gathers(fn)
        ok = rel < 1e-4 and (allowed is None or ag <= allowed)
        if name == "spawn" and int(ref[2]):
            ok, rel = False, f"the reference dropped {int(ref[2])} children (raise the Queue)"
        report(f"{name} (k={k})", ok, f"relerr {rel if isinstance(rel, str) else f'{rel:.2g}'}, all-gathers {ag}")
    except Exception as e:  # noqa: BLE001
        report(f"{name} (k={k})", False, f"RAISE {type(e).__name__}: {str(e)[:300]}")

# negative control: the rule off -> the items are all-gathered again, and the check above must see it
_md = _primitive._multi_device
_primitive._multi_device = lambda ctx: False
try:
    ag = gathers(lambda xs, ks, th: f(xs, ks, th).outputs["hist"] + 0.0)    # (a new function: a fresh lowering)
    report("negative control: rule off", ag > 0, f"all-gathers {ag} (must be > 0)")
finally:
    _primitive._multi_device = _md

print("TOTAL", len(RES), "PASS", sum(ok for _, ok in RES), flush=True)
sys.exit(0 if all(ok for _, ok in RES) else 1)
