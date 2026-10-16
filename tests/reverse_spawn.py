"""The MECHANICS of reverse mode through spawn (the tree adjoint) on toy cascades.

Toy: a branching process.  Each particle: up to MAXN steps (lax.while_loop); per step a pathwise flight
(-log(1-u) * lam), a deposit E*f emitted with a DiCE magic-box weight into hard bins and a soft-edged bin, and an
inner fori_loop over 2 split trials: split ~ Bernoulli(sigmoid(c + d*E)) scored into the causal score `s`; a child
gets E*r*(0.5+0.5v), inherits z and the score, and has generation g+1 (spawn only while g < DEPTH).
Roots also return a per-item result (final z) that enters the loss.

The DENSE reference is the SAME item code with a plain-JAX DenseTally (emit = .at[].add, spawn = write into a
fixed [KMAX] child buffer) and a masked fixed-trip loop, run generation by generation (python-unrolled, fixed
capacity per generation, children compacted in (parent, ordinal) order).  jax.grad of it is the definition the
tree adjoint must reproduce (ulp-level).  Also compared: tallymap jacfwd / jvp through spawn at the same keys.

python tests/reverse_spawn.py
env: TREE_BACKEND (lockstep|cuda|auto)  TREE_X64 (0|1)  TREE_N roots (48)  TREE_CG (512)  TREE_TOT (4096)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax
import numpy as np

X64 = (os.environ.get("TREE_X64") or os.environ.get("TEST_X64") or "0") == "1"
jax.config.update("jax_enable_x64", X64)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
from jax import lax

import tallymap as tm

if os.environ.get("TX_UNFUSED") == "1":                 # run_all --unfused: the two-node jvp design
    from tallymap import _primitive as _P0
    _P0.FUSE_JVP = False

BK = os.environ.get("TREE_BACKEND", "lockstep")
N = int(os.environ.get("TREE_N", 128))
CG = int(os.environ.get("TREE_CG", 2048))
TOT = int(os.environ.get("TREE_TOT", 32768))
QCAP = int(os.environ.get("TREE_Q", TOT))
FT = jnp.float64 if X64 else jnp.float32
NB, ZMAX, MAXN, DEPTH, KMAX = 16, 8.0, 6, 30, 12
ECUT = 1e-12
print("jax", jax.__version__, jax.devices(), "backend", BK, "x64", X64, "N", N, "Cg", CG, "TOT", TOT, flush=True)

RESULTS = []


def report(name, ok, **kw):
    RESULTS.append(ok)
    num = lambda v: float(v) if np.ndim(v) == 0 and not isinstance(v, (bool, str)) else v
    print(("PASS " if ok else "FAIL ") + name + " " + json.dumps({k: num(v) for k, v in kw.items()}), flush=True)


def magic(x):
    return jnp.exp(x - lax.stop_gradient(x))


def make_item(loop, fori):
    def item(t, x, th):
        def cond(c):
            s, n, t = c
            return (s["E"] > ECUT) & (n < MAXN)

        def body(c):
            s, n, t = c
            k = jax.random.fold_in(s["key"], n)
            u = jax.random.uniform(k, (2,), dtype=FT)
            z2 = s["z"] + (-jnp.log1p(-u[0])) * th["lam"]
            dep = s["E"] * th["f"]
            w = magic(s["s"])
            ib = jnp.clip(jnp.floor(z2 * (NB / ZMAX)).astype(jnp.int32), 0, NB - 1)
            t = t.add("h", ib, dep * w)
            t = t.add("soft", jnp.int32(0), dep * w * jax.nn.sigmoid((3.0 - z2) / 0.5))
            E = s["E"] - dep

            def kid(j, c2):
                E, cum, t = c2
                kj = jax.random.fold_in(k, 10 + j)
                v = jax.random.uniform(kj, (2,), dtype=FT)
                p = jax.nn.sigmoid(th["c"] + th["d"] * E)
                split = (v[0] < lax.stop_gradient(p)) & (s["g"] < DEPTH)
                cum2 = cum + jnp.where(split, jnp.log(p), jnp.log1p(-p))
                Ec = E * th["r"] * (0.5 + 0.5 * v[1])
                child = dict(E=Ec, z=z2, s=cum2, key=kj, g=s["g"] + 1)
                t = t.spawn(child, when=split)
                return jnp.where(split, E - Ec, E), cum2, t
            E, cum, t = fori(0, 2, kid, (E, s["s"], t))
            return dict(E=E, z=z2, s=cum, key=s["key"], g=s["g"]), n + 1, t
        s, n, t = loop(cond, body, (x, jnp.int32(0), t))
        return t, s["z"]
    return item


# ------------------------------------------------------------------------------------------- dense reference
@jax.tree_util.register_pytree_node_class
class DenseTally:
    def __init__(self, a, kids, nk):
        self.a, self.kids, self.nk = a, kids, nk

    def tree_flatten(self):
        return (self.a, self.kids, self.nk), None

    @classmethod
    def tree_unflatten(cls, aux, ch):
        return cls(*ch)

    def add(self, name, idx, val):
        a = dict(self.a)
        S = a[name].shape[0]
        i = jnp.asarray(idx)
        a[name] = a[name].at[jnp.where((i >= 0) & (i < S), i, S)].add(val, mode="drop")
        return DenseTally(a, self.kids, self.nk)

    def spawn(self, child, when=True):
        kids = jax.tree_util.tree_map(lambda b, c: jnp.where(when, b.at[self.nk].set(c, mode="drop"), b), self.kids,
                                      child)
        return DenseTally(self.a, kids, self.nk + jnp.asarray(when).astype(jnp.int32))


def masked_loop(cond, body, init):
    def f(i, c):
        a = cond(c)
        return jax.tree_util.tree_map(lambda x, y: jnp.where(a, x, y), body(c), c)
    return lax.fori_loop(0, MAXN, f, init)


ITEM_TALLYMAP = make_item(lax.while_loop, lax.fori_loop)
ITEM_REF = make_item(masked_loop, lax.fori_loop)


def proto(n):
    return dict(E=jnp.zeros((n,), FT), z=jnp.zeros((n,), FT), s=jnp.zeros((n,), FT),
                key=jnp.zeros((n, 2), jnp.uint32), g=jnp.zeros((n,), jnp.int32))


def reference(th, X0, depth=DEPTH + 1, cap=CG):
    """-> (outputs dict, per-root y, generation sizes)"""
    tot = dict(h=jnp.zeros((NB,), FT), soft=jnp.zeros((1,), FT))
    X, sizes, y0 = X0, [], None
    for gen in range(depth + 1):
        def one(x):
            t = DenseTally(dict(h=jnp.zeros((NB,), FT), soft=jnp.zeros((1,), FT)),
                           jax.tree_util.tree_map(lambda a: jnp.zeros((KMAX,) + a.shape[1:], a.dtype), proto(1)),
                           jnp.int32(0))
            return ITEM_REF(t, x, th)
        t, y = jax.vmap(one)(X)
        if gen == 0:
            y0 = y
        tot = {k: tot[k] + t.a[k].sum(0) for k in tot}
        nk = t.nk
        sizes.append(nk.sum())
        off = jnp.cumsum(nk) - nk
        slot = off[:, None] + jnp.arange(KMAX)[None, :]
        slot = jnp.where(jnp.arange(KMAX)[None, :] < nk[:, None], slot, cap)
        X = {k: jnp.zeros((cap,) + v.shape[2:], v.dtype).at[slot.reshape(-1)].set(v.reshape((-1,) + v.shape[2:]),
                                                                                  mode="drop")
             for k, v in t.kids.items()}
    return tot, y0, sizes


# ------------------------------------------------------------------------------------------- tallymap
def tallymap_fn(cap_q=QCAP, cg=CG, backend=BK, **neg):
    return tm.vmap(ITEM_TALLYMAP, outputs={"h": tm.Sum((NB,), FT), "soft": tm.Sum((1,), FT)}, in_axes=(0, None),
                    spawn=tm.Queue(cap_q), backend=backend, _rev_gen_capacity=cg, rev_stats=True, **neg)


rng = np.random.default_rng(0)
WH = jnp.asarray(rng.normal(size=NB), FT)
VY = jnp.asarray(rng.normal(size=N) * 0.1, FT)


def loss_of(out, y):
    return jnp.dot(WH, out["h"]) + 3.0 * out["soft"][0] + jnp.dot(VY, y)


def L_tallymap(th, X0, f=None):
    r = (f or tallymap_fn())(X0, th)
    return loss_of(r.outputs, r.per_item)


def L_ref(th, X0):
    o, y, _ = reference(th, X0)
    return loss_of(o, y)


TH = dict(lam=jnp.asarray(0.7, FT), f=jnp.asarray(0.15, FT), c=jnp.asarray(-2.3, FT), d=jnp.asarray(0.2, FT),
          r=jnp.asarray(0.8, FT))
X0 = dict(E=jnp.asarray(rng.uniform(0.5, 1.5, N), FT), z=jnp.asarray(rng.uniform(0, 0.5, N), FT), s=jnp.zeros((N,), FT),
          key=jax.random.split(jax.random.PRNGKey(3), N), g=jnp.zeros((N,), jnp.int32))
DIFF = lambda th, x: (th, x["E"], x["z"])


def flat(g):
    th, gE, gz = g
    return np.concatenate([np.asarray([th[k] for k in sorted(th)], np.float64), np.asarray(gE, np.float64),
                           np.asarray(gz, np.float64)])


def cmp(a, b):
    a, b = flat(a) if not isinstance(a, np.ndarray) else a, flat(b) if not isinstance(b, np.ndarray) else b
    eps = np.finfo(np.float64 if X64 else np.float32).eps
    d = np.abs(a - b)
    sc = np.maximum(np.abs(b), np.abs(b).max() * 1e-3)
    return dict(max_abs=float(d.max()), max_rel=float((d / sc).max()), ulps=float((d / (sc * eps)).max()),
                bitwise=float(np.mean(a == b)), n=len(a))


def grad_tallymap(th, X0, f=None):
    def L(th, E, z):
        return L_tallymap(th, dict(X0, E=E, z=z), f)
    return jax.jit(jax.value_and_grad(L, argnums=(0, 1, 2)))(th, X0["E"], X0["z"])


def grad_ref(th, X0):
    def L(th, E, z):
        return L_ref(th, dict(X0, E=E, z=z))
    return jax.jit(jax.value_and_grad(L, argnums=(0, 1, 2)))(th, X0["E"], X0["z"])


# 1. forward: tallymap vs dense reference
t0 = time.time()
o_ref, y_ref, sizes = jax.jit(reference)(TH, X0)
sizes = [int(s) for s in sizes]
r = jax.jit(lambda th, x: tallymap_fn()(x, th))(TH, X0)
gens = sum(1 for s in sizes if s > 0)
hr = np.asarray(o_ref["h"])
fd = dict(h=float(np.abs(np.asarray(r.outputs["h"]) - hr).max() / np.abs(hr).max()),
          y_equal=bool(np.array_equal(np.asarray(r.per_item), np.asarray(y_ref))), spawned=int(r.stats["spawned"]),
          ref_children=int(sum(sizes)), generations=gens, max_gen=max(sizes), dropped=int(r.stats["dropped"]))
report("forward tallymap == dense reference",
       fd["h"] < (1e-9 if X64 else 1e-4) and fd["y_equal"] and fd["spawned"] == fd["ref_children"], **fd)
print("generation sizes", sizes, flush=True)
assert max(sizes) <= CG and sum(sizes) <= TOT, "toy too big for the capacities"
assert gens >= 25, f"toy not deep enough: {gens} generations"

# 2. reverse (tree adjoint) vs jax.grad of the dense reference
tm.diagnostics(reset=True)
v_s, g_s = grad_tallymap(TH, X0)
jax.block_until_ready(g_s)
st = tm.diagnostics()["reverse"]["tree"] or {}
v_r, g_r = grad_ref(TH, X0)
c = cmp(g_s, g_r)
tol = 64 if X64 else 2e3
report("reverse(tree) == grad(dense reference)", c["ulps"] < tol and st.get("replay_mismatch", 1) == 0,
       loss_tallymap=float(v_s), loss_ref=float(v_r), **c, **{k: (np.asarray(v).tolist()) for k, v in st.items()})

# 3. reverse vs jacfwd through spawn (tallymap forward mode) at the same keys: theta columns + jvp along root E and z
keys_th = sorted(TH)


def L_vec(p, E, z):
    th = {k: p[i] for i, k in enumerate(keys_th)}
    return L_tallymap(th, dict(X0, E=E, z=z))
p0 = jnp.stack([TH[k] for k in keys_th])
J = jax.jit(jax.jacfwd(L_vec))(p0, X0["E"], X0["z"])
vE = jnp.asarray(rng.normal(size=N), FT)
vz = jnp.asarray(rng.normal(size=N), FT)
_, jE = jax.jit(lambda E: jax.jvp(lambda e: L_vec(p0, e, X0["z"]), (E,), (vE,)))(X0["E"])
_, jz = jax.jit(lambda z: jax.jvp(lambda q: L_vec(p0, X0["E"], q), (z,), (vz,)))(X0["z"])
a = np.concatenate([np.asarray([g_s[0][k] for k in keys_th], np.float64),
                    [float(jnp.dot(g_s[1], vE)), float(jnp.dot(g_s[2], vz))]])
b = np.concatenate([np.asarray(J, np.float64), [float(jE), float(jz)]])
c = cmp(a, b)
report("reverse(tree) == jacfwd/jvp through spawn", c["max_rel"] < (1e-10 if X64 else 1e-4), **c)

# 4. the tree matters: detached children (cotangents of children not injected) must differ -- NEGATIVE CONTROL
_, g_d = grad_tallymap(TH, X0, tallymap_fn(_tree_neg="detach"))
c = cmp(g_d, g_r)
report("NEG detach: detached-children gradient differs from the reference", c["max_rel"] > 1e-3, **c)

# 5. replay fingerprint: the adjoint replays a 1-ulp perturbed item -> mismatches counted, gradient NaN
#    (NEGATIVE CONTROL)
tm.diagnostics(reset=True)
_, g_p = grad_tallymap(TH, X0, tallymap_fn(_tree_neg="replay"))
jax.block_until_ready(g_p)
st = tm.diagnostics()["reverse"]["tree"]
report("NEG replay: fingerprint detects a replay that differs from the forward", int(st["replay_mismatch"]) > 0
       and np.isnan(flat(g_p)).all(), replay_mismatch=int(st["replay_mismatch"]))

# 6. overflow: generation capacity too small -> NaN + overflow flag (never silently wrong)
tm.diagnostics(reset=True)
_, g_o = grad_tallymap(TH, X0, tallymap_fn(cg=max(8, max(sizes) // 2)))
jax.block_until_ready(g_o)
st = tm.diagnostics()["reverse"]["tree"]
report("overflow (_rev_gen_capacity < largest generation) -> NaN", bool(st["overflow"]) and np.isnan(flat(g_o)).all(),
       overflow=bool(st["overflow"]))

# 7. queue capacity below the tree size -> the forward may have dropped children -> NaN (drop_risk)
tm.diagnostics(reset=True)
fq = tallymap_fn(cap_q=max(4, sum(sizes) // 2))
_, g_q = grad_tallymap(TH, X0, fq)
jax.block_until_ready(g_q)
st = tm.diagnostics()["reverse"]["tree"]
report("drop risk (tree > Queue capacity) -> NaN", bool(st["drop_risk"]) and np.isnan(flat(g_q)).all(),
       drop_risk=bool(st["drop_risk"]))

# 8. jit cache: a second evaluation point (passed as a jit argument) reuses the compiled program, and differs
TH2 = dict(TH, lam=TH["lam"] * 1.01)
v2, g2 = grad_tallymap(TH2, X0)
_, g2r = grad_ref(TH2, X0)
c = cmp(g2, g2r)
report("reverse(tree) == grad(dense) at a second point", c["ulps"] < tol, **c)
print("ALL", "PASS" if all(RESULTS) else "FAIL", f"{sum(RESULTS)}/{len(RESULTS)}", f"{time.time() - t0:.0f}s",
      flush=True)
