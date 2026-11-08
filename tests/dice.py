"""DiCE through tallymap: gradients of DISCRETE outcomes, with float score leaves carried through spawn.
python tests/dice.py   (env: BACKENDS=lockstep,cuda  TEST_X64=1  ONLY=substr)

Ported from the private dtrax line (tests_wavefront/test_dice.py a49dd83 + 3e8fd86, test_dice_baseline.py
e9c99ac + 02da99e + 68c0c6e, programs from grad_progs.py), adapted to this suite's cell format.  Two programs whose
pathwise derivative is EXACTLY zero (theta enters only through a comparison), so the whole gradient is the DiCE score
and a closed form tells a right answer from a plausible one:

  slab    a particle crosses a slab in K steps, surviving each with p = exp(-mu dz).  No spawn.
          E[transmitted] = exp(-mu L); dE/dmu = -L exp(-mu L).
  branch  an item spawns at most one child with probability sigmoid(theta), to depth D; the child INHERITS its
          ancestors' score as a per-item FLOAT LEAF of the spawn payload.  E[n] = sum_g p^g.

References: R1 the lockstep backend (cuda's tangent must equal it to float tolerance; GPU only), R3 the analytic
derivative as a z-score against the estimator's OWN standard error.  A z-score cell whose standard error cannot
separate the answer from zero (se / |scale| > 0.34) is UNRESOLVED, which is not a PASS.
The factored-score cells (R2) check, in this order: the score tally is exactly 0.0 forward; E[dS_k] = 0 per decision
(the licence of a baseline); the gradient, total and per decision; the two baseline forms -- item-attributed
(f - b) * sv, which is NOT a control variate on the cascade (graded against its PREDICTED bias -b dE: a CONTROL), and
decision-attributed f * sv - b * S, unbiased on both.  Two plants must fire: 'wrongp' (score log p whatever the
outcome) breaks the slab's licence, 'noinherit' (the child forgets its ancestors' score) breaks the cascade's
gradient.
Summary: TOTAL n PASS k (then FAIL / RAISE / UNRESOLVED counts).
"""
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.environ.get("TEST_X64"):
    import jax
    jax.config.update("jax_enable_x64", True)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402

import tallymap as tm  # noqa: E402

if os.environ.get("TX_UNFUSED") == "1":                 # run_all --unfused: the two-node jvp design
    from tallymap import _primitive as _P0
    _P0.FUSE_JVP = False
GPU = jax.default_backend() == "gpu"
BACKENDS = os.environ.get("BACKENDS", "lockstep,cuda" if GPU else "lockstep").split(",")
B0 = "cuda" if "cuda" in BACKENDS else "lockstep"       # the backend under test; lockstep is R1's reference
ONLY = os.environ.get("ONLY", "")
EX = bool(jax.config.jax_enable_x64)                    # exact tallies need x64 on the lockstep backend
F = jnp.float64 if EX else jnp.float32
TOL = 3e-5 if EX else 3e-3
print("jax", jax.__version__, jax.__file__, jax.devices(), "x64", EX, "backend", B0, flush=True)
RES = []


class Fail(Exception):
    pass


class Unresolved(Exception):
    pass


def case(name, gpu_only=False):
    def deco(fn):
        if (ONLY and ONLY not in name) or (gpu_only and "cuda" not in BACKENDS):
            return fn
        t0 = time.perf_counter()
        try:
            r = fn()
            msg = "PASS" + (f" ({r})" if isinstance(r, str) and r else "")
        except Fail as e:
            msg = f"FAIL {e}"
        except Unresolved as e:
            msg = f"UNRESOLVED {e}"
        except Exception as e:
            msg = f"RAISE {type(e).__name__}: {str(e)[:300]}"
            traceback.print_exc()
        RES.append((name, msg))
        print(f"{name:66s} {msg}  [{time.perf_counter() - t0:.1f}s]", flush=True)
        return fn
    return deco


def graded(est, se, truth, scale=None, lim=3.5):
    """z of est against truth in units of the estimator's own se; UNRESOLVED when se cannot separate the answer
    from zero (scale: the effect size to separate from, for a truth of 0)."""
    sc = abs(scale if scale is not None else truth)
    res = se / sc if sc else float("inf")
    z = (est - truth) / se if se > 0 else float("inf")
    d = f"est {est:+.6g} truth {truth:+.6g} se {se:.3g} z {z:+.2f} se/|scale| {res:.3g}"
    if res > 0.34:
        raise Unresolved(d)
    if abs(z) > lim:
        raise Fail(d)
    return d


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def jvp_of(g, th, dth):
    return jax.block_until_ready(jax.jit(lambda t, d: jax.jvp(g, (t,), (d,)))(th, dth))


# ================================================================================================ the programs
SLAB_L, SLAB_K = 1.0, 16
BRANCH_D = 8
NG = 32                          # independent root groups (R2): the error of every mean


def slab(t, key, th, dice=True):
    dz = jnp.asarray(SLAB_L / SLAB_K, F)
    p = jnp.exp(-th[0] * dz)

    def body(i, c):
        alive, tau = c
        u = jax.random.uniform(jax.random.fold_in(key, i), (), F)
        hit = u < p
        lp = jnp.where(hit, jnp.log(p), jnp.log1p(-p))
        return alive & hit, tau + jnp.where(alive, lp, jnp.asarray(0.0, F))
    alive, tau = lax.fori_loop(0, SLAB_K, body, (jnp.asarray(True), jnp.asarray(0.0, F)))
    mb = jnp.exp(tau - lax.stop_gradient(tau)) if dice else jnp.asarray(1.0, F)    # DiCE magic box
    t = t.add("T", 0, jnp.where(alive, jnp.asarray(1.0, F), jnp.asarray(0.0, F)) * mb)
    return t.add("T", 1, mb)                             # the control variate: E = 1, dE/dmu = 0


def slab_truth(mu):
    return float(np.exp(-mu * SLAB_L)), float(-SLAB_L * np.exp(-mu * SLAB_L))


def branch(t, s, th):
    p = jnp.asarray(1.0, F) / (jnp.asarray(1.0, F) + jnp.exp(-th[0]))
    t = t.add("n", 0, jnp.exp(s["tau"] - lax.stop_gradient(s["tau"])))
    u = jax.random.uniform(jax.random.fold_in(s["key"], 0), (), F)
    split = (u < p) & (s["g"] < BRANCH_D)
    return t.spawn(dict(g=s["g"] + 1, tau=s["tau"] + jnp.log(p), key=jax.random.fold_in(s["key"], 1)), when=split)


def branch_inputs(n, seed=5):
    return dict(g=jnp.zeros((n,), jnp.int32), tau=jnp.zeros((n,), F), key=jax.random.split(jax.random.PRNGKey(seed), n))


def branch_truth(th0):
    p = 1.0 / (1.0 + np.exp(-th0))
    return (float(sum(p ** g for g in range(BRANCH_D + 1))),
            float(sum(g * p ** (g - 1) for g in range(1, BRANCH_D + 1)) * p * (1.0 - p)))


def slab_fac(t, key, grp, th, b=0.0, plant=""):
    K = SLAB_K
    dz = jnp.asarray(SLAB_L / K, F)
    p = jnp.exp(-th[0] * dz)
    Z = jnp.asarray(0.0, F)

    def body(i, c):
        alive, sv = c
        u = jax.random.uniform(jax.random.fold_in(key, i), (), F)
        hit = u < p
        lp = jnp.log(p) if plant == "wrongp" else jnp.where(hit, jnp.log(p), jnp.log1p(-p))
        s = (lp - lax.stop_gradient(lp)) * jnp.where(alive, jnp.asarray(1.0, F), Z)
        return alive & hit, sv.at[i].set(s)
    alive, sv = lax.fori_loop(0, K, body, (jnp.asarray(True), jnp.zeros((K,), F)))
    f = jnp.where(alive, jnp.asarray(1.0, F), Z)
    idx = grp * jnp.int32(K) + jnp.arange(K, dtype=jnp.int32)
    t = t.add("S", idx, sv).add("FS", idx, f * sv).add("NS", idx, (f - jnp.asarray(b, F)) * sv)
    return t.add("m", grp * jnp.int32(2) + jnp.arange(2, dtype=jnp.int32), jnp.stack([f, jnp.asarray(1.0, F)]))


def slab_fac_truth(mu):
    E = float(np.exp(-mu * SLAB_L))
    return E, float(-SLAB_L * E), [float(-SLAB_L / SLAB_K * E)] * SLAB_K


def branch_fac(t, s, th, b=0.0, plant=""):
    D = BRANCH_D
    p = jnp.asarray(1.0, F) / (jnp.asarray(1.0, F) + jnp.exp(-th[0]))
    sv, g, grp = s["sv"], s["g"], s["grp"]
    f = jnp.asarray(1.0, F)
    idx = grp * jnp.int32(D) + jnp.arange(D, dtype=jnp.int32)
    t = t.add("FS", idx, f * sv).add("NS", idx, (f - jnp.asarray(b, F)) * sv)
    u = jax.random.uniform(jax.random.fold_in(s["key"], 0), (), F)
    live = g < jnp.int32(D)
    split = (u < p) & live
    lq = jnp.where(split, jnp.log(p), jnp.log1p(-p))
    hot = (jnp.arange(D, dtype=jnp.int32) == jnp.clip(g, 0, D - 1)).astype(F) * live.astype(F)
    t = t.add("S", idx, (lq - lax.stop_gradient(lq)) * hot)
    t = t.add("m", grp * jnp.int32(2) + jnp.arange(2, dtype=jnp.int32), jnp.stack([f, jnp.asarray(1.0, F)]))
    lp = jnp.log(p)
    child_sv = jnp.zeros((D,), F) if plant == "noinherit" else sv + (lp - lax.stop_gradient(lp)) * hot
    return t.spawn(dict(g=g + 1, grp=grp, key=jax.random.fold_in(s["key"], 1), sv=child_sv), when=split)


def branch_fac_truth(th0):
    D = BRANCH_D
    p = 1.0 / (1.0 + np.exp(-th0))
    dp = p * (1.0 - p)
    E = sum(p ** g for g in range(D + 1))
    dE = sum(g * p ** (g - 1) for g in range(1, D + 1)) * dp
    return float(E), float(dE), [float(dp / p * sum(p ** m for m in range(g + 1, D + 1))) for g in range(D)]


def fac_outputs(K):
    o = {n: tm.Sum((NG * K,), F, exact=EX) for n in ("S", "FS", "NS")}
    o["m"] = tm.Sum((NG * 2,), F, exact=EX)
    return o


# ================================================================================================ G4: slab
N_SLAB = 1 << 16
SKEYS = jax.random.split(jax.random.PRNGKey(4), N_SLAB)
TH_S, DTH = jnp.asarray([0.9, 0.0, 0.0], F), jnp.asarray([1.0, 0.0, 0.0], F)


def slab_run(backend, dice=True):
    f = tm.vmap(lambda t, k, th: slab(t, k, th, dice), outputs={"T": tm.Sum((2,), F, exact=EX)}, in_axes=(0, None),
                backend=backend)
    p, d = jvp_of(lambda th: f(SKEYS, th).outputs["T"], TH_S, DTH)
    return np.asarray(p, np.float64), np.asarray(d, np.float64)


def slab_per_item():
    """the per-item tangents in plain JAX (jax.vmap, not tallymap), for the standard errors"""
    def one(k):
        def h(th):
            r = slab(_Acc(), k, th)
            return jnp.stack([r.v[0], r.v[1]])
        return jax.jvp(h, (TH_S,), (DTH,))
    p, d = jax.jit(jax.vmap(one))(SKEYS)
    return np.asarray(p, np.float64), np.asarray(d, np.float64)


class _Acc:
    """a two-bin accumulator with a Tally's .add, so the same item function runs under jax.vmap"""
    def __init__(self, v=None):
        self.v = jnp.zeros((2,), F) if v is None else v

    def add(self, _name, i, x):
        return _Acc(self.v.at[i].add(x))


_SLAB = {}


def _slab():
    if not _SLAB:
        _SLAB.update(main=slab_run(B0), per=slab_per_item())
    return _SLAB


@case("[G4] slab R1: cuda tangent == lockstep", gpu_only=True)
def _():
    pc, tc = _slab()["main"]
    pl, tl = slab_run("lockstep")
    r = rel(tc, tl)
    if r >= TOL:
        raise Fail(f"max rel {r:.3e}")
    return f"max rel {r:.3e}"


@case("[G4] slab forward E[transmit] vs exp(-mu L); control variate E = 1 exactly, dE = 0 in mean")
def _():
    (pc, tc), (pi, ti) = _slab()["main"], _slab()["per"]
    E0, _ = slab_truth(0.9)
    se0 = float(np.std(pi[:, 0], ddof=1) / np.sqrt(N_SLAB))
    if abs(pc[0] / N_SLAB - E0) > 4.0 * se0:
        raise Fail(f"E[transmit] {pc[0] / N_SLAB:.6f} vs {E0:.6f} (se {se0:.2g})")
    if abs(pc[1] / N_SLAB - 1.0) > 1e-6:
        raise Fail(f"E[magic box] {pc[1] / N_SLAB:.8f} != 1")
    se1 = float(np.std(ti[:, 1], ddof=1) / np.sqrt(N_SLAB))
    z1 = (tc[1] / N_SLAB) / se1
    if abs(z1) > 3.5:
        raise Fail(f"control dE {tc[1] / N_SLAB:+.3e}, its own se {se1:.3e}, z {z1:+.2f}")
    return f"E {pc[0] / N_SLAB:.5f} vs {E0:.5f}; control dE z {z1:+.2f}"


@case("[G4] slab R3: dE/dmu vs -L exp(-mu L)")
def _():
    tc, ti = _slab()["main"][1], _slab()["per"][1]
    return graded(float(tc[0]) / N_SLAB, float(np.std(ti[:, 0], ddof=1) / np.sqrt(N_SLAB)), slab_truth(0.9)[1])


@case("[G4] slab WITHOUT the DiCE box: tangent exactly 0, same forward value (the trap)")
def _():
    pc, tc = _slab()["main"]
    pz, tz = slab_run(B0, dice=False)
    if tz[0] != 0.0 or abs(pz[0] - pc[0]) > 1e-6 * max(1.0, abs(pc[0])):
        raise Fail(f"|tangent| {abs(tz[0]):.3e}, forward {pz[0]} vs {pc[0]}")
    return f"DiCE tangent {abs(tc[0]):.3e} vs 0"


# ================================================================================================ G4: branch
N_BR = 1 << 14
TH_B = jnp.asarray([0.4, 0.0, 0.0], F)


def branch_f(backend, cap=1 << 20):
    return tm.vmap(branch, outputs={"n": tm.Sum((1,), F, exact=EX)}, in_axes=(0, None), spawn=tm.Queue(cap),
                   backend=backend)


def branch_run(backend, s, cap=1 << 20):
    f = branch_f(backend, cap)
    p, d = jvp_of(lambda th: f(s, th).outputs["n"], TH_B, DTH)
    return np.asarray(p, np.float64), np.asarray(d, np.float64)


_BR = {}


def _branch():
    if not _BR:
        _BR.update(main=branch_run(B0, branch_inputs(N_BR)))
    return _BR["main"]


@case("[G4] branch R1: cuda tangent == lockstep (spawn, inherited float score leaf)", gpu_only=True)
def _():
    pc, tc = _branch()
    pl, tl = branch_run("lockstep", branch_inputs(N_BR))
    r = rel(tc, tl)
    if r >= TOL:
        raise Fail(f"max rel {r:.3e} (cuda {tc[0] / N_BR:+.6g}, lockstep {tl[0] / N_BR:+.6g} per root)")
    return f"max rel {r:.3e}"


@case("[G4] branch forward E[n] vs sum p^g; R3 dE/dtheta (block standard error)")
def _():
    pc, tc = _branch()
    E0, dE0 = branch_truth(0.4)
    if abs(pc[0] / N_BR - E0) > 0.02 * E0:
        raise Fail(f"E[n] {pc[0] / N_BR:.5f} vs {E0:.5f}")
    p0, Bk = branch_inputs(N_BR), 64
    blk = [float(branch_run(B0, {k: v[j::Bk] for k, v in p0.items()}, cap=1 << 18)[1][0]) / (N_BR // Bk)
           for j in range(Bk)]
    se = float(np.std(blk, ddof=1) / np.sqrt(Bk))
    return f"E {pc[0] / N_BR:.4f} vs {E0:.4f}; " + graded(float(tc[0]) / N_BR, se, dE0)


@case("[G4] branch: reverse through the spawn tree (tree adjoint) == jacfwd")
def _():
    f, s = branch_f(B0), branch_inputs(N_BR)
    gc = lambda th: f(s, th).outputs["n"][0]                     # noqa: E731
    gr = np.asarray(jax.jit(jax.grad(gc))(TH_B), np.float64)
    jf = np.asarray(jax.jit(jax.jacfwd(gc))(TH_B), np.float64)
    r = rel(gr, jf)
    if r >= TOL:
        raise Fail(f"max rel {r:.3e}: grad {gr[0]:+.6g} jacfwd {jf[0]:+.6g}")
    return f"max rel {r:.3e}, grad {gr[0]:+.6g}"


# ============================================================================================= R2: the factored score
def r2_run(tag, plant=""):
    if tag == "slab":
        n, K, th0, b = 1 << 18, SLAB_K, 0.9, float(np.exp(-0.9 * SLAB_L))
        keys = jax.random.split(jax.random.PRNGKey(4), n)
        grp = jnp.arange(n, dtype=jnp.int32) % jnp.int32(NG)
        f = tm.vmap(lambda t, k, g_, x: slab_fac(t, k, g_, x, b=b, plant=plant), outputs=fac_outputs(K),
                    in_axes=(0, 0, None), backend=B0)
        g = lambda x: f(keys, grp, x).outputs              # noqa: E731
        truth = slab_fac_truth(th0)
    else:
        n, K, th0, b = 1 << 15, BRANCH_D, 0.4, 1.0
        s = dict(g=jnp.zeros((n,), jnp.int32), sv=jnp.zeros((n, K), F),
                 grp=jnp.arange(n, dtype=jnp.int32) % jnp.int32(NG), key=jax.random.split(jax.random.PRNGKey(5), n))
        f = tm.vmap(lambda t, ss, x: branch_fac(t, ss, x, b=b, plant=plant), outputs=fac_outputs(K),
                    in_axes=(0, None), spawn=tm.Queue(1 << 17), backend=B0)
        g = lambda x: f(s, x).outputs                      # noqa: E731
        truth = branch_fac_truth(th0)
    o, do = jvp_of(g, jnp.asarray([th0], F), jnp.asarray([1.0], F))
    A = {k: np.asarray(v, np.float64).reshape(NG, -1) for k, v in o.items()}
    dA = {k: np.asarray(v, np.float64).reshape(NG, -1) for k, v in do.items()}
    return dict(A=A, dA=dA, npg=n // NG, K=K, b=b, truth=truth, raw_S=np.asarray(o["S"]))


def stats(a, npg):
    v = np.asarray(a, np.float64) / float(npg)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


_R2 = {}


def r2(tag, plant=""):
    if (tag, plant) not in _R2:
        _R2[(tag, plant)] = r2_run(tag, plant)
    return _R2[(tag, plant)]


def licence(r):
    """-> (n RED, n unresolved, worst |z|, detail) of E[dS_k] = 0 over the decisions k"""
    dS, npg, dsh = r["dA"]["S"], r["npg"], r["truth"][2]
    nbad, nun, worst = 0, 0, 0.0
    for k in range(r["K"]):
        m, se = stats(dS[:, k], npg)
        z = m / se if se > 0 else float("inf")
        if se > 0.34 * max(abs(dsh[k]), 1e-30):
            nun += 1
        elif abs(z) > 3.5:
            nbad += 1
        worst = max(worst, abs(z))
    return nbad, nun, worst, f"worst |z| {worst:.2f} over {r['K']} decisions, {nbad} RED, {nun} unresolved"


def gradient(r):
    m, se = stats(r["dA"]["FS"].sum(1), r["npg"])
    return m, se


for _tag in ("slab", "branch"):
    @case(f"[R2] {_tag} A: the score tally is EXACTLY 0.0 forward (value neutrality, bitwise)")
    def _(tag=_tag):
        S = r2(tag)["raw_S"]
        if not np.all(S.view(np.uint64 if S.dtype == np.float64 else np.uint32) == 0):
            raise Fail(f"max |S| = {np.abs(S).max():.3g}")
        return f"{S.size} bins"

    @case(f"[R2] {_tag} B: E[dS_k] = 0 for every decision k (the baseline's licence)")
    def _(tag=_tag):
        nbad, nun, worst, d = licence(r2(tag))
        if nbad:
            raise Fail(d)
        if nun == r2(tag)["K"]:
            raise Unresolved(d)
        return d

    @case(f"[R2] {_tag} C: gradient sum_k E[f ds_k] vs the analytic dE/dtheta, and every per-decision share")
    def _(tag=_tag):
        r = r2(tag)
        m, se = gradient(r)
        d = graded(m, se, r["truth"][1])
        wz, wk = 0.0, -1
        for k in range(r["K"]):
            mk, sek = stats(r["dA"]["FS"][:, k], r["npg"])
            z = (mk - r["truth"][2][k]) / sek
            if abs(z) > abs(wz):
                wz, wk = z, k
        if abs(wz) > 3.5:
            raise Fail(f"per-decision share {wk}: z {wz:+.2f}")
        return d + f"; worst share |z| {abs(wz):.2f}"

    @case(f"[R2] {_tag} D: baselines -- item-attributed (f-b)*sv as predicted, decision-attributed f*sv-b*S unbiased")
    def _(tag=_tag):
        r = r2(tag)
        npg, b, dE = r["npg"], r["b"], r["truth"][1]
        nm, nse = stats(r["dA"]["NS"].sum(1), npg)
        pred = -b * dE if tag == "branch" else 0.0      # the term count IS the discrete outcome on the cascade
        if abs(pred) > max(3.5 * nse, 1e-12):
            if abs((nm - dE) - pred) > max(3.5 * nse, 1e-9 * abs(pred)):
                raise Fail(f"item-attributed bias {nm - dE:+.6g}, predicted {pred:+.6g} (se {nse:.3g})")
            d = f"item-attributed: CONTROL fired, bias {nm - dE:+.4g} = predicted {pred:+.4g}"
        else:
            d = "item-attributed: " + graded(nm, nse, dE)
        dC = r["dA"]["FS"] - b * r["dA"]["S"]
        cm, cse = stats(dC.sum(1), npg)
        v0, vd = (r["dA"]["FS"].sum(1) / npg).std(ddof=1), (dC.sum(1) / npg).std(ddof=1)
        return d + "; decision-attributed: " + graded(cm, cse, dE) + f"; sigma ratio {vd / v0:.3f}"


@case("[R2] plant 'wrongp' (score log p whatever the outcome) must break the slab's licence")
def _():
    nbad, nun, worst, d = licence(r2("slab", "wrongp"))
    if not nbad:
        raise Fail("the plant did not fire: " + d)
    return "fired: " + d


@case("[R2] plant 'noinherit' (the child forgets its ancestors' score) must break the cascade's gradient")
def _():
    r = r2("branch", "noinherit")
    m, se = gradient(r)
    z = (m - r["truth"][1]) / se
    if abs(z) <= 3.5:
        raise Fail(f"the plant did not fire: gradient {m:+.5g} vs {r['truth'][1]:+.5g}, z {z:+.2f}")
    return f"fired: gradient {m:+.5g} vs {r['truth'][1]:+.5g}, z {z:+.1f}"


print("\n================ SUMMARY", jax.__version__, "x64" if EX else "", jax.devices()[0].device_kind)
for n_, m_ in RES:
    print(f"{n_:66s} {m_[:200]}")
cnt = lambda p: sum(m.startswith(p) for _, m in RES)         # noqa: E731
print("TOTAL", len(RES), "PASS", cnt("PASS"), "FAIL", cnt("FAIL"), "RAISE", cnt("RAISE "), "UNRESOLVED",
      cnt("UNRESOLVED"))
