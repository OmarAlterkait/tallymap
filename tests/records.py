"""Records outputs (tallymap.Records + t.append): one pass (the default) and count-then-fill (spawn, capacity=None).
python tests/records.py  (env: BACKENDS=xla,auto  TEST_X64=1  ONLY=substr)

The toy: a 2-D lattice walk in an L x L box.  Every DECISION is integer-exact (RNG bits), so the backends and the
reference agree on which records exist; the float fields (path time t = s / v, weight w attenuated by exp(-ell / mu_a)
and raised by a roulette at each scatter) carry pathwise derivatives in mu_a and v.  A wall hit appends a record
(sensor, t, w, item, k) when w >= wcut: wcut = 0 makes existence independent of the parameters (finite differences
are valid), wcut > 0 is a parameter-dependent cut (checked against the reference's own jvp).

Cells: the canonical (item, k) layout and the dense reference; bitwise determinism (twice, step vs item schedule,
xla batch sizes); capacity / overflow / capacity=None; the fill guard and the count-mismatch poison (negative
control through _api._COUNTS_HOOK); jvp / vjp / jacfwd / jacrev against central differences and the reference;
jax.checkpoint, the replay adjoint (small rev_checkpoints) and the single-loop adjoint; jit, n_active, jax.vmap of
the call, max_steps; refusals (t.append inside jax.vmap within an item) and misuse errors.  One pass: bitwise equal to
count-then-fill (_api.ONE_PASS = False) in every output and derivative; the count report (stats['appended'] on
overflow); contention; the unfused jvp and the fill twin's mismatch poison; subnormal fields.
"""
import os
import sys
import time
import traceback
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.environ.get("TEST_X64"):
    import jax
    jax.config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import numpy as np
from ab_rule import patched
from jax import lax

import tallymap as tm
from tallymap import _api

if os.environ.get("TX_UNFUSED") == "1":
    from tallymap import _primitive as _P0
    _P0.FUSE_JVP = False
if os.environ.get("TX_T3") == "1":
    from tallymap import _reverse as _R0
    _R0.SINGLE = "force"

BACKENDS = os.environ.get("BACKENDS", "xla,auto").split(",")
ONLY = os.environ.get("ONLY", "")
F = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
print("jax", jax.__version__, jax.__file__, jax.devices(), "x64", jax.config.jax_enable_x64, flush=True)
RES = []


class Fail(Exception):
    pass


def case(name, backends=None):
    def deco(fn):
        for b in (backends or BACKENDS):
            if b not in BACKENDS or (ONLY and ONLY not in name):
                continue
            t0 = time.perf_counter()
            try:
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    r = fn(b)
                msg = "PASS" + (f" ({r})" if isinstance(r, str) else "")
            except Fail as e:
                msg = f"FAIL {e}"
            except Exception as e:
                msg = f"RAISE {type(e).__name__}: {str(e)[:200]}"
                traceback.print_exc()
            RES.append((name, b, msg))
            print(f"{name:60s} {b:6s} {msg}  [{time.perf_counter() - t0:.1f}s]", flush=True)
        return fn
    return deco


def close(a, b, rtol=2e-4, atol=1e-6, what=""):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        raise Fail(f"{what} shape {a.shape} vs {b.shape}")
    err = np.max(np.abs(a - b)) if a.size else 0.0
    tol = rtol * (np.max(np.abs(b)) if b.size else 0.0) + atol
    if not err <= tol:
        raise Fail(f"{what} max err {err:.3g} > tol {tol:.3g}")


def same(a, b, what=""):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
        raise Fail(f"{what} not bitwise equal")


def raises(exc, f, contains=""):
    try:
        f()
    except exc as e:
        if contains and contains not in str(e):
            raise Fail(f"raised {type(e).__name__} but message lacks {contains!r}: {str(e)[:150]}")
        return
    raise Fail(f"did not raise {exc.__name__}")


# ------------------------------------------------------------------------------------------------ the toy
L, NSTEP, CAP, N = 6, 40, 4096, 256
FIELDS = {"sensor": jnp.int32, "t": F, "w": F, "item": jnp.int32, "k": jnp.int32}
INT = ("sensor", "item", "k")
FLT = ("t", "w")


def walk(t, i, key, mu_a, v, wcut):
    def body(c):
        n, x, y, s, w, k, alive, t = c
        b = jax.random.bits(jax.random.fold_in(key, n), (3,))
        d = (b[0] % 4).astype(jnp.int32)
        x = x + jnp.array([1, -1, 0, 0], jnp.int32)[d]
        y = y + jnp.array([0, 0, 1, -1], jnp.int32)[d]
        hit = (x <= 0) | (x >= L - 1) | (y <= 0) | (y >= L - 1)
        u = ((b[1] >> 8).astype(F) + 0.5) * F(2.0 ** -24)
        ell = -jnp.log(u) * 0.5
        s = s + ell
        w = w * jnp.exp(-ell / mu_a)
        sensor = x * L + y
        rec = hit & (w >= wcut)
        t = t.append("hits", {"sensor": sensor, "t": s / v, "w": w, "item": i, "k": k}, when=rec)
        t = t.add("Q", sensor, jnp.where(hit, w, 0.0))
        k = k + rec.astype(jnp.int32)
        r10 = (b[2] % 10).astype(jnp.int32)
        alive = jnp.where(hit, r10 < 6, r10 < 9) & (n < NSTEP)
        x, y = jnp.clip(x, 1, L - 2), jnp.clip(y, 1, L - 2)
        w = jnp.where(hit, w, w / 0.9)                   # roulette: a scatter survives with 0.9, w / 0.9
        return n + 1, x, y, s, w, k, alive, t
    c = (jnp.int32(0), jnp.int32(L // 2), jnp.int32(L // 2), F(0.0), F(1.0), jnp.int32(0), jnp.bool_(True), t)
    c = lax.while_loop(lambda c: c[6], body, c)
    return c[-1], (c[5], c[4])


AX = (0, 0, None, None, None)
IDS = jnp.arange(N, dtype=jnp.int32)
KEYS = jax.random.split(jax.random.PRNGKey(3), N)
MU, VV = F(2.0), F(1.5)


def outs(cap=CAP):
    return {"Q": tm.Sum((L * L,), F), "hits": tm.Records(FIELDS, capacity=cap)}


def V(b, cap=CAP, fn=walk, **kw):
    return tm.vmap(fn, outputs=outs(cap), in_axes=AX, backend=b, **kw)


def R(cap=CAP, fn=walk, **kw):
    return tm.reference(fn, outputs=outs(cap), in_axes=AX, **kw)


def loss_of(f, wcut=0.0):
    def loss(th):
        o = f(IDS, KEYS, th[0], th[1], F(wcut)).outputs
        h = o["hits"]
        return (jnp.sum(h["w"] * jnp.cos(0.3 * h["t"]) * (1.0 + (h["sensor"] % 5).astype(F)))
                + 0.1 * jnp.sum(o["Q"]))
    return loss


TH = jnp.array([MU, VV], F)


def check_layout(res, ref, what=""):
    for f in INT:
        same(res.outputs["hits"][f], ref.outputs["hits"][f], f"{what} {f}")
    for f in FLT:
        close(res.outputs["hits"][f], ref.outputs["hits"][f], 1e-5, 1e-7, f"{what} {f}")
    if int(res.stats["appended"]["hits"]) != int(ref.stats["appended"]["hits"]):
        raise Fail(f"{what} appended {res.stats['appended']} vs {ref.stats['appended']}")


# ------------------------------------------------------------------------------------------------ cells
@case("forward == reference; canonical (item, k) layout")
def _(b):
    res = V(b)(IDS, KEYS, MU, VV, F(0.0))
    ref = R()(IDS, KEYS, MU, VV, F(0.0))
    check_layout(res, ref)
    close(res.outputs["Q"], ref.outputs["Q"], 1e-5, 1e-6, "Q")
    same(res.per_item[0], ref.per_item[0], "per-item k")
    h = res.outputs["hits"]
    tot = int(res.stats["appended"]["hits"])
    cnt = np.asarray(res.per_item[0])
    if tot != cnt.sum() or tot == 0 or tot > CAP:
        raise Fail(f"appended {tot} vs per-item sum {cnt.sum()}")
    item, k = np.asarray(h["item"])[:tot], np.asarray(h["k"])[:tot]
    want_i = np.repeat(np.arange(N), cnt)
    want_k = np.concatenate([np.arange(c) for c in cnt])
    if not (np.array_equal(item, want_i) and np.array_equal(k, want_k)):
        raise Fail("records are not in (item, k) order")
    for f in FIELDS:
        if np.any(np.asarray(h[f])[tot:] != 0):
            raise Fail(f"padding of {f} is not zero")
    if int(res.stats["records_mismatch"]["hits"]) or int(res.stats["records_dropped"]["hits"]) or int(
            res.stats["dropped_emits"]):
        raise Fail(f"stats {res.stats}")
    return f"{tot} records, max {cnt.max()} per item"


@case("bitwise: twice, jit, schedule / batch size (lane scheduling)")
def _(b):
    f = V(b)
    a = f(IDS, KEYS, MU, VV, F(0.0)).outputs["hits"]
    c = f(IDS, KEYS, MU, VV, F(0.0)).outputs["hits"]
    d = jax.jit(lambda *x: f(*x).outputs["hits"])(IDS, KEYS, MU, VV, F(0.0))
    alt = V(b, schedule="step") if b != "xla" else V(b, xla_batch_size=7)
    e = alt(IDS, KEYS, MU, VV, F(0.0)).outputs["hits"]
    for n in FIELDS:
        same(a[n], c[n], f"twice {n}")
        same(a[n], d[n], f"jit {n}")
        same(a[n], e[n], f"schedule {n}")


@case("derivatives BITWISE run to run and across schedule / batch size: grad, jacrev, hessian (records + Sum loss)")
def _(b):
    # a broadcast operand's cotangent is an exact sum (every backend, x64 or not), so the order in which items add
    # into it (atomics, lanes, batches) cannot change a bit; a float32 scatter-add there flaked on a GPU (xla, no x64)
    def run(f):
        wq = 1.0 + (jnp.arange(L * L) % 3).astype(F)
        jr = lambda th: (lambda o: jnp.stack([jnp.sum(o["hits"]["w"]), jnp.sum(o["Q"] * wq)]))(
            f(IDS, KEYS, th[0], th[1], F(0.0)).outputs)
        return [jax.grad(loss_of(f))(TH), jax.jacrev(jr)(TH), jax.hessian(loss_of(f))(TH)]
    ref = run(V(b))
    alt = V(b, schedule="step") if b != "xla" else V(b, xla_batch_size=7)
    for nm, f in (("rerun", V(b)), ("rerun", V(b)), ("schedule / batch size", alt)):
        for k, x, y in zip(("grad", "jacrev", "hessian"), run(f), ref):
            same(x, y, f"{nm} {k}")
    if not np.all(np.isfinite(np.asarray(ref[2]))):
        raise Fail("hessian not finite")
    return "grad, jacrev, hessian bitwise in 4 runs"


@case("only Records outputs (no Sum); zero records")
def _(b):
    def item(t, i, key, mu_a, v, wcut):
        t = t.append("r", {"a": i.astype(F) * mu_a, "b": i}, when=i % 3 == 1)
        return t.append("r", {"a": -mu_a, "b": -i}, when=i % 5 == 0)
    f = tm.vmap(item, outputs={"r": tm.Records({"a": F, "b": jnp.int32}, capacity=256)}, in_axes=AX, backend=b)
    res = f(IDS, KEYS, MU, VV, F(0.0))
    ref = tm.reference(item, outputs={"r": tm.Records({"a": F, "b": jnp.int32}, capacity=256)}, in_axes=AX)(
        IDS, KEYS, MU, VV, F(0.0))
    same(res.outputs["r"]["b"], ref.outputs["r"]["b"], "b")
    close(res.outputs["r"]["a"], ref.outputs["r"]["a"], what="a")
    want = [x for i in range(N) for x in ([i] if i % 3 == 1 else []) + ([-i] if i % 5 == 0 else [])]
    if list(np.asarray(res.outputs["r"]["b"])[:len(want)]) != want:
        raise Fail("order")
    z = f(IDS, KEYS, MU, VV, F(0.0), n_active=0)
    if int(z.stats["appended"]["r"]) != 0 or np.any(np.asarray(z.outputs["r"]["a"]) != 0):
        raise Fail("n_active=0 should append nothing")


@case("capacity: exact fits, one short poisons (NaN / -1 / stats), None")
def _(b):
    tot = int(V(b)(IDS, KEYS, MU, VV, F(0.0)).stats["appended"]["hits"])
    ok = V(b, cap=tot)(IDS, KEYS, MU, VV, F(0.0))
    if not np.all(np.isfinite(np.asarray(ok.outputs["hits"]["w"]))) or int(ok.stats["records_dropped"]["hits"]):
        raise Fail("capacity == total poisoned")
    check_layout(ok, R(cap=tot)(IDS, KEYS, MU, VV, F(0.0)), "exact")
    bad = V(b, cap=tot - 1)(IDS, KEYS, MU, VV, F(0.0))
    h = bad.outputs["hits"]
    if not (np.all(np.isnan(np.asarray(h["w"]))) and np.all(np.isnan(np.asarray(h["t"])))
            and np.all(np.asarray(h["sensor"]) == -1) and np.all(np.isnan(np.asarray(bad.outputs["Q"])))):
        raise Fail("overflow did not poison")
    if int(bad.stats["records_dropped"]["hits"]) != 1 or int(bad.stats["appended"]["hits"]) != tot:
        raise Fail(f"stats {bad.stats}")
    rb = R(cap=tot - 1)(IDS, KEYS, MU, VV, F(0.0))
    if not np.all(np.asarray(rb.outputs["hits"]["k"]) == -1):
        raise Fail("reference overflow")
    g = jax.grad(loss_of(V(b, cap=tot - 1)))(TH)
    if not np.all(np.isnan(np.asarray(g))):
        raise Fail(f"gradient of an overflowed call {g}")
    j = jax.jvp(loss_of(V(b, cap=tot - 1)), (TH,), (jnp.array([1.0, 0.0], F),))[1]
    if not np.isnan(float(j)):
        raise Fail("jvp of an overflowed call")
    n = V(b, cap=None)(IDS, KEYS, MU, VV, F(0.0))
    C = 1 << (tot - 1).bit_length()
    if n.outputs["hits"]["w"].shape != (C,):
        raise Fail(f"capacity None shape {n.outputs['hits']['w'].shape}, want {C}")
    check_layout(n, R(cap=None)(IDS, KEYS, MU, VV, F(0.0)), "None")
    raises(ValueError, lambda: jax.jit(lambda *x: V(b, cap=None)(*x).outputs)(IDS, KEYS, MU, VV, F(0.0)),
           "capacity")
    return f"total {tot}"


def _hook(name, c):
    """count of item 0 one short (its fill appends one more than counted), item 3 one long (one fewer)."""
    return c.at[0].add(-1).at[3].add(1)


@case("count-then-fill count mismatch: the fill guard k < count, and the poison (negative control)")
def _(b):
    ref = V(b)(IDS, KEYS, MU, VV, F(0.0))
    cnt = np.asarray(ref.per_item[0])
    if cnt[0] < 1:
        raise Fail("toy: item 0 has no record")
    with patched(_api, _COUNTS_HOOK=_hook, ONE_PASS=False):
        r = V(b)(IDS, KEYS, MU, VV, F(0.0))
        h = r.outputs["hits"]
        if int(r.stats["records_mismatch"]["hits"]) != 2:
            raise Fail(f"records_mismatch {r.stats['records_mismatch']}")
        if not (np.all(np.isnan(np.asarray(h["w"]))) and np.all(np.asarray(h["item"]) == -1)
                and np.all(np.isnan(np.asarray(r.outputs["Q"])))):
            raise Fail("mismatch did not poison")
        j = jax.jvp(loss_of(V(b)), (TH,), (jnp.array([1.0, 1.0], F),))[1]
        g = jax.grad(loss_of(V(b)))(TH)
        if not (np.isnan(float(j)) and np.all(np.isnan(np.asarray(g)))):
            raise Fail(f"derivatives of a mismatched call: jvp {j} grad {g}")
        # the guard itself, poison switched off: item 0's extra record must NOT land in item 1's first slot
        ident = lambda o, bad: o
        with patched(_api, _poison=ident, _poison_primal=ident):
            raw = V(b)(IDS, KEYS, MU, VV, F(0.0)).outputs["hits"]
    # the layout the guard must give: each item's own records in slots [off'_i, off'_i + count'_i) of the hooked
    # counts: item 0 loses its last record (not written past its count), item 3 keeps an empty (zero) last slot
    off = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    want = {f: [] for f in FIELDS}
    for i in range(N):
        c2 = cnt[i] - (i == 0) + (i == 3)
        for f in FIELDS:
            mine = np.asarray(ref.outputs["hits"][f])[off[i]:off[i] + cnt[i]]
            want[f].append(np.concatenate([mine, np.zeros(1, mine.dtype)])[:c2] if i == 3 else mine[:c2])
    for f in INT:                                        # (the integer poison is not switched off)
        if not np.all(np.asarray(raw[f]) == -1):
            raise Fail(f"int field {f} not poisoned")
    for f in FLT:
        w = np.concatenate(want[f])
        got = np.asarray(raw[f])[:len(w)]
        if not np.array_equal(got, w):
            bad = np.nonzero(got != w)[0][:5]
            raise Fail(f"guard: field {f} differs at slots {bad}")
    return f"item 0 count {cnt[0]}"


@case("jvp / grad / jacfwd / jacrev vs central differences (records gated by control flow)")
def _(b):
    loss = loss_of(V(b))
    v, g = jax.value_and_grad(loss)(TH)
    e = F(1e-2) if F == jnp.float32 else F(1e-5)
    fd = np.array([(loss(TH + e * u) - loss(TH - e * u)) / (2 * e) for u in jnp.eye(2, dtype=F)])
    for u in jnp.eye(2, dtype=F):     # existence must not change under the step (FD would be meaningless)
        a, c = V(b)(IDS, KEYS, TH[0] + e * u[0], TH[1] + e * u[1], F(0.0)), V(b)(IDS, KEYS, MU, VV, F(0.0))
        same(a.outputs["hits"]["sensor"], c.outputs["hits"]["sensor"], "existence under the FD step")
    close(g, fd, 2e-3, 1e-5, "grad vs FD")
    jv = np.array([jax.jvp(loss, (TH,), (u,))[1] for u in jnp.eye(2, dtype=F)])
    close(jv, fd, 2e-3, 1e-5, "jvp vs FD")
    close(jv, g, 1e-4, 1e-6, "jvp vs grad")
    close(jax.jacfwd(loss)(TH), g, 1e-4, 1e-6, "jacfwd")
    close(jax.jacrev(loss)(TH), g, 1e-4, 1e-6, "jacrev")
    close(g, jax.jacfwd(loss_of(R()))(TH), 1e-4, 1e-6, "grad vs reference jacfwd")
    return f"grad {np.asarray(g)} fd {fd}"


@case("record-field derivatives per slot (jacfwd of the fields) == reference")
def _(b):
    f = lambda th: V(b)(IDS, KEYS, th[0], th[1], F(0.0)).outputs["hits"]
    r = lambda th: R()(IDS, KEYS, th[0], th[1], F(0.0)).outputs["hits"]
    jt, jr = jax.jacfwd(f)(TH), jax.jacfwd(r)(TH)
    for n in FLT:
        close(jt[n], jr[n], 1e-4, 1e-6, f"d{n}")
    # vjp with a random cotangent on every slot == the jvp transpose
    ct = {n: (jax.random.normal(jax.random.PRNGKey(1), (CAP,), F) if n in FLT else None) for n in FIELDS}
    _, pull = jax.vjp(lambda th: {n: f(th)[n] for n in FLT}, TH)
    g = pull({n: ct[n] for n in FLT})[0]
    want = sum(jnp.tensordot(ct[n], jt[n], 1) for n in FLT)
    close(g, want, 1e-4, 1e-6, "vjp of fields")


@case("parameter-dependent cut (w >= wcut): jvp == reference, grad == jvp")
def _(b):
    loss = loss_of(V(b), wcut=0.3)
    ref = loss_of(R(), wcut=0.3)
    jv = np.array([jax.jvp(loss, (TH,), (u,))[1] for u in jnp.eye(2, dtype=F)])
    jr = np.array([jax.jvp(ref, (TH,), (u,))[1] for u in jnp.eye(2, dtype=F)])
    close(jv, jr, 1e-4, 1e-6, "jvp vs reference")
    close(jax.grad(loss)(TH), jv, 1e-4, 1e-6, "grad vs jvp")
    a = V(b)(IDS, KEYS, MU, VV, F(0.3))
    check_layout(a, R()(IDS, KEYS, MU, VV, F(0.3)), "cut")
    return f"{int(a.stats['appended']['hits'])} records"


@case("jax.checkpoint, replay adjoint (rev_checkpoints=2), single-loop adjoint")
def _(b):
    from tallymap import _reverse
    loss = loss_of(V(b))
    v0, g0 = jax.value_and_grad(loss)(TH)
    v1, g1 = jax.value_and_grad(jax.checkpoint(loss))(TH)
    v2, g2 = jax.jit(jax.value_and_grad(jax.checkpoint(loss)))(TH)
    v3, g3 = jax.value_and_grad(loss_of(V(b, rev_checkpoints=2)))(TH)
    with patched(_reverse, SINGLE="force"):
        v4, g4 = jax.value_and_grad(loss_of(V(b)))(TH)
    with patched(_reverse, SINGLE=False):
        v5, g5 = jax.value_and_grad(loss_of(V(b)))(TH)
    for nm, vv, gg in (("checkpoint", v1, g1), ("jit checkpoint", v2, g2), ("rev_checkpoints=2", v3, g3),
                       ("single-loop", v4, g4), ("no single-loop", v5, g5)):
        close(vv, v0, 1e-5, 1e-6, nm + " value")
        close(gg, g0, 1e-4, 1e-6, nm + " grad")


@case("jit, n_active, jax.vmap of the call, max_steps")
def _(b):
    f = V(b)
    full = f(IDS, KEYS, MU, VV, F(0.0))
    half = jax.jit(lambda n: f(IDS, KEYS, MU, VV, F(0.0), n_active=n).outputs["hits"])(jnp.int32(N // 2))
    cnt = np.asarray(full.per_item[0])
    m = int(cnt[:N // 2].sum())
    for n in INT:
        same(np.asarray(half[n])[:m], np.asarray(full.outputs["hits"][n])[:m], f"n_active {n}")
        if np.any(np.asarray(half[n])[m:] != 0):
            raise Fail("n_active: records past the active items")
    mus = jnp.array([1.0, 2.0, 3.0], F)
    vm = jax.vmap(lambda mu: f(IDS, KEYS, mu, VV, F(0.0)).outputs["hits"]["w"])(mus)
    for j in range(3):
        close(vm[j], f(IDS, KEYS, mus[j], VV, F(0.0)).outputs["hits"]["w"], 1e-6, 0.0, f"vmap member {j}")
    gv = jax.vmap(jax.grad(loss_of(f)))(jnp.stack([TH, TH * 1.1]))
    close(gv[0], jax.grad(loss_of(f))(TH), 1e-4, 1e-6, "vmap(grad)")
    capped = V(b, max_steps=5)(IDS, KEYS, MU, VV, F(0.0))
    if not (int(capped.stats["n_truncated"]) > 0 and np.all(np.isnan(np.asarray(capped.outputs["hits"]["w"])))
            and np.all(np.asarray(capped.outputs["hits"]["k"]) == -1)):
        raise Fail("max_steps truncation did not poison the records")
    big = V(b, max_steps=1000)(IDS, KEYS, MU, VV, F(0.0))
    check_layout(big, full, "max_steps not reached")
    gb = jax.grad(loss_of(V(b, max_steps=1000)))(TH)
    close(gb, jax.grad(loss_of(f))(TH), 1e-4, 1e-6, "grad with max_steps")


@case("refusals and misuse: append inside jax.vmap / a vmapped loop, fields", ["xla"])
def _(b):
    def laned_loop(t, i, key, mu_a, v, wcut):
        def one(x, t):
            def body(c):
                j, t = c
                return j + 1, t.append("hits", {"sensor": j, "t": x, "w": x, "item": i, "k": j}, when=x > 0.5)
            return lax.while_loop(lambda c: c[0] < (x * 4).astype(jnp.int32), body, (jnp.int32(0), t))[1]
        return jax.vmap(one, in_axes=(0, None))(jnp.array([0.3, 0.9], F), t)
    raises(TypeError, lambda: V(b, fn=laned_loop)(IDS, KEYS, MU, VV, F(0.0)), "jax.vmap")

    def laned(t, i, key, mu_a, v, wcut):
        return jax.vmap(lambda x, t: t.append("hits", {"sensor": i, "t": x, "w": x, "item": i, "k": i}),
                        in_axes=(0, None))(jnp.array([0.3, 0.9], F), t)
    raises(TypeError, lambda: V(b, fn=laned)(IDS, KEYS, MU, VV, F(0.0)), "jax.vmap")

    def wrong(t, i, key, mu_a, v, wcut):
        return t.append("hits", {"sensor": i})
    raises(TypeError, lambda: V(b, fn=wrong)(IDS, KEYS, MU, VV, F(0.0)), "declared fields")

    def floatint(t, i, key, mu_a, v, wcut):
        return t.append("hits", {"sensor": mu_a, "t": mu_a, "w": mu_a, "item": i, "k": i})
    raises(TypeError, lambda: V(b, fn=floatint)(IDS, KEYS, MU, VV, F(0.0)), "integer field")

    def add_rec(t, i, key, mu_a, v, wcut):
        return t.add("hits", 0, 1.0)
    raises(KeyError, lambda: V(b, fn=add_rec)(IDS, KEYS, MU, VV, F(0.0)), "t.append")

    def app_sum(t, i, key, mu_a, v, wcut):
        return t.append("Q", {"a": 1.0})
    raises(KeyError, lambda: V(b, fn=app_sum)(IDS, KEYS, MU, VV, F(0.0)), "t.add")
    raises(TypeError, lambda: tm.Records({"b": bool}), "integer")
    raises(ValueError, lambda: tm.Records({"a": F}, capacity=0), "capacity")
    if tm.Records(16, {"a": F}) != tm.Records({"a": F}, capacity=16):
        raise Fail("Records(capacity, fields) != Records(fields, capacity=)")


# ------------------------------------------------------------------------------------------------ one pass
# The default implementation is one pass (_records.py); _api.ONE_PASS = False runs the same calls as count-then-fill
# (step 1).  The two must agree BITWISE: the layout, the values, the stats, and every derivative.  (The toy's Sum Q is
# an ordinary float sum, whose atomics add in a run-dependent order: it is compared to a tolerance, and the losses
# here read the records only.)
def _lossr(f):
    return lambda th: (lambda h: jnp.sum(h["w"] * jnp.cos(0.3 * h["t"]) * (1.0 + (h["sensor"] % 5).astype(F))))(
        f(IDS, KEYS, th[0], th[1], F(0.0)).outputs["hits"])


def _fl(f):
    return lambda th: (lambda h: (h["t"], h["w"]))(f(IDS, KEYS, th[0], th[1], F(0.0)).outputs["hits"])


def _red2(f):
    return lambda th: (lambda h: jnp.stack([jnp.sum(h["w"]), jnp.sum(h["t"] * h["w"])]))(
        f(IDS, KEYS, th[0], th[1], F(0.0)).outputs["hits"])


def _everything(b, **kw):
    f = V(b, **kw)
    r = f(IDS, KEYS, MU, VV, F(0.0))
    o = {f"fwd {n}": r.outputs["hits"][n] for n in FIELDS}
    o.update({"per-item k": r.per_item[0], "per-item w": r.per_item[1],
              "appended": r.stats["appended"]["hits"], "dropped": r.stats["records_dropped"]["hits"]})
    p, t = jax.jvp(_fl(f), (TH,), (jnp.array([1.0, 0.5], F),))
    o.update({"jvp primal t": p[0], "jvp primal w": p[1], "jvp tangent t": t[0], "jvp tangent w": t[1]})
    v, g = jax.value_and_grad(_lossr(f))(TH)
    o.update({"value": v, "grad": g, "grad (checkpoint)": jax.grad(jax.checkpoint(_lossr(f)))(TH)})
    jf = jax.jacfwd(_fl(f))(TH)
    o.update({"jacfwd t": jf[0], "jacfwd w": jf[1], "jacrev": jax.jacrev(_red2(f))(TH)})
    o["grad (rev_checkpoints=2)"] = jax.grad(_lossr(V(b, rev_checkpoints=2, **kw)))(TH)
    o["hessian"] = jax.hessian(_lossr(f))(TH)
    return o, r.outputs["Q"]


@case("one pass == count-then-fill (step 1) BITWISE: layout, stats, jvp, grad, jacfwd, jacrev, checkpoint, hessian")
def _(b):
    one, q1 = _everything(b)
    with patched(_api, ONE_PASS=False):
        ctf, q2 = _everything(b)
    for k in one:
        same(one[k], ctf[k], k)
    close(q1, q2, 1e-6, 0.0, "Q")
    if not np.all(np.isfinite(np.asarray(one["grad"]))):
        raise Fail("grad not finite")
    return f"{len(one)} outputs bitwise"


@case("one pass == count-then-fill BITWISE with max_steps, n_active and an overflowed call")
def _(b):
    def run():
        r1 = V(b, max_steps=1000)(IDS, KEYS, MU, VV, F(0.0))
        r2 = jax.jit(lambda n: V(b)(IDS, KEYS, MU, VV, F(0.0), n_active=n))(jnp.int32(N // 3))
        r3 = V(b, cap=100)(IDS, KEYS, MU, VV, F(0.0))
        g3 = jax.grad(_lossr(V(b, cap=100)))(TH)
        return [x for r in (r1, r2, r3) for x in jax.tree_util.tree_leaves((r.outputs["hits"], r.stats, r.truncated,
                                                                           r.per_item))] + [g3]
    one = run()
    with patched(_api, ONE_PASS=False):
        ctf = run()
    for j, (a, c) in enumerate(zip(one, ctf)):
        same(a, c, f"leaf {j}")


@case("the count report: stats['appended'] of an overflowed call sizes the rerun (next power of two)")
def _(b):
    full = V(b)(IDS, KEYS, MU, VV, F(0.0))
    tot = int(full.stats["appended"]["hits"])
    small = V(b, cap=64)(IDS, KEYS, MU, VV, F(0.0))
    a = int(small.stats["appended"]["hits"])
    if a != tot or int(small.stats["records_dropped"]["hits"]) != tot - 64:
        raise Fail(f"appended {a} (want {tot}), dropped {small.stats['records_dropped']}")
    h = small.outputs["hits"]
    if not (np.all(np.isnan(np.asarray(h["w"]))) and np.all(np.asarray(h["sensor"]) == -1)):
        raise Fail("overflow not poisoned")
    again = V(b, cap=64)(IDS, KEYS, MU, VV, F(0.0))        # overflow is deterministic: the poisoned output and stats
    for n in FIELDS:
        same(again.outputs["hits"][n], h[n], f"overflowed twice {n}")
    same(again.stats["appended"]["hits"], small.stats["appended"]["hits"], "appended twice")
    C = 1 << (a - 1).bit_length()
    rr = V(b, cap=C)(IDS, KEYS, MU, VV, F(0.0))
    for n in FIELDS:
        same(np.asarray(rr.outputs["hits"][n])[:tot], np.asarray(full.outputs["hits"][n])[:tot], f"rerun {n}")
    if int(rr.stats["records_dropped"]["hits"]):
        raise Fail("rerun dropped")
    j = jax.jit(lambda: V(b, cap=64)(IDS, KEYS, MU, VV, F(0.0)).stats["appended"]["hits"])()
    if int(j) != tot:
        raise Fail(f"jit appended {int(j)}")
    return f"appended {a} at capacity 64 -> rerun at {C}"


@case("counts past 2^31: the exact two-word totals, the overflow decision, the clipped offsets and the reported "
      "(int64 with x64, saturated int32 without) counts at and around 2^31 - 1 and past 2^32", ["xla"])
def _(b):
    """_limits (the one rule every Records path uses: one pass, count-then-fill, canon, its fill twin's tables,
    sharded canon; and the spawn Queue's record rule) on per-item counts that sum near the int32 limits: a naive
    int32 sum wraps, and a two-word sum that drops a carry fails the 2^32 + 1 case."""
    from tallymap import _limits as L
    M = (1 << 31) - 1
    x64 = jax.config.jax_enable_x64
    rep = lambda v: v if x64 else min(v, M)                                               # noqa: E731
    cases = [([M - 1, 1], M), ([M], M), ([M, 1], 1 << 31), ([M, M, 3], (1 << 32) + 1), ([0, M, M, M, 0, 7], 3 * M + 7)]
    for cnt, want in cases:
        c = jnp.asarray(np.asarray(cnt, np.int64).astype(np.uint32).view(np.int32))
        n = L.total(c)
        got = int(np.asarray(n.hi, np.uint64)) << 32 | int(np.asarray(n.lo, np.uint64))
        if got != want:
            raise Fail(f"total {cnt}: {got} != {want}")
        for cap in (16, M - 1, M):
            ovf, app, drop = _api._record_limits({"r": c}, {"r": cap})
            if bool(ovf) != (want > cap) or int(app["r"]) != rep(want) or int(drop["r"]) != rep(max(want - cap, 0)):
                raise Fail(f"{cnt} capacity {cap}: overflow {bool(ovf)}, appended {int(app['r'])} (want "
                           f"{rep(want)}), dropped {int(drop['r'])} (want {rep(max(want - cap, 0))})")
            if (app["r"].dtype, drop["r"].dtype) != ((np.int64,) * 2 if x64 else (np.int32,) * 2):
                raise Fail(f"stats dtypes {app['r'].dtype} {drop['r'].dtype} (x64 {x64})")
            pre = np.concatenate([[0], np.cumsum(np.asarray(cnt, np.int64))[:-1]])
            if not np.array_equal(np.asarray(L.offsets(c, cap)), np.minimum(pre, cap)):
                raise Fail(f"offsets {cnt} cap {cap}: {np.asarray(L.offsets(c, cap))} != {np.minimum(pre, cap)}")
    big = jnp.full((1 << 20,), 1 << 12, jnp.int32)                     # 2^32 exactly: lo word 0, hi word 1
    ovf, app, _ = _api._record_limits({"r": big}, {"r": 1 << 20})
    if not bool(ovf) or int(app["r"]) != rep(1 << 32):
        raise Fail(f"2^20 x 2^12: overflow {bool(ovf)} appended {int(app['r'])}")
    st = jnp.zeros((len(_api.STATS),), jnp.int32).at[_api.ST["spawned"]].set(jnp.int32(-5))   # a wrapped counter
    if not bool(L.queue_lost(st, 1 << 20, record=True)) or bool(L.queue_lost(st, 1 << 20, record=False)):
        raise Fail("a spawned counter past 2^31 (wrapped int32) did not exceed the Queue for a record path")
    return f"{len(cases)} sums x 3 capacities, 2^32, a wrapped spawn counter (x64 {x64})"


@case("a real call with 2^32 + 2^15 appends into capacity 2^16 poisons (an int32 total wraps to 2^15) and reports "
      "the count saturated (int64 with x64); count-then-fill too", ["auto"])
def _(b):
    if jax.default_backend() != "gpu":
        return "skipped: GPU only (2^32 appends)"
    n_it, K, C = (1 << 17) + 1, 1 << 15, 1 << 16

    def item(t, x):
        t = lax.fori_loop(0, K, lambda k, t: t.append("rec", {"v": x + k.astype(jnp.float32)}, when=True), t)
        return t.add("S", 0, x)
    want = n_it * K if jax.config.jax_enable_x64 else (1 << 31) - 1
    msg = []
    outs = {"S": tm.Sum((1,), jnp.float32), "rec": tm.Records({"v": jnp.float32}, capacity=C)}
    for one in (True, False):
        with patched(_api, ONE_PASS=one):
            r = tm.vmap(item, outputs=outs, backend=b)(jnp.ones((n_it,), jnp.float32))
            v = np.asarray(r.outputs["rec"]["v"])
            app, drop = int(r.stats["appended"]["rec"]), int(r.stats["records_dropped"]["rec"])
        if not (np.all(np.isnan(v)) and np.all(np.isnan(np.asarray(r.outputs["S"])))):
            raise Fail(f"{'one pass' if one else 'count-then-fill'}: not poisoned (appended {app}, dropped {drop})")
        if app != want or drop != (want - C if jax.config.jax_enable_x64 else want):
            raise Fail(f"{'one pass' if one else 'count-then-fill'}: appended {app} (want {want}), dropped {drop}")
        msg.append(f"{'one' if one else 'ctf'}: appended {app} dropped {drop}")
    return "; ".join(msg)


def burst(t, i, key, mu_a, v, wcut):
    """every item appends 1 + i % 7 records back to back: all lanes contend for the slot counter at once."""
    def body(c):
        j, t = c
        return j + 1, t.append("r", {"a": i.astype(F) * mu_a + j.astype(F), "i": i, "j": j})
    return lax.while_loop(lambda c: c[0] < 1 + i % 7, body, (jnp.int32(0), t))[1]


@case("contention: every item appends at once; capacity exact / one short; schedule='step'")
def _(b):
    nb = np.arange(N) % 7 + 1
    tot = int(nb.sum())
    want_i = np.repeat(np.arange(N), nb)
    want_j = np.concatenate([np.arange(c) for c in nb])
    RFs = {"a": F, "i": jnp.int32, "j": jnp.int32}
    mk = lambda cap, **kw: tm.vmap(burst, outputs={"r": tm.Records(RFs, capacity=cap)}, in_axes=AX, backend=b, **kw)
    for kw in ({}, {"schedule": "step"} if b != "xla" else {"xla_batch_size": 33}):
        r = mk(tot, **kw)(IDS, KEYS, MU, VV, F(0.0)).outputs["r"]
        if not (np.array_equal(np.asarray(r["i"]), want_i) and np.array_equal(np.asarray(r["j"]), want_j)):
            raise Fail(f"layout {kw}")
        close(r["a"], want_i * float(MU) + want_j, 1e-6, 0.0, f"values {kw}")
        s = mk(tot - 1, **kw)(IDS, KEYS, MU, VV, F(0.0))
        if not (np.all(np.isnan(np.asarray(s.outputs["r"]["a"]))) and np.all(np.asarray(s.outputs["r"]["i"]) == -1)
                and int(s.stats["records_dropped"]["r"]) == 1):
            raise Fail(f"one short {kw}")
    return f"{tot} records"


@case("one pass: unfused jvp (FUSE_JVP=False) == fused; the twin's count-mismatch poison (negative control)")
def _(b):
    from tallymap import _primitive as _P
    f = V(b)
    fused = jax.jvp(_fl(f), (TH,), (jnp.array([1.0, 0.5], F),))
    with patched(_P, FUSE_JVP=False):
        unf = jax.jvp(_fl(f), (TH,), (jnp.array([1.0, 0.5], F),))
    for a, c, n in zip(jax.tree_util.tree_leaves(fused), jax.tree_util.tree_leaves(unf), ("pt", "pw", "tt", "tw")):
        same(a, c, f"unfused {n}")
    base = f(IDS, KEYS, MU, VV, F(0.0)).outputs["hits"]
    g0 = jax.grad(loss_of(f))(TH)
    with patched(_api, _COUNTS_HOOK=_hook):
        r = f(IDS, KEYS, MU, VV, F(0.0))
        for n in FIELDS:              # (one pass: the forward hands no counts over, so nothing changes)
            same(r.outputs["hits"][n], base[n], f"hooked forward {n}")
        j = jax.jvp(_fl(f), (TH,), (jnp.array([1.0, 0.5], F),))     # (fused: one launch, no record either)
        same(j[1][1], fused[1][1], "hooked fused jvp")
        g = jax.grad(loss_of(f))(TH)          # the twin replays with the hooked counts: mismatch -> NaN
        with patched(_P, FUSE_JVP=False):
            ju = jax.jvp(_fl(f), (TH,), (jnp.array([1.0, 0.5], F),))[1][1]
    if not np.all(np.isnan(np.asarray(g))):
        raise Fail(f"grad with mismatched counts {g} (unhooked {g0})")
    if not np.all(np.isnan(np.asarray(ju))):
        raise Fail("unfused jvp with mismatched counts not NaN")


@case("subnormal record fields: one pass == count-then-fill == the emitted values, primal / tangent / cotangent")
def _(b):
    if b == "xla" and jax.devices()[0].platform == "cpu":
        return "skip: XLA:CPU flushes subnormals in its own arithmetic (docs/float_semantics.md item 6)"
    vals = jnp.array([1.4e-44, 2.0 ** -140, -(2.0 ** -130), 2.0 ** -126, 1.0, 0.0, 3.0 * 2.0 ** -149, 2.0 ** -100],
                     jnp.float32)

    def run():
        f = tm.vmap(lambda t, x, s: t.append("r", {"q": x * s}), outputs={"r": tm.Records({"q": jnp.float32},
                    capacity=16)}, in_axes=(0, None), backend=b)
        q = f(vals, jnp.float32(1.0)).outputs["r"]["q"]
        tg = jax.jvp(lambda s: f(vals, s).outputs["r"]["q"], (jnp.float32(1.0),), (jnp.float32(1.0),))[1]
        ct = jax.grad(lambda x: jnp.sum(f(x, jnp.float32(1.0)).outputs["r"]["q"] * jnp.arange(16.0)))(vals)
        return q, tg, ct
    one = run()
    with patched(_api, ONE_PASS=False):
        ctf = run()
    for a, c, n in zip(one, ctf, ("primal", "tangent", "cotangent")):
        same(a, c, n)
    same(np.asarray(one[0])[:8], np.asarray(vals), "primal == the emitted values")
    same(np.asarray(one[1])[:8], np.asarray(vals), "tangent == the emitted values")




# ------------------------------------------------------------------------------------------------ spawn
# A deterministic tree: node (root, gen, code, x); h = hash(code) decides its records (h % 3 of them, j = 0, 1) and
# its children (gen < 3: (h >> 4) % 3 of them, ordinal o -> code * 3 + o + 1, x * th).  The canonical order is the
# tree's breadth-first order: roots in item order, then each generation by (parent, spawn ordinal); numpy rebuilds it.
NR, QCAP, SCAP = 16, 512, 1024
SFIELDS = {"root": jnp.int32, "gen": jnp.int32, "code": jnp.int32, "j": jnp.int32, "val": F}


def _h(code):
    return (code.astype(jnp.uint32) * jnp.uint32(2654435761)) >> 7


def sitem(t, root, gen, code, x, th):
    h = _h(code)
    nrec = (h % 3).astype(jnp.int32)
    nch = jnp.where(gen < 3, ((h >> 4) % 3).astype(jnp.int32), 0)
    for j in range(2):
        t = t.append("rec", {"root": root, "gen": gen, "code": code, "j": jnp.int32(j),
                             "val": x * jnp.cos(0.5 * j) * th}, when=j < nrec)
    for o in range(2):
        t = t.spawn((root, gen + 1, code * 3 + o + 1, x * th), when=o < nch)
    return t.add("S", root % 4, x)


ROOTS = (jnp.arange(NR, dtype=jnp.int32), jnp.zeros(NR, jnp.int32), jnp.arange(NR, dtype=jnp.int32) + 1000,
         jnp.linspace(0.5, 1.5, NR).astype(F))
STH = F(0.9)


def SV(b, q=QCAP, cap=SCAP, **kw):
    return tm.vmap(sitem, outputs={"S": tm.Sum((4,), F), "rec": tm.Records(SFIELDS, capacity=cap)},
                   in_axes=(0, 0, 0, 0, None), spawn=tm.Queue(q), backend=b, **kw)


def bfs_records(th=0.9):
    h = lambda c: (np.uint32(c) * np.uint32(2654435761)) >> np.uint32(7)
    queue = [(int(r), 0, 1000 + int(r), float(x)) for r, x in zip(np.asarray(ROOTS[0]), np.asarray(ROOTS[3]))]
    recs, i = [], 0
    while i < len(queue):
        root, gen, code, x = queue[i]
        i += 1
        hc = int(h(code))
        for j in range(hc % 3):
            recs.append((root, gen, code, j, x * np.cos(0.5 * j) * th))
        nch = (hc >> 4) % 3 if gen < 3 else 0
        for o in range(nch):
            queue.append((root, gen + 1, code * 3 + o + 1, x * th))
    return recs, len(queue) - NR


@case("spawn: canonical breadth-first tree order, bitwise determinism")
def _(b):
    want, nchild = bfs_records()
    res = SV(b)(*ROOTS, STH)
    h = res.outputs["rec"]
    n = int(res.stats["appended"]["rec"])
    if n != len(want) or int(res.stats["spawned"]) != nchild:
        raise Fail(f"appended {n} vs {len(want)}, spawned {int(res.stats['spawned'])} vs {nchild}")
    got = list(zip(*[np.asarray(h[f])[:n].tolist() for f in ("root", "gen", "code", "j")]))
    if got != [w[:4] for w in want]:
        bad = next(k for k, (g, w) in enumerate(zip(got, want)) if g != w[:4])
        raise Fail(f"order differs at record {bad}: {got[bad]} vs {want[bad][:4]}")
    close(np.asarray(h["val"])[:n], np.array([w[4] for w in want]), 1e-5, 1e-6, "val")
    again = SV(b)(*ROOTS, STH).outputs["rec"]
    for f in SFIELDS:
        same(h[f], again[f], f"twice {f}")
    if int(res.stats["records_mismatch"]["rec"]) or int(res.stats["children_dropped"]):
        raise Fail(f"stats {res.stats}")
    return f"{n} records from {NR} roots + {nchild} children"


@case("spawn: xla backend == kernel backend bitwise (integer fields)", ["auto"])
def _(b):
    a, x = SV("auto")(*ROOTS, STH).outputs["rec"], SV("xla")(*ROOTS, STH).outputs["rec"]
    for f in ("root", "gen", "code", "j"):
        same(a[f], x[f], f)
    close(a["val"], x["val"], 1e-6, 0.0, "val")


@case("spawn: jvp / grad (tree adjoint) vs central differences")
def _(b):
    def loss(th):
        h = SV(b)(*ROOTS, th).outputs
        return jnp.sum(h["rec"]["val"] * (1.0 + (h["rec"]["code"] % 5).astype(F)))
    g = jax.grad(loss)(STH)
    j = jax.jvp(loss, (STH,), (F(1.0),))[1]
    e = F(1e-3) if F == jnp.float32 else F(1e-6)
    fd = (loss(STH + e) - loss(STH - e)) / (2 * e)
    close(g, fd, 2e-3, 1e-5, "grad vs FD")
    close(j, g, 1e-4, 1e-6, "jvp vs grad")
    wantg = sum(w[4] * (w[1] + 1) / 0.9 * (1 + w[2] % 5) for w in bfs_records()[0])   # d/dth of x0 th^(gen+1) ...
    close(g, wantg, 1e-4, 1e-5, "grad vs closed form")
    return f"grad {float(g):.6g} fd {float(fd):.6g}"


@case("spawn: Queue too small / capacity short / count mismatch poison")
def _(b):
    want, nchild = bfs_records()
    small = SV(b, q=nchild - 1)(*ROOTS, STH)
    if not (np.all(np.isnan(np.asarray(small.outputs["rec"]["val"])))
            and np.all(np.asarray(small.outputs["rec"]["root"]) == -1)):
        raise Fail("a dropped child did not poison")
    short = SV(b, cap=len(want) - 1)(*ROOTS, STH)
    if not (np.all(np.isnan(np.asarray(short.outputs["S"]))) and int(short.stats["records_dropped"]["rec"]) == 1):
        raise Fail("capacity overflow did not poison")
    with patched(_api, _COUNTS_HOOK=lambda r, c: c.at[NR + 2].add(1)):     # a child's count one long
        mm = SV(b)(*ROOTS, STH)
        g = jax.grad(lambda th: jnp.sum(SV(b)(*ROOTS, th).outputs["rec"]["val"]))(STH)
    if int(mm.stats["records_mismatch"]["rec"]) != 1 or not np.all(np.isnan(np.asarray(mm.outputs["rec"]["val"]))):
        raise Fail(f"mismatch: {mm.stats['records_mismatch']}")
    if not np.isnan(float(g)):
        raise Fail(f"gradient of a mismatched spawning call {g}")


CN, CDEPTH, CQ = 3, 9, 4         # 3 chains of 9 children each: 27 children in all, at most 3 pending in a Queue of 4


def chain(t, root, gen, x, th):
    t = t.append("rec", {"root": root, "gen": gen, "val": x}, when=True)
    t = t.spawn((root, gen + 1, x * th), when=gen < CDEPTH)
    return t.add("S", root % 4, x)


def chain_plain(t, root, gen, x, th):
    return t.spawn((root, gen + 1, x * th), when=gen < CDEPTH).add("S", root % 4, x)


@case("spawn: Records with more children in all than the Queue holds (none dropped: each item spawns at most one) "
      "poison; the same Queue without Records is exact")
def _(b):
    """Records + spawn reads the spawn tree from the count pass's queue ring (_tree_order), which holds every child
    only when the call has at most Q children in all; a ring that reused a slot has lost a node.  Without Records the
    forward Queue bounds only the PENDING children (here at most CN), so the plain call is exact."""
    roots = (jnp.arange(CN, dtype=jnp.int32), jnp.zeros(CN, jnp.int32), jnp.linspace(1.0, 2.0, CN).astype(F))
    th = F(0.5)
    fl = {"root": jnp.int32, "gen": jnp.int32, "val": F}
    rv = tm.vmap(chain, outputs={"S": tm.Sum((4,), F), "rec": tm.Records(fl, capacity=64)},
                 in_axes=(0, 0, 0, None), spawn=tm.Queue(CQ), backend=b)
    res = rv(*roots, th)
    x0 = np.asarray(roots[2], np.float64)
    want = [(r, g, x0[r] * 0.5 ** g) for g in range(CDEPTH + 1) for r in range(CN)]    # breadth-first order
    plain = tm.vmap(chain_plain, outputs={"S": tm.Sum((4,), F)}, in_axes=(0, 0, 0, None), spawn=tm.Queue(CQ),
                    backend=b)(*roots, th)
    sw = np.zeros(4)
    for r, g, v in want:
        sw[r % 4] += v
    close(plain.outputs["S"], sw, 1e-5, 1e-6, "plain call (no Records) with total children > Q")
    if int(plain.stats["children_dropped"]) != 0:
        raise Fail(f"the toy dropped children: {int(plain.stats['children_dropped'])} (its pending count must fit)")
    h = res.outputs["rec"]
    val, root, gen = np.asarray(h["val"]), np.asarray(h["root"]), np.asarray(h["gen"])
    nsp, ndr, app = int(res.stats["spawned"]), int(res.stats["children_dropped"]), int(res.stats["appended"]["rec"])
    poisoned = (np.all(np.isnan(val)) and np.all(root == -1) and np.all(gen == -1)
                and np.all(np.isnan(np.asarray(res.outputs["S"]))))
    if not poisoned:
        n = len(want)
        ok = (app == n and [(int(a), int(c)) for a, c in zip(root[:n], gen[:n])] == [w[:2] for w in want]
              and np.allclose(val[:n], [w[2] for w in want], rtol=1e-5) and np.all(val[n:] == 0))
        if not ok:
            bad = [k for k in range(64) if (k < n and ((int(root[k]), int(gen[k])) != want[k][:2]
                                                       or not np.isclose(val[k], want[k][2], rtol=1e-5)))
                   or (k >= n and val[k] != 0)]
            raise Fail(f"silently wrong, not poisoned: spawned {nsp} > Q {CQ}, dropped {ndr}, appended {app} (want "
                       f"{n}); {len(bad)} wrong slots, first {bad[:4]}: (root, gen, val) "
                       f"{[(int(root[k]), int(gen[k]), float(val[k])) for k in bad[:4]]}")
        return f"exact although spawned {nsp} > Q {CQ}"
    g = jax.grad(lambda th: jnp.sum(rv(*roots, th).outputs["rec"]["val"]))(th)
    if not np.isnan(float(g)):
        raise Fail(f"the gradient of a call whose ring reused slots is {float(g)}, not NaN")
    return f"poisoned (spawned {nsp} > Q {CQ}, dropped {ndr}); gradient NaN; plain call exact"


@case("spawn + Records (count-then-fill) and one-pass Records with jax_enable_x64 toggled on in-process: the same "
      "records, sums, stats and gradient bit for bit as with it off (the tree order's while carry stays int32)")
def _(b):
    f32 = {"root": jnp.int32, "gen": jnp.int32, "code": jnp.int32, "j": jnp.int32, "val": jnp.float32}
    roots = [np.asarray(r).astype(np.int32 if i < 3 else np.float32) for i, r in enumerate(ROOTS)]
    th = np.float32(0.9)

    def run():
        sv = tm.vmap(sitem, outputs={"S": tm.Sum((4,), jnp.float32, fixed_point=True),
                                     "rec": tm.Records(f32, capacity=SCAP)},
                     in_axes=(0, 0, 0, 0, None), spawn=tm.Queue(QCAP), backend=b)
        one = tm.vmap(lambda t, x: t.append("r", {"x": x, "k": jnp.int32(1)}, when=x > 1.0), backend=b,
                      outputs={"r": tm.Records({"x": jnp.float32, "k": jnp.int32}, capacity=64)})
        r = sv(*roots, th)
        g = jax.grad(lambda th: jnp.sum(sv(*roots, th).outputs["rec"]["val"]))(th)
        o = one(roots[3])
        cnt = [r.stats["appended"]["rec"], o.stats["appended"]["r"]]   # (int64 with x64, int32 without: _limits)
        want = np.int64 if jax.config.jax_enable_x64 else np.int32
        if any(np.asarray(c).dtype != want for c in cnt):
            raise Fail(f"stats['appended'] dtypes {[np.asarray(c).dtype for c in cnt]}, want {np.dtype(want)}")
        return (r.outputs, [np.asarray(c).astype(np.int64) for c in cnt], r.stats["spawned"], r.stats["n_truncated"],
                g, o.outputs)
    x0 = jax.config.jax_enable_x64
    res = []
    try:
        for on in (False, True):
            jax.config.update("jax_enable_x64", on)
            res.append([np.asarray(v) for v in jax.tree_util.tree_leaves(run())])
    finally:
        jax.config.update("jax_enable_x64", x0)
    for k, (a, c) in enumerate(zip(*res)):
        if a.dtype != c.dtype or a.tobytes() != c.tobytes():
            raise Fail(f"leaf {k}: x64 off {a.dtype} {a.ravel()[:6]} != on {c.dtype} {c.ravel()[:6]}")
    return f"{len(res[0])} leaves bitwise (records, sums, stats, grad; spawn and one pass)"


@case("big tally: grad of an emit inside lax.cond in a loop, and of Records, with 2^17 bins / slots")
def _(b):
    S = 1 << 17

    def mk(op):
        def item(t, i, key, mu_a, v, wcut):
            def body(c):
                j, x, t = c
                u = jax.random.uniform(jax.random.fold_in(key, j), (), F)
                t = op(t, (i * 7919 + j * 104729) % S, x * mu_a, u > 0.3)
                return j + 1, x * (0.5 + 0.5 * u), t
            return lax.while_loop(lambda c: c[1] > 0.05, body, (jnp.int32(0), F(1.0), t))[2]
        return item
    cond = lambda t, k, v, ok: lax.cond(ok, lambda t: t.add("Q", k, v), lambda t: t, t)
    idx = lambda t, k, v, ok: t.add("Q", jnp.where(ok, k, -1), v)
    app = lambda t, k, v, ok: t.append("R", {"v": v}, when=ok)
    g = {}
    for nm, op, o in (("cond", cond, {"Q": tm.Sum((S,), F)}), ("index", idx, {"Q": tm.Sum((S,), F)}),
                      ("records", app, {"R": tm.Records({"v": F}, capacity=S)})):
        f = tm.vmap(mk(op), outputs=o, in_axes=AX, backend=b)
        red = (lambda o: jnp.sum(o["Q"] ** 2)) if nm != "records" else (lambda o: jnp.sum(o["R"]["v"] ** 2))
        g[nm] = jax.grad(lambda m: red(f(IDS, KEYS, m, VV, F(0.0)).outputs))(MU)
    close(g["cond"], g["index"], 1e-5, 0.0, "cond vs index")
    if not np.isfinite(float(g["records"])):
        raise Fail("records grad")
    return f"grads {float(g['cond']):.6g} {float(g['records']):.6g}"


def multi(t, i, key, mu_a, v, wcut):
    """the walk (one append per loop step), then appends in straight-line code (a chain of gated appends at the
    item's top level) and a loop of two appends per step: the reverse of every gated append passes the output's
    cotangent on to the previous one, which must never become a per-item array of the capacity."""
    t, (k, w) = walk(t, i, key, mu_a, v, wcut)
    for j in range(3):
        u = jax.random.uniform(jax.random.fold_in(key, 1000 + j), (), F)
        t = t.append("hits", {"sensor": -1 - j, "t": u * v, "w": w * u / mu_a, "item": i, "k": k + j}, when=u > 0.25)

    def body(c):
        n, t = c
        u = jax.random.uniform(jax.random.fold_in(key, 2000 + n), (2,), F)
        for q in range(2):
            t = t.append("hits", {"sensor": -10 - q, "t": u[q] / v, "w": w * mu_a * u[1 - q], "item": i, "k": n},
                         when=u[q] > 0.4)
        return n + 1, t
    t = lax.while_loop(lambda c: c[0] < 3, body, (jnp.int32(0), t))[1]
    return t, (k, w)


BIGC, SMALLC, NB = 1 << 22, 4096, 128


@case("large capacity (2^22): grad / vjp / jacrev of Records == capacity 4096 BITWISE (f64 reverse: to 1e-13), and vs "
      "central differences")
def _(b):
    ids, keys = IDS[:NB], KEYS[:NB]
    wt = jax.random.uniform(jax.random.PRNGKey(5), (3, SMALLC), F)

    def fields(cap):                  # th -> (t, w of the first SMALLC slots: the rest is padding), records appended
        f = tm.vmap(multi, outputs=outs(cap), in_axes=AX, backend=b)

        def run(th):
            r = f(ids, keys, th[0], th[1], F(0.0))
            h = r.outputs["hits"]
            return {k: h[k][:SMALLC] for k in ("t", "w")}, r.stats["appended"]["hits"]
        return run

    def red(cap):
        run = fields(cap)

        def g(th):                    # 3 outputs
            h = run(th)[0]
            t, w = h["t"], h["w"]
            return jnp.stack([jnp.sum(wt[0] * w * jnp.cos(0.3 * t)), jnp.sum(wt[1] * t * t), jnp.sum(wt[2] * w * t)])
        return g
    n = int(fields(SMALLC)(TH)[1])
    if not 0 < n <= SMALLC:
        raise Fail(f"{n} records: the small capacity must hold them all")
    res = {}
    for cap in (SMALLC, BIGC):
        g, run = red(cap), fields(cap)
        _, pull = jax.vjp(lambda th: run(th)[0], TH)
        res[cap] = dict(grad=jax.grad(lambda th: jnp.sum(g(th)))(TH), vjp=pull({"t": wt[1], "w": wt[2]})[0],
                        jacrev=jax.jacrev(g)(TH), jacfwd=jax.jacfwd(g)(TH))
    for k in res[SMALLC]:
        # the reverse modes sum each item's contribution to the broadcast operands' cotangents: exactly (so bitwise,
        # whatever the order) for float32, in the operand's own dtype otherwise (docs/float_semantics.md item 4), and
        # the kernel backend's f64 atomic adds are then in schedule order: a few ulp from run to run at ANY capacity
        # (round9/recovf 46215731: vjp 2-3 of 8 runs distinct at 2^16 .. 2^22, jacrev at 2^18 / 2^20)
        if k == "jacfwd" or np.dtype(F) == np.float32:
            same(res[BIGC][k], res[SMALLC][k], f"{k}: capacity 2^22 vs 4096")
        elif not np.allclose(np.asarray(res[BIGC][k]), np.asarray(res[SMALLC][k]), rtol=1e-13, atol=0):
            raise Fail(f"{k}: capacity 2^22 vs 4096 beyond 1e-13 relative: {np.asarray(res[BIGC][k]).ravel()} vs "
                       f"{np.asarray(res[SMALLC][k]).ravel()}")
    loss = lambda th: jnp.sum(red(SMALLC)(th))                                                 # noqa: E731
    e = F(1e-2) if F == jnp.float32 else F(1e-5)
    fd = np.array([(loss(TH + e * u) - loss(TH - e * u)) / (2 * e) for u in jnp.eye(2, dtype=F)])
    close(res[BIGC]["grad"], fd, 2e-3, 1e-5, "grad vs FD")
    close(res[BIGC]["jacrev"], res[BIGC]["jacfwd"], 1e-4, 1e-6, "jacrev vs jacfwd")
    return f"{n} records; grad {np.asarray(res[BIGC]['grad'])} fd {fd}"


@case("memory: Records grad / vjp scale like the forward, linear in the capacity (compiled memory analysis)")
def _(b):
    ids, keys = IDS[:NB], KEYS[:NB]

    def mem(cap, deriv):
        f = tm.vmap(multi, outputs=outs(cap), in_axes=AX, backend=b)
        wt = jnp.ones((cap,), F)
        fw = lambda th: f(ids, keys, th[0], th[1], F(0.0)).outputs["hits"]["w"]                # noqa: E731
        if deriv == "fwd":
            fn = fw
        elif deriv == "grad":
            fn = jax.grad(lambda th: jnp.sum(fw(th) * wt))
        else:
            fn = lambda th: jax.vjp(fw, th)[1](wt)[0]                                          # noqa: E731
        ma = jax.jit(fn).lower(TH).compile().memory_analysis()
        if ma is None:
            return None
        return ma.temp_size_in_bytes + ma.argument_size_in_bytes + ma.output_size_in_bytes - ma.alias_size_in_bytes
    caps = (1 << 18, 1 << 20, 1 << 22)
    m = {d: [mem(c, d) for c in caps] for d in ("fwd", "grad", "vjp")}
    if any(x is None for v in m.values() for x in v):
        return "SKIP: no memory analysis on this platform"
    sl = {d: [(v[i + 1] - v[i]) / (caps[i + 1] - caps[i]) for i in range(2)] for d, v in m.items()}
    for d in ("grad", "vjp"):
        if not sl[d][1] <= 2 * sl["fwd"][1] + 8:
            raise Fail(f"{d}: {sl[d][1]:.1f} B per slot vs the forward's {sl['fwd'][1]:.1f}")
        if not abs(sl[d][1] - sl[d][0]) <= 0.1 * sl[d][1] + 1:
            raise Fail(f"{d}: not linear in the capacity: {sl[d][0]:.1f} then {sl[d][1]:.1f} B per slot")
    return "B/slot " + ", ".join(f"{d} {v[1]:.1f}" for d, v in sl.items()) + "; MB at 2^22 " + \
        ", ".join(f"{d} {v[2] / 1e6:.0f}" for d, v in m.items())


print("\n================ SUMMARY", jax.__version__, "x64" if jax.config.jax_enable_x64 else "",
      jax.devices()[0].device_kind)
for n, b, m in RES:
    print(f"{n:60s} {b:6s} {m[:160]}")
print("TOTAL", len(RES), "PASS", sum(m.startswith("PASS") for _, _, m in RES))
