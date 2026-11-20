"""Memory per output cell (tallymap.cell_bytes / tallymap.max_cells) and large cell counts.
python tests/cellbytes.py  (env: BACKENDS=xla,auto  TEST_X64=1  ONLY=substr  CELLBYTES_BIG=1)

Cells: the API's numbers against the measured peak device memory of fresh processes (runtime peak and XLA's compiled
memory analysis: forward, jvp, grad, vjp; exact, f32, fixed point, Records); the API's refusals and limits; the
in-place carry step of a large exact tally (bitwise the small tally's lax.cond path and the kernel backend); outputs
past 2^31 / W cells LOWER on the xla backend (the int32 index the HK run overflowed); results at normal sizes
bitwise the reference.  CELLBYTES_BIG=1 (a separate job: tens of GB): RUN calls past 2^31 / W cells on both backends
and check the bins past the int32 boundary.
"""
import os
import subprocess
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

import tallymap as tm
from tallymap.backends import xla as X

BACKENDS = os.environ.get("BACKENDS", "xla,auto").split(",")
ONLY = os.environ.get("ONLY", "")
BIG = os.environ.get("CELLBYTES_BIG") == "1"
X64 = bool(jax.config.jax_enable_x64)
GPU = jax.default_backend() == "gpu"
print("jax", jax.__version__, jax.__file__, jax.devices(), "x64", X64, "big", BIG, flush=True)
RES = []
F32 = jnp.float32
MN, MK = 1 << 14, 8                           # the measured toy: items x emits per item


def _out(kind, S):
    return {"exact": lambda: tm.Sum((S,), F32, fixed_point=True), "f32": lambda: tm.Sum((S,), F32),
            "i32": lambda: tm.Sum((S,), jnp.int32),
            "fxi32": lambda: tm.Sum((S,), jnp.int32, fixed_point=True, scale=2.0 ** 10),
            "fx16": lambda: tm.Sum((S,), jnp.float16, fixed_point=True, scale=2.0 ** 10),
            "rec": lambda: tm.Records({"s": jnp.int32, "t": F32}, capacity=S)}[kind]()


def measure(backend, kind, deriv, S):
    """(in a fresh process) -> dict(peak=, compiled=): the runtime peak device memory of one jitted call of the toy
    (GPU: memory_stats peak_bytes_in_use over the bytes in use before it; CPU: the process's VmHWM growth over its
    VmRSS before it) and XLA's compiled memory analysis (temp + arguments + outputs - aliased)."""
    import json

    def status(key):
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
        return 0
    out = _out(kind, S)
    isint = kind in ("i32", "fxi32")

    def item(t, x, k, th):
        for j in range(MK):
            idx = (k * 7919 + j * 104729) % S
            v = x * th + F32(0.37) * j
            t = t.append("Q", {"s": idx, "t": v}) if kind == "rec" else t.add("Q", idx, (idx & 7) if isint else v)
        return t
    rng = np.random.default_rng(0)
    xs = jnp.asarray(rng.uniform(0.5, 1.5, MN).astype(np.float32))
    ks = jnp.asarray(np.arange(MN, dtype=np.int32))
    th = F32(0.999)
    xo = {"xla_batch_size": 4096} if backend == "xla" else {}
    v = tm.vmap(item, outputs={"Q": out}, in_axes=(0, 0, None), backend=backend, **xo)
    get = (lambda r: r.outputs["Q"]["t"]) if kind == "rec" else (lambda r: r.outputs["Q"])
    args = (xs, ks, th)
    if deriv == "forward":
        f = lambda x, k, th: get(v(x, k, th))                                                 # noqa: E731
    elif deriv == "jvp":
        f = lambda x, k, th: jax.jvp(lambda th: get(v(x, k, th)), (th,), (F32(1),))           # noqa: E731
    elif deriv == "grad":
        f = jax.grad(lambda th, x, k: jnp.sum(jnp.sin(get(v(x, k, th)))).astype(F32))     # (in the output dtype)
        args = (th, xs, ks)
    else:                                     # vjp: the output and the cotangent's pullback (cotangent an input)
        def f(th, x, k, ct):
            o, b = jax.vjp(lambda th: get(v(x, k, th)), th)
            return o, b(ct)[0]
        dt = jnp.float16 if kind == "fx16" else F32
        args = (th, xs, ks, jnp.asarray(rng.uniform(0, 1, S).astype(np.float32)).astype(dt))
    c = jax.jit(f).lower(*args).compile()
    ma = c.memory_analysis()
    res = {"compiled": None if ma is None else ma.temp_size_in_bytes + ma.argument_size_in_bytes
           + ma.output_size_in_bytes - ma.alias_size_in_bytes}
    argb = sum(int(np.prod(np.shape(a))) * np.asarray(a).dtype.itemsize for a in args)
    jax.block_until_ready(args)
    dev = jax.devices()[0]
    if dev.platform == "gpu":
        m0 = dev.memory_stats()
        jax.block_until_ready(c(*args))
        res["peak"] = dev.memory_stats()["peak_bytes_in_use"] - m0["bytes_in_use"] + argb
    else:
        rss0, hwm0 = status("VmRSS"), status("VmHWM")
        jax.block_until_ready(c(*args))
        hwm1 = status("VmHWM")
        res["peak"] = hwm1 - rss0 + argb if hwm1 > hwm0 else 0
    print("MEASURED " + json.dumps(res), flush=True)


if len(sys.argv) > 1 and sys.argv[1] == "--measure":
    measure(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    sys.exit(0)


def measured(backend, kind, deriv, S):
    import json
    p = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--measure", backend, kind, deriv, str(S)],
                       capture_output=True, text=True, timeout=1200, env=dict(os.environ))
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("MEASURED ")]
    if not line:
        raise Fail(f"measurement {backend} {kind} {deriv} {S} crashed: {(p.stdout + p.stderr)[-500:]}")
    return json.loads(line[0][9:])


class Fail(Exception):
    pass


def case(name, backends=None, big=False):
    def deco(fn):
        if big != BIG:
            return fn
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
                msg = f"RAISE {type(e).__name__}: {str(e)[:300]}"
                traceback.print_exc()
            RES.append((name, b, msg))
            print(f"{name:60s} {b:6s} {msg}  [{time.perf_counter() - t0:.1f}s]", flush=True)
        return fn
    return deco


def bitwise(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


# ------------------------------------------------------------------ the API against measured memory
DER = ("forward", "jvp", "grad", "vjp")
MCFG = {"auto": [("exact", d) for d in DER] + [("f32", d) for d in DER]
        + [("i32", "forward"), ("fxi32", "forward")] + [("rec", d) for d in DER],
        "xla": [("exact", d) for d in DER] + [("f32", d) for d in DER] + [("fx16", d) for d in DER]
        + [("i32", "forward"), ("fxi32", "forward")] + [("rec", d) for d in DER]}
MARGIN, SLACK = 1.3, 16 << 20      # slope x cells <= MARGIN x measured + SLACK; measured <= total + SLACK
# (xla plain float sums, Records: partial rows and canonical copies whose holding XLA schedules differently per
# size, platform and jax version: 2x - 3x)
LOOSE = {("xla", "f32"): 2.0, ("xla", "rec"): 3.0, ("auto", "rec"): 2.0}


@case("cell_bytes >= the measured peak memory, within 30% (xla plain sums, Records: 2-3x) + 16 MiB (S = 2^23, 2^19)")
def _(b):
    bad, rows = [], []
    cfg = [(k, d, 1 << 23) for k, d in MCFG[b]] + [("exact", "forward", 1 << 19)]
    for kind, d, S in cfg:
        m = measured(b, kind, d, S)
        om = tm.output_memory(_out(kind, S), b, None if d == "forward" else d)
        api = om.total(S)
        got = [m["peak"]] + ([m["compiled"]] if GPU and m["compiled"] is not None else [])
        if not m["peak"] and m["compiled"] is not None:      # (CPU: the high-water mark did not rise: unmeasured)
            got = [m["compiled"]]
        hi = max(got)
        # the bound holds (measured <= per_cell S + fixed), and its slope is tight (per_cell S <= MARGIN measured)
        ok = hi <= api + SLACK and om.per_cell * S <= LOOSE.get((b, kind), MARGIN) * hi + SLACK
        rows.append(f"{kind}/{d}/{S}: api {api / S:.1f} meas {hi / S:.1f}")
        if not ok:
            bad.append(rows[-1])
    print("   ", "; ".join(rows), flush=True)
    if os.environ.get("CELLBYTES_OUT"):                 # (measurement record: one line per platform / backend)
        with open(os.environ["CELLBYTES_OUT"], "a") as fo:
            fo.write(f"{jax.__version__} {jax.default_backend()} x64={X64} {b}: " + "; ".join(rows) + "\n")
    if bad:
        raise Fail("; ".join(bad))
    return f"{len(cfg)} configurations: " + ", ".join(rows[:4]) + " ..."


@case("cell_bytes / max_cells: arguments, limits, monotone; a Sum past 2^31 - 1 cells raises", ["xla"])
def _(b):
    ex = tm.Sum((1 << 20,), F32, fixed_point=True)
    k = tm.cell_bytes(ex, "auto", platform="gpu")
    if k != 44.125 or tm.cell_bytes(ex, "auto", "jvp", platform="cpu") != 88.25:
        raise Fail(f"kernel exact {k}")
    for d in (None, "jvp", "vjp", "grad"):
        for be in ("auto", "xla"):
            for pl in ("gpu", "cpu"):
                for x in (False, True):
                    for o in (ex, tm.Sum(1000), tm.Sum((7, 9), jnp.int32), tm.Sum(5, jnp.int32, fixed_point=True),
                              tm.Sum(64, jnp.float16, fixed_point=True, scale=2.0 ** 4)):
                        v = tm.cell_bytes(o, be, d, platform=pl, x64=x)
                        if not (v > 0 and np.isfinite(v)):
                            raise Fail(f"{o} {be} {d} {pl} {x}: {v}")
    if tm.max_cells(ex, "xla") != (1 << 31) - 1:
        raise Fail("max_cells without memory")
    mem = 4 << 30
    for be in ("auto", "xla"):
        n = tm.max_cells(ex, be, "jvp", memory=mem, platform="gpu")
        if not (tm.cell_bytes(ex, be, "jvp", cells=n, platform="gpu") * n <= mem
                < tm.cell_bytes(ex, be, "jvp", cells=n + 1, platform="gpu") * (n + 1)):
            raise Fail(f"max_cells({be}) = {n} is not the largest fitting count")
    rec = tm.Records({"s": jnp.int32, "t": F32}, capacity=1 << 20)
    for be in ("auto", "xla"):                  # (Records reverse mode: modelled since 0103ac3)
        for d in (None, "jvp", "vjp", "grad"):
            m = tm.output_memory(rec, be, d)
            if not abs(m.total(1 << 20) / (1 << 20) - tm.cell_bytes(rec, be, d)) < 1e-6:
                raise Fail(f"output_memory vs cell_bytes ({be}, {d})")
    outs = {"a": ex, "b": tm.Sum((3, 1 << 18), F32), "r": rec}
    for be in ("auto", "xla"):                  # one call: readouts and the launch counted once
        c = tm.call_memory(outs, be, "jvp")
        sz = ((ex, 1 << 20), (outs["b"], 3 << 18), (rec, 1 << 20))
        s = sum(tm.output_memory(o, be, "jvp").total(m_) for o, m_ in sz)
        if not 0 < c <= s:
            raise Fail(f"call_memory {c} vs the outputs' sum {s}")
    raises = [lambda: tm.cell_bytes(ex, "cuda"),
              lambda: tm.cell_bytes(ex, "xla", "hessian"), lambda: tm.cell_bytes(ex, cells=0),
              lambda: tm.cell_bytes(tm.Records({"s": jnp.int32}), "xla"), lambda: tm.cell_bytes(ex, platform="tpux")]
    for f in raises:
        try:
            f()
        except (ValueError, NotImplementedError):
            continue
        raise Fail("an invalid argument did not raise")
    try:
        tm.vmap(lambda t, x: t.add("Q", 0, x), outputs={"Q": tm.Sum((1 << 16, 1 << 15))}, backend=b)(
            jnp.ones(4, F32))
    except ValueError as e:
        if "2^31 - 1" not in str(e):
            raise Fail(f"wrong message: {e}") from None
    else:
        raise Fail("a Sum of 2^31 cells did not raise")
    return (f"kernel exact 44.125 / jvp 88.25; xla exact gpu {tm.cell_bytes(ex, 'xla', platform='gpu', x64=False):.1f}"
            f" / jvp {tm.cell_bytes(ex, 'xla', 'jvp', platform='gpu', x64=False):.1f} (2^20 cells)")


# ------------------------------------------------------------------ the carry step of a large exact tally
# values whose integer sums carry and borrow across many words: huge and tiny magnitudes, both signs, cancellation
VALS = np.array([3.0e38, -3.0e38, 1.0e-45, -2.0e-38, 1.5, -0.75, 6.1e7, -6.1e7 + 4.0, 2.0 ** 100, -(2.0 ** -120),
                 1.17549435e-38, 3.4028235e38], np.float32)
NB_SMALL = 8


def _carry_item(bins_off):
    def item(t, x, i):
        for j in range(4):
            b = (i * 3 + j) % 5 + bins_off
            t = t.add("Q", b, x * jnp.asarray(VALS)[(i + j) % len(VALS)])
        return t
    return item


def _carry_run(b, S, off, n=1 << 16, **kw):
    xs = jnp.asarray(np.random.default_rng(1).uniform(0.5, 2.0, n).astype(np.float32))
    ids = jnp.arange(n, dtype=jnp.int32)
    v = tm.vmap(_carry_item(off), outputs={"Q": tm.Sum((S,), F32, fixed_point=True)}, backend=b, **kw)
    r = jax.jit(lambda x, i: v(x, i))(xs, ids)
    return np.asarray(r.outputs["Q"]), int(r.stats["fixed_point_overflow"])


@case("large exact tally: in-place carry step == small tally (lax.cond) == kernel, bitwise", ["xla"])
def _(b):
    # 2^16 items x 4 emits into 5 bins: ~52k values per bin > XNORM32, so the carry step runs (several times per bin
    # with 4096-item batches).  S = 2^17 bins (2.5M int32 words > FUSED_WORDS): _xnorm_loop; S = 8: lax.cond.
    big, ob = _carry_run(b, 1 << 17, 70000, xla_batch_size=4096)
    small, os_ = _carry_run(b, NB_SMALL, 2, xla_batch_size=4096)
    if not bitwise(big[70000:70005], small[2:7]) or ob != os_:
        raise Fail(f"in place {big[70000:70005]} ({ob}) vs cond {small[2:7]} ({os_})")
    if np.any(big[:70000] != 0) or np.any(big[70005:] != 0):
        raise Fail("bins outside the emitted ones are not zero")
    out = "loop == cond"
    saved = X.FUSED_WORDS
    try:                                      # the loop path on the SMALL tally too (FUSED_WORDS = 0)
        X.FUSED_WORDS = 0
        forced, of = _carry_run(b, NB_SMALL, 2, xla_batch_size=4096)
    finally:
        X.FUSED_WORDS = saved
    if not bitwise(forced, small) or of != os_:
        raise Fail(f"forced loop {forced[2:7]} vs cond {small[2:7]}")
    out += " == forced loop (small)"
    if "auto" in BACKENDS:
        k, ok_ = _carry_run("auto", NB_SMALL, 2)
        if not bitwise(k, small) or ok_ != os_:
            raise Fail(f"kernel {k[2:7]} ({ok_}) vs xla {small[2:7]} ({os_})")
        out += " == kernel"
    return out + f"; overflow count {os_}; bins {small[2:7]}"


# ------------------------------------------------------------------ outputs past 2^31 / W cells lower (xla)
def _lower(b, out, S):
    def item(t, x, i):
        return t.add("Q", (i * 40503) % S, x)
    xs = jax.ShapeDtypeStruct((1024,), F32)
    ids = jax.ShapeDtypeStruct((1024,), jnp.int32)
    v = tm.vmap(item, outputs={"Q": out}, backend=b)
    jax.jit(lambda x, i: v(x, i).outputs["Q"]).lower(xs, ids)      # (traces and lowers; allocates nothing)


@case("xla: outputs past 2^31 / words cells lower (the HK int32 index overflow)", ["xla"])
def _(b):
    done = []
    W = X._xfmt()[0]
    S = (1 << 31) // W + 12345                # exact: W words per cell (19 without x64, 9 with)
    _lower(b, tm.Sum((S,), F32, fixed_point=True), S)
    done.append(f"exact {S} cells x {W} words")
    S2 = (1 << 30) + 77                       # f32 (2 rows: the LOW row) and the x64 fixed point pair: 2 S > 2^31
    _lower(b, tm.Sum((S2,), F32), S2)
    done.append(f"f32 {S2}")
    if X64:
        _lower(b, tm.Sum((S2,), jnp.float64, fixed_point=True), S2)
        done.append(f"f64 fixed point {S2}")
    else:
        S3 = (1 << 31) // X.FIXED_WORDS32 + 999
        _lower(b, tm.Sum((S3,), jnp.float16, fixed_point=True, scale=2.0 ** 8), S3)
        done.append(f"f16 fixed point {S3} x {X.FIXED_WORDS32} words")
    return "; ".join(done)


# ------------------------------------------------------------------ BIG: run past the int32 boundary
def _big_run(b, out, S, bins, vals):
    """emit vals[k] into bins[k] (one item each, and the same again from a second item) -> the output."""
    n = len(bins)

    def item(t, i, v):
        return t.add("Q", i, v)
    ib = np.array(list(bins) * 2, np.int64)
    if np.any(ib >= S):
        raise Fail("bad test bins")
    iv = np.array(list(vals) * 2, np.float32)
    v = tm.vmap(item, outputs={"Q": out}, backend=b)
    o = v(jnp.asarray(ib.astype(np.int32)), jnp.asarray(iv)).outputs["Q"]
    del n
    return o


def _check_big(o, S, bins, want):
    got = np.asarray(o[np.array(bins)])
    if not bitwise(got, np.asarray(want, got.dtype)):
        raise Fail(f"bins {bins}: {got} vs {want}")
    tot = float(jax.jit(lambda o: jnp.sum(jnp.abs(o.astype(jnp.float32))))(o))     # (fused: no temporary)
    if not np.isclose(tot, float(np.sum(np.abs(np.asarray(want, np.float64)))), rtol=1e-5):
        raise Fail(f"other bins are not zero: total |o| {tot}")


def _bins(S, W):
    edge = (1 << 31) // W                     # the first cell whose flat word index passes int32
    return [0, 1, edge - 1, edge, edge + 1, S // 2, S - 2, S - 1]


@case("BIG: exact Sum past 2^31 / W cells runs, bins past the boundary exact", None, big=True)
def _(b):
    W = X._xfmt()[0] if b == "xla" else 5     # (the kernel backend: 5 int64 words per cell)
    S = (1 << 31) // W + 4096
    bins = _bins(S, W)
    vals = [1.5, -2.0 ** -149, 1.0e38, 0.1, -7.25, 2.0 ** 60, 1e-3, -1.0]
    o = _big_run(b, tm.Sum((S,), F32, fixed_point=True), S, bins, vals)
    want = [np.float32(2 * np.float64(np.float32(v))) for v in vals]
    _check_big(o, S, bins, want)
    return f"{S} cells x {W} words: {len(bins)} bins exact"


@case("BIG: f32 Sum past 2^30 cells runs (the LOW row: 2 S > 2^31)", None, big=True)
def _(b):
    S = (1 << 30) + 4096 if b == "xla" else (1 << 31) - 1     # (the kernel: up to the int32 index limit)
    bins = [0, 1, (1 << 30) - 1, 1 << 30, (1 << 30) + 1, S - 2, S - 1]
    vals = [1.5, 1e-32, 2.5, -3.0, 2.0 ** -110, 7.0, -1.0]     # (normal: XLA:CPU flushes subnormal f32 sums)
    o = _big_run(b, tm.Sum((S,), F32), S, bins, vals)
    want = [np.float32(2 * np.float64(np.float32(v))) for v in vals]
    _check_big(o, S, bins, want)
    return f"{S} cells: {len(bins)} bins exact"


print("\n================ SUMMARY", jax.__version__, "x64" if X64 else "", jax.devices()[0].device_kind)
for n, b, m in RES:
    print(f"{n:60s} {b:8s} {m[:200]}")
print("TOTAL", len(RES), "PASS", sum(m.startswith("PASS") for _, _, m in RES))
