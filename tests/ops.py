"""Per-operation parity of the kernel op table (backends/kernel/op_table.py), the main maintenance surface.

    python tests/ops.py              # judge every op against its stated bound, compare with the baseline
    python tests/ops.py --record     # (re)write this (device, JAX) entry of tests/ops_baseline.json

It runs the target JAX places the computation on: the cuda target on a GPU (both math modes), the CPU target
elsewhere (one mode, 'ieee': the host target is IEEE whatever cuda_math says; judged by the ieee bounds, with the
libm math library where the GPU has libdevice, so `lib` holds glibc to libdevice's documented bounds).
    env OPS=add,exp (subset)  N (262144 items)  OPS_BASELINE (another baseline file)

Each op is one item (= one element): tm.vmap(op, backend='auto') for cuda_math in {ieee, approx}, against
    XLA       jax.jit(jax.vmap(op)) on the same device, and
    EXACT     (classes cr / lib only) the op evaluated in float64 on the host CPU, rounded to float32.
Inputs mix wide random ranges with the specials 0, -0, +-inf, NaN, the smallest subnormal, 1, -1, 3e38, 0.5.
NaN == NaN, and +0 == -0 only where the class allows it (it never does for exact / cr).

Classes and their bounds (STATED BEFORE THE FIRST RUN; a deliberate change of a bound is a commit of its own):
  exact  bitwise equal to XLA in both math modes: integer / bit / compare / select / convert / floor / round / sign /
         abs / neg / min / max / clip / nextafter / RNG / gathers / sorting, f64 add / mul / div / sqrt (IEEE in both).
  cr     correctly-rounded f32 arithmetic, judged in ulp against EXACT:
           ieee:   add sub mul div rcp sqrt rem <= 1 ulp (correct rounding gives 0); rsqrt = rcp(sqrt) <= 1;
                   x**3 (two roundings) <= 2
           approx: add sub mul rem unchanged (<= 1); div / rcp (div.full) <= 2; sqrt (sqrt.approx) <= 2;
                   rsqrt (rsqrt.approx) <= 3; x**3 <= 2
  lib    libdevice transcendentals, judged in ulp against EXACT with the maximum errors documented in the CUDA C++
         Programming Guide (single precision, full range, no -ftz): exp 2, exp2 2, log 1, log2 1, log1p 1, expm1 1,
         sin 2, cos 2, tan 4, asin 2, acos 2, atan 2, atan2 3, sinh 3, cosh 2, tanh 2, asinh 3, acosh 4, atanh 3,
         pow 4, cbrt 1, erf 2, erfc 4, erfinv 2, lgamma 6 (positive arguments); logistic = 1/(1+exp(-x)): 4 (ieee),
         6 (approx: div.full).  Non-finite results must match EXACT exactly.
         CORRECTED AFTER THE FIRST RUN (disclosed, commit of its own): log2 approx 1 -> 2.  On JAX 0.5.3 jnp.log2 is
         log(x) / log(2), so cuda_math='approx' adds div.full's error to logf's (measured 2 ulp on L40S and 2080 Ti;
         ieee 1 ulp; on JAX 0.11.2 it is a primitive and passed the original bound).
  info   compositions (fused patterns, complex arithmetic, dot / reductions, the RNG's normal) and f64 libdevice:
         reported against XLA only (mismatching elements, max ulp), not judged against a bound.
Every row (all classes) is also compared with tests/ops_baseline.json for this (GPU model, JAX version): the same
cubin on the same inputs must give the same numbers, so ANY change is a FAIL until it is re-recorded on purpose.
The baseline holds tallymap's own output (`tm_sha`, a hash of the per-item result) and the EXACT-judged fields, NOT
the XLA comparison: XLA's side is not reproducible across processes.  MEASURED on L40S, JAX 0.11.2: in about 1 process
in 6, XLA's fusion autotuner compiles the cumsum8 row reduction with its Triton emitter instead of the native one,
which sums in a different order (xla_mismatch 107625 -> 99300), while tallymap's output was bitwise identical in all
of them.  The `exact` class still requires xla_mismatch == 0 in every run.
Prints 'OP {json}' per (op, math) and 'ALL PASS k/n' (or FAIL).
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

jax.config.update("jax_default_matmul_precision", "highest")
import tallymap as tm

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("OPS_BASELINE") or os.path.join(HERE, "ops_baseline.json")
N = int(os.environ.get("N", 1 << 18))
SEL = [o for o in os.environ.get("OPS", "").split(",") if o]
RECORD = "--record" in sys.argv[1:]
rng = np.random.default_rng(0)
SPECIAL = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-45, 1.0, -1.0, 3e38, 0.5], np.float32)


def inp(kind):
    if kind == "wide":
        x = (rng.standard_normal(N) * np.exp(rng.uniform(-20, 20, N))).astype(np.float32)
    elif kind == "unit":
        x = rng.uniform(-1, 1, N).astype(np.float32)
    elif kind == "pos":
        x = np.exp(rng.uniform(-30, 30, N)).astype(np.float32)
    elif kind == "mid":
        x = rng.uniform(-12, 12, N).astype(np.float32)
    elif kind == "ge1":
        x = (1 + np.exp(rng.uniform(-10, 10, N))).astype(np.float32)
    elif kind == "i32":
        return rng.integers(-2 ** 31, 2 ** 31, N, dtype=np.int64).astype(np.int32)
    elif kind == "sh":
        return rng.integers(-4, 40, N).astype(np.int32)
    elif kind == "u32":
        return rng.integers(0, 2 ** 32, N, dtype=np.uint64).astype(np.uint32)
    elif kind == "key":
        return np.asarray(jax.random.split(jax.random.PRNGKey(1), N))
    elif kind == "f64":
        return rng.standard_normal(N) * np.exp(rng.uniform(-20, 20, N))
    elif kind == "f64pos":
        return np.exp(rng.uniform(-30, 30, N))
    else:
        raise ValueError(kind)
    x[:len(SPECIAL)] = SPECIAL
    return x


sp = jax.scipy.special
TAB = jnp.asarray(np.sin(np.arange(128.0) * 0.37).astype(np.float32))
SORTED = jnp.asarray(np.cumsum(np.abs(np.sin(np.arange(128.0)))).astype(np.float32))


def _ri(z):
    return jnp.stack([jnp.real(z), jnp.imag(z)])


def _fresnel(c, k):
    n = (1.33 + 0j) + 1j * 0.01 * jnp.abs(k)
    m = (1.0 + 0j) / n
    c = jnp.abs(c) + 0j
    s = jnp.sqrt(1 - m ** 2 * (1 - c * c))
    rs = (c - n * s) / (c + n * s)
    rp = (n * c - s) / (n * c + s)
    return 0.5 * (jnp.abs(rs) ** 2 + jnp.abs(rp) ** 2)


def B(ieee, approx=None):
    return (ieee, ieee if approx is None else approx)


E, I = "exact", "info"
# name: (fn, input kinds, class, (bound ieee, bound approx) for cr / lib)
OPS = {
    # ---- exact: bitwise vs XLA
    "neg": (lambda a: -a, ("wide",), E, None), "abs": (jnp.abs, ("wide",), E, None),
    "sign": (jnp.sign, ("wide",), E, None), "floor": (jnp.floor, ("mid",), E, None),
    "ceil": (jnp.ceil, ("mid",), E, None), "round_away": (lambda a: lax.round(a), ("mid",), E, None),
    "round_even": (jnp.round, ("mid",), E, None), "is_finite": (jnp.isfinite, ("wide",), E, None),
    "max": (jnp.maximum, ("wide", "wide"), E, None), "min": (jnp.minimum, ("wide", "wide"), E, None),
    "clip": (lambda a: jnp.clip(a, -1.0, 1.0), ("wide",), E, None),
    "nextafter": (jnp.nextafter, ("wide", "wide"), E, None),
    "lt": (lambda a, b: a < b, ("wide", "wide"), E, None), "eq": (lambda a, b: a == b, ("wide", "wide"), E, None),
    "where": (lambda a, b: jnp.where(a > b, a, b * 2.0), ("wide", "wide"), E, None),
    "f2i": (lambda a: a.astype(jnp.int32), ("wide",), E, None),
    "f2u": (lambda a: a.astype(jnp.uint32), ("wide",), E, None),
    "i2f": (lambda a: a.astype(jnp.float32), ("i32",), E, None),
    "u2f": (lambda a: a.astype(jnp.float32), ("u32",), E, None),
    "bitcast_u2f": (lambda a: lax.bitcast_convert_type(a, jnp.float32), ("u32",), E, None),
    "bitcast_f2u": (lambda a: lax.bitcast_convert_type(a, jnp.uint32), ("wide",), E, None),
    "iadd": (lambda a, b: a + b, ("i32", "i32"), E, None), "imul": (lambda a, b: a * b, ("i32", "i32"), E, None),
    "idiv": (lambda a, b: lax.div(a, b), ("i32", "sh"), E, None),
    "irem": (lambda a, b: lax.rem(a, b), ("i32", "sh"), E, None),
    "ifloordiv": (lambda a, b: a // b, ("i32", "sh"), E, None),
    "imax": (jnp.maximum, ("i32", "i32"), E, None),
    "and": (lambda a, b: a & b, ("i32", "i32"), E, None), "xor": (lambda a, b: a ^ b, ("u32", "u32"), E, None),
    "not": (lambda a: ~a, ("i32",), E, None),
    "shl": (lax.shift_left, ("i32", "sh"), E, None), "shrl": (lax.shift_right_logical, ("i32", "sh"), E, None),
    "shra": (lax.shift_right_arithmetic, ("i32", "sh"), E, None),
    "popc": (lax.population_count, ("u32",), E, None), "clz": (lax.clz, ("i32",), E, None),
    "gather": (lambda i: TAB[jnp.abs(i) % 128], ("i32",), E, None),
    "gather_oob_clip": (lambda i: TAB[i % 200], ("i32",), E, None),
    "dynamic_slice": (lambda i: lax.dynamic_slice(TAB, (jnp.abs(i) % 125,), (3,)), ("i32",), E, None),
    "searchsorted": (lambda a: jnp.searchsorted(SORTED, jnp.abs(a) * 5.0), ("mid",), E, None),
    "argmax": (lambda a, b: jnp.argmax(jnp.stack([a, b, a - b, b - a])), ("mid", "mid"), E, None),
    "sort3": (lambda a, b, c: jnp.sort(jnp.stack([a, b, c])), ("wide", "wide", "wide"), E, None),
    "cumsum_i": (lambda a: jnp.cumsum(jnp.stack([a, a ^ 7, a >> 3]))[2], ("i32",), E, None),
    "rng_bits": (lambda k: jax.random.bits(k, (2,)), ("key",), E, None),
    "rng_uniform": (lambda k: jax.random.uniform(jax.random.fold_in(k, 3), (2,)), ("key",), E, None),
    "rng_split": (lambda k: jax.random.split(k)[1], ("key",), E, None),
    "rng_randint": (lambda k: jax.random.randint(k, (), 0, 37), ("key",), E, None),
    "f64_add": (lambda a, b: a + b, ("f64", "f64"), E, None), "f64_mul": (lambda a, b: a * b, ("f64", "f64"), E, None),
    "f64_div": (lambda a, b: a / b, ("f64", "f64"), E, None), "f64_sqrt": (jnp.sqrt, ("f64pos",), E, None),
    # ---- cr: correctly rounded arithmetic vs the host-exact result
    "add": (lambda a, b: a + b, ("wide", "wide"), "cr", B(1)),
    "sub": (lambda a, b: a - b, ("wide", "wide"), "cr", B(1)),
    "mul": (lambda a, b: a * b, ("wide", "wide"), "cr", B(1)),
    "div": (lambda a, b: a / b, ("wide", "wide"), "cr", B(1, 2)),
    "div_const": (lambda a: a / 3.0, ("wide",), "cr", B(1, 2)),
    "rcp": (lambda a: 1.0 / a, ("wide",), "cr", B(1, 2)), "sqrt": (jnp.sqrt, ("pos",), "cr", B(1, 2)),
    "rsqrt": (lax.rsqrt, ("pos",), "cr", B(1, 3)), "rem": (lambda a, b: lax.rem(a, b), ("mid", "mid"), "cr", B(1)),
    "ipow3": (lambda a: a ** 3, ("mid",), "cr", B(2)),
    # ---- lib: libdevice, CUDA Programming Guide maximum ulp errors
    "exp": (jnp.exp, ("mid",), "lib", B(2)), "exp2": (jnp.exp2, ("mid",), "lib", B(2)),
    "log": (jnp.log, ("pos",), "lib", B(1)), "log2": (jnp.log2, ("pos",), "lib", B(1, 2)),
    "log1p": (jnp.log1p, ("pos",), "lib", B(1)), "expm1": (jnp.expm1, ("mid",), "lib", B(1)),
    "sin": (jnp.sin, ("mid",), "lib", B(2)), "cos": (jnp.cos, ("mid",), "lib", B(2)),
    "tan": (jnp.tan, ("mid",), "lib", B(4)), "asin": (jnp.arcsin, ("unit",), "lib", B(2)),
    "acos": (jnp.arccos, ("unit",), "lib", B(2)), "atan": (jnp.arctan, ("wide",), "lib", B(2)),
    "atan2": (jnp.arctan2, ("wide", "wide"), "lib", B(3)), "sinh": (jnp.sinh, ("mid",), "lib", B(3)),
    "cosh": (jnp.cosh, ("mid",), "lib", B(2)), "tanh": (jnp.tanh, ("mid",), "lib", B(2)),
    "asinh": (jnp.arcsinh, ("wide",), "lib", B(3)), "acosh": (jnp.arccosh, ("ge1",), "lib", B(4)),
    "atanh": (jnp.arctanh, ("unit",), "lib", B(3)), "pow": (jnp.power, ("pos", "mid"), "lib", B(4)),
    "cbrt": (jnp.cbrt, ("wide",), "lib", B(1)), "erf": (sp.erf, ("mid",), "lib", B(2)),
    "erfc": (sp.erfc, ("mid",), "lib", B(4)), "erf_inv": (sp.erfinv, ("unit",), "lib", B(2)),
    "lgamma": (sp.gammaln, ("pos",), "lib", B(6)), "logistic": (jax.nn.sigmoid, ("mid",), "lib", B(4, 6)),
    # ---- info: compositions, reported vs XLA only
    "div_sqrt": (lambda a, b: a / jnp.sqrt(b), ("wide", "pos"), I, None),
    "div_sqrt_shared": (lambda a, b: (a / jnp.sqrt(b)) + (b / jnp.sqrt(b)), ("wide", "pos"), I, None),
    "div_div": (lambda a, b, c: (a / b) / c, ("mid", "mid", "mid"), I, None),
    "cond_divsqrt": (lambda a, b: lax.cond(a > 0, lambda: a / jnp.sqrt(jnp.abs(b)), lambda: b / jnp.sqrt(jnp.abs(a))),
                     ("mid", "mid"), I, None),
    "while_divsqrt": (lambda a, b: lax.while_loop(lambda c: c[1] < 3, lambda c: (c[0] + a / jnp.sqrt(jnp.abs(b) + c[0]),
                                                                                c[1] + 1), (jnp.float32(0), 0))[0],
                      ("mid", "mid"), I, None),
    "vec_norm": (lambda a, b, c: (jnp.stack([a, b, c]) / jnp.sqrt(a * a + b * b + c * c))[1], ("mid",) * 3, I, None),
    "vec_norm_lin": (lambda a, b, c: (jnp.stack([a, b, c]) / jnp.linalg.norm(jnp.stack([a, b, c])))[2], ("mid",) * 3, I,
                     None),
    "div_exp": (lambda a, b: a / jnp.exp(b), ("mid", "mid"), I, None),
    "exp_negdiv": (lambda a, b: jnp.exp(-a / b), ("pos", "pos"), I, None),
    "exp_mul_exp": (lambda a, b: jnp.exp(b) * jnp.exp(a), ("mid", "mid"), I, None),
    "square": (jnp.square, ("wide",), I, None),
    "dot3": (lambda a, b, c: jnp.dot(jnp.stack([a, b, c]), jnp.stack([c, a, b])), ("mid",) * 3, I, None),
    "matvec3": (lambda a, b, c: (jnp.stack([jnp.stack([a, b, c]), jnp.stack([b, c, a]), jnp.stack([c, a, b])])
                                 @ jnp.stack([c, b, a]))[1], ("mid",) * 3, I, None),
    "sum3": (lambda a, b, c: jnp.sum(jnp.stack([a, b, c])), ("mid",) * 3, I, None),
    "cumsum8": (lambda a, b: jnp.cumsum(a * jnp.arange(1.0, 9.0) + b)[7], ("mid", "mid"), I, None),
    "c_mul": (lambda a, b: _ri((a + 1j * b) * (b - 0.5j * a)), ("mid", "mid"), I, None),
    "c_div": (lambda a, b: _ri((a + 1j * b) / (b - 0.5j * a)), ("mid", "mid"), I, None),
    "c_abs": (lambda a, b: jnp.abs(a + 1j * b), ("wide", "wide"), I, None),
    "c_sqrt": (lambda a, b: _ri(jnp.sqrt(a + 1j * b)), ("mid", "mid"), I, None),
    "c_fresnel": (lambda a, b: _fresnel(a, b), ("unit", "mid"), I, None),
    "rng_normal": (lambda k: jax.random.normal(k, (2,)), ("key",), I, None),
    "f64_exp": (lambda a: jnp.exp(a / 1e6), ("f64",), I, None),
    "f64_log": (jnp.log, ("f64pos",), I, None),
}


# ------------------------------------------------------------------------------------------- measurement
def _ordered(x):
    """float bits -> integers in value order (+0 and -0 adjacent: one ulp apart)."""
    if x.dtype == np.float32:
        b = x.view(np.int32).astype(np.int64)
        return np.where(b < 0, -(b & 0x7FFFFFFF) - 1, b)
    b = x.view(np.int64)
    return np.where(b < 0, -(b & 0x7FFFFFFFFFFFFFFF) - 1, b)


def diff(ref, got):
    """-> (mismatching elements, max ulp over elements finite in both or None, non-finite pattern mismatches)"""
    ref, got = np.asarray(ref), np.asarray(got)
    if ref.dtype != got.dtype or ref.shape != got.shape:
        return ref.size, None, ref.size
    if ref.dtype.kind != "f":
        return int((ref != got).sum()), None, 0
    same = (ref.view(np.uint8).reshape(ref.size, -1) == got.view(np.uint8).reshape(got.size, -1)).all(1)
    same |= (np.isnan(ref) & np.isnan(got)).reshape(-1)
    fin = (np.isfinite(ref) & np.isfinite(got)).reshape(-1)
    nonfin = (~same & ~fin).reshape(-1)
    ulp = None
    if (~same & fin).any():
        r, g = ref.reshape(-1)[~same & fin], got.reshape(-1)[~same & fin]
        ulp = int(np.max(np.abs(_ordered(r) - _ordered(g))))
    return int((~same).sum()), ulp, int(nonfin.sum())


def exact_ref(f, xs):
    """f evaluated in float64 on the host CPU, rounded to float32 (correctly rounded for + - * / sqrt fmod)."""
    cpu = jax.devices("cpu")[0]
    x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)          # (jax.experimental.enable_x64 is gone on 0.11)
    try:
        xd = [jax.device_put(np.asarray(x, np.float64) if np.asarray(x).dtype == np.float32 else x, cpu) for x in xs]
        y = jax.jit(jax.vmap(f))(*xd)                  # runs on the CPU: its inputs are committed there
        return np.asarray(y).astype(np.float32)
    finally:
        jax.config.update("jax_enable_x64", x64)


def run_op(name, f, kinds, cls, bound):
    xs = [inp(k) for k in kinds]
    xj = [jnp.asarray(x) for x in xs]
    xla = np.asarray(jax.jit(jax.vmap(f))(*xj))
    ex = exact_ref(f, xs) if cls in ("cr", "lib") else None
    rows = []
    for k, math in enumerate(MATHS):
        g = tm.vmap(lambda t, *a: (t, f(*a)), outputs={"_": tm.Sum((1,))}, in_axes=(0,) * len(xs), backend="auto",
                    cuda_math=math)
        got = np.asarray(jax.jit(g)(*xj).per_item)
        mx, ux, nx = diff(xla, got)
        row = dict(op=name, cls=cls, math=math, n=int(np.asarray(xla).size), xla_mismatch=mx, xla_max_ulp=ux,
                   tm_sha=hashlib.sha1(np.ascontiguousarray(got).tobytes()).hexdigest()[:16])
        if cls == "exact":
            row["ok"] = mx == 0
        elif cls in ("cr", "lib"):
            me, ue, ne = diff(ex, got)
            row.update(exact_mismatch=me, exact_max_ulp=ue, exact_nonfinite=ne, bound=bound[k])
            row["ok"] = ne == 0 and (ue or 0) <= bound[k]
        else:
            row["ok"] = True
        rows.append(row)
    return rows


def check_names():
    """D16: every op-table name is a primitive jax.extend.core.primitives exports, or explained in
    backends/cuda/census.NOT_IN_JAX_EXTEND (a JAX rename shows up here, not in a user's UnsupportedError)."""
    from jax.extend.core import Primitive
    from jax.extend.core import primitives as P

    from tallymap.backends.kernel import canon, census, op_table
    jaxnames = {v.name for v in vars(P).values() if isinstance(v, Primitive)}
    ours = set(op_table.RULES) | set(canon.SUGAR)
    bad = sorted(ours - jaxnames - set(census.NOT_IN_JAX_EXTEND))
    print("NAMES", json.dumps(dict(ours=len(ours), exported=len(ours & jaxnames), explained=len(ours - jaxnames),
                                   unexplained=bad)), flush=True)
    return not bad


GPU = jax.default_backend() == "gpu"
MATHS = ("ieee", "approx") if GPU else ("ieee",)


def _device():
    if GPU:
        return jax.devices()[0].device_kind
    try:                                       # the host target: the CPU model (the kernels are -march=native)
        with open("/proc/cpuinfo") as f:
            return "cpu " + next(ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name"))
    except (OSError, StopIteration):
        import platform
        return "cpu " + platform.machine()


def main():
    names_ok = check_names()
    dev = _device()
    key = f"{dev} | jax {jax.__version__}"
    print("ENV", json.dumps(dict(jax=jax.__version__, device=dev, n=N)), flush=True)
    rows = []
    for x64 in (False, True):
        jax.config.update("jax_enable_x64", x64)
        for name, (f, kinds, cls, bound) in OPS.items():
            if x64 != name.startswith("f64") or (SEL and name not in SEL):
                continue
            t0 = time.perf_counter()
            try:
                rs = run_op(name, f, kinds, cls, bound)
            except Exception as e:
                rs = [dict(op=name, cls=cls, math="-", ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")]
            for r in rs:
                r["s"] = round(time.perf_counter() - t0, 1)
                print("OP", json.dumps(r), flush=True)
            rows += rs
    jax.config.update("jax_enable_x64", False)
    meas = ("tm_sha", "exact_mismatch", "exact_max_ulp", "exact_nonfinite", "error")    # not xla_*: see the docstring
    cur = {f"{r['op']}/{r['math']}": {k: r.get(k) for k in meas if k in r} for r in rows}
    base = json.load(open(BASE)) if os.path.exists(BASE) else {}
    if RECORD:
        base[key] = dict(date=time.strftime("%Y-%m-%d"), n=N, rows=cur)
        json.dump(base, open(BASE, "w"), indent=1, sort_keys=True)
        print("RECORDED", key, len(cur), "rows", flush=True)
    ref = base.get(key)
    changed = []
    if ref is None:
        print(f"no baseline for {key!r} (have {sorted(base)}): bounds only", flush=True)
    elif not SEL and ref.get("n") == N:
        changed = sorted(k for k in set(cur) | set(ref["rows"]) if cur.get(k) != ref["rows"].get(k))
        for k in changed:
            print("CHANGED", k, "baseline", json.dumps(ref["rows"].get(k)), "now", json.dumps(cur.get(k)), flush=True)
    bad = [f"{r['op']}/{r['math']}" for r in rows if not r["ok"]]
    for b in bad:
        print("OUT OF BOUND", b, flush=True)
    judged = [r for r in rows if r.get("cls") != "info"]
    n_ok = sum(r["ok"] for r in judged) - len(changed)
    tot = len(judged)
    print(f"OPS {len(OPS)} ops x {len(MATHS)} math modes; judged {tot}, out of bound {len(bad)}, changed vs baseline "
          f"{len(changed)}", flush=True)
    ok = not bad and not changed and names_ok
    print(f"ALL {'PASS' if ok else 'FAIL'} {max(n_ok, 0)}/{tot}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
