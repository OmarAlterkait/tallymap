"""Measurement harness shared by every problem in the plain-JAX comparison suite.

One measurement per PROCESS.  The device peak-memory counter is cumulative for the life of a process, so two arms
measured in one process report each other's peak; `run.py` therefore launches a fresh subprocess per cell and this
module prints exactly one `ROW {json}` line per cell.

Timing protocol (the same one benchmarks/toy_bench.py uses):
  * a warm-up call, then the MEDIAN of at least `reps` calls, each with jax.block_until_ready on the result;
  * `jax.jit` wall time per call, which is what a user sees (an AOT-compiled call hides host overhead);
  * compile time reported separately as `lower_s` (trace and lower, including any kernel build) and
    `xla_compile_s`;
  * peak device memory from the device allocator's own `peak_bytes_in_use` counter.

Every row also carries the device name, the JAX version and the git commit, so a results file identifies the
machine it was measured on.
"""
import json
import os
import statistics
import subprocess
import sys
import time

import jax
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def device():
    return jax.local_devices()[0]


def peak_mb():
    try:
        return (device().memory_stats() or {}).get("peak_bytes_in_use", -1) / 2 ** 20
    except Exception:
        return -1.0


def platform_tag():
    """A short, stable name for the machine class: 'l40s', 'rtx2080ti', 'cpu'."""
    d = device()
    if d.platform != "gpu":
        return "cpu"
    name = (getattr(d, "device_kind", "") or "gpu").lower()
    for k, v in (("l40s", "l40s"), ("2080", "rtx2080ti"), ("a100", "a100"), ("h200", "h200")):
        if k in name:
            return v
    return "".join(c for c in name.replace(" ", "") if c.isalnum())


def _git(args):
    try:
        return subprocess.run(["git", "-C", ROOT] + args, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""


def environment():
    """Identity of the machine and the code under test; goes on every row."""
    commit = _git(["rev-parse", "--short=10", "HEAD"]) or os.path.basename(ROOT)
    if _git(["status", "--porcelain", "--", "tallymap", "benchmarks"]):
        commit += "+dirty"
    import tallymap as tm
    d = device()
    return dict(platform=platform_tag(), device=getattr(d, "device_kind", d.platform), backend=d.platform,
                jax=jax.__version__, tallymap=tm.__version__, commit=commit,
                date=time.strftime("%Y-%m-%d"))


def measure(fn, args, reps=7, min_total_s=0.5, max_reps=40, compile_time=True):
    """-> (output, row).  Median warm jit wall time, compile time separately, peak device memory."""
    r = {}
    jf = jax.jit(fn)
    if compile_time:
        t = time.perf_counter()
        lowered = jf.lower(*args)
        r["lower_s"] = time.perf_counter() - t
        t = time.perf_counter()
        compiled = lowered.compile()
        r["xla_compile_s"] = time.perf_counter() - t
        del compiled, lowered
    t = time.perf_counter()
    out = jax.block_until_ready(jf(*args))
    r["first_call_s"] = time.perf_counter() - t
    ts = []
    t0 = time.perf_counter()
    while len(ts) < reps or (time.perf_counter() - t0 < min_total_s and len(ts) < max_reps):
        t = time.perf_counter()
        out = jax.block_until_ready(jf(*args))
        ts.append((time.perf_counter() - t) * 1e3)
    pk = peak_mb()
    r.update(ms=float(statistics.median(ts)), ms_min=float(min(ts)), ms_max=float(max(ts)), reps=len(ts),
             peak_mb=pk if pk > 0 else None)      # None on CPU: the host allocator reports no peak
    return out, r


def encode(x):
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(f"{float(x):.6g}")        # 6 significant digits: never round a 1e-7 relative error to 0
    if isinstance(x, (list, tuple)):
        return [encode(v) for v in x]
    if isinstance(x, dict):
        return {k: encode(v) for k, v in x.items()}
    return x


def row(**kw):
    print("ROW " + json.dumps(encode(kw)), flush=True)


def rel(a, b):
    """max |a - b| / max |b|, computed in f64."""
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if a.shape != b.shape:
        return float("nan")
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-300))


class Unavailable(Exception):
    """This arm cannot run here: no GPU for the cuda backend, or an optional dependency is missing.  Reported as
    status='SKIP', which is not a failure."""


def is_oom(e):
    s = f"{type(e).__name__}: {e}"
    return any(k in s for k in ("RESOURCE_EXHAUSTED", "out of memory", "Out of memory", "OUT_OF_MEMORY",
                                "CUDA_ERROR_OUT_OF_MEMORY", "Unable to allocate"))


def guard(f, base):
    """Run f(); report an OOM, a SKIP or an error as a row instead of crashing the sweep."""
    try:
        f()
    except Unavailable as e:
        row(**base, status="SKIP", error=str(e)[:200])
        sys.exit(0)
    except Exception as e:                                                      # noqa: BLE001
        kind = "OOM" if is_oom(e) else ("SKIP" if type(e).__name__ == "UnsupportedError" else "ERROR")
        # the LAST 400 characters: a library that raises through a host callback (diffrax does) puts its own
        # message at the end of the traceback, and truncating from the front throws it away
        row(**base, status=kind, error=f"{type(e).__name__}: {e}"[-400:])
        sys.exit(0)


def trip_stats(k):
    """Trip-count telemetry from a per-item step counter."""
    k = np.asarray(k)
    return dict(steps_min=int(k.min()), steps_mean=float(k.mean()), steps_p99=float(np.percentile(k, 99)),
                steps_max=int(k.max()))
