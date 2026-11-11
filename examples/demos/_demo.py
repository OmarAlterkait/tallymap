"""The few helpers the demos share: a timer, a relative error, a check table, and the library guards."""
import importlib
import statistics
import sys
import time

import jax
import numpy as np


def need(lib, what="this demo"):
    """Import `lib`, or print that `what` needs it and exit 0: a missing optional library is not a failure."""
    try:
        return importlib.import_module(lib)
    except ImportError:
        print(f"SKIPPED: {what} needs {lib} (pip install -r requirements.txt)")
        sys.exit(0)


def have(lib):
    """Import `lib` if it is installed, else say its arm is skipped and return None."""
    try:
        return importlib.import_module(lib)
    except ImportError:
        print(f"skipping the {lib} arm: it needs {lib} (pip install -r requirements.txt)")
        return None


def rel(a, b):
    """max |a - b| / max |b|, in f64."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-300))


def ms(fn, *args, reps=5):
    """Median warm wall time of `jax.jit(fn)(*args)` in milliseconds, compile excluded."""
    f = jax.jit(fn)
    jax.block_until_ready(f(*args))
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(f(*args))
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def timings(n, arms):
    """arms: {label: (fn, args)}, tallymap first.  Prints one line per arm with its ratio to the first."""
    t = {k: ms(f, *a) for k, (f, a) in arms.items()}
    first = next(iter(t))
    print(f"\ntiming at N = {n:,} on {jax.devices()[0].device_kind} (median ms per call, compile excluded)")
    for k, v in t.items():
        print(f"  {k:<30s} {v:10.2f} ms" + ("" if k == first else f"   {v / t[first]:.1f}x {first}"))


def verdict(checks):
    """checks: [(name, ok, detail)].  Prints the table and returns the exit code."""
    bad = [c for c in checks if not c[1]]
    w = max(len(c[0]) for c in checks)
    print()
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{w}}  {detail}")
    print(f"\n{'PASS' if not bad else 'FAIL'}: {len(checks) - len(bad)}/{len(checks)} checks passed")
    return 0 if not bad else 1
