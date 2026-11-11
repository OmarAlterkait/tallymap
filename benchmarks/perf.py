"""Performance safety net: re-measure a small fixed set of timings and flag regressions > 50% (1.5x) against the
recorded baseline for THIS GPU model ($PERF_BASELINE, else perf_baseline.json in the tallymap cache
directory; not in git).

    python benchmarks/perf.py              # compare (exit 1 if any row is > 1.5x slower than the baseline)
    python benchmarks/perf.py --record     # (re)write this GPU's baseline entry
    env PERF_TOL=0.5     PERF_MATH=approx|ieee     PERF_BASELINE (the baseline file)
    (benchmarks/perf_negative_control.py: the comparator's negative control -- a known >= 2x slowdown must FAIL)

The tolerance is the project's bar, 1.5x: a speedup or a slowdown under 1.5x does not matter, and identical reruns
of the same commit already move these rows by ~10% (toy_jacfwd8 is bimodal across processes, see main()), so a
tighter tolerance only reports noise.  The tripwire is for real regressions (a lost fast path is 2x and more).

Rows = jax.jit WALL time per call (median; NOT AOT, because a host callback in the jitted program drops jax.jit off
its C++ fast path and only the wall time shows it); toy rows: min over 3 processes; cuda_math="approx", the default:
    toy    walk over a 256-bin table, 1M items: fwd, jvp, jacfwd T=8, grad              (benchmarks/toy_bench.py)
"""
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MATH = os.environ.get("PERF_MATH", "approx")
ROOT = os.path.dirname(HERE)
CACHE = os.environ.get("TALLYMAP_CACHE") or os.path.expanduser("~/.cache/tallymap")
BASE = os.environ.get("PERF_BASELINE") or os.path.join(CACHE, "perf_baseline.json")   # recorded per machine, not in git


def run(cmd, env):
    e = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""), **env)
    r = subprocess.run([sys.executable, "-u"] + cmd, env=e, capture_output=True, text=True, timeout=3600)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
        raise RuntimeError(f"{cmd[0]} failed ({r.returncode})")
    return r.stdout


def toy():
    out = run([os.path.join(HERE, "toy_bench.py")], dict(MODES="fwd,jvp,jacfwd,grad", BK="auto", MATH=MATH))
    names = {"fwd": "fwd", "jvp (primal+tangent)": "jvp", "jacfwd T=8": "jacfwd8", "grad": "grad"}
    rows = {}
    for m in re.finditer(r"^BENCH \S+\s+(.+?)\s+([\d.]+) ms", out, re.M):
        if m.group(1).strip() in names:
            rows["toy_" + names[m.group(1).strip()]] = float(m.group(2))
    return rows


def hardware():
    import jax
    gpu = os.popen("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader").read().strip()
    commit = os.popen(f"git -C {ROOT} rev-parse --short HEAD 2>/dev/null").read().strip()
    dirty = bool(commit) and bool(os.popen(f"git -C {ROOT} status --porcelain -- tallymap").read().strip())
    commit = commit or os.path.basename(ROOT)          # a jobs/ snapshot (.ci/snap/<sha>) has no .git
    # the node name is deliberately not recorded: the GPU model is what the baseline is keyed on
    return dict(gpu=gpu, jax=jax.__version__, jax_file=jax.__file__, commit=commit + ("+dirty" if dirty else ""),
                date=time.strftime("%Y-%m-%d"))


def main(argv):
    tol = float(os.environ.get("PERF_TOL", 0.5))
    hw = hardware()
    key = hw["gpu"].split(",")[0].strip()
    rows = {}
    # toy timings are BIMODAL across processes (jacfwd T=8: 4.96 or 6.84 ms, same compiled program, same node):
    # each row is the MIN over 3 processes (false alarm only if all 3 land in the slow mode)
    t0 = time.time()
    for _ in range(3):
        for k, v in toy().items():
            rows[k] = min(v, rows.get(k, v))
    print(f"[toy {time.time() - t0:.0f}s]", flush=True)
    base = json.load(open(BASE)) if os.path.exists(BASE) else {}
    if "--record" in argv:
        base[key] = dict(hardware=hw, ms=rows, note="jax.jit wall ms per call (min over 3 processes); a host callback "
                         "in the jitted program would show up here")
        json.dump(base, open(BASE, "w"), indent=1, sort_keys=True)
        print("RECORDED", key, json.dumps(rows), flush=True)
        return 0
    ref = base.get(key)
    if ref is None:
        print(f"no baseline for {key!r} (have {sorted(base)}); run with --record", flush=True)
        return 1
    bad = []
    print(f"{'row':18s} {'baseline':>9s} {'now':>9s} {'ratio':>6s}   (baseline {ref['hardware']['commit']}, "
          f"{ref['hardware']['date']}, jax {ref['hardware']['jax']})", flush=True)
    for k in sorted(set(rows) | set(ref["ms"])):
        b, n = ref["ms"].get(k), rows.get(k)
        if b is None or n is None:
            print(f"{k:18s} {b!s:>9} {n!s:>9}   (missing)", flush=True)
            continue
        r = n / b
        flag = "REGRESSION" if r > 1 + tol else ("faster" if r < 1 / (1 + tol) else "")
        if flag == "REGRESSION":
            bad.append(k)
        print(f"{k:18s} {b:9.2f} {n:9.2f} {r:6.2f}  {flag}", flush=True)
    print("PERF", "FAIL " + ",".join(bad) if bad else "PASS", json.dumps(hw), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] + os.environ.get("PERF_ARGS", "").split()))
