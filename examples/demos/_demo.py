"""Measurement, bookkeeping and plotting shared by the demos in this directory.

Each demo's `run.py` is a driver with three entry points:

  --quick   one process, CPU, small N: runs every arm, checks agreement, prints PASS or FAIL.
  --full    the sweep.  It re-execs `run.py --one ...` once per (arm, mode, N) so that each measurement
            gets a fresh process and the device peak-memory counter means something, collects the one
            `ROW {json}` line each subprocess prints, and writes `results/<demo>.json` plus the plots.
  --one     a single measurement.  Prints exactly one `ROW {json}` line.

Timing protocol, the same one the library's own benchmarks use: median `jax.jit` wall time per call after a
warm-up, with `jax.block_until_ready` on the result; compile time measured separately and split into
`lower_s` (trace, plus any kernel build) and `xla_compile_s`; peak device memory from the device allocator's
own `peak_bytes_in_use` counter.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import jax
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------------------------------ measurement
def peak_mb():
    try:
        s = jax.local_devices()[0].memory_stats() or {}
    except Exception:
        return -1.0
    return s.get("peak_bytes_in_use", -1) / 2 ** 20


def measure(fn, args, reps=7, min_total_s=0.5, max_reps=40):
    """-> (output, row dict).  Warm median jit wall time, compile time separately, peak device bytes."""
    r = {}
    jf = jax.jit(fn)
    t = time.perf_counter()
    lo = jf.lower(*args)
    r["lower_s"] = time.perf_counter() - t
    t = time.perf_counter()
    c = lo.compile()
    r["xla_compile_s"] = time.perf_counter() - t
    del c, lo
    o = jax.block_until_ready(jf(*args))
    ts = []
    t0 = time.perf_counter()
    while len(ts) < reps or (time.perf_counter() - t0 < min_total_s and len(ts) < max_reps):
        t = time.perf_counter()
        o = jax.block_until_ready(jf(*args))
        ts.append((time.perf_counter() - t) * 1e3)
    r.update(ms=float(statistics.median(ts)), ms_min=float(min(ts)), ms_max=float(max(ts)),
             reps=len(ts), peak_mb=peak_mb())
    return o, r


def rel(a, b):
    """max |a - b| / max |b|, in f64."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-300))


def _enc(x):
    if isinstance(x, (np.floating, float)):
        return float(f"{float(x):.6g}")
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return [_enc(v) for v in x.ravel().tolist()]
    if isinstance(x, (list, tuple)):
        return [_enc(v) for v in x]
    return x


def row(**kw):
    """Print one measurement as a single `ROW {json}` line, which the --full driver collects."""
    print("ROW " + json.dumps({k: _enc(v) for k, v in kw.items()}), flush=True)


def oom(e):
    s = f"{type(e).__name__}: {e}"
    return any(x in s for x in ("RESOURCE_EXHAUSTED", "out of memory", "Out of memory",
                                "OUT_OF_MEMORY", "CUDA_ERROR_OUT_OF_MEMORY"))


def guard(f, base=None):
    """Run f(); turn an OOM or an unsupported program into a reported row instead of a dead sweep."""
    try:
        f()
    except Exception as e:                                                    # noqa: BLE001
        row(**(base or {}), status="OOM" if oom(e) else "ERROR",
            error=f"{type(e).__name__}: {e}"[:400])
        sys.exit(0)


# ------------------------------------------------------------------------------------------ environment
# a few of the incumbents carry no __version__ attribute, so fall back to the installed distribution
_DIST = {"crn_jax": "crn_jax", "gymnax": "gymnax", "ray_trax": None}


def versions(mods):
    """-> {module: version string}.  'not installed' if the import fails, '?' if neither route gives one."""
    import importlib.metadata as md
    out = {}
    for m in mods:
        try:
            mod = __import__(m)
        except Exception:
            out[m] = "not installed"
            continue
        v = getattr(mod, "__version__", None)
        if v is None:
            try:
                v = md.version(_DIST.get(m, m))
            except Exception:
                v = "?"
        out[m] = v
    return out


def commit():
    """The commit the demo tree was measured from: $TALLYMAP_DEMO_COMMIT, then a `.commit` file written by a
    `git archive` snapshot, then `git rev-parse` when running from a checkout."""
    v = os.environ.get("TALLYMAP_DEMO_COMMIT", "").strip()
    if v:
        return v
    f = os.path.join(HERE, "..", "..", ".commit")
    if os.path.exists(f):
        with open(f) as fh:
            return fh.read().strip() or "unknown"
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=20).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def env(libs=()):
    """What the numbers were measured on.  Goes into results/<demo>.json and into the README."""
    import tallymap as tm
    try:
        d = jax.local_devices()[0]
        dev = f"{d.device_kind} ({d.platform})"
    except Exception:
        dev = "unknown"
    git = commit()
    return dict(device=dev, jax=jax.__version__, tallymap=tm.__version__, tallymap_commit=git,
                python=sys.version.split()[0], libs=versions(libs))


# ------------------------------------------------------------------------------------------ --full driver
def run_jobs(script, jobs, timeout=1800):
    """Re-exec `script --one ...` once per job and collect the ROW lines.

    jobs: list of dicts of command-line flags, e.g. {"arm": "tm", "mode": "fwd", "n": 65536}.
    """
    rows = []
    for j in jobs:
        cmd = [sys.executable, script, "--one"]
        for k, v in j.items():
            if isinstance(v, bool):
                if v:
                    cmd.append(f"--{k}")
            else:
                cmd += [f"--{k}", str(v)]
        print("+ " + " ".join(cmd), flush=True)
        # One retry, because a shared filesystem occasionally fails an import and a lost measurement is a
        # hole in the table.  A measurement that fails twice is reported as a row, not swallowed.
        for attempt in (1, 2):
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            got = [json.loads(ln[4:]) for ln in p.stdout.splitlines() if ln.startswith("ROW ")]
            for ln in p.stdout.splitlines():
                if ln.startswith("#"):
                    print("  " + ln, flush=True)
            if got and all(r.get("status") == "ok" for r in got):
                break
            tail = (p.stdout[-600:] + p.stderr[-1500:]).strip()
            why = got[0].get("error", got[0].get("status")) if got else f"no ROW (exit {p.returncode})"
            print(f"  attempt {attempt} failed: {why}", flush=True)
            if attempt == 2:
                print(f"  giving up:\n{tail}", flush=True)
        rows += got or [dict(j, status="NOROW", error=tail[-400:])]
    return rows


def write_results(out_json, rows, libs=(), extra_meta=None):
    meta = env(libs)
    meta.update(extra_meta or {})
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1, sort_keys=True)
    print(f"\nwrote {out_json}  ({len(rows)} rows)", flush=True)
    return rows, meta


def sweep(script, jobs, out_json, libs=(), extra_meta=None, timeout=1800):
    """run_jobs + write_results: the usual one-stage sweep."""
    return write_results(out_json, run_jobs(script, jobs, timeout), libs, extra_meta)


def load(out_json):
    with open(out_json) as f:
        d = json.load(f)
    return d["rows"], d["meta"]


def sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


# ------------------------------------------------------------------------------------------ plotting
# One panel per mode: forward time, gradient time and peak memory are never mixed in one axes, there is no
# second y axis anywhere, and identity is carried by colour AND marker AND a direct end label.
COLOR = {"tm": "#1f78b4", "tm_exact": "#7fb8da",
         "jax_while": "#d95f02", "jax_scan": "#7570b3", "jax_ckpt": "#66a61e",
         "bj_vmap": "#d95f02", "gx_scan": "#d95f02", "gx_while": "#7570b3",
         "jax_scan_shared": "#7570b3",
         "crnjax": "#d95f02", "crnjax_lean": "#7570b3", "jax_flat": "#66a61e",
         "diffrax": "#e7298a", "quadax": "#e7298a", "raytrax": "#e7298a", "raytrax_b": "#f59bc8",
         "jax_front": "#d95f02", "jax_gen": "#66a61e",
         "while": "#d95f02", "scan": "#7570b3", "ckpt": "#66a61e"}
MARKER = {"tm": "o", "tm_exact": "D", "jax_while": "s", "jax_scan": "^", "jax_ckpt": "v",
          "bj_vmap": "s", "gx_scan": "s", "gx_while": "^", "crnjax": "s", "crnjax_lean": "^",
          "jax_flat": "v", "diffrax": "X", "quadax": "X", "raytrax": "X", "raytrax_b": "P",
          "jax_front": "s", "jax_gen": "v", "while": "s", "scan": "^", "ckpt": "v",
          "jax_scan_shared": "^"}


def panel(path, rows, arms, labels, title, ylabel, xlabel="items N", ykey="ms", xkey="n",
          logx=True, logy=True, note=None):
    """One PNG, one quantity.  `arms`: [arm key]; `labels`: {arm: legend text}.  Returns the path or None."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    ends, fails = [], []
    for a in arms:
        s = sorted([r for r in rows if r.get("arm") == a and r.get("status") == "ok"
                    and r.get(ykey) is not None], key=lambda r: r[xkey])
        for r in rows:
            if r.get("arm") == a and r.get("status") in ("OOM", "ERROR", "NOROW"):
                fails.append((r.get(xkey), a, r.get("status")))
        if not s:
            continue
        xs, ys = [r[xkey] for r in s], [r[ykey] for r in s]
        ax.plot(xs, ys, marker=MARKER.get(a, "o"), color=COLOR.get(a, "k"), lw=1.8, ms=6)
        ends.append((ys[-1], xs[-1], labels.get(a, a), COLOR.get(a, "k")))
    if not ends:
        plt.close(fig)
        return None
    ends.sort(reverse=True)
    for k, (y, x, lbl, col) in enumerate(ends):
        ax.annotate(lbl, xy=(x, y), xytext=(1.03, 0.93 - 0.115 * k), textcoords="axes fraction",
                    color=col, fontsize=8, va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.6, alpha=0.6, shrinkA=2, shrinkB=3))
    if fails:
        ytop = max(e[0] for e in ends)
        for x, a, st in sorted(set(fails), key=lambda z: (z[0] is None, z[0])):
            if x is not None:
                ax.plot([x], [ytop], marker="X", color=COLOR.get(a, "k"), ms=9, clip_on=False)
        note = (note + "  " if note else "") + "X marks an arm that could not run at that N."
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(alpha=0.3, which="both")
    if note:
        ax.annotate(note, (0.0, -0.19), xycoords="axes fraction", fontsize=7, va="top", color="#444444")
    fig.tight_layout(rect=(0, 0.03 if note else 0.0, 0.70, 1))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    kb = os.path.getsize(path) / 1024
    print(f"wrote {path}  ({kb:.0f} KB)", flush=True)
    return path


# ------------------------------------------------------------------------------------------ argument parsing
def parser(**extra):
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true", help="CPU, small N, agreement check, PASS/FAIL")
    g.add_argument("--full", action="store_true", help="the GPU sweep that regenerates results/ and the plots")
    g.add_argument("--one", action="store_true", help="one measurement; prints one ROW line")
    g.add_argument("--plots", action="store_true",
                   help="redraw plots/ from the committed results/, without measuring anything")
    p.add_argument("--arm", default="tm")
    p.add_argument("--mode", default="fwd")
    p.add_argument("--n", type=int, default=1 << 16)
    p.add_argument("--div", default="mid")
    p.add_argument("--reps", type=int, default=7)
    for k, v in extra.items():
        p.add_argument("--" + k, type=type(v), default=v)
    return p


def banner(tag, libs=()):
    import tallymap as tm
    v = " ".join(f"{k} {w}" for k, w in versions(libs).items())
    print(f"# {tag} | jax {jax.__version__} tallymap {tm.__version__} {v} | {jax.devices()} | "
          f"pid {os.getpid()}", flush=True)


def verdict(checks):
    """checks: [(name, ok, detail)].  Prints a table and returns the process exit code."""
    bad = [c for c in checks if not c[1]]
    print()
    w = max(len(c[0]) for c in checks)
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{w}}  {detail}")
    print(f"\n{'PASS' if not bad else 'FAIL'}: {len(checks) - len(bad)}/{len(checks)} checks passed")
    return 0 if not bad else 1


# ------------------------------------------------------------------------------------------ markdown tables
def table(rows, index, arms, labels, ykey="ms", fmt="{:.3g}", index_label="N", best=None, note=""):
    """Render one markdown table: one row per `index` value, one column per arm.

    `index`: (key, [values]).  `best`: a callable over the arm values giving a trailing "x vs best" column.
    An arm that did not run shows its status (OOM, ERROR) instead of a number.  Returns the markdown string.
    """
    key, vals = index
    head = f"| {index_label} | " + " | ".join(labels.get(a, a) for a in arms) + " |"
    if best:
        head += " x vs best baseline |"
    out = [head, "|" + "---|" * (len(arms) + 1 + (1 if best else 0))]
    for v in vals:
        cells, nums = [], {}
        for a in arms:
            s = [r for r in rows if r.get("arm") == a and r.get(key) == v]
            if not s:
                cells.append("-")
                continue
            r = s[0]
            if r.get("status") != "ok" or r.get(ykey) is None:
                cells.append(f"**{r.get('status', 'missing')}**")
                continue
            nums[a] = r[ykey]
            cells.append(fmt.format(r[ykey]))
        line = f"| {v:,} | " if isinstance(v, int) else f"| {v} | "
        line += " | ".join(cells) + " |"
        if best:
            b = best(nums)
            line += (f" **{b:.3g}x** |" if b is not None else " - |")
        out.append(line)
    if note:
        out.append("")
        out.append(note)
    return "\n".join(out)


def ratio_vs_best(tm_arm="tm"):
    """-> f({arm: value}) giving tm's value divided by the best (smallest) other value."""
    def f(nums):
        others = [v for a, v in nums.items() if a != tm_arm]
        if tm_arm not in nums or not others:
            return None
        return min(others) / nums[tm_arm]
    return f
