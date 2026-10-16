"""Runner for the plain-JAX comparison suite.

    python benchmarks/vs_jax/run.py --problem random_walk
    python benchmarks/vs_jax/run.py --problem all --quick          # the CPU smoke sweep (minutes)
    python benchmarks/vs_jax/run.py --list

One measurement is one CELL, and every cell runs in a fresh subprocess, because the device peak-memory counter is
cumulative for the life of a process.  A cell prints exactly one `ROW {json}` line; the driver collects them and
merges them into `results/<platform>_<jax>.json`, keyed by the cell's identity, newest run winning.
(Set `VSJ_RESULTS` to write that file somewhere else; a batch job runs from a snapshot of the commit, so it
points it back at the checkout.)

The sweep is deliberately asymmetric, so that it finishes.  The FIRST divergence level of a problem gets the
whole size sweep in every mode; the other levels get one size (the problem's `DIV_VAL`) in the forward and
gradient modes only.  That is enough for the two questions the tables ask: how the cost scales with N, and how
it moves with divergence.

For each (axis value, divergence level) the driver first runs a `probe` cell, which measures the trip count of the
item loop with an UNBOUNDED loop construct.  The padded-scan arms are then told that observed maximum: the
`jax_scan` / `jax_ckpt` arms pad to exactly it (an oracle baseline that a real program cannot achieve) and the
`...2x` arms pad to twice it (what a careful user picks).  No arm is padded to a number pulled out of the air, and
every row carries the padding it used.

A problem module in `problems/` defines:
    NAME TITLE GROUP AXIS SWEEP DIVS ARMS MODES CHECK_VAL [PROBE_KEYS]
    setup(val, div)              -> ctx            build the inputs, measure nothing
    telemetry(ctx)               -> dict           trip-count statistics from an unbounded loop
    measured(ctx, arm, mode, maxs) -> (fn, args, extra)    the callable a cell measures
    check(ctx, maxs)             -> dict           cross-arm agreement at CHECK_VAL

`PROBE_KEYS` names fields of the probe row that the measurement cells need as well (the shower's per-generation
populations, for instance): the driver copies them from the probe row into every later cell's `ctx`, which is
how an arm can be given an oracle it could not have computed itself.

GROUP is `natural` (tallymap is expected to win), `bad-fit` (measured so the documentation can say when not to
use it), `interacting` (items that read a shared state rebuilt between calls) or `floor` (no divergence at all:
what is left when the scheduling advantage is zero).
"""
import argparse
import importlib
import json
import os
import subprocess
import sys
import time

import harness as H
import jax

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("VSJ_RESULTS") or os.path.join(HERE, "results")
PROBLEMS = ["random_walk", "transport", "walk_on_spheres", "ode_ensemble", "gillespie", "shower", "barrier",
            "uniform", "dense_state", "trajectory", "small_n", "interacting_md", "particle_filter"]
KEYS = ("problem", "arm", "mode", "val", "div")
# --quick is a smoke test, not a measurement: one divergence level, one small size, forward and gradient only,
# and one baseline of each kind.  It has to finish on a CPU in under five minutes.
QUICK_ARMS = ("tm_cuda", "tm_lockstep", "jax_while", "jax_scan2x", "jax_pad2x", "jax_gen2x")
QUICK_MODES = ("fwd", "grad", "score")


def load(name):
    return importlib.import_module("problems." + name)


def cell(args):
    """One measurement, in this process.  Prints one ROW line."""
    p = load(args.problem)
    # the environment goes on the row BEFORE anything can fail, so an OOM or a SKIP row still identifies its
    # machine and its JAX version: the results files are keyed on those
    base = dict(problem=p.NAME, group=p.GROUP, axis=p.AXIS, arm=args.arm, mode=args.mode, val=args.val,
                div=args.div, maxs=args.maxs, **H.environment())

    def go():
        if args.mode == "probe":
            ctx = p.setup(args.val, args.div)
            H.row(**base, status="ok", **p.telemetry(ctx))
            return
        if args.mode == "check":
            ctx = p.setup(args.val, args.div)
            ctx.update(json.loads(args.probe))
            H.row(**base, status="ok", **p.check(ctx, args.maxs))
            return
        ctx = p.setup(args.val, args.div)
        ctx.update(json.loads(args.probe))
        fn, fargs, extra = p.measured(ctx, args.arm, args.mode, args.maxs)
        out, r = H.measure(fn, fargs, reps=args.reps)
        leaves = [x for x in jax.tree.leaves(out) if hasattr(x, "shape")]
        r.update(base, status="ok", **extra,
                 out_norm=float(sum(float((x.astype("float64") ** 2).sum()) for x in leaves) ** 0.5))
        H.row(**r)
    H.guard(go, base)


def one(py, pyargs, timeout):
    """Run one cell subprocess; -> (row | None, stderr tail)."""
    r = subprocess.run([py, "-u", os.path.join(HERE, "run.py"), "--cell"] + pyargs,
                       capture_output=True, text=True, timeout=timeout, cwd=HERE)
    m = None
    for line in r.stdout.splitlines():
        if line.startswith("ROW "):
            m = json.loads(line[4:])
    tail = (r.stdout[-1500:] + r.stderr[-1500:]) if m is None else ""
    return m, tail


def merge(rows):
    """Write/update results/<platform>_<jax>.json from the rows of this run.

    Several jobs may sweep different problems of the same platform at the same time, so each sets VSJ_SUFFIX and
    writes its own file; `run.py --merge` then consolidates them into the canonical name."""
    if not rows:
        return None
    os.makedirs(RESULTS, exist_ok=True)
    rows = [r for r in rows if "platform" in r]
    if not rows:
        return None
    tag = f"{rows[0]['platform']}_{rows[0]['jax'].replace('.', '')}{os.environ.get('VSJ_SUFFIX', '')}"
    path = os.path.join(RESULTS, tag + ".json")
    old = json.load(open(path))["rows"] if os.path.exists(path) else []
    by = {tuple(str(r.get(k)) for k in KEYS): r for r in old}
    for r in rows:
        by[tuple(str(r.get(k)) for k in KEYS)] = r
    out = sorted(by.values(), key=lambda r: tuple(str(r.get(k)) for k in KEYS))
    json.dump({"generated": time.strftime("%Y-%m-%d"), "rows": out}, open(path, "w"), indent=1, sort_keys=True)
    return path


def consolidate():
    """Fold every results/*.json into one file per (platform, jax), keyed by cell identity, and drop the parts."""
    groups, parts = {}, []
    for f in sorted(os.listdir(RESULTS)):
        if not f.endswith(".json"):
            continue
        path = os.path.join(RESULTS, f)
        for r in json.load(open(path))["rows"]:
            if "platform" not in r:
                continue                      # a row from before the environment was put on failure rows
            tag = f"{r['platform']}_{r['jax'].replace('.', '')}"
            groups.setdefault(tag, {})[tuple(str(r.get(k)) for k in KEYS)] = r
        if f[:-5] not in groups:
            parts.append(path)
    for tag, by in groups.items():
        out = sorted(by.values(), key=lambda r: tuple(str(r.get(k)) for k in KEYS))
        json.dump({"generated": time.strftime("%Y-%m-%d"), "rows": out},
                  open(os.path.join(RESULTS, tag + ".json"), "w"), indent=1, sort_keys=True)
        print(f"{tag}.json  {len(out)} rows")
    for path in parts:
        if os.path.basename(path)[:-5] not in groups:
            os.remove(path)
            print("removed part " + os.path.basename(path))
    return 0


def drive(args):
    import arms as A
    names = PROBLEMS if args.problem == "all" else args.problem.split(",")
    py = sys.executable
    rows, fails = [], []
    for name in names:
        p = load(name)
        vals = p.SWEEP["quick" if args.quick else "full"]
        if args.max_val:
            vals = tuple(v for v in vals if v <= args.max_val) or vals[:1]
        divs = p.DIVS[:1] if args.quick and len(p.DIVS) > 1 else p.DIVS
        modes = [m for m in p.MODES if args.modes is None or m in args.modes.split(",")]
        plist = [a for a in p.ARMS if args.arms is None or a in args.arms.split(",")]
        if args.quick:
            vals = vals[-1:]                   # one size: this is a smoke test, not a measurement
            modes = [m for m in modes if m in QUICK_MODES]
            plist = [a for a in plist if a in QUICK_ARMS]
        for i_div, div in enumerate(divs):
            # the size sweep runs at the first divergence level only; the others get one size, fwd and grad
            dv = getattr(p, "DIV_VAL", vals[-2 if len(vals) > 1 else -1])
            dvals = vals if i_div == 0 else (min(dv, max(vals)),)
            dmodes = modes if i_div == 0 else [m for m in modes if m in ("fwd", "grad", "score")]
            maxs_by_val, probe_by_val = {}, {}
            pvals = list(dvals)
            if args.check and hasattr(p, "check") and p.CHECK_VAL not in pvals:
                pvals.append(p.CHECK_VAL)      # the agreement cell needs its own trip counts too
            for v in pvals:
                r, tail = one(py, ["--problem", name, "--arm", p.ARMS[0], "--mode", "probe", "--val", str(v),
                                   "--div", div, "--maxs", "0"], args.timeout)
                if r is None or r.get("status") != "ok":
                    fails.append(f"{name}/{div}/val={v} probe: {(r or {}).get('error', tail)[:200]}")
                    continue
                rows.append(r)
                maxs_by_val[v] = int(r["steps_max"])
                probe_by_val[v] = json.dumps({k: r[k] for k in getattr(p, "PROBE_KEYS", ()) if k in r})
                if r.get("backend") != "gpu" and "tm_cuda" in plist:
                    plist = [a for a in plist if a != "tm_cuda"]      # CPU: only the lockstep backend exists
                print(f"  probe {name:16s} {div:5s} val={v:<9} steps mean {r['steps_mean']:.1f} "
                      f"max {r['steps_max']}", flush=True)
            if args.check and hasattr(p, "check"):
                r, tail = one(py, ["--problem", name, "--arm", p.ARMS[0], "--mode", "check", "--val",
                                   str(p.CHECK_VAL), "--div", div, "--maxs",
                                   str(maxs_by_val.get(p.CHECK_VAL, 1 << 14)),
                                   "--probe", probe_by_val.get(p.CHECK_VAL, "{}")], args.timeout)
                if r is None:
                    fails.append(f"{name}/{div} check: {tail[:200]}")
                else:
                    rows.append(r)
                    print(f"  check {name:16s} {div:5s} {json.dumps({k: v for k, v in r.items() if 'rel' in k})}",
                          flush=True)
            for v in dvals:
                if v not in maxs_by_val:
                    continue
                for arm in plist:
                    for mode in dmodes:
                        if mode != "fwd" and arm == "jax_while":
                            continue      # JAX cannot reverse-differentiate a data-dependent while_loop
                        maxs = A.pad_for(arm, maxs_by_val[v])
                        t0 = time.time()
                        r, tail = one(py, ["--problem", name, "--arm", arm, "--mode", mode, "--val", str(v),
                                           "--div", div, "--maxs", str(maxs), "--reps", str(args.reps),
                                           "--probe", probe_by_val[v]], args.timeout)
                        if r is None:
                            fails.append(f"{name}/{div}/{arm}/{mode}/val={v}: {tail[:300]}")
                            print(f"  CELL-FAIL {name} {div} {arm} {mode} val={v}", flush=True)
                            continue
                        rows.append(r)
                        st = r.get("status")
                        mb = r.get("peak_mb")
                        print(f"  {name:16s} {div:5s} {arm:12s} {mode:7s} val={v:<9} "
                              + (f"{r['ms']:9.3f} ms  {'      -' if mb is None else format(mb, '7.1f')} MB  "
                                 f"pad={r.get('padded_steps', '-')}"
                                 if st == "ok" else f"{st}: {str(r.get('error'))[:90]}")
                              + f"   [{time.time() - t0:.0f}s]", flush=True)
    path = merge(rows)
    print(f"\n{len(rows)} rows -> {path}", flush=True)
    for f in fails:
        print("FAIL " + f, flush=True)
    skipped = sum(r.get("status") == "SKIP" for r in rows)
    print(f"VS_JAX {'FAIL' if fails else 'PASS'} {len(rows)} rows, {skipped} skipped, {len(fails)} failed cells",
          flush=True)
    return 1 if fails else 0


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="all")
    ap.add_argument("--quick", action="store_true", help="the small sweep: one divergence level, small sizes")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--merge", action="store_true",
                    help="fold results/*.json into one file per (platform, jax) and drop the parts")
    ap.add_argument("--arms", default=None, help="comma list; default every arm the problem supports")
    ap.add_argument("--modes", default=None)
    ap.add_argument("--reps", type=int, default=0, help="0 = 7, or 3 with --quick")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--max-val", dest="max_val", type=int, default=0,
                    help="skip sweep points above this (the CPU sweep, where only the lockstep backend exists)")
    ap.add_argument("--check", type=int, default=1)
    ap.add_argument("--cell", action="store_true", help="internal: run ONE measurement in this process")
    ap.add_argument("--arm", default="tm_cuda")
    ap.add_argument("--mode", default="fwd")
    ap.add_argument("--val", type=int, default=1 << 16)
    ap.add_argument("--div", default="mid")
    ap.add_argument("--maxs", type=int, default=0)
    ap.add_argument("--probe", default="{}", help="internal: probe-row fields the problem's PROBE_KEYS asks for")
    a = ap.parse_args(argv)
    a.reps = a.reps or (3 if a.quick else 7)
    if a.list:
        for n in PROBLEMS:
            p = load(n)
            print(f"{n:16s} {p.GROUP:12s} axis={p.AXIS} divs={','.join(p.DIVS)} arms={len(p.ARMS)}  {p.TITLE}")
        return 0
    if a.merge:
        return consolidate()
    if a.cell:
        cell(a)
        return 0
    return drive(a)
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
