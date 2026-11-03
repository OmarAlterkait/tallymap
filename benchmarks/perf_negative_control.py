"""The perf comparator's negative control (run once per baseline), on a GPU.

benchmarks/perf.py flags a row only above PERF_TOL = 0.5 (the project's 1.5x bar), so the control must inject a
slowdown well above 1.5x:

  DEFECT: the same toy program on 4x the items (N = 4M instead of 1M).  Both reference GPUs are saturated at 1M
     items (2080 Ti: 68 SMs x 1024 = 69,632 resident threads; L40S: 142 x 1536), so every row is ~4x slower: a
     known >= 2x slowdown.  Instrument (docs/validation.md: a defect must be shown to change the observable): at
     least two rows must move by >= 2x, else the control is VOID.  Verdict: PASS iff perf.py flags a row other
     than the bimodal toy_jacfwd8 (4.9 or 6.9 ms across processes for the same program).
  INFO: cuda_math='ieee' against the approx baseline.  The A/B rule (tm.assert_arms_differ on fwd / jvp / grad)
     first shows the two arms compile to different kernels; then the ratios are printed.  MEASURED 2026-10-14: the
     toy rows do not move under ieee (well inside 10%), so ieee is NOT a usable defect at any tolerance; it is kept
     as a report only.
Verdict line: "NEGCTRL PASS|VOID|FAIL ...".
"""
import os
import re
import subprocess
import sys

import jax
import jax.numpy as jnp

import tallymap as tm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("N", str(1 << 16))
os.environ["MODES"] = "none"                     # import the toy program only (run no benchmark rows)
import toy_bench as T  # noqa: E402  (the toy rows' own program)

N_DEFAULT = 1 << 20                              # toy_bench's default N: what the baseline was recorded with
SCALE = 4


def arm(math):
    f = tm.vmap(T.walk, outputs={"Q": tm.Sum((T.NB,))}, in_axes=(0, 0, None, None), backend="cuda", cuda_math=math)
    return dict(fwd=lambda th: f(T.xs, T.keys, T.tab, th).outputs["Q"],
                jvp=lambda th: jax.jvp(lambda s: f(T.xs, T.keys, T.tab, s).outputs["Q"], (th,), (th,))[1],
                grad=jax.grad(lambda th: jnp.sum(f(T.xs, T.keys, T.tab, th).outputs["Q"] * T.WQ)))


def perf(**env):
    """run benchmarks/perf.py with these extra environment variables -> (output, {row: ratio}, [flagged rows])"""
    e = dict(os.environ)
    for k in ("N", "MODES"):                     # (this process's import settings, not perf.py's)
        e.pop(k, None)
    e.update(env)
    r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "perf.py")], env=e, capture_output=True, text=True)
    ratios = {m.group(1): float(m.group(2)) for m in
              re.finditer(r"^(toy_\w+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)", r.stdout, re.M)}
    flagged = [line.split()[0] for line in r.stdout.splitlines() if line.rstrip().endswith("REGRESSION")]
    return r.stdout[-3000:] + r.stderr[-2000:], ratios, flagged


# INFO: the ieee arm (A/B rule first: the two arms must be different programs)
a, b = arm("approx"), arm("ieee")
for k in a:
    (ka, _), (kb, _) = tm.assert_arms_differ(a[k], b[k], T.TH)
    print("AB", k, "approx kernels", ka, "ieee kernels", kb, flush=True)
out, ratios, flagged = perf(PERF_MATH="ieee")
print(out, flush=True)
print("INFO ieee arm: ratios", ratios, "flagged", flagged or "none", flush=True)

# DEFECT: SCALE x the items, same program
out, ratios, flagged = perf(N=str(SCALE * N_DEFAULT))
print(out, flush=True)
moved = [k for k, r in ratios.items() if r >= 2.0]
# toy_jacfwd8 is BIMODAL across processes (benchmarks/perf.py): a flag on it alone does not show that the comparator
# catches a real slowdown
real = [k for k in flagged if k != "toy_jacfwd8"]
if len(moved) < 2:
    print(f"NEGCTRL VOID (instrument: {SCALE}x the items moved only {moved} by >= 2x; ratios {ratios})", flush=True)
    sys.exit(1)
if real:
    print(f"NEGCTRL PASS ({SCALE}x items: ratios {ratios}; flagged {', '.join(real)})", flush=True)
    sys.exit(0)
print(f"NEGCTRL FAIL (the comparator did not flag a {SCALE}x-items slowdown; ratios {ratios})", flush=True)
sys.exit(1)
