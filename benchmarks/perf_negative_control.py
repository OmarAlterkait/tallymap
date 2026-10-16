"""The perf comparator's negative control (run once per baseline), on a GPU:

  1. A/B rule: the toy walk compiled with cuda_math='ieee' and 'approx' must be DIFFERENT programs (else the control
     is void): tm.assert_arms_differ on fwd / jvp / grad.
  2. The defect must change the observable: benchmarks/perf.py with PERF_MATH=ieee against the approx baseline must
     flag a row (ieee div / sqrt are slower).  Prints the ratios.  A flag on the bimodal toy_jacfwd8 row alone is
     VOID (measured 2026-10-14: the toy rows do not move under ieee).
Verdict line: "NEGCTRL PASS|VOID|FAIL ...".
"""
import os
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


def arm(math):
    f = tm.vmap(T.walk, outputs={"Q": tm.Sum((T.NB,))}, in_axes=(0, 0, None, None), backend="cuda", cuda_math=math)
    return dict(fwd=lambda th: f(T.xs, T.keys, T.tab, th).outputs["Q"],
                jvp=lambda th: jax.jvp(lambda s: f(T.xs, T.keys, T.tab, s).outputs["Q"], (th,), (th,))[1],
                grad=jax.grad(lambda th: jnp.sum(f(T.xs, T.keys, T.tab, th).outputs["Q"] * T.WQ)))


a, b = arm("approx"), arm("ieee")
for k in a:
    (ka, _), (kb, _) = tm.assert_arms_differ(a[k], b[k], T.TH)
    print("AB", k, "approx kernels", ka, "ieee kernels", kb, flush=True)
env = dict(os.environ, PERF_MATH="ieee")
env.pop("N", None)
env.pop("MODES", None)
r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "perf.py")], env=env, capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-2000:], flush=True)
flagged = [line.split()[0] for line in r.stdout.splitlines() if line.rstrip().endswith("REGRESSION")]
# toy_jacfwd8 is BIMODAL across processes (4.9 or 6.9 ms, the same program: benchmarks/perf.py): a flag on it alone
# does not show that the defect changed the observable
real = [k for k in flagged if k != "toy_jacfwd8"]
if real:
    print("NEGCTRL PASS (the ieee arm is flagged on", ", ".join(real) + ")", flush=True)
elif flagged:
    print("NEGCTRL VOID (only the bimodal toy_jacfwd8 row moved: the ieee defect does not change the toy rows, so"
          " this control cannot show that the comparator would catch a real slowdown)", flush=True)
else:
    print("NEGCTRL FAIL (the comparator did not flag the ieee arm)", flush=True)
sys.exit(0 if real else 1)
