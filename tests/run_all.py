"""The ONE test entry point.

    python tests/run_all.py            # xla backend AND the kernel backend on the platform JAX picks (CPU or GPU)
    python tests/run_all.py --gpu      # the same; --gpu only labels the summary line
    python tests/run_all.py --x64      # with jax_enable_x64
    python tests/run_all.py --unfused  # the two-node jvp design (what runs if JAX drops the partial-eval registries)
    python tests/run_all.py --t3       # the single-loop adjoint forced wherever a program has its shape (TX_T3=1)
    python tests/run_all.py --canary   # DeprecationWarning / FutureWarning are errors (the JAX-upgrade canary)
    python tests/run_all.py --only reverse,reverse_spawn
    env RUN_ALL="--gpu --x64" (same flags), TEST_LOG=<prefix> (full output of a failing suite -> <prefix>.<suite>.log)

Suites: edge (edge cases), transforms (composability matrix), reverse (reverse mode), reverse_spawn (tree adjoint),
invariants (schedule invariance, cache-key soundness, run-or-raise, AD self-consistency, truncation),
dice (DiCE gradients of discrete outcomes against closed forms, float score leaves through spawn),
ops (kernel op table vs jit(vmap(op)) and the host-exact result), mulchain (a 64-bit multiply chain compiles fast
and matches xla bitwise), examples (examples/*.py run as scripts).
Each suite runs in its own process (fresh JAX state).  A suite passes iff its summary line was printed and every cell
is PASS, an EXPECTED raise (RAISES-OK) or an expected failure (XFAIL: a known library defect pinned by the suite);
anything else (FAIL, unexpected RAISE, XPASS -- an XFAIL cell that now passes, so its marker must go --, crash,
timeout) fails the run.  Summary line: TOTAL n PASS k [RAISES-OK m] [XFAIL x].
Exit status 0 = all suites pass.  The full suite takes about 15 minutes on CPU.
On GPU the suites run with XLA_PYTHON_CLIENT_PREALLOCATE=false unless you set it (see main()).
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUITES = ["edge", "transforms", "reverse", "reverse_spawn", "invariants", "dice", "ops", "mulchain", "examples"]
EXAMPLES = ["quickstart", "photon_toy", "shower_toy", "ensemble_toy", "walk_on_spheres",
            "ode_ensemble", "gillespie", "interacting_particles", "particle_filter", "gradient_pitfall"]


def verdict(name, out, rc):
    if name in ("reverse_spawn", "ops", "examples"):
        m = re.search(r"^ALL (PASS|FAIL) (\d+)/(\d+)", out, re.M)
        if not m:
            return False, f"no summary (exit {rc})"
        return m.group(1) == "PASS" and rc == 0, f"{m.group(2)}/{m.group(3)} pass"
    m = re.search(r"^TOTAL (\d+) PASS (\d+)(?: RAISES-OK (\d+))?(?: XFAIL (\d+))?", out, re.M)
    if not m:
        return False, f"no summary (exit {rc})"
    tot, ok, exp, xf = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0), int(m.group(4) or 0)
    # XFAIL (a pinned, known defect) is OK; XPASS is not counted in ok + exp + xf, so it fails the suite
    return (tot > 0 and ok + exp + xf == tot and rc == 0,
            f"{ok} pass + {exp} expected raises" + (f" + {xf} xfail" if xf else "") + f" of {tot}")


def run(cmd, env, timeout):
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        return r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        return (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), "timeout"


def examples(env, timeout):
    """each example must exit 0 and print '<NAME> OK'; -> (output, rc) with an 'ALL PASS k/n' summary."""
    outs, ok = [], 0
    names = [n for n in EXAMPLES if os.path.exists(os.path.join(ROOT, "examples", n + ".py"))]
    for n in names:
        out, rc = run([sys.executable, "-u", os.path.join(ROOT, "examples", n + ".py")], env, timeout)
        good = rc == 0 and re.search(rf"^{n.upper()} OK", out, re.M) is not None
        ok += good
        outs.append(out + f"\n{n} {'PASS' if good else 'FAIL'} (exit {rc})\n")
    s = "".join(outs) + f"ALL {'PASS' if ok == len(names) else 'FAIL'} {ok}/{len(names)}\n"
    return s, 0


def main(argv):
    argv = list(argv) + os.environ.get("RUN_ALL", "").split()        # (batch jobs pass flags through the env)
    gpu, x64, unfused, canary = "--gpu" in argv, "--x64" in argv, "--unfused" in argv, "--canary" in argv
    t3 = "--t3" in argv
    only = next((a.split("=", 1)[1] if "=" in a else argv[argv.index(a) + 1] for a in argv if a.startswith("--only")),
                None)
    suites = [s for s in SUITES if (only is None or s in only.split(","))]
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    # the kernel backend runs on every platform JAX can lower it for (backend='auto': cuda on a GPU, the CPU target
    # elsewhere), against the xla backend as the reference
    env["BACKENDS"] = "xla,auto"
    env["TREE_BACKEND"] = "auto"
    # JAX preallocates 75% of the GPU; the forced-dense reverse cases launch a kernel whose per-thread local memory
    # (~17 GiB on an L40S) the CUDA driver reserves outside that pool, so they need it off.  A value you set wins.
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if x64:
        env["TEST_X64"] = "1"
    if unfused:
        env["TX_UNFUSED"] = "1"
    if t3:
        env["TX_T3"] = "1"
    if canary:      # every suite process: a deprecation in the APIs tallymap (or the tests) use fails the run
        env["PYTHONWARNINGS"] = "error::DeprecationWarning,error::FutureWarning"
    timeout = int(os.environ.get("TEST_TIMEOUT", 3600))
    res = []
    for s in suites:
        t0 = time.time()
        if s == "examples":
            out, rc = examples(env, timeout)
        else:
            out, rc = run([sys.executable, "-u", os.path.join(HERE, f"{s}.py")], env, timeout)
        ok, msg = verdict(s, out, rc)
        bad = [l for l in out.splitlines() if re.match(r".*\s(FAIL|RAISE|XPASS)\s", l) and "RAISES-OK" not in l][:10]
        res.append((s, ok, msg, time.time() - t0))
        print(f"{s:13s} {'PASS' if ok else 'FAIL'}  {msg}  [{time.time() - t0:.0f}s]", flush=True)
        for l in bad if not ok else []:
            print("    " + l[:200], flush=True)
        if not ok and os.environ.get("TEST_LOG"):
            open(f"{os.environ['TEST_LOG']}.{s}.log", "w").write(out)
    import jax
    print(f"ALL {'PASS' if all(r[1] for r in res) else 'FAIL'}  jax {jax.__version__} ({jax.__file__})"
          f"  {'gpu' if gpu else 'cpu'}{' x64' if x64 else ''}{' unfused' if unfused else ''}{' t3' if t3 else ''}"
          f"{' canary' if canary else ''}", flush=True)
    return 0 if all(r[1] for r in res) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
