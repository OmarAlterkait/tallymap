"""Wide-multiply chains on cuda.  python tests/mulchain.py   (GPU; enables jax_enable_x64 itself)

NVVM is exponential in the depth of a serial chain of 64-bit integer multiplies inside one function (127 s at depth
16, > 300 s at 20), so the cuda emitter cuts such a chain every codegen.MULCHAIN_K links with a zero-instruction
barrier (sj_brk).  The item below is ONE chain of DEPTH modular multiply links (the 64x64->128 product in 32-bit
limbs, as a JAX program has to spell it).  This is a STRUCTURAL test: the cuts are in the source and the results
are BITWISE equal to the lockstep backend (integer arithmetic: any difference is a bug, not rounding); the compile
bound is only a sanity limit.  It does not reproduce the cliff: with the rule patched off this item also compiles in
under a second through NVRTC on the CI GPUs (job 43047843).  The off / on compile measurement is a one-off job,
recorded in docs/design.md section 10.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE = tempfile.mkdtemp(prefix="tm_mulchain_cache_", dir=os.environ.get("LSCRATCH") or None)
os.environ["TALLYMAP_CUDA_CACHE"] = _CACHE          # a fresh cache: the kernel is really compiled here
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import tallymap as tm  # noqa: E402
from tallymap.backends.cuda import codegen as CG  # noqa: E402

DEPTH, NK, N = 24, 4, 4096
M61 = (1 << 61) - 1
U = jnp.uint64
RES = []


def mulhi64(a, b):
    al, ah = a & U(0xFFFFFFFF), a >> U(32)
    bl, bh = b & U(0xFFFFFFFF), b >> U(32)
    t0 = al * bl
    t1 = ah * bl + (t0 >> U(32))
    t2 = al * bh + (t1 & U(0xFFFFFFFF))
    return ah * bh + (t1 >> U(32)) + (t2 >> U(32))


def link(cum, a, b):
    """(a * b + cum) mod 2^61 - 1: one link of the chain."""
    lo, hi = a * b, mulhi64(a, b)
    lo2 = lo + cum
    hi = hi + (lo2 < cum).astype(U)
    s1 = (lo2 & U(M61)) + (hi * U(8)) + (lo2 >> U(61))
    return (s1 & U(M61)) + (s1 >> U(61))


def item(t, x, kind):
    cum = U(1) + x
    a = x * U(0x9E3779B97F4A7C15)
    for j in range(DEPTH):
        cum = link(cum, a + U(j + 1), x + U(2 * j + 3))
    return t.add("s", kind, cum).add("n", 0, jnp.int32(1)), cum


def main():
    if jax.default_backend() != "gpu":
        print("SUITE SKIP (no GPU)")
        print("TOTAL 0 PASS 0")
        return 0
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.integers(1, 1 << 62, N, dtype=np.int64)).astype(U)
    kind = jnp.asarray(rng.integers(0, NK, N), jnp.int32)
    out = {"s": tm.Sum((NK,), jnp.uint64), "n": tm.Sum((1,), jnp.int32)}
    run = lambda b: jax.jit(tm.vmap(item, outputs=out, backend=b, cuda_math="ieee"))(x, kind)

    def cell(name, fn):
        try:
            msg = "PASS (" + fn() + ")"
        except AssertionError as e:
            msg = f"FAIL {e}"
        except Exception as e:
            msg = f"RAISE {type(e).__name__}: {str(e)[:300]}"
        RES.append(msg)
        print(f"{name:70s} cuda     {msg}", flush=True)

    tm.diagnostics(reset=True)
    rc = run("cuda")
    kern = tm.diagnostics()["kernels"]

    def compiled():
        assert len(kern) == 1 and not kern[0]["cached"], f"expected one freshly compiled kernel, got {kern}"
        src = open(kern[0]["source"]).read()
        ncut = src.count("= sj_brk(")
        assert ncut >= DEPTH // CG.MULCHAIN_K - 1, f"{ncut} chain cuts for a {DEPTH}-link chain"
        assert kern[0]["compile_s"] < 60, f"compile {kern[0]['compile_s']} s (sanity bound)"
        return f"{ncut} cuts, compile {kern[0]['compile_s']} s"

    def bitwise():
        rl = run("lockstep")
        for k in out:
            assert np.array_equal(np.asarray(rc.outputs[k]), np.asarray(rl.outputs[k])), f"tally {k} differs"
        assert np.array_equal(np.asarray(rc.per_item), np.asarray(rl.per_item)), "per-item chain value differs"
        return f"tallies and {N} per-item values bitwise equal"

    cell(f"wide-multiply chain of {DEPTH} links: cut every {CG.MULCHAIN_K} (structural)", compiled)
    cell("wide-multiply chain: cuda == lockstep bitwise", bitwise)
    print("TOTAL", len(RES), "PASS", sum(m.startswith("PASS") for m in RES))
    return 0 if all(m.startswith("PASS") for m in RES) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        shutil.rmtree(_CACHE, True)
