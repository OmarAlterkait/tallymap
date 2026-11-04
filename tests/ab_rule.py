"""The A/B rule (docs/validation.md): before comparing two arms of an option (timings, numerics), assert that they
compiled to different programs -- different cuda kernels, or for the lockstep backend different StableHLO.
Comparing two identical programs measures noise.  Measurement hygiene for the library's own developers, not API.

    from ab_rule import assert_arms_differ
    (ka, ha), (kb, hb) = assert_arms_differ(f_a, f_b, *args)
"""
import tallymap as tm


def artifacts(fn, *args, **kwargs):
    """(sorted cubin hashes of the cuda kernels, StableHLO sha256) that jax.jit(fn)(*args) compiles to."""
    d = tm.diagnostics(fn, *args, **kwargs)
    return tuple(sorted(k["hash"] for k in d["kernels"])), d["stablehlo_sha256"]


def assert_arms_differ(fn_a, fn_b, *args, **kwargs):
    """The A/B rule: two arms of an option comparison must compile to different programs (else the comparison is
    void).  cuda: the kernels' cubin hashes differ; lockstep (no kernels): the StableHLO differs.  -> the two
    artifacts; raises AssertionError when they are identical."""
    ka, ha = artifacts(fn_a, *args, **kwargs)
    kb, hb = artifacts(fn_b, *args, **kwargs)
    same = (ka == kb) if (ka or kb) else (ha == hb)
    if same:
        raise AssertionError("A/B rule: the two arms compiled to identical programs "
                             f"(kernels {list(ka)}, StableHLO {ha[:12]}); the comparison would be void")
    return (ka, ha), (kb, hb)
