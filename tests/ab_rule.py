"""The A/B rule (docs/validation.md): before comparing two arms of an option (timings, numerics), assert that they
compiled to different programs -- different cuda kernels, or for the xla backend different StableHLO.
Comparing two identical programs measures noise.  Measurement hygiene for the library's own developers, not API.

    from ab_rule import assert_arms_differ, patched
    (ka, ha), (kb, hb) = assert_arms_differ(f_a, f_b, *args)
    with patched(codegen, INPLACE=False):     # an arm that is a TEST-ONLY module constant (codegen.FILL / INPLACE,
        ...                                   # _primitive.LANES, _reverse.SPARSE_MIN / NODENSE, ...)
"""
import contextlib

import tallymap as tm


@contextlib.contextmanager
def patched(mod, **values):
    """Set test-only module constants of `mod` for the block and restore them after.  A constant is read when a
    program is traced (lanes, folding, the reverse program) or lowered (the cuda emitter), so everything the arm
    traces, lowers and runs belongs inside the block.  The eager-call cache (_primitive._EAGER) is keyed on the
    program, not on these constants, so it is cleared on entry and on exit."""
    from tallymap import _primitive
    old = {k: getattr(mod, k) for k in values}
    _primitive._EAGER.clear()
    for k, v in values.items():
        setattr(mod, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(mod, k, v)
        _primitive._EAGER.clear()


def artifacts(fn, *args, **kwargs):
    """(sorted cubin hashes of the cuda kernels, StableHLO sha256) that jax.jit(fn)(*args) compiles to."""
    d = tm.diagnostics(fn, *args, **kwargs)
    return tuple(sorted(k["hash"] for k in d["kernels"])), d["stablehlo_sha256"]


def assert_arms_differ(fn_a, fn_b, *args, patch_b=None, **kwargs):
    """The A/B rule: two arms of an option comparison must compile to different programs (else the comparison is
    void).  cuda: the kernels' cubin hashes differ; xla (no kernels): the StableHLO differs.  patch_b: (module,
    {constant: value}) set while arm b is lowered (an arm that is a test-only module constant).  -> the two
    artifacts; raises AssertionError when they are identical."""
    ka, ha = artifacts(fn_a, *args, **kwargs)
    if patch_b is None:
        kb, hb = artifacts(fn_b, *args, **kwargs)
    else:
        with patched(patch_b[0], **patch_b[1]):
            kb, hb = artifacts(fn_b, *args, **kwargs)
    same = (ka == kb) if (ka or kb) else (ha == hb)
    if same:
        raise AssertionError("A/B rule: the two arms compiled to identical programs "
                             f"(kernels {list(ka)}, StableHLO {ha[:12]}); the comparison would be void")
    return (ka, ha), (kb, hb)
