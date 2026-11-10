"""The kernel backend: item jaxpr -> C (codegen, kernel) -> one compiled kernel, for one of two TARGETS:

  'cuda'  the GPU: NVRTC -> cubin, launched by shim.cc's jax.ffi CUDA handler (compile.py).
  'cpu'   the host: the same source with host.HOST_PRELUDE, the system C++ compiler -> a shared object, run by
          host_shim.cc's jax.ffi CPU handler on one host thread per CUDA thread (host.py).

The target is the platform JAX lowers the computation for (_primitive), never a tallymap decision.

Float semantics on cuda (cuda_math=): 'approx' (DEFAULT) = XLA:GPU's own f32 div/sqrt/rsqrt PTX (div.full /
sqrt.approx / rsqrt.approx), everything else IEEE round-to-nearest without FMA contraction; 'ieee' = correctly
rounded div/sqrt as well: portable.  On cpu every operation is IEEE without FMA contraction, with libm's math
functions; cuda_math is ignored.

run(prog, pi, bc, target=, cuda_math=, schedule=) -> [sum tallies] + [stats i32[5]] + [per-item results]
Unsupported constructs raise tallymap.UnsupportedError on both targets (backend='xla' runs them).
"""
import hashlib
import os

import jax
import jax.numpy as jnp
import numpy as np

from ..._api import UnsupportedError
from ..._compat import key_impl
from ..._jaxpr import digest
from . import codegen
from . import compile as C
from .codegen import CodegenError, is_key_dtype
from .kernel import BLOCK, build_source

_KERNELS = {}


def _const_sig(c):
    if isinstance(c, (int, float, bool, np.generic)) or np.ndim(c) == 0:
        a = np.asarray(c)
        return (a.shape, str(a.dtype), a.tobytes())
    a = np.asarray(c)
    # the VALUE is part of the key: a launch passes the constants of the program its kernel was built from
    # (Layout.cjs / Layout.consts), so two programs that differ only in a constant's value must not share it
    return (a.shape, str(a.dtype), hashlib.sha1(a.tobytes()).hexdigest())


def _key(p, opts):
    # the structural part is _jaxpr.digest (was str(p.cj.jaxpr): JAX's pretty-printer, 95 s per build on DTRAX)
    return (digest(p.cj.jaxpr), tuple(_const_sig(c) for c in p.cj.consts),
            tuple((s.S, str(np.dtype(s.dtype)), tuple(s.lead), s.fixed_point, s.scale, s.kind) for s in p.tallies),
            p.nP, p.nB, p.trunc, None if p.spawn is None else int(p.spawn.capacity), p.nact,
            tuple(sorted(opts.items())),
            codegen.FILL, codegen.INPLACE)          # (the test-only fill byte / in-place switch change the source)


MATH_OPTS = {"ieee": [], "approx": ["-DTALLYMAP_APPROX_DIVSQRT", "--prec-div=false", "--prec-sqrt=false"]}


class Kernel:
    def __init__(self, p, target="cuda", schedule="item", strict=False, math="approx"):
        self.block = BLOCK
        self.target = target
        try:
            self.src, self.L = build_source(p, schedule=schedule, strict=strict)
        except CodegenError as e:
            raise UnsupportedError(str(e)) from None
        note = schedule_note(self.L)
        if target == "cpu":
            from . import host
            self.src = host.source(self.src, self.L)
            so, self.info = host.compile_so(self.src, p.where, note)   # run-or-raise on the stack bound
            self.kid = host.register(so, self.info)
            self.ext = ".cc"
            return
        if math not in MATH_OPTS:
            raise ValueError("tallymap: cuda_math must be 'ieee' or 'approx'")
        cubin, self.info = C.compile_cubin(self.src, C.device_arch(), MATH_OPTS[math])
        C.check_frame(cubin, p.where, note)      # run-or-raise: over CUDA's per-thread local limit never launches
        self.kid = C.register_kernel(cubin, note)
        self.ext = ".cu"


def schedule_note(L):
    """the schedule part of a frame-size or launch out-of-memory error: which schedule ran, and when the item
    schedule ran on a program the step schedule can split, that schedule='step' gives a smaller frame."""
    s = f"The kernel ran the {L.mode} schedule"
    if L.mode == "item" and L.splittable:
        s += (" on an item with one top-level while loop: schedule='step' gives a smaller frame (only the loop "
              "state stays live from step to step)")
    return s + "."


def kernel(p, **opts):
    k = _key(p, opts)
    if k not in _KERNELS:
        _KERNELS[k] = Kernel(p, **opts)
    kern = _KERNELS[k]
    from ..._diagnostics import record_kernel
    src = (os.path.join(C.cache_dir(), "host", f"k_{kern.info['hash']}.cc") if kern.target == "cpu" else
           os.path.join(C.cache_dir(), kern.info["hash"] + ".cu"))
    record_kernel(dict(tag=p.tag, target=kern.target, schedule=kern.L.mode, hash=kern.info["hash"],
                       cached=kern.info["cached"], kid=kern.kid, compile_s=round(kern.info["seconds"], 1),
                       source=src, prims=list(kern.L.prims), options=dict(opts)))
    return kern


def _phys_in(x):
    return jax.random.key_data(x) if is_key_dtype(x.dtype) else x


def run(p, pi, bc, target="cuda", cuda_math="approx", schedule="item", **_):
    """schedule: 'item' (DEFAULT: a thread runs one whole item at a time) | 'step' (a thread is refilled with a new
    item at loop-STEP granularity: the item program must have exactly one top-level while loop, else
    UnsupportedError; a program derived from it (jvp, lanes, an adjoint) uses the step schedule
    where it splits and the item schedule otherwise; tm.diagnostics()['kernels'][i]['schedule'] is what ran).
    Measurements: docs/design.md section 1b.
    cuda_math: 'approx' (DEFAULT: div.full / sqrt.approx / rsqrt.approx, the PTX XLA:GPU itself emits for f32;
    measured 1.5x fwd / 1.65x jvp faster than ieee; validated by the pre-registered compare runs,
    docs/validation.md) |
    'ieee' (correctly rounded f32 div/sqrt: portable)."""
    math = cuda_math or "approx"
    if target == "cpu":
        math = "ieee"                       # (no approx PTX on the host: every operation is IEEE)
    # strict: the user's own item program (tag 'primal') must split under schedule='step'
    kern = kernel(p, target=target, schedule=schedule, strict=schedule == "step" and p.tag == "primal", math=math)
    L = kern.L
    N = int(pi[0].shape[0])
    consts = []
    for c in L.consts:
        if c[0] == "static":
            consts.append(jnp.asarray(c[2]))
        else:
            _, fn, k, shape, dt = c
            consts.append(jnp.asarray(L.cjs[fn].consts[k]).astype(dt).reshape(shape))
    args = [_phys_in(jnp.asarray(x)) for x in pi] + [_phys_in(jnp.asarray(x)) for x in bc] + consts
    rs = []
    outs = [k for k, s in enumerate(p.tallies) if s.kind == "sum"]
    for k in outs:
        s = p.tallies[k]
        rs.append(jax.ShapeDtypeStruct(tuple(s.lead) + (s.S,), s.dtype))
    for k in outs:
        if k in L.r_scr:
            s = p.tallies[k]
            rs.append(jax.ShapeDtypeStruct((3 * int(np.prod(s.lead, dtype=np.int64)) * s.S,), jnp.uint32))  # u64 + flag
    rs.append(jax.ShapeDtypeStruct((2,), jnp.uint32))          # claim counter (u64)
    rs.append(jax.ShapeDtypeStruct((5,), jnp.int32))           # stats
    if p.spawn is not None:
        Q = int(p.spawn.capacity)
        for sh, dt, _ in L.pav:
            rs.append(jax.ShapeDtypeStruct((Q * max(1, int(np.prod(sh, dtype=np.int64))),), dt))
        rs.append(jax.ShapeDtypeStruct((Q,), jnp.uint32))
        rs.append(jax.ShapeDtypeStruct((8,), jnp.uint32))
    for sh, dt, _ in L.yav:
        rs.append(jax.ShapeDtypeStruct((N,) + tuple(sh), dt))
    if target == "cpu":
        from . import host
        call = jax.ffi.ffi_call(host.TARGET, rs, has_side_effect=False)
        res = call(*args, kernel_id=np.int64(kern.kid), n_zero=np.int32(L.n_zero), flags=np.int32(L.flags),
                   n_items=np.int64(N))
    else:
        call = jax.ffi.ffi_call(C.TARGET, rs, has_side_effect=False)
        res = call(*args, kernel_id=np.int64(kern.kid), block=np.int32(kern.block), n_zero=np.int32(L.n_zero),
                   flags=np.int32(L.flags), grid=np.int32(0), n_items=np.int64(N))
    nO = len(outs)
    nscr = len(L.r_scr)
    stats = res[nO + nscr + 1]
    ys = list(res[len(res) - len(L.yav):]) if L.yav else []
    yav = [v.aval for v in p.cj.jaxpr.outvars[p.nT:]]
    ys = [jax.random.wrap_key_data(y, impl=key_impl(a.dtype).name) if is_key_dtype(a.dtype) else y
          for y, a in zip(ys, yav)]
    return list(res[:nO]) + [stats] + ys
