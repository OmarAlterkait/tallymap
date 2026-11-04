"""cuda backend: item jaxpr -> CUDA C -> NVRTC -> one generic jax.ffi launch handler.

Float semantics (cuda_math=): 'approx' (DEFAULT) = XLA:GPU's own f32 div/sqrt/rsqrt PTX (div.full / sqrt.approx /
rsqrt.approx), everything else IEEE round-to-nearest without FMA contraction; 'ieee' = correctly rounded div/sqrt as
well: the portable reference.

run(prog, pi, bc, cuda_math=None) -> [sum tallies] + [stats i32[5]] + [per-item results]
Unsupported constructs raise tallymap.UnsupportedError (backend='auto' too; backend='lockstep' runs them).
"""
import hashlib
import os

import jax
import jax.numpy as jnp
import numpy as np

from ..._api import UnsupportedError
from ..._compat import key_impl
from . import compile as C
from .codegen import CodegenError, is_key_dtype
from .kernel import BLOCK, build_source

_KERNELS = {}


def _const_sig(c):
    if isinstance(c, (int, float, bool, np.generic)) or np.ndim(c) == 0:
        a = np.asarray(c)
        return (a.shape, str(a.dtype), a.tobytes())
    a = np.asarray(c)
    # the VALUE is part of the key: constant folding bakes closed-over arrays into the generated source
    return (a.shape, str(a.dtype), hashlib.sha1(a.tobytes()).hexdigest())


def _key(p, opts):
    return (str(p.cj.jaxpr), tuple(_const_sig(c) for c in p.cj.consts),
            tuple((s.S, str(np.dtype(s.dtype)), tuple(s.lead), s.exact, s.scale, s.kind) for s in p.tallies),
            p.nP, p.nB, p.trunc, None if p.spawn is None else int(p.spawn.capacity), tuple(sorted(opts.items())))


MATH_OPTS = {"ieee": [], "approx": ["-DTALLYMAP_APPROX_DIVSQRT", "--prec-div=false", "--prec-sqrt=false"]}


class Kernel:
    def __init__(self, p, split=True, math="approx", inplace=True):
        self.block = BLOCK
        try:
            self.src, self.L = build_source(p, split=bool(split), inplace=bool(inplace))
        except CodegenError as e:
            raise UnsupportedError(str(e)) from None
        if math not in MATH_OPTS:
            raise ValueError("tallymap: cuda_math must be 'ieee' or 'approx'")
        cubin, self.info = C.compile_cubin(self.src, C.device_arch(), MATH_OPTS[math])
        self.kid = C.register_kernel(cubin)


def kernel(p, **opts):
    k = _key(p, opts)
    if k not in _KERNELS:
        _KERNELS[k] = Kernel(p, **opts)
    kern = _KERNELS[k]
    from ..._diagnostics import record_kernel
    record_kernel(dict(tag=p.tag, mode=kern.L.mode, hash=kern.info["hash"], cached=kern.info["cached"],
                       kid=kern.kid, compile_s=round(kern.info["seconds"], 1),
                       source=os.path.join(C.cache_dir(), kern.info["hash"] + ".cu"), prims=list(kern.L.prims),
                       options=dict(opts)))
    return kern


def _phys_in(x):
    return jax.random.key_data(x) if is_key_dtype(x.dtype) else x


def run(p, pi, bc, cuda_math="approx", _schedule=None, _cuda_inplace=True, **_):
    """Scheduling: a lane is refilled with a new item at STEP granularity (needs one top-level while loop;
    1.35x faster than whole items on a photon-transport item) unless the program spawns (whole items: 1.5x faster
    on a particle shower); private _schedule
    'step' | 'item' overrides it (measurements and A/B checks only).
    Private _cuda_inplace=False: never update a
    per-item array in place (every dynamic_update_slice copies; the A/B control of the emitter's aliasing analysis).
    cuda_math: 'approx' (DEFAULT: div.full / sqrt.approx / rsqrt.approx, the PTX XLA:GPU itself emits for f32;
    measured 1.5x fwd / 1.65x jvp faster than ieee; validated by the pre-registered compare runs,
    docs/validation.md) |
    'ieee' (correctly rounded f32 div/sqrt: the portable reference)."""
    math = cuda_math or "approx"
    if _schedule not in (None, "step", "item"):
        raise ValueError("tallymap: _schedule must be 'step' or 'item'")
    split = _schedule == "step" or (_schedule is None and p.spawn is None)
    extra = {} if _cuda_inplace else dict(inplace=False)        # (the default leaves the kernel key unchanged)
    kern = kernel(p, split=split, math=math, **extra)
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
            rs.append(jax.ShapeDtypeStruct((2 * int(np.prod(s.lead, dtype=np.int64)) * s.S,), jnp.uint32))  # u64
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
