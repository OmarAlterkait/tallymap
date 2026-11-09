"""tm.diagnostics(): the ONE place to look when something is wrong (and what a bug report should contain).

    tm.diagnostics()                       # environment + the host-side counters + the last programs / kernels
    tm.diagnostics(f, *args)               # ... and what jax.jit(f).lower(*args) builds: item jaxprs, cuda kernels
    tm.diagnostics(reset=True)             # read, then clear the counters and the history

Keys:
  versions   tallymap / jax / jaxlib / numpy / python
  env        TALLYMAP_CUDA_CACHE, JAX_PLATFORMS, XLA_FLAGS, CUDA_VISIBLE_DEVICES (when set), machine
  platform   jax.default_backend(), devices; on a GPU: name, compute capability, driver
  cuda       compiler (NVRTC build or nvcc), cubin cache directory
  reverse    single = whether the last adjoint built is the single-loop one (T3), and if not why not;
             tree   = stats of the last tree adjoint (reverse through spawn), filled only by tm.vmap(...,
                      rev_stats=True) calls (a host callback costs jax.jit's C++ fast path): generations, particles,
                      launches, overflow, drop_risk, replay_mismatch, fwd_mismatch (any of the last four -> NaN
                      gradient)
  programs   the last item programs bound by tm.vmap: tag, item jaxpr (text), and baked_floats: [(shape, dtype,
             size)] of the FLOAT constants the item function closed over, the values a derivative cannot reach
             (a constant's tangent is zero, silently): anything you differentiate must be an argument
  kernels    the last cuda kernels used: tag, schedule, cubin hash, cached CUDA source (.cu), primitives, compile
             seconds, and once a kernel has run `resources`: registers and local memory per thread, static shared
             memory, resident blocks per SM, and the local memory the CUDA driver reserves for every resident
             thread outside JAX's pool (local_reserved_gib: what an out-of-memory error at launch is about)
With f: `programs` / `kernels` are those of f only (JAX's caches are cleared first so that everything is re-traced
and re-lowered), plus `stablehlo_sha256` of the lowered program: of its text with every host-callback descriptor
(the ADDRESS of a registered Python callable, new on every lowering of the same program) replaced by its ordinal.
"""
import collections
import contextlib
import hashlib
import os
import platform
import re
import sys

import jax
import numpy as np

HISTORY = 16
PROGRAMS = collections.deque(maxlen=HISTORY)       # (tag, jaxpr object) of item programs bound by tm.vmap
KERNELS = collections.deque(maxlen=HISTORY)        # dicts recorded by backends/cuda.kernel()
_CAPTURE = []                                      # active diagnostics(f, ...) captures: dict(programs=, kernels=)


def record_program(prog):
    # the jaxpr OBJECT, printed only when diagnostics() is read: str(jaxpr) at every bind was 102 s of a 260 s fully
    # cached build on DTRAX (JAX's pretty-printer).  The object is alive anyway, in the program being bound.
    rec = dict(tag=prog.tag, jaxpr=prog.cj.jaxpr, consts=prog.cj.consts)
    PROGRAMS.append(rec)
    for c in _CAPTURE:
        c["programs"].append(rec)


def _render(rec):
    """a program record as returned by diagnostics(): the jaxpr as text and the float constants (built now, not at
    bind time)."""
    return dict(tag=rec["tag"], jaxpr=str(rec["jaxpr"]), baked_floats=float_consts(rec["consts"]))


def float_consts(consts):
    """[(shape, dtype, size)] of the FLOAT ones among an item program's constants: the values its item function
    closed over, which a derivative cannot reach (a constant's tangent is zero, silently)."""
    from ._rewrite import isfloat
    out = []
    for c in consts:
        dt = c.dtype if hasattr(c, "dtype") else np.asarray(c).dtype
        if isfloat(dt):
            sh = tuple(np.shape(c))
            out.append((sh, str(dt), int(np.prod(sh, dtype=np.int64))))
    return out


def record_kernel(rec):
    KERNELS.append(rec)
    for c in _CAPTURE:
        c["kernels"].append(rec)


@contextlib.contextmanager
def _capture():
    c = dict(programs=[], kernels=[])
    _CAPTURE.append(c)
    try:
        yield c
    finally:
        _CAPTURE.remove(c)


def _versions():
    import jaxlib

    from . import __version__
    return dict(tallymap=__version__, jax=jax.__version__, jaxlib=getattr(jaxlib, "__version__", "?"),
                numpy=np.__version__, python=sys.version.split()[0])


_PLATFORM = {}


def _platform():
    if _PLATFORM:
        return dict(_PLATFORM)
    d = dict(backend=jax.default_backend(), devices=[str(x) for x in jax.devices()])
    if jax.default_backend() == "gpu":
        dev = jax.devices()[0]
        d.update(gpu=dev.device_kind, compute_capability=getattr(dev, "compute_capability", None))
        try:
            import subprocess
            d["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                         capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        except Exception:
            d["driver"] = None
    _PLATFORM.update(d)
    return d


def _cuda():
    try:
        from .backends.cuda import compile as C
        comp = C.compiler()
        return dict(compiler=C.compiler_identity(comp), cache_dir=C.cache_dir())
    except Exception as e:                                   # no CUDA toolchain on this machine
        return dict(error=f"{type(e).__name__}: {e}")


def _reverse():
    from . import _reverse as R
    from . import _reverse_spawn as RS
    return dict(single=dict(R.LAST_SINGLE), tree=RS.LAST[-1] if RS.LAST else None)


_CB_LINE = re.compile(r"^.*python_\w*callback.*$", re.M)
_CB_CFG = re.compile(r'backend_config = "(\d+)"')


def canon_hlo(txt):
    """StableHLO text with every host-callback descriptor replaced by its first-appearance ordinal.  JAX < 0.7 lowers
    a host callback (jax.debug.callback, pure_callback) with the address of the registered Python callable as a
    constant operand and as the custom_call's backend_config; it is new on every lowering, so the raw text of one
    program hashes differently every time.  Programs with a different number or placement of callbacks still
    differ."""
    ds = []
    for line in _CB_LINE.findall(txt):
        for d in _CB_CFG.findall(line):
            if d not in ds:
                ds.append(d)
    for k, d in enumerate(ds):
        txt = re.sub(rf"(?<!\d){d}(?!\d)", f"TM_HOST_CALLBACK_{k}", txt)
    return txt


def diagnostics(fn=None, *args, reset=False, **kwargs):
    """-> dict (see the module docstring).  fn, args: also lower jax.jit(fn)(*args, **kwargs) and report its
    programs / kernels.  reset: clear the counters and the history after reading them."""
    d = dict(versions=_versions(), platform=_platform(), env=_env(),
             cuda=_cuda() if jax.default_backend() == "gpu" else None, reverse=_reverse())
    if fn is not None:
        jax.clear_caches()
        with _capture() as c:
            low = jax.jit(fn).lower(*args, **kwargs)
        txt = canon_hlo(low.as_text())
        d.update(programs=[_render(r) for r in c["programs"]], kernels=c["kernels"],
                 stablehlo_sha256=hashlib.sha256(txt.encode()).hexdigest())
    else:
        d.update(programs=[_render(r) for r in PROGRAMS], kernels=list(KERNELS))
    d["kernels"] = [_with_resources(k) for k in d["kernels"]]
    if reset:
        from . import _reverse as R
        from . import _reverse_spawn as RS
        R.LAST_SINGLE.clear()
        RS.LAST.clear()
        PROGRAMS.clear()
        KERNELS.clear()
    return d


def _with_resources(rec):
    """a kernel record plus sj_main's measured resources (registers, local memory, occupancy, the driver's
    local-memory reservation) once the kernel has run; the record is not modified."""
    if "kid" not in rec:
        return rec
    from .backends.cuda.compile import kernel_info
    info = kernel_info(rec["kid"])
    return dict(rec, resources=info) if info is not None else rec


def _env():                                                    # the environment variables that matter
    keys = ("TALLYMAP_CUDA_CACHE", "JAX_PLATFORMS", "XLA_FLAGS", "CUDA_VISIBLE_DEVICES")
    return {k: os.environ.get(k) for k in keys if os.environ.get(k) is not None} | {"machine": platform.machine()}
