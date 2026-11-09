"""CUDA C -> CUBIN (NVRTC via ctypes), the on-disk cubin cache, and the generic FFI launch shim.

Cache keys contain everything that determines the binary besides the source text: the NVRTC BUILD (realpath +
version + size/mtime), arch, options, and the tallymap / jax / jaxlib versions (the jaxpr -> C mapping depends on
JAX's primitive semantics).  The shim key adds the FFI headers, the host compiler and libc.  Cache directory: env
TALLYMAP_CUDA_CACHE, else ~/.cache/tallymap (cubins are small).  NVRTC comes from the nvidia-cuda-nvrtc-cu12 wheel
(tallymap's `cuda` extra), else a CUDA toolkit's libnvrtc; $CXX names the HOST C++ compiler that builds the FFI shim.
"""
import ctypes
import ctypes.util
import glob
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
_LOCK = threading.Lock()
TARGET = "tallymap_cuda_launch"
OPTS = ["-fmad=false", "--std=c++17", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"]   # IEEE


def cache_dir():
    d = os.environ.get("TALLYMAP_CUDA_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "tallymap")
    os.makedirs(d, exist_ok=True)
    return d


def _atomic_write(path, data):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _cmd(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=60)
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # pragma: no cover
        return f"<{type(e).__name__}>"


_NVRTC = []


def nvrtc():
    """-> (ctypes library, path) of NVRTC: the nvidia-cuda-nvrtc-cu12 wheel, else a CUDA toolkit's libnvrtc.  Raises
    a tallymap error naming the wheel when there is none (the cuda backend compiles with NVRTC only)."""
    if not _NVRTC:
        cands = []
        try:
            import nvidia.cuda_nvrtc as m
            cands += sorted(glob.glob(os.path.join(list(m.__path__)[0], "lib", "libnvrtc.so*")))
        except Exception:
            pass
        cands += ["/usr/local/cuda/lib64/libnvrtc.so", "/usr/local/cuda/lib64/libnvrtc.so.12"]
        if ctypes.util.find_library("nvrtc"):
            cands.append(ctypes.util.find_library("nvrtc"))
        found = (None, None)
        for c in cands:
            if "alt" in os.path.basename(c) or "builtins" in os.path.basename(c):
                continue
            try:
                found = (ctypes.CDLL(c), c)
                break
            except OSError:
                continue
        _NVRTC.append(found)
    if _NVRTC[0][0] is None:
        raise RuntimeError("tallymap: the cuda backend compiles with NVRTC, and no libnvrtc was found (neither the "
                           "nvidia-cuda-nvrtc-cu12 wheel nor /usr/local/cuda/lib64 nor the loader path).  Fix: pip "
                           "install 'tallymap[cuda]' (or nvidia-cuda-nvrtc-cu12), or put a CUDA 12 toolkit's lib64 "
                           "on LD_LIBRARY_PATH; or run with backend='lockstep'.")
    return _NVRTC[0]


def host_cxx():
    """the host C++ compiler that builds the FFI shim: $CXX when set, else g++ / c++ from PATH.  Raises a
    tallymap error naming $CXX when there is none (the cuda backend cannot run without the shim)."""
    env = os.environ.get("CXX")
    if env:
        p = shutil.which(env) or (env if os.path.isfile(env) and os.access(env, os.X_OK) else None)
        if p is None:
            raise RuntimeError(f"tallymap: $CXX is set to {env!r}, which is not an executable C++ compiler, and "
                               "the cuda backend needs one to build its FFI shim (built once, then cached).  Fix: "
                               "point CXX at a working compiler (export CXX=$(command -v g++)), or unset CXX to "
                               "use g++ from PATH.")
        return p
    p = shutil.which("g++") or shutil.which("c++")
    if p is None:
        raise RuntimeError("tallymap: no host C++ compiler on PATH (no g++, no c++), and the cuda backend needs "
                           "one to build its FFI shim (built once, then cached).  Fix: install g++, or set $CXX "
                           "to a C++ compiler (export CXX=/path/to/g++).")
    return p


_IDENT = []


def compiler_identity():
    """the NVRTC build: version, realpath, size and mtime (part of every cubin cache key)."""
    if not _IDENT:
        lib, path = nvrtc()
        a, b = ctypes.c_int(), ctypes.c_int()
        lib.nvrtcVersion(ctypes.byref(a), ctypes.byref(b))
        rp = os.path.realpath(path) if path and os.path.exists(path) else str(path)
        st = os.stat(rp) if os.path.exists(rp) else None
        stamp = f"{st.st_size}:{int(st.st_mtime)}" if st else "?:?"
        _IDENT.append(f"nvrtc|{a.value}.{b.value}|{rp}|{stamp}")
    return _IDENT[0]


def _nvrtc_compile(src, arch, opts):
    lib, _ = nvrtc()
    prog = ctypes.c_void_p()
    if lib.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"tallymap.cu", 0, None, None):
        raise RuntimeError("nvrtcCreateProgram failed")
    o = [f"--gpu-architecture=sm_{arch}".encode()] + [x.encode() for x in opts]
    r = lib.nvrtcCompileProgram(prog, len(o), (ctypes.c_char_p * len(o))(*o))
    n = ctypes.c_size_t()
    lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
    log = ctypes.create_string_buffer(n.value)
    lib.nvrtcGetProgramLog(prog, log)
    if r:
        lib.nvrtcDestroyProgram(ctypes.byref(prog))
        raise RuntimeError(f"NVRTC compile failed ({r}):\n{log.value.decode(errors='replace')}")
    lib.nvrtcGetCUBINSize(prog, ctypes.byref(n))
    buf = ctypes.create_string_buffer(n.value)
    lib.nvrtcGetCUBIN(prog, buf)
    lib.nvrtcDestroyProgram(ctypes.byref(prog))
    return buf.raw


def _versions():
    import jax
    import jaxlib

    from ... import __version__ as sv
    return f"tallymap={sv};jax={jax.__version__};jaxlib={getattr(jaxlib, '__version__', '?')}"


def compile_cubin(src, arch, extra=()):
    """-> (cubin bytes, info).  Disk-cached."""
    opts = [o for o in OPTS if o.split("=")[0] not in {e.split("=")[0] for e in extra}] + extra
    h = hashlib.sha256(f"{compiler_identity()}|{arch}|{opts}|{_versions()}\n{src}".encode()).hexdigest()[:24]
    path = os.path.join(cache_dir(), f"{h}.cubin")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read(), dict(hash=h, cached=True, seconds=0.0)
    t = time.perf_counter()
    cubin = _nvrtc_compile(src, arch, opts)
    dt = time.perf_counter() - t
    _atomic_write(os.path.join(cache_dir(), f"{h}.cu"), src.encode())
    _atomic_write(path, cubin)
    return cubin, dict(hash=h, cached=False, seconds=dt)


_SHIM = []


def shim():
    """build (once per toolchain/jaxlib) and load the FFI launch handler; register it with XLA."""
    with _LOCK:
        if _SHIM:
            return _SHIM[0]
        import jax
        import jax.ffi
        import jaxlib
        src = os.path.join(HERE, "shim.cc")
        inc = jax.ffi.include_dir()
        cxx = host_cxx()
        with open(src) as f:
            parts = [f.read(), inc, jax.__version__, getattr(jaxlib, "__version__", "?"),
                     os.path.realpath(jaxlib.__file__), str(cxx),
                     (_cmd([cxx, "--version"]).splitlines() or [""])[0],
                     " ".join(platform.libc_ver()), platform.machine(), sys.version.split()[0]]
        for hdr in ("xla/ffi/api/ffi.h", "xla/ffi/api/c_api.h", "xla/ffi/api/api.h"):
            p = os.path.join(inc, hdr)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    parts.append(hashlib.sha256(f.read()).hexdigest())
        h = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]
        d = os.path.join(cache_dir(), "shim")
        os.makedirs(d, exist_ok=True)
        so = os.path.join(d, f"libtallymap_cuda_{h}.so")
        if not os.path.exists(so):
            tmp = so + f".tmp{os.getpid()}"
            cmd = [cxx, "-O2", "-std=c++17", "-shared", "-fPIC", "-w", f"-I{inc}", src, "-o", tmp, "-ldl"]
            r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
            if r.returncode:
                raise RuntimeError(f"tallymap: building the cuda FFI shim with the host C++ compiler {cxx!r} "
                                   f"failed (set $CXX to a working compiler):\n{' '.join(cmd)}\n{r.stdout}\n"
                                   f"{r.stderr}")
            os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        lib.tallymap_register.argtypes = [ctypes.c_int64, ctypes.c_char_p, ctypes.c_size_t]
        jax.ffi.register_ffi_target(TARGET, jax.ffi.pycapsule(lib.TallymapLaunch), platform="CUDA")
        _SHIM.append(lib)
        return lib


def kernel_info(kid):
    """-> the resources of a registered kernel's sj_main once it has been launched (regs, local_bytes per thread,
    shared_bytes per block, blocks_per_sm, the local memory the driver reserves for every resident thread in GiB),
    or None (not launched yet, or no shim in this process)."""
    if not _SHIM:
        return None
    out = (ctypes.c_longlong * 7)()
    lib = _SHIM[0]
    lib.tallymap_kernel_info.argtypes = [ctypes.c_int64, ctypes.POINTER(ctypes.c_longlong)]
    if not lib.tallymap_kernel_info(ctypes.c_int64(kid), out):
        return None
    regs, local, shared, occ, sms, per_sm, block = list(out)
    return dict(regs=regs, local_bytes=local, shared_bytes=shared, blocks_per_sm=occ,
                resident_threads=occ * block * sms, local_reserved_gib=round(local * per_sm * sms / 2 ** 30, 3))


# CUDA's per-thread local memory limit: 512 KiB on every compute capability (CUDA C++ Programming Guide, "Technical
# Specifications per Compute Capability": maximum amount of local memory per thread).  A kernel over it compiles, and
# its launch fails with `CUDA driver error 1: invalid argument` (DTRAX sEM1_lar_e at 557,248 B, 2080 Ti and L40S).
LOCAL_LIMIT = 512 * 1024


def frame_bytes(cubin):
    """-> {kernel name: per-thread local memory bytes} read from the cubin itself (no device, no module load): each
    function's EIATTR_MIN_STACK_SIZE (0x12, its frame plus what its callees need) in the ELF .nv.info section, else
    its EIATTR_FRAME_SIZE (0x11).  These equal cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES) (checked on
    DTRAX's kernels: 70,160 / 178,160 / 557,248 B).  {} when the cubin carries no such record."""
    import struct
    b = bytes(cubin)
    if b[:4] != b"\x7fELF" or b[4] != 2:
        return {}
    shoff, = struct.unpack_from("<Q", b, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x3A)
    sh = [struct.unpack_from("<IIQQQQIIQQ", b, shoff + i * shentsize) for i in range(shnum)]

    def cstr(off):
        return b[off:b.index(b"\0", off)].decode(errors="replace")
    names = [cstr(sh[shstrndx][4] + s[0]) for s in sh]
    sec = {n: b[s[4]:s[4] + s[5]] for n, s in zip(names, sh)}
    syms = []
    if ".symtab" in sec and ".strtab" in names:
        st, so = sec[".symtab"], sh[names.index(".strtab")][4]
        syms = [cstr(so + struct.unpack_from("<I", st, o)[0]) for o in range(0, len(st) - 23, 24)]
    got = {}
    d, i = sec.get(".nv.info", b""), 0
    while i + 2 <= len(d):
        fmt, attr = d[i], d[i + 1]
        i += 2
        if fmt == 4:                                     # EIFMT_SVAL: u16 size + payload
            n, = struct.unpack_from("<H", d, i)
            if attr in (0x11, 0x12) and n == 8:
                sym, val = struct.unpack_from("<II", d, i + 2)
                name = syms[sym] if sym < len(syms) else f"symbol {sym}"
                if attr == 0x12 or name not in got:
                    got[name] = val
            i += 2 + n
        elif fmt == 3:                                   # EIFMT_HVAL: u16
            i += 2
        elif fmt == 2:                                   # EIFMT_BVAL: u8
            i += 1
        elif fmt != 1:                                   # unknown format: stop (the reading is best effort)
            break
    return got


def check_frame(cubin, where=""):
    """run-or-raise: a kernel whose per-thread local memory is over LOCAL_LIMIT cannot launch on any GPU; raise
    UnsupportedError naming the bytes and the limit BEFORE the launch, instead of the driver's `invalid argument`."""
    over = {k: v for k, v in frame_bytes(cubin).items() if v > LOCAL_LIMIT}
    if over:
        from ..._api import UnsupportedError
        k, v = max(over.items(), key=lambda kv: kv[1])
        raise UnsupportedError(
            f"tallymap: the cuda kernel{' for ' + where if where else ''} needs {v:,} bytes of local memory per "
            f"thread ({k}: the item's per-thread frame, its local arrays and register spills), over CUDA's limit "
            f"of {LOCAL_LIMIT:,} bytes (512 KiB) per thread, so it cannot launch on any GPU.  Make the per-item "
            f"state smaller (fewer or smaller per-item arrays live at once), or run it with backend='lockstep'.")


def register_kernel(cubin):
    kid = int.from_bytes(hashlib.sha256(cubin).digest()[:8], "little", signed=True)
    shim().tallymap_register(ctypes.c_int64(kid), cubin, len(cubin))
    return kid


def device_arch():
    import jax
    cc = getattr(jax.devices()[0], "compute_capability", None)
    if not cc:
        raise RuntimeError("cannot determine the GPU compute capability")
    return cc.replace(".", "")
