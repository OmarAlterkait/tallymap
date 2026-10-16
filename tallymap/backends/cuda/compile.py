"""CUDA C -> CUBIN (NVRTC via ctypes, or nvcc), the on-disk cubin cache, and the generic FFI launch shim.

Cache keys contain everything that determines the binary besides the source text: the compiler BUILD (NVRTC
realpath + version + size/mtime, or `nvcc --version`), arch, options, and the tallymap / jax / jaxlib versions (the
jaxpr -> C mapping depends on JAX's primitive semantics).  The shim key adds the FFI headers, the host compiler and
libc.  Cache directory: env TALLYMAP_CUDA_CACHE, else ~/.cache/tallymap (cubins are small); toolchain discovery:
NVCC, CXX (NVRTC is used when found, else nvcc).  $CXX names the HOST C++ compiler that builds the FFI shim.
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
    return _NVRTC[0]


def find_nvcc():
    for p in (os.environ.get("NVCC"), "/usr/local/cuda/bin/nvcc", shutil.which("nvcc")):
        if p and os.path.exists(p):
            return p
    return None


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


def compiler():
    return "nvrtc" if nvrtc()[0] is not None else "nvcc"


_IDENT = {}


def compiler_identity(comp):
    if comp not in _IDENT:
        if comp == "nvrtc":
            lib, path = nvrtc()
            a, b = ctypes.c_int(), ctypes.c_int()
            lib.nvrtcVersion(ctypes.byref(a), ctypes.byref(b))
            rp = os.path.realpath(path) if path and os.path.exists(path) else str(path)
            st = os.stat(rp) if os.path.exists(rp) else None
            stamp = f"{st.st_size}:{int(st.st_mtime)}" if st else "?:?"
            _IDENT[comp] = f"nvrtc|{a.value}.{b.value}|{rp}|{stamp}"
        else:
            nv = find_nvcc()
            _IDENT[comp] = f"nvcc|{nv}|{_cmd([nv, '--version']) if nv else None}"
    return _IDENT[comp]


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


def _nvcc_compile(src, arch, opts):
    nv = find_nvcc()
    if nv is None:
        raise RuntimeError("no nvcc found and NVRTC unavailable")
    with tempfile.TemporaryDirectory(dir=os.environ.get("LSCRATCH") or None) as td:
        cu, out = os.path.join(td, "tallymap.cu"), os.path.join(td, "tallymap.cubin")
        with open(cu, "w") as f:
            f.write(src)
        cmd = [nv, "-cubin", f"-arch=sm_{arch}", *[x.replace("--std=", "-std=") for x in opts], cu, "-o", out]
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if r.returncode:
            raise RuntimeError(f"nvcc failed:\n{' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
        with open(out, "rb") as f:
            return f.read()


def _versions():
    import jax
    import jaxlib

    from ... import __version__ as sv
    return f"tallymap={sv};jax={jax.__version__};jaxlib={getattr(jaxlib, '__version__', '?')}"


def compile_cubin(src, arch, extra=()):
    """-> (cubin bytes, info).  Disk-cached."""
    comp = compiler()
    opts = [o for o in OPTS if o.split("=")[0] not in {e.split("=")[0] for e in extra}] + extra
    h = hashlib.sha256(f"{compiler_identity(comp)}|{arch}|{opts}|{_versions()}\n{src}".encode()).hexdigest()[:24]
    path = os.path.join(cache_dir(), f"{h}.cubin")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read(), dict(hash=h, cached=True, seconds=0.0)
    t = time.perf_counter()
    cubin = _nvrtc_compile(src, arch, opts) if comp == "nvrtc" else _nvcc_compile(src, arch, opts)
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
