"""The CPU target of the kernel backend: the SAME generated C as the cuda target, compiled for the host with the
system C++ compiler and run by host_shim.cc's generic jax.ffi CPU handler.

Each host thread plays one CUDA thread of a one-thread block (blockIdx.x = thread id, gridDim.x = thread count,
lane 0 of a one-lane warp): sj_main is unchanged and claims items from the shared counter.  What is CUDA-only in the
generated source is mapped by HOST_PRELUDE: the __device__/__global__/__shared__ qualifiers (a privatised __shared__
accumulator becomes a per-thread stack array, flushed atomically at the end, as it is per block on a GPU), the
warp primitives of the work claim (one lane: they collapse to identities), atomics (GCC builtins; float add is a
compare-and-swap loop), the _rn/_rz arithmetic and conversion intrinsics, and the math library: libm instead of
libdevice, so f32 division and square root are always IEEE (cuda_math has no meaning here) and results agree with
the GPU to rounding, not bitwise.

A kernel's shared object includes no XLA header and needs no OpenMP: it exports one C entry point,
sj_host_run(phase, buffers, n, n_items, tid, nthreads), which the shim calls on every thread.  The shim (built once
per toolchain, like the cuda shim) owns the threads and the FFI binding.
"""
import ctypes
import glob
import hashlib
import os
import platform
import re
import subprocess
import sys
import threading
import time

from . import compile as C

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = "tallymap_host_launch"
# -O3 -march=native: the kernel is built on and for this machine (the CPU identity is in the cache key).
# -ffp-contract=off: no FMA contraction, as on the cuda target (-fmad=false); -fno-math-errno changes no value.
OPTS = ["-O3", "-march=native", "-ffp-contract=off", "-fno-math-errno", "-std=c++17", "-shared", "-fPIC", "-w"]
# per-thread stack the shim reserves (host_shim.cc HOST_STACK); a kernel whose frames could exceed half of it raises
# before it runs, as a cuda kernel over CUDA's local-memory limit does (compile.check_frame)
HOST_STACK = 16 << 20
_LOCK = threading.Lock()

HOST_PRELUDE = r"""
#include <cmath>
#include <math.h>
#include <cstring>
#include <cstdint>
#include <limits>
#include <sched.h>
#define __device__
#define __global__
// __forceinline__ is a hint here: GCC's own inliner decides.  Forcing it (always_inline) inlined four threefry2x32
// bodies into one loop and sent GCC 11's value-range pass past 100 s at -O2 on a 389-line kernel (2.2 s without)
#define __forceinline__ inline
#define __noinline__ __attribute__((noinline))
#define __launch_bounds__(...)
#define __shared__
#define __restrict__ __restrict
struct sj_dim3 { unsigned x, y, z; };
static thread_local sj_dim3 blockIdx = {0, 0, 0}, gridDim = {1, 1, 1};
static const sj_dim3 threadIdx = {0, 0, 0}, blockDim = {1, 1, 1};
static inline void __syncthreads() {}
static inline void __threadfence() { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
static inline void __nanosleep(unsigned) { sched_yield(); }      // the spawn queue's back-off: let another thread run
static inline unsigned __activemask() { return 1u; }
template <class T> static inline T __shfl_sync(unsigned, T v, int) { return v; }
static inline int __ffs(unsigned x) { return __builtin_ffs((int)x); }
static inline int __popc(unsigned x) { return __builtin_popcount(x); }
static inline int __popcll(unsigned long long x) { return __builtin_popcountll(x); }
static inline int __clz(int x) { return x == 0 ? 32 : __builtin_clz((unsigned)x); }
static inline int __clzll(long long x) { return x == 0 ? 64 : __builtin_clzll((unsigned long long)x); }
static inline unsigned __umulhi(unsigned a, unsigned b) { return (unsigned)(((unsigned long long)a * b) >> 32); }
static inline int __mulhi(int a, int b) { return (int)(((long long)a * b) >> 32); }
static inline unsigned long long __umul64hi(unsigned long long a, unsigned long long b) { return (unsigned long long)(((unsigned __int128)a * b) >> 64); }
static inline long long __mul64hi(long long a, long long b) { return (long long)(((__int128)a * b) >> 64); }
template <class T> static inline T __ldg(const T* p) { return *p; }
struct float2 { float x, y; };
struct double2 { double x, y; };
template <class To, class From> static inline To sj_hbc(From f) { To t; std::memcpy(&t, &f, sizeof(t)); return t; }
static inline float __int_as_float(int x) { return sj_hbc<float>(x); }
static inline float __uint_as_float(unsigned x) { return sj_hbc<float>(x); }
static inline int __float_as_int(float x) { return sj_hbc<int>(x); }
static inline unsigned __float_as_uint(float x) { return sj_hbc<unsigned>(x); }
static inline double __longlong_as_double(long long x) { return sj_hbc<double>(x); }
static inline long long __double_as_longlong(double x) { return sj_hbc<long long>(x); }
static inline float __fadd_rn(float a, float b) { return a + b; }
static inline float __fsub_rn(float a, float b) { return a - b; }
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fdiv_rn(float a, float b) { return a / b; }
static inline float __fsqrt_rn(float a) { return sqrtf(a); }
static inline float __frcp_rn(float a) { return 1.0f / a; }
static inline double __dadd_rn(double a, double b) { return a + b; }
static inline double __dsub_rn(double a, double b) { return a - b; }
static inline double __dmul_rn(double a, double b) { return a * b; }
static inline double __ddiv_rn(double a, double b) { return a / b; }
static inline double __dsqrt_rn(double a) { return sqrt(a); }
static inline float __double2float_rn(double a) { return (float)a; }
static inline long long __double2ll_rn(double a) { return llrint(a); }
static inline double rsqrt(double x) { return 1.0 / sqrt(x); }
// erfinv (libdevice has it, libm does not): Giles (2010)'s single-precision polynomial as the starting point,
// evaluated in double, then Halley steps on erf (erfc in the upper half, where 1 - |y| is exact) until the step is
// below an ulp: two from the f32-accurate start, up to five in the far tail (1 - |y| < 1e-12, where the polynomial
// is outside its range).  f32 rounds the double result.
static inline double erfinv(double y) {
  if (!(y > -1.0 && y < 1.0)) return y == 1.0 ? INFINITY : (y == -1.0 ? -INFINITY : NAN);
  if (y == 0.0) return y;
  const double a = fabs(y);
  double w = -log((1.0 - a) * (1.0 + a)), p;
  if (w < 5.0) { w -= 2.5; p = 2.81022636e-08; p = 3.43273939e-07 + p * w; p = -3.5233877e-06 + p * w;
    p = -4.39150654e-06 + p * w; p = 0.00021858087 + p * w; p = -0.00125372503 + p * w; p = -0.00417768164 + p * w;
    p = 0.246640727 + p * w; p = 1.50140941 + p * w; }
  else { w = sqrt(w) - 3.0; p = -0.000200214257; p = 0.000100950558 + p * w; p = 0.00134934322 + p * w;
    p = -0.00367342844 + p * w; p = 0.00573950773 + p * w; p = -0.0076224613 + p * w; p = 0.00943887047 + p * w;
    p = 1.00167406 + p * w; p = 2.83297682 + p * w; }
  double x = p * a;
  for (int i = 0; i < 8; ++i) {
    const double f = a < 0.5 ? erf(x) - a : (1.0 - a) - erfc(x);
    const double d = 1.1283791670955126 * exp(-x * x);         // erf'(x) = 2/sqrt(pi) exp(-x^2)
    const double dx = f / (d * (1.0 + x * f / d));             // Halley: erf'' = -2x erf'
    x -= dx;
    if (fabs(dx) <= 2.3e-16 * x) break;
  }
  return copysign(x, y); }
static inline float erfinvf(float x) { return (float)erfinv((double)x); }
template <class T, class F> static inline T sj_f2i(F x) {   // CUDA's _rz conversions saturate; NaN -> 0
  if (x != x) return (T)0;
  const F lo = (F)std::numeric_limits<T>::min(), hi = (F)std::numeric_limits<T>::max();
  if (x <= lo) return std::numeric_limits<T>::min();
  if (x >= hi) return std::numeric_limits<T>::max();
  return (T)x; }
#define __float2int_rz(x) sj_f2i<int, float>(x)
#define __float2uint_rz(x) sj_f2i<unsigned, float>(x)
#define __float2ll_rz(x) sj_f2i<long long, float>(x)
#define __float2ull_rz(x) sj_f2i<unsigned long long, float>(x)
#define __double2int_rz(x) sj_f2i<int, double>(x)
#define __double2uint_rz(x) sj_f2i<unsigned, double>(x)
#define __double2ll_rz(x) sj_f2i<long long, double>(x)
#define __double2ull_rz(x) sj_f2i<unsigned long long, double>(x)
#define __int2float_rn(x) ((float)(x))
#define __uint2float_rn(x) ((float)(x))
#define __ll2float_rn(x) ((float)(x))
#define __ull2float_rn(x) ((float)(x))
#define __int2double_rn(x) ((double)(x))
#define __uint2double_rn(x) ((double)(x))
#define __ll2double_rn(x) ((double)(x))
#define __ull2double_rn(x) ((double)(x))
template <class T> static inline T max(T a, T b) { return a > b ? a : b; }
template <class T> static inline T min(T a, T b) { return a < b ? a : b; }
static inline int atomicAdd(int* a, int v) { return __atomic_fetch_add(a, v, __ATOMIC_RELAXED); }
static inline unsigned atomicAdd(unsigned* a, unsigned v) { return __atomic_fetch_add(a, v, __ATOMIC_RELAXED); }
static inline unsigned long long atomicAdd(unsigned long long* a, unsigned long long v) { return __atomic_fetch_add(a, v, __ATOMIC_RELAXED); }
static inline float atomicAdd(float* a, float v) {
  unsigned* u = (unsigned*)a; unsigned o = __atomic_load_n(u, __ATOMIC_RELAXED), n;
  do { n = __float_as_uint(__uint_as_float(o) + v); } while (!__atomic_compare_exchange_n(u, &o, n, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
  return __uint_as_float(o); }
static inline double atomicAdd(double* a, double v) {
  unsigned long long* u = (unsigned long long*)a; unsigned long long o = __atomic_load_n(u, __ATOMIC_RELAXED), n;
  do { n = sj_hbc<unsigned long long>(sj_hbc<double>(o) + v); } while (!__atomic_compare_exchange_n(u, &o, n, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
  return sj_hbc<double>(o); }
static inline int atomicSub(int* a, int v) { return __atomic_fetch_sub(a, v, __ATOMIC_RELAXED); }
static inline unsigned atomicSub(unsigned* a, unsigned v) { return __atomic_fetch_sub(a, v, __ATOMIC_RELAXED); }
static inline unsigned atomicExch(unsigned* a, unsigned v) { return __atomic_exchange_n(a, v, __ATOMIC_SEQ_CST); }
static inline int atomicExch(int* a, int v) { return __atomic_exchange_n(a, v, __ATOMIC_SEQ_CST); }
template <class T> static inline T sj_amax(T* a, T v) { T o = __atomic_load_n(a, __ATOMIC_RELAXED); while (o < v && !__atomic_compare_exchange_n(a, &o, v, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {} return o; }
template <class T> static inline T sj_amin(T* a, T v) { T o = __atomic_load_n(a, __ATOMIC_RELAXED); while (o > v && !__atomic_compare_exchange_n(a, &o, v, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {} return o; }
static inline int atomicMax(int* a, int v) { return sj_amax(a, v); }
static inline unsigned atomicMax(unsigned* a, unsigned v) { return sj_amax(a, v); }
static inline long long atomicMax(long long* a, long long v) { return sj_amax(a, v); }
static inline unsigned long long atomicMax(unsigned long long* a, unsigned long long v) { return sj_amax(a, v); }
static inline int atomicMin(int* a, int v) { return sj_amin(a, v); }
static inline unsigned atomicMin(unsigned* a, unsigned v) { return sj_amin(a, v); }
static inline long long atomicMin(long long* a, long long v) { return sj_amin(a, v); }
static inline unsigned long long atomicMin(unsigned long long* a, unsigned long long v) { return sj_amin(a, v); }
"""

# kernel.BRK with portable constraints ("+g": any register or memory the compiler picks; the cut is still opaque)
HOST_BRK = r"""
template <class T> static inline T sj_brk(T x) { asm("" : "+g"(x)); return x; }
"""

ENTRY = r"""
extern "C" void sj_host_run(int phase, void** p, int np, long long n_items, int tid, int nthreads) {
  SjParams P;
  const int r = (int)(sizeof(P.p) / sizeof(void*));
  for (int i = 0; i < r; ++i) P.p[i] = i < np ? p[i] : nullptr;
  blockIdx.x = (unsigned)tid; gridDim.x = (unsigned)nthreads;
  if (phase == 0) SJ_PREP; else if (phase == 1) sj_main(P, n_items); else SJ_POST;
}
"""


def source(src, L):
    """the cuda target's generated source -> the host translation unit."""
    from .kernel import BRK
    if BRK in src:
        src = src.replace(BRK, HOST_BRK)
    entry = (ENTRY.replace("SJ_PREP", "sj_prep(P, n_items)" if L.flags & 1 else "(void)0")
             .replace("SJ_POST", "sj_post(P, n_items)" if L.flags & 2 else "(void)0"))
    return HOST_PRELUDE + src + "\n" + entry


_IDENT = []


def compiler_identity():
    """the host compiler build and the exact target -march=native expands to on this machine (its predefined
    macros), part of every host kernel's cache key: a cached kernel never runs on a CPU it was not built for."""
    if not _IDENT:
        cxx = C.host_cxx()
        ver = (C._cmd([cxx, "--version"]).splitlines() or [""])[0]
        r = subprocess.run([cxx, "-march=native", "-dM", "-E", "-x", "c++", "-"], input="", capture_output=True,
                           text=True, errors="replace")
        tgt = hashlib.sha256(r.stdout.encode()).hexdigest()[:16]
        _IDENT.append(f"{os.path.realpath(cxx)}|{ver}|march-native:{tgt}|{platform.machine()}")
    return _IDENT[0]


def _stack_bytes(su):
    """upper bound on a kernel's stack from gcc/clang -fstack-usage: the sum of every function's frame (the call
    tree has no recursion, so the deepest chain is at most the sum)."""
    tot = 0
    try:
        with open(su) as f:
            for ln in f:
                m = re.search(r"\t(\d+)\t", ln)
                if m:
                    tot += int(m.group(1))
    except OSError:
        return None
    return tot


def compile_so(src, where="", note=""):
    """-> (path of the kernel's shared object, info).  Disk-cached under <cache>/host."""
    h = hashlib.sha256(f"{compiler_identity()}|{OPTS}|{C._versions()}\n{src}".encode()).hexdigest()[:24]
    d = os.path.join(C.cache_dir(), "host")
    os.makedirs(d, exist_ok=True)
    so, su = os.path.join(d, f"k_{h}.so"), os.path.join(d, f"k_{h}.su")
    cached = os.path.exists(so)
    t = time.perf_counter()
    if not cached:
        cc = os.path.join(d, f"k_{h}.cc")
        C._atomic_write(cc, src.encode())
        tmp = so + f".tmp{os.getpid()}"
        cmd = [C.host_cxx(), *OPTS, "-fstack-usage", cc, "-o", tmp]
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", cwd=d)
        if r.returncode:
            raise RuntimeError(f"tallymap: compiling the host kernel failed (internal error; backend='xla' runs the "
                               f"program meanwhile):\n{' '.join(cmd)}\n{r.stderr[-4000:]}")
        # -fstack-usage names its file after the output and the source (GCC 11: <tmp>-k_<h>.su); take it by pattern
        for cand in glob.glob(os.path.join(d, f"*{h}*.su")):
            if cand != su:
                os.replace(cand, su)
        os.replace(tmp, so)
    dt = time.perf_counter() - t
    stack = _stack_bytes(su)
    if stack is not None and stack > HOST_STACK // 2:
        from ..._api import UnsupportedError
        raise UnsupportedError(
            f"tallymap: the host kernel{' for ' + where if where else ''} may need up to {stack:,} bytes of stack "
            f"per thread (the item's frame and its local arrays), over the {HOST_STACK // 2:,} bytes it may use.  "
            f"{note + '  ' if note else ''}Make the per-item state smaller, or run it with backend='xla'.")
    return so, dict(hash=h, cached=cached, seconds=dt, stack=stack)


_SHIM = []


def shim():
    """build (once per toolchain/jaxlib) and load the host FFI handler; register it with XLA for the CPU."""
    with _LOCK:
        if _SHIM:
            return _SHIM[0]
        import jax
        import jax.ffi
        import jaxlib
        src = os.path.join(HERE, "host_shim.cc")
        inc = jax.ffi.include_dir()
        cxx = C.host_cxx()
        with open(src) as f:
            parts = [f.read(), inc, jax.__version__, getattr(jaxlib, "__version__", "?"),
                     os.path.realpath(jaxlib.__file__), compiler_identity(),
                     " ".join(platform.libc_ver()), sys.version.split()[0]]
        for hdr in ("xla/ffi/api/ffi.h", "xla/ffi/api/c_api.h", "xla/ffi/api/api.h"):
            p = os.path.join(inc, hdr)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    parts.append(hashlib.sha256(f.read()).hexdigest())
        h = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]
        d = os.path.join(C.cache_dir(), "shim")
        os.makedirs(d, exist_ok=True)
        so = os.path.join(d, f"libtallymap_host_{h}.so")
        if not os.path.exists(so):
            tmp = so + f".tmp{os.getpid()}"
            cmd = [cxx, "-O2", "-std=c++17", "-shared", "-fPIC", "-w", "-pthread", f"-I{inc}", src, "-o", tmp]
            r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
            if r.returncode:
                raise RuntimeError(f"tallymap: building the host FFI shim with the C++ compiler {cxx!r} failed "
                                   f"(set $CXX to a working compiler, or run with backend='xla'):\n"
                                   f"{' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
            os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        lib.tallymap_host_register.argtypes = [ctypes.c_int64, ctypes.c_void_p]
        lib.tallymap_host_threads.restype = ctypes.c_int
        jax.ffi.register_ffi_target(TARGET, jax.ffi.pycapsule(lib.TallymapHostLaunch), platform="cpu")
        _SHIM.append(lib)
        return lib


_LIBS = {}        # kernel id -> the loaded shared object (kept alive: the shim holds its entry point)


def register(so, info):
    """load a kernel's shared object and register its entry point with the shim under its content id."""
    kid = int.from_bytes(hashlib.sha256(info["hash"].encode()).digest()[:8], "little", signed=True)
    if kid not in _LIBS:
        lib = ctypes.CDLL(so)
        fn = ctypes.cast(lib.sj_host_run, ctypes.c_void_p).value
        shim().tallymap_host_register(ctypes.c_int64(kid), ctypes.c_void_p(fn))
        _LIBS[kid] = lib
    return kid


def threads():
    """the thread count the shim uses: $OMP_NUM_THREADS when set, else the CPUs this process may run on."""
    return int(shim().tallymap_host_threads())
