"""CUDA source for one tallymap program: the WHOLE item function is one device function (its loops are C loops),
run by persistent threads that claim items with warp-aggregated atomics.  Emits are atomics into the output
(small outputs privatised in shared memory); spawned children go through a bounded MPMC ticket queue.

Buffers (one by-value struct of pointers): [per-item inputs] [broadcast inputs] [hoisted constants]
| results: [sum tallies] [scratch accumulators (fixed point: u64 low words, i32 carry words, u32 flags;
           exact: DENSE_WORDS int64 words per bin, then flag bits)]
           [claim counter u64] [stats i32[5]]
           [spawn: queue leaves, seq, qctl] [per-item results].
N (the item count) is a kernel ARGUMENT: one cubin serves every N.
"""
import numpy as np

from ..._api import DENSE_WORDS, STATS
from . import ranges
from .codegen import CodegenError, Emitter, Local, Scalar, Table, VTally, _size, ctype, elem, fill_decl, unravel
from .op_table import phys

# Every kernel takes ONE by-value struct of buffer pointers, sized to the program's own buffer count r
# (`struct SjParams { void* p[r]; }`).  The limits are the driver's: 8 r + 8 bytes (with n_items) within the classic
# 4 KB kernel-parameter limit (r <= 511) everywhere, within 32,764 bytes (r <= SJ_MAXP = 4,094) with CUDA 12.1+
# large kernel parameters (compute capability 7.0+, 12.1+ toolkit and driver).
SJ_MAXP = 4094


PRELUDE = r"""
typedef signed char i8; typedef unsigned char u8; typedef short i16; typedef unsigned short u16;
typedef int i32; typedef unsigned int u32; typedef long long i64; typedef unsigned long long u64;
// the f32 math library behind ONE __noinline__ wrapper per function: one body per kernel instead of one per
// call site (inlined at every site, the libdevice bodies made most of the compile time: NVRTC on the
// benchmarks/compile_scaling 100-stage forward 195 -> 5.3 s, LUCiD with the scan change 1.5x; run time
// within -6%..+2% on lumen and LUCiD, outputs bitwise; docs/design_notes.md section 9)
#define SJ_NI1(f) __device__ __noinline__ float sj_n_##f(float x) { return f(x); }
#define SJ_NI2(f) __device__ __noinline__ float sj_n_##f(float x, float y) { return f(x, y); }
SJ_NI1(sinf) SJ_NI1(cosf) SJ_NI1(tanf) SJ_NI1(expf) SJ_NI1(logf) SJ_NI1(log1pf) SJ_NI1(expm1f) SJ_NI1(acosf) SJ_NI1(asinf)
SJ_NI1(atanf) SJ_NI1(tanhf) SJ_NI1(sinhf) SJ_NI1(coshf) SJ_NI1(erff) SJ_NI1(erfcf) SJ_NI1(lgammaf) SJ_NI1(cbrtf)
SJ_NI2(powf) SJ_NI2(atan2f) SJ_NI2(fmodf)
#define sinf(x) sj_n_sinf(x)
#define cosf(x) sj_n_cosf(x)
#define tanf(x) sj_n_tanf(x)
#define expf(x) sj_n_expf(x)
#define logf(x) sj_n_logf(x)
#define log1pf(x) sj_n_log1pf(x)
#define expm1f(x) sj_n_expm1f(x)
#define acosf(x) sj_n_acosf(x)
#define asinf(x) sj_n_asinf(x)
#define atanf(x) sj_n_atanf(x)
#define tanhf(x) sj_n_tanhf(x)
#define sinhf(x) sj_n_sinhf(x)
#define coshf(x) sj_n_coshf(x)
#define erff(x) sj_n_erff(x)
#define erfcf(x) sj_n_erfcf(x)
#define lgammaf(x) sj_n_lgammaf(x)
#define cbrtf(x) sj_n_cbrtf(x)
#define powf(x, y) sj_n_powf(x, y)
#define atan2f(x, y) sj_n_atan2f(x, y)
#define fmodf(x, y) sj_n_fmodf(x, y)
#ifndef TALLYMAP_APPROX_DIVSQRT   // cuda_math='ieee': correctly rounded (portable)
#define SJ_FDIV(a, b) __fdiv_rn(a, b)
#define SJ_FSQRT(x) __fsqrt_rn(x)
#define SJ_FRSQRT(x) __frcp_rn(__fsqrt_rn(x))
#else                            // cuda_math='approx' (default, -DTALLYMAP_APPROX_DIVSQRT): the f32 PTX XLA:GPU emits
__device__ __forceinline__ float sj_div_full(float a, float b) { float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float sj_sqrt_approx(float b) { float r; asm("sqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(b)); return r; }
__device__ __forceinline__ float sj_rsqrt_approx(float b) { float r; asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(b)); return r; }
#define SJ_FDIV(a, b) sj_div_full(a, b)
#define SJ_FSQRT(x) sj_sqrt_approx(x)
#define SJ_FRSQRT(x) sj_rsqrt_approx(x)
#endif
#ifdef TALLYMAP_FAST             // cuda_math='fast': division by a divisor proven away from zero (op_table.b_div)
__device__ __forceinline__ float sj_div_approx(float a, float b) { float r; asm("div.approx.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
#endif

template <class T> __device__ __forceinline__ T sj_ld(const T* p) { return __ldg(p); }
template <> __device__ __forceinline__ bool sj_ld<bool>(const bool* p) { return __ldg((const u8*)p) != 0; }
// a table staged in shared memory (kernel._stage_tables): a plain load, never __ldg (global only)
template <class T> __device__ __forceinline__ T sj_lds(const T* p) { return *p; }
// lax.max/min: NaN-propagating
template <class T> __device__ __forceinline__ T sj_fmax(T a, T b) { return (a >= b || a != a) ? a : b; }
template <class T> __device__ __forceinline__ T sj_fmin(T a, T b) { return (a <= b || a != a) ? a : b; }
// the same where both operands are proven non-NaN (ranges.float_facts): no NaN test; and, f32 with no signed-zero tie
// the two forms break differently (fmaxf(-0, +0) = +0 where sj_fmax gives -0), one FMNMX on the device
template <class T> __device__ __forceinline__ T sj_fmaxn(T a, T b) { return a >= b ? a : b; }
template <class T> __device__ __forceinline__ T sj_fminn(T a, T b) { return a <= b ? a : b; }
#ifdef __CUDA_ARCH__
__device__ __forceinline__ float sj_fmaxz(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ float sj_fminz(float a, float b) { return fminf(a, b); }
#else
__device__ __forceinline__ float sj_fmaxz(float a, float b) { return a >= b ? a : b; }
__device__ __forceinline__ float sj_fminz(float a, float b) { return a <= b ? a : b; }
#endif
template <class T> __device__ __forceinline__ T sj_imax(T a, T b) { return a > b ? a : b; }
template <class T> __device__ __forceinline__ T sj_imin(T a, T b) { return a < b ? a : b; }
template <class T> __device__ __forceinline__ T sj_fsign(T a) { return a > (T)0 ? (T)1 : (a < (T)0 ? (T)-1 : a); }
// lax integer division: x/0 = -1, INT_MIN/-1 = INT_MIN; x%0 = x, INT_MIN%-1 = 0
#define SJ_IDIV(T, MN) \
  __device__ __forceinline__ T sj_idiv(T a, T b) { return b == 0 ? (T)-1 : ((b == (T)-1 && a == (T)(MN)) ? a : (T)(a / b)); } \
  __device__ __forceinline__ T sj_irem(T a, T b) { return b == 0 ? a : ((b == (T)-1 && a == (T)(MN)) ? (T)0 : (T)(a % b)); }
SJ_IDIV(i8, -128) SJ_IDIV(i16, -32768) SJ_IDIV(i32, 0x80000000u) SJ_IDIV(i64, 0x8000000000000000ull)
#define SJ_UDIV(T) \
  __device__ __forceinline__ T sj_idiv(T a, T b) { return b == 0 ? (T)~(T)0 : (T)(a / b); } \
  __device__ __forceinline__ T sj_irem(T a, T b) { return b == 0 ? a : (T)(a % b); }
SJ_UDIV(u8) SJ_UDIV(u16) SJ_UDIV(u32) SJ_UDIV(u64)
template <class T, class U> __device__ __forceinline__ T sj_shl(T a, T b) {
  return ((U)b >= (U)(sizeof(T) * 8)) ? (T)0 : (T)((U)a << (U)b); }
template <class T, class U> __device__ __forceinline__ T sj_shrl(T a, T b) {
  return ((U)b >= (U)(sizeof(T) * 8)) ? (T)0 : (T)((U)a >> (U)b); }
template <class T, class U, class S> __device__ __forceinline__ T sj_shra(T a, T b) {
  return ((U)b >= (U)(sizeof(T) * 8)) ? (T)(((S)a) < 0 ? (S)-1 : (S)0) : (T)(((S)a) >> (U)b); }
template <class To, class From> __device__ __forceinline__ To sj_bitcast(From x) {
  union { From f; To t; } u; u.f = x; return u.t; }
__device__ __forceinline__ float sj_canon_f(float x) { return x == 0.0f ? 0.0f : (x != x ? __int_as_float(0x7fc00000) : x); }
__device__ __forceinline__ double sj_canon_d(double x) { return x == 0.0 ? 0.0 : (x != x ? __longlong_as_double(0x7ff8000000000000ll) : x); }
__device__ __forceinline__ i32 sj_tkey_f(float x) { i32 i = __float_as_int(x); return i ^ ((i >> 31) & 0x7fffffff); }
__device__ __forceinline__ i64 sj_tkey_d(double x) { i64 i = __double_as_longlong(x); return i ^ ((i >> 63) & 0x7fffffffffffffffll); }
__device__ __forceinline__ i64 sj_clampi(i64 x, i64 hi) { return x < 0 ? 0 : (x > hi ? hi : x); }
// the same clamp of a 32-bit index (two IMNMX instead of 64-bit compare pairs and selects): op_table._clamp_expr
__device__ __forceinline__ i32 sj_clamp32(i32 x, i32 hi) { return x < 0 ? 0 : (x > hi ? hi : x); }
__device__ __forceinline__ bool sj_inb(i64 x, i64 hi) { return x >= 0 && x <= hi; }
// an f32 sum's atomic add, keeping subnormals.  The GPU's f32 atomic add into global memory (red / atom.add.f32)
// flushes subnormal inputs AND results to zero whatever --ftz says (PTX ISA; measured on a 2080 Ti: 4 x 1.4e-44 summed
// to 0, 1.5 x 2^-126 - 2^-126 to 0).  An add of |v| >= 2^-102 never meets a subnormal: a subnormal bin is below half
// an ulp of v (ulp(v) >= 2^-125), so IEEE gives v as the flushed add does; and v + bin is a multiple of
// min(ulp(v), ulp(bin)) >= 2^-126, or |bin| < 2^-103 and |v + bin| > 2^-126: never a subnormal.  So that add (and a
// NaN / inf) keeps the one-instruction atomic, bitwise IEEE; a smaller v is added by a compare-and-swap loop of the
// IEEE add (__fadd_rn: no ftz); adding +-0 changes no bin (a bin starts at +0 and never becomes -0), so it is skipped.
#ifdef __CUDA_ARCH__
__device__ __forceinline__ void sj_fatom(float* a, float v) {
  if (v == 0.0f) return;
  if (!(fabsf(v) < __int_as_float(0x0c800000))) { atomicAdd(a, v); return; }    // 0x0c800000 = 2^-102
  u32* const u = (u32*)a; u32 o = *(volatile u32*)u, w;
  do { w = o; o = atomicCAS(u, w, __float_as_uint(__fadd_rn(__uint_as_float(w), v))); } while (o != w); }
#else
__device__ __forceinline__ void sj_fatom(float* a, float v) { atomicAdd(a, v); }   // the host's: an IEEE CAS loop
#endif

// threefry2x32 (20 rounds), bitwise identical to jax.random's threefry2x32_p
__device__ __forceinline__ u32 sj_rotl(u32 x, int d) { return (x << d) | (x >> (32 - d)); }
__device__ __forceinline__ void sj_threefry2x32(u32 k0, u32 k1, u32 c0, u32 c1, u32& o0, u32& o1) {
  const u32 k2 = k0 ^ k1 ^ 0x1BD11BDAu;
  u32 x0 = c0 + k0, x1 = c1 + k1;
#define SJ_R(r) x0 += x1; x1 = sj_rotl(x1, r); x1 ^= x0;
  SJ_R(13) SJ_R(15) SJ_R(26) SJ_R(6)  x0 += k1; x1 += k2 + 1u;
  SJ_R(17) SJ_R(29) SJ_R(16) SJ_R(24) x0 += k2; x1 += k0 + 2u;
  SJ_R(13) SJ_R(15) SJ_R(26) SJ_R(6)  x0 += k0; x1 += k1 + 3u;
  SJ_R(17) SJ_R(29) SJ_R(16) SJ_R(24) x0 += k1; x1 += k2 + 4u;
  SJ_R(13) SJ_R(15) SJ_R(26) SJ_R(6)  x0 += k2; x1 += k0 + 5u;
#undef SJ_R
  o0 = x0; o1 = x1;
}

// complex (no FMA, Smith division)
struct sj_c64 { float re, im; };
struct sj_c128 { double re, im; };
__device__ __forceinline__ sj_c64 sj_mkc64(float r, float i) { sj_c64 z; z.re = r; z.im = i; return z; }
__device__ __forceinline__ sj_c128 sj_mkc128(double r, double i) { sj_c128 z; z.re = r; z.im = i; return z; }
template <> __device__ __forceinline__ sj_c64 sj_ld<sj_c64>(const sj_c64* p) { float2 v = __ldg((const float2*)p); return sj_mkc64(v.x, v.y); }
template <> __device__ __forceinline__ sj_c128 sj_ld<sj_c128>(const sj_c128* p) { double2 v = __ldg((const double2*)p); return sj_mkc128(v.x, v.y); }
__device__ __forceinline__ bool operator==(sj_c64 a, sj_c64 b) { return a.re == b.re && a.im == b.im; }
__device__ __forceinline__ bool operator!=(sj_c64 a, sj_c64 b) { return !(a == b); }
__device__ __forceinline__ bool operator==(sj_c128 a, sj_c128 b) { return a.re == b.re && a.im == b.im; }
__device__ __forceinline__ bool operator!=(sj_c128 a, sj_c128 b) { return !(a == b); }
#define SJ_CPLX(C, F, ADD, SUB, MUL, DIV, ABS, MK) \
__device__ __forceinline__ C sj_cadd(C a, C b) { return MK(ADD(a.re, b.re), ADD(a.im, b.im)); } \
__device__ __forceinline__ C sj_csub(C a, C b) { return MK(SUB(a.re, b.re), SUB(a.im, b.im)); } \
__device__ __forceinline__ C sj_cneg(C a) { return MK(-a.re, -a.im); } \
__device__ __forceinline__ C sj_cconj(C a) { return MK(a.re, -a.im); } \
__device__ __forceinline__ C sj_cmul(C a, C b) { return MK(SUB(MUL(a.re, b.re), MUL(a.im, b.im)), ADD(MUL(a.re, b.im), MUL(a.im, b.re))); } \
__device__ __forceinline__ C sj_cdiv(C a, C b) { \
  if (ABS(b.re) >= ABS(b.im)) { const F r = DIV(b.im, b.re), d = ADD(b.re, MUL(b.im, r)); \
    return MK(DIV(ADD(a.re, MUL(a.im, r)), d), DIV(SUB(a.im, MUL(a.re, r)), d)); } \
  const F r = DIV(b.re, b.im), d = ADD(b.im, MUL(b.re, r)); \
  return MK(DIV(ADD(MUL(a.re, r), a.im), d), DIV(SUB(MUL(a.im, r), a.re), d)); }
SJ_CPLX(sj_c64, float, __fadd_rn, __fsub_rn, __fmul_rn, __fdiv_rn, fabsf, sj_mkc64)
SJ_CPLX(sj_c128, double, __dadd_rn, __dsub_rn, __dmul_rn, __ddiv_rn, fabs, sj_mkc128)
__device__ __forceinline__ float sj_cabs(sj_c64 z) { return hypotf(z.re, z.im); }
__device__ __forceinline__ double sj_cabs(sj_c128 z) { return hypot(z.re, z.im); }
__device__ __forceinline__ sj_c64 sj_csqrt(sj_c64 z) {
  const float r = __fsqrt_rn(sj_cabs(z)), t = __fmul_rn(atan2f(z.im, z.re), 0.5f); return sj_mkc64(__fmul_rn(r, cosf(t)), __fmul_rn(r, sinf(t))); }
__device__ __forceinline__ sj_c128 sj_csqrt(sj_c128 z) {
  const double r = sqrt(sj_cabs(z)), t = 0.5 * atan2(z.im, z.re); return sj_mkc128(r * cos(t), r * sin(t)); }
__device__ __forceinline__ sj_c64 sj_cexp(sj_c64 z) { const float e = expf(z.re); return sj_mkc64(__fmul_rn(e, cosf(z.im)), __fmul_rn(e, sinf(z.im))); }
__device__ __forceinline__ sj_c128 sj_cexp(sj_c128 z) { const double e = exp(z.re); return sj_mkc128(e * cos(z.im), e * sin(z.im)); }
__device__ __forceinline__ sj_c64 sj_clog(sj_c64 z) { return sj_mkc64(logf(sj_cabs(z)), atan2f(z.im, z.re)); }
__device__ __forceinline__ sj_c128 sj_clog(sj_c128 z) { return sj_mkc128(log(sj_cabs(z)), atan2(z.im, z.re)); }

#define SJ_LDV(p) (*((volatile u32*)(p)))
"""

# the wide-multiply chain cut (codegen.MULCHAIN_K), emitted only into kernels that use it (so every other kernel's
# source, and cubin, is unchanged)
BRK = r"""
// the wide-multiply chain cut (codegen.MULCHAIN_K): the same value, through an empty asm the front end cannot see
// through; the register allocator ties output to input, so it is zero instructions
__device__ __forceinline__ u64 sj_brk(u64 x) { asm("" : "+l"(x)); return x; }
__device__ __forceinline__ i64 sj_brk(i64 x) { asm("" : "+l"(x)); return x; }
__device__ __forceinline__ u32 sj_brk(u32 x) { asm("" : "+r"(x)); return x; }
__device__ __forceinline__ i32 sj_brk(i32 x) { asm("" : "+r"(x)); return x; }
__device__ __forceinline__ u16 sj_brk(u16 x) { u32 t = x; asm("" : "+r"(t)); return (u16)t; }
__device__ __forceinline__ i16 sj_brk(i16 x) { u32 t = (u16)x; asm("" : "+r"(t)); return (i16)t; }
__device__ __forceinline__ u8 sj_brk(u8 x) { u32 t = x; asm("" : "+r"(t)); return (u8)t; }
__device__ __forceinline__ i8 sj_brk(i8 x) { u32 t = (u8)x; asm("" : "+r"(t)); return (i8)t; }
__device__ __forceinline__ double sj_brk(double x) { asm("" : "+d"(x)); return x; }
__device__ __forceinline__ float sj_brk(float x) { asm("" : "+f"(x)); return x; }
"""


class Layout:
    pass


def _decl(name, shape, dt, const=True):
    ct = ctype(dt)
    if shape == ():
        return f"const {ct} {name}" if const else f"{ct}& {name}"
    return f"{'const ' if const else ''}{ct} (&{name})[{max(1, _size(shape))}]"


def _val(name, shape, dt, impl):
    v = Scalar(name, dt) if shape == () else Local(name, shape, dt)
    v.key_impl = impl
    return v


def split_blocker(jx):
    """why the step schedule cannot split the item jaxpr `jx` (None: it can).  The split needs exactly one
    top-level while loop."""
    ws = [e for e in jx.eqns if e.primitive.name == "while"]
    n = len(ws)
    if n == 1:
        from ..._jaxpr import while_parts
        if while_parts(ws[0]).cond.jaxpr.outvars[0].aval.shape != ():
            return "the item's top-level while loop has a batched (per-lane) predicate"
        return None
    return (f"the item has {n} top-level while loops; the step schedule needs exactly one (a "
            "while loop inside a cond, a scan or a call does not count)")


def _split(cj):
    """The step schedule: split an item with exactly ONE top-level while loop into pre / (cond, body) / post, so the
    kernel refills a thread with a new item at STEP granularity (a thread whose item finished does not idle until
    the whole warp's items finish; measurements: docs/design_notes.md section 1b).  -> None (split_blocker says why) or a
    dict of closed jaxprs + wiring."""
    import jax
    from jax.extend.core import Literal

    from ..._jaxpr import bind, is_dropvar, lit, while_parts
    jx = cj.jaxpr
    if split_blocker(jx) is not None:
        return None
    w = next(i for i, e in enumerate(jx.eqns) if e.primitive.name == "while")
    we = jx.eqns[w]
    pre_e, post_e = jx.eqns[:w], jx.eqns[w + 1:]
    base = set(jx.invars) | set(jx.constvars)
    defined = [v for e in pre_e for v in e.outvars if not is_dropvar(v)]
    used_later = {v for e in post_e for v in e.invars if not isinstance(v, Literal)}
    used_later |= {v for v in jx.outvars if not isinstance(v, Literal)}
    live = [v for v in defined if v in used_later]
    wp = while_parts(we)
    nc, nb = wp.nc, wp.nb
    wops = list(we.invars)
    init = wops[nc + nb:]          # loop carries: ALWAYS per-lane state (even a literal or an input initialises one)
    state = []                     # other values that must persist: computed loop consts and post-loop live values
    for v in wops[:nc + nb] + live:
        if not isinstance(v, Literal) and v not in base and v not in state:
            state.append(v)
    consts = dict(zip(jx.constvars, cj.consts))

    def ev(eqns, env):
        rd = lambda v: lit(v) if isinstance(v, Literal) else env[v]
        for e in eqns:
            for v, x in zip(e.outvars, bind(e, [rd(v) for v in e.invars])):
                env[v] = x
        return rd

    def pre(*a):
        env = dict(consts)
        env.update(zip(jx.invars, a))
        rd = ev(pre_e, env)
        return [rd(v) for v in init] + [rd(v) for v in state]

    def post(*a):
        n_in = len(jx.invars)
        env = dict(consts)
        env.update(zip(jx.invars, a[:n_in]))
        env.update(zip(state, a[n_in:n_in + len(state)]))
        env.update((v, x) for v, x in zip(we.outvars, a[n_in + len(state):]) if not is_dropvar(v))
        rd = ev(post_e, env)
        return [rd(v) for v in jx.outvars]
    sds = lambda v: jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype)
    in_av = [sds(v) for v in jx.invars]
    pre_cj = jax.make_jaxpr(pre)(*in_av)
    post_cj = jax.make_jaxpr(post)(*in_av, *[sds(v) for v in state], *[sds(v) for v in we.outvars])
    return dict(pre=pre_cj, post=post_cj, cond=wp.cond, body=wp.body, ops=wops, state=state,
                nc=nc, nb=nb, invars=list(jx.invars), constvars=consts, eqn=we)


BLOCK = 128          # threads per block (measured: 64 / 128 / 256 within 1%)
STEP_INPLACE_MIN = 64    # step schedule: array loop carries of at least this many elements are updated in place
PRIV_BYTES = 4096    # shared-memory budget for privatised sum tallies per block
# Small read-only tables (broadcast inputs and hoisted constants) copied into shared memory once per block at kernel
# start (kernel._stage_tables): their loads become LDS at a constant address instead of LDG through a pointer read
# from the parameter bank.  A table of at most SHARED_TABLE_MAX bytes is staged, most-referenced first, while the
# block's total stays within SHARED_TABLES bytes (0: off; a test-only A/B control, part of the kernel key).
SHARED_TABLES = 4096
SHARED_TABLE_MAX = 2048
# the spawn queue's 5 control words (claim / head / pending / occupancy / in flight) each on its own 128-byte line:
# every idle thread polls two of them, every spawn and pop updates the others, and on one shared line the polls
# queued every atomic behind them (docs/design_notes.md section 1c)
QCTL_STRIDE = 32


def privatised(tallies, budget=PRIV_BYTES):
    """indices of the sum tallies the kernel keeps in shared memory (in order, while they fit in `budget`)."""
    out = []
    for k, s in enumerate(tallies):
        if s.kind != "sum":
            continue
        nbytes = (int(np.prod(s.lead, dtype=np.int64)) * s.S * (8 if _acc_t(s) in ("u64", "double") else 4)
                  * (DENSE_WORDS if s.exact else 1))
        if 0 < nbytes <= budget:
            budget -= nbytes
            out.append(k)
    return out


def _stage_tables(fdefs, tables, budget=None, tmax=None):
    """copy small read-only tables into one shared-memory array per block (SHARED_TABLES, SHARED_TABLE_MAX).
    tables: [(P.p index, pointer spelling in the source ("P.p[i]" / "SJ_CONST(j)"), bytes)].  The function
    definitions `fdefs` are rewritten in place: every load `sj_ld(((const T*)<spelling>) + o)` of a staged table
    reads `sj_lds(((const T*)(sj_tab + off)) + o)` (the same bytes, from shared memory).  -> (file-scope
    declaration, per-block copy lines for the kernel's start; both empty when nothing is staged)."""
    import re
    budget = SHARED_TABLES if budget is None else budget
    tmax = SHARED_TABLE_MAX if tmax is None else tmax
    if budget <= 0 or not tables:
        return "", []
    text = "\n".join(fdefs)
    cand = []
    for idx, sp, nb in tables:
        refs = len(re.findall(r"sj_ld\(\(?\(const \w+\*\)\(?\(const \w+\*\)" + re.escape(sp) + r"\)", text))
        refs += len(re.findall(r"sj_ld\(\(\(const \w+\*\)" + re.escape(sp) + r"\)", text))
        if 0 < nb <= tmax and refs:
            cand.append((-refs, nb, idx, sp))
    cand.sort()
    chosen, off = [], 0
    for _, nb, idx, sp in cand:
        if off + nb > budget:
            continue
        chosen.append((idx, sp, nb, off))
        off += (nb + 15) // 16 * 16
    if not chosen:
        return "", []
    pats = []
    for idx, sp, nb, o in chosen:
        e = re.escape(sp)
        pats.append((re.compile(r"sj_ld\(\(\(const (\w+)\*\)" + e + r"\)"), rf"sj_lds(((const \1*)(sj_tab + {o}))"))
        pats.append((re.compile(r"sj_ld\(\(const u8\*\)\(\(const u8\*\)" + e + r"\)"),
                     rf"sj_lds((const u8*)((const u8*)(sj_tab + {o}))"))
    for k, ln in enumerate(fdefs):
        if "sj_ld(" in ln:
            for pat, rep in pats:
                ln = pat.sub(rep, ln)
            fdefs[k] = ln
    decl = f"__shared__ __align__(16) unsigned char sj_tab[{off}];   // staged tables (kernel._stage_tables)"
    copy = []
    for idx, sp, nb, o in chosen:
        if nb % 4 == 0:
            copy.append(f"  for (int i = threadIdx.x; i < {nb // 4}; i += blockDim.x) "
                        f"((u32*)(sj_tab + {o}))[i] = ((const u32*)P.p[{idx}])[i];")
        else:
            copy.append(f"  for (int i = threadIdx.x; i < {nb}; i += blockDim.x) sj_tab[{o} + i] = ((const u8*)P.p[{idx}])[i];")
    return decl, copy


def build_source(p, *, block=BLOCK, schedule="item", strict=False, target="cuda", fast=False):
    """-> (CUDA source, Layout).  p: tallymap._api.ItemProgram.  Its jaxpr is first rewritten into the
    canonical primitives (canon.py).  schedule: 'item' (one whole item per thread at a time) | 'step' (the step
    schedule wherever the program splits, else 'item'; strict: raise when it does not split).  Layout.mode is the
    schedule that ran; Layout.splittable says whether the program splits (error hints)."""
    import dataclasses

    from .canon import canonical
    p = dataclasses.replace(p, cj=canonical(p.cj))
    em = Emitter()
    em.fast = fast                 # cuda_math='fast': FMA-contractible f32 / f64 arithmetic (op_table.b_add ...)
    em.where = p.where
    L = Layout()
    jx = p.cj.jaxpr
    nT, nP, nB = p.nT, p.nP, p.nB
    pav = [phys(v.aval) for v in jx.invars[nT:nT + nP]]
    bav = [phys(v.aval) for v in jx.invars[nT + nP:]]
    yav = [phys(v.aval) for v in jx.outvars[nT:]]
    spawn = p.spawn
    L.i_pi = list(range(nP))
    L.i_bc = list(range(nP, nP + nB))

    def tptr(i, dt):
        return f"((const {'u8' if np.dtype(dt) == np.bool_ else ctype(dt)}*)P.p[{i}])"

    outs = p.out_tallies
    base_in, base_par, base_arg = [], ["const SjParams& P", "const i64 item"], ["P", "item"]
    base_par += [f"{_acc_t(p.tallies[k])}* const sj_acc{k}" for k in outs]
    base_arg += [f"sj_acc{k}" for k in outs]
    for k, s in enumerate(p.tallies):
        base_in.append(VTally(k, tuple(s.lead) + (s.S,), s.dtype))
    for i, (sh, dt, impl) in enumerate(pav):
        if spawn is None:
            t = Table(tptr(L.i_pi[i], dt), sh, dt, off=f"(item)*{_size(sh)}")
            t.key_impl = impl
        else:
            base_par.append(_decl(f"x{i}", sh, dt))
            base_arg.append(f"x{i}")
            t = _val(f"x{i}", sh, dt, impl)
        base_in.append(t)
    for i, (sh, dt, impl) in zip(L.i_bc, bav):
        t = Table(tptr(i, dt), sh, dt)
        t.key_impl = impl
        base_in.append(t)
    em.hooks["tallymap_emit"] = lambda em_, eqn, ins: _emit_rule(em_, p, ins)
    em.hooks["tallymap_spawn"] = lambda em_, eqn, ins: _spawn_rule(em_, p, ins, pav)
    claims = {}                    # Records one-pass: claim token (key tally) slot -> its counter (SJ_APPC index)
    em.hooks["tallymap_claim"] = lambda em_, eqn, ins, r: _claim_rule(em_, p, ins, claims)
    L.cjs = {}
    fdefs = []

    def function(name, cj, ins, inplace_out=None):
        """emit `cj` as a device function taking the base params + (decl, Val) inputs; non-tally results are
        written to reference params o<j>.  inplace_out {j: name}: result j may be the mutable
        reference parameter `name` itself, updated in place (then it has no o<j>, and its spec is "inplace").
        -> (results, out (shape, dtype, impl) | None | "inplace")"""
        L.cjs[name] = cj
        em.lines, em.ind = [], 1
        inplace_out = dict(inplace_out or {})
        res = em.run_top(cj, [v for _, v in ins], name, mutable=set(inplace_out.values()))
        odecl, ospec = [], []
        for j, (v, r) in enumerate(zip(cj.jaxpr.outvars, res)):
            if isinstance(r, VTally):
                ospec.append(None)
                continue
            if j in inplace_out and isinstance(r, Local) and r.name == inplace_out[j]:
                ospec.append("inplace")
                continue
            sh, dt, impl = phys(v.aval)
            odecl.append(_decl(f"o{j}", sh, dt, const=False))
            em.assign(Local(f"o{j}", sh, dt), r)
            ospec.append((sh, dt, impl))
        par = base_par + [d for d, _ in ins if d] + odecl
        fdefs.extend([f"__device__ __forceinline__ void sj_{name}({', '.join(par)}) {{"] + em.lines + ["}"])
        return res, ospec

    def svar(prefix, j, spec):
        sh, dt, impl = spec
        return f"  {ctype(dt)} {prefix}{j}{'' if sh == () else f'[{max(1, _size(sh))}]'};{fill_decl(f'{prefix}{j}')}\n"

    L.splittable = split_blocker(p.cj.jaxpr) is None
    sp = _split(p.cj) if schedule == "step" else None
    if sp is None and schedule == "step" and strict:
        raise CodegenError(f"schedule='step': {split_blocker(p.cj.jaxpr)}.  Pass schedule='item' (the default).")
    L.mode = "step" if sp is not None else "item"
    decls = ""
    if sp is None:
        res, ospec = function("item", p.cj, [(None, v) for v in base_in])
        ys_names = [f"y{j}" for j in range(len(yav))]
        decls += "".join(svar("y", j, phys(v.aval)) for j, v in enumerate(jx.outvars[nT:]))
        start = f"sj_item({', '.join(base_arg + ys_names)}); done = true;"
        advance = ""
    else:
        # ---- pre: item inputs -> (loop carries, persistent state)
        ops = sp["ops"]
        nc, nb = sp["nc"], sp["nb"]
        cc, bc, init = ops[:nc], ops[nc:nc + nb], ops[nc + nb:]
        res, ospec = function("pre", sp["pre"], [(None, v) for v in base_in])
        env = dict(zip(sp["invars"], base_in))
        pre_args, kvals = [], []
        for j, (r, spc) in enumerate(zip(res[:len(init)], ospec[:len(init)])):
            if spc is None:
                kvals.append(r)                                  # a tally carry stays virtual
            else:
                decls += svar("k", j, spc)
                kvals.append(_val(f"k{j}", spc[0], spc[1], spc[2]))
                pre_args.append(f"k{j}")
        for j, (v, r, spc) in enumerate(zip(sp["state"], res[len(init):], ospec[len(init):])):
            if spc is None:
                env[v] = r
            else:
                decls += svar("s", j, spc)
                env[v] = _val(f"s{j}", spc[0], spc[1], spc[2])
                pre_args.append(f"s{j}")
        start = f"sj_pre({', '.join(base_arg + pre_args)});"

        def val_of(v):
            if type(v).__name__ == "Literal":
                return em.const_val(v.val, v.aval)
            if v in env:
                return env[v]
            return em.const_val(sp["constvars"][v], v.aval)

        def as_param(x, name):
            """-> (param decl | None, Val inside the function, caller argument | None)"""
            if isinstance(x, (VTally, Table)) or (isinstance(x, Scalar) and x.cval is not None) or \
                    not isinstance(x, (Scalar, Local)):
                if not isinstance(x, (VTally, Table, Scalar, Local)):
                    raise CodegenError(f"schedule='step': a loop constant of kind {type(x).__name__} cannot cross "
                                       "the step schedule's split.  Pass schedule='item' (the default).")
                return None, x, None
            sh, dt = tuple(x.shape), x.dtype
            return _decl(name, sh, dt), _val(name, sh, dt, x.key_impl), (x.name if isinstance(x, Local) else x.text)
        pc = [as_param(val_of(v), f"c{j}") for j, v in enumerate(cc)]
        pb = [as_param(val_of(v), f"b{j}") for j, v in enumerate(bc)]
        pk = [as_param(x, f"w{j}") for j, x in enumerate(kvals)]
        # integer ranges across the split (ranges.py): the loop's operands from the item jaxpr, its carries'
        # invariant in cond / post and the predicate-narrowed one in the body
        renv = {}
        ranges.jaxpr_ranges(p.cj, [None] * len(jx.invars), renv)
        wr = [ranges.lit_range(v.val, v.aval.dtype) if type(v).__name__ == "Literal" else renv.get(v)
              for v in sp["eqn"].invars]
        body_r, inv_r = ranges.while_ranges(sp["eqn"], wr)
        for (d, x, _), r in zip(pc + pb + pk, wr[:nc] + wr[nc:nc + nb] + list(inv_r)):
            if d is not None:
                x.rng = r
        pk_body = []
        for j, ((d, x, a), r) in enumerate(zip(pk, body_r)):
            if d is not None:
                x = _val(f"w{j}", tuple(x.shape), x.dtype, x.key_impl)
                x.rng = r
            pk_body.append((d, x, a))
        function("cond", sp["cond"], [(d, x) for d, x, _ in pc + pk])
        decls += "  bool sj_pr;\n"
        cargs = [a for _, _, a in pc + pk if a is not None]
        # A LARGE array carry (>= STEP_INPLACE_MIN elements: a checkpoint buffer, a state history) is passed to the
        # body as a mutable reference and updated in place, instead of being rebuilt into n<j> and copied back to
        # k<j> every step (two full copies per step: for the single-loop adjoint's checkpoint buffer, 2C+1 loop
        # states, more traffic than the step itself).  Small carries keep the copy (their code is unchanged).
        ipc = {j for j, x in enumerate(kvals) if em.inplace and isinstance(x, Local) and x.shape != ()
               and _size(x.shape) >= STEP_INPLACE_MIN}
        pkb = [(_decl(f"w{j}", tuple(x.shape), x.dtype, const=False), _val(f"w{j}", tuple(x.shape), x.dtype,
                x.key_impl), f"k{j}") if j in ipc else q for j, (x, q) in enumerate(zip(kvals, pk_body))]
        for j in ipc:
            pkb[j][1].rng = body_r[j]
        bargs = [a for _, _, a in pb + pkb if a is not None]
        bres, bspec = function("body", sp["body"], [(d, x) for d, x, _ in pb + pkb],
                               inplace_out={j: f"w{j}" for j in ipc})
        nargs, upd = [], []
        for j, spc in enumerate(bspec):
            if spc is None or spc == "inplace":
                continue
            decls += svar("n", j, spc)
            nargs.append(f"n{j}")
            if spc[0] == ():
                upd.append(f"k{j} = n{j};")
            else:
                upd.append(f"for (int e = 0; e < {_size(spc[0])}; ++e) k{j}[e] = n{j}[e];")
        # ---- post: (item inputs, persistent state, final carries) -> item outputs
        pq = [as_param(val_of(v), f"q{j}") for j, v in enumerate(sp["state"])]
        function("post", sp["post"], [(None, v) for v in base_in] + [(d, x) for d, x, _ in pq + pk])
        ys_names = [f"y{j}" for j in range(len(yav))]
        decls += "".join(svar("y", j, phys(v.aval)) for j, v in enumerate(jx.outvars[nT:]))
        post_args = [a for _, _, a in pq + pk if a is not None]
        advance = (f"sj_cond({', '.join(base_arg + cargs + ['sj_pr'])});\n"
                   f"      if (sj_pr) {{ sj_body({', '.join(base_arg + bargs + nargs)}); {' '.join(upd)} }}\n"
                   f"      else {{ sj_post({', '.join(base_arg + post_args + ys_names)}); done = true; }}")
    L.consts = list(em.consts)
    base_const = nP + nB
    nin = base_const + len(L.consts)
    L.n_in = nin
    # the result buffers, written once: L.res[i] = (shape, dtype, per_item) of buffer nin + i (run() builds the FFI
    # results from it, the item axis prepended where per_item), and each L.r_* is a buffer's index in P.p[]
    L.res = []

    def result(shape, dt, per_item=False):
        L.res.append((tuple(shape), np.dtype(dt), per_item))
        return nin + len(L.res) - 1
    L.r_out = {k: result(tuple(p.tallies[k].lead) + (p.tallies[k].S,), p.tallies[k].dtype) for k in outs}
    L.r_scr = {k: result((_scratch_words(p.tallies[k]),), np.uint32)
               for k in outs if _scratch(p.tallies[k]) is not None}
    L.r_ctr = result((2,), np.uint32)                       # the claim counter (u64)
    if claims:                                              # Records one-pass slot counters (u64 each)
        L.r_app = result((2 * len(claims),), np.uint32)
    L.r_stats = result((len(STATS),), np.int32)
    if spawn is not None:
        L.r_q = [result((int(spawn.capacity) * max(1, _size(sh)),), dt) for sh, dt, _ in pav]
        L.r_seq = result((int(spawn.capacity),), np.uint32)
        L.r_qctl = result((8 * QCTL_STRIDE,), np.uint32)   # the queue's control words, one line each
    L.n_zero = len(L.res)                                   # (the shim zeroes these before the launch)
    L.r_ys = [result(sh, dt, per_item=True) for sh, dt, _ in yav]
    r = nin + len(L.res)
    if r > SJ_MAXP:
        raise CodegenError(f"too many buffers ({r} > {SJ_MAXP}: the kernel parameter struct would exceed the "
                           "driver's 32,764-byte limit)")
    L.pav, L.yav = pav, yav

    # SJ_STATS[i] is slot i of _api.STATS: 0 spawned, 1 children_dropped, 2 children_truncated, 3 fixed_point_overflow,
    # 4 dropped_emits
    src = [PRELUDE, f"struct SjParams {{ void* p[{r}]; }};",
           f"#define SJ_CONST(j) (P.p[{base_const} + (j)])", f"#define SJ_STATS ((i32*)P.p[{L.r_stats}])"]
    if claims:
        src.append(f"#define SJ_APPC ((u64*)P.p[{L.r_app}])")
    if spawn is not None:
        src.append(f"#define SJ_Q(i) (((u32*)P.p[{L.r_qctl}])[(i) * {QCTL_STRIDE}])\n#define SJ_SEQ ((u32*)P.p[{L.r_seq}])\n"
                   f"#define SJ_QBUF(i) (P.p[{L.r_q[0] if L.r_q else 0} + (i)])")
    flags, prep, post = 0, [], []
    priv = privatised(p.tallies)
    acc_decl, acc_flush = [], []
    for k in outs:
        s = p.tallies[k]
        ot = ctype(s.dtype)
        tot = int(np.prod(s.lead, dtype=np.int64)) * s.S
        dst = f"(({ot}*)P.p[{L.r_out[k]}])"
        at = _acc_t(s)
        nacc = tot * (DENSE_WORDS if s.exact else 1)      # accumulator words
        if s.exact:
            # the exact f32 sum (DENSE): DENSE_WORDS int64 words per bin (bin i's at a[i * W]), then one flag bit per
            # bin: a non-finite emit, or a carry out of the bin's top word, flags it (counted in stats[3]; the bin is
            # NaN), so a bin is exact or NaN, never silently wrong
            if DENSE not in src:
                src.append(DENSE)
            W = DENSE_WORDS
            gp = f"((u64*)P.p[{L.r_scr[k]}])"
            fl = f"((u32*)P.p[{L.r_scr[k]}] + {2 * nacc})"
            src.append(f"__device__ __forceinline__ void sj_emit{k}(const SjParams& P, u64* a, i64 i, float v) {{ "
                       f"if (sj_demit(&a[i * {W}], v)) {{ atomicAdd(&SJ_STATS[3], 1); "
                       f"atomicOr(&{fl}[i >> 5], 1u << (i & 31)); }} }}")
            post.append(f"for (i64 i = g; i < {tot}; i += s) {{ bool bad = false; "
                        f"const float o = sj_dread(&{gp}[i * {W}], &bad); const bool f = (({fl}[i >> 5] >> (i & 31)) & 1u) "
                        f"!= 0u; if (bad && !f) atomicAdd(&SJ_STATS[3], 1); "
                        f"{dst}[i] = (f || bad) ? __int_as_float(0x7fc00000) : o; }}")
            # a privatised copy's flush: each of its signed words added into the same word of the global bin
            op = lambda a, i, v, W=W, fl=fl: (f"if (sj_dadd(&{a}[({i}) / {W} * {W}], (int)(({i}) % {W}), (i64){v})) {{ "
                                              f"atomicAdd(&SJ_STATS[3], 1); atomicOr(&{fl}[({i}) / {W} >> 5], "
                                              f"1u << ((({i}) / {W}) & 31)); }}")
        elif s.fixed_point:
            if WIDE not in src:
                src.append(WIDE)
            gp = f"((u64*)P.p[{L.r_scr[k]}])"
            sc = float(s.scale)
            # an emit that cannot be represented is dropped, counted, and flags its bin (u32 flags after the
            # accumulators): post makes a flagged bin NaN, so each output (primal or tangent, any node) is poisoned
            # by its own overflows only
            fl = f"((u32*)P.p[{L.r_scr[k]}] + {2 * tot})"
            # each bin's signed high word: the net carry out of its int64 low word (WIDE).  Every add into a bin
            # (an emit, or a privatised accumulator's flush) adds its carry, so the bin's exact total is
            # low + carry * 2^64 whatever order the adds ran in.  A privatised emit counts its shared-memory add's
            # carry here too (the flush then adds the block's low word with its own carry).
            cw = f"((i32*)P.p[{L.r_scr[k]}] + {3 * tot})"
            src.append(f"__device__ __forceinline__ void sj_emit{k}(const SjParams& P, u64* a, i64 i, {ot} v) {{ "
                       f"const double x = (double)v * {sc!r}; if (!(fabs(x) < 9.2e18)) {{ atomicAdd(&SJ_STATS[3], 1); "
                       f"{fl}[i] = 1u; return; }} "
                       f"sj_wadd(&a[i], {cw} + i, __double2ll_rn(x)); }}")
            # readout: the exact low + carry * 2^64, rounded once to double (exactly today's value when the carry is
            # 0), times 1/scale, cast to the output type.  A bin whose value does not fit the output type is counted
            # with the dropped emits (stats[3]) and, for a float output, is NaN like a flagged bin.
            val = f"sj_wide_f64({gp}[i], {cw}[i]) * {1.0 / sc!r}"
            if ot in ("float", "double"):
                mx = "3.4028234663852886e38" if ot == "float" else "1.7976931348623157e308"
                post.append(f"for (i64 i = g; i < {tot}; i += s) {{ const double v = {val}; const {ot} o = ({ot})v; "
                            f"const bool big = !(fabs((double)o) <= {mx}); if (big && !{fl}[i]) atomicAdd(&SJ_STATS[3], 1); "
                            f"{dst}[i] = ({fl}[i] || big) ? ({ot})__longlong_as_double(0x7ff8000000000000LL) : o; }}")
            elif ot in _IRANGE:
                lo, hi = _IRANGE[ot]
                post.append(f"for (i64 i = g; i < {tot}; i += s) {{ const double v = {val}; "
                            f"if (!(v >= {lo} && v < {hi})) atomicAdd(&SJ_STATS[3], 1); {dst}[i] = ({ot})(v); }}")
            else:
                post.append(f"for (i64 i = g; i < {tot}; i += s) {dst}[i] = ({ot})({val});")
            op = lambda a, i, v, cw=cw: f"sj_wadd(&{a}[{i}], {cw} + {i}, (i64){v});"
        elif ot in ("i64", "u64"):
            gp = f"((u64*)P.p[{L.r_out[k]}])"
            src.append(f"__device__ __forceinline__ void sj_emit{k}(const SjParams& P, u64* a, i64 i, {ot} v) {{ "
                       f"atomicAdd((unsigned long long*)&a[i], (unsigned long long)v); }}")
            op = lambda a, i, v: f"atomicAdd((unsigned long long*)&{a}[{i}], (unsigned long long){v});"
        elif ot in ("float", "double", "i32", "u32"):
            gp = dst
            # f32: sj_fatom keeps subnormals (the hardware f32 atomic flushes them); f64 / int atomics are exact
            fa = "sj_fatom" if ot == "float" else "atomicAdd"
            src.append(f"__device__ __forceinline__ void sj_emit{k}(const SjParams& P, {ot}* a, i64 i, {ot} v) {{ "
                       f"{fa}(&a[i], v); }}")
            op = lambda a, i, v, fa=fa: f"{fa}(&{a}[{i}], {v});"
        else:
            raise CodegenError(f"sum output of dtype {s.dtype} is not supported")
        if k in priv:
            acc_decl += [f"  __shared__ {at} sj_sh{k}[{nacc}];",
                         f"  for (int i = threadIdx.x; i < {nacc}; i += blockDim.x) sj_sh{k}[i] = ({at})0;",
                         f"  {at}* const sj_acc{k} = sj_sh{k};"]
            acc_flush.append(f"  for (int i = threadIdx.x; i < {nacc}; i += blockDim.x) {{ const {at} x = sj_sh{k}[i]; "
                             f"if (x != ({at})0) {{ {at}* const g = {gp}; {op('g', 'i', 'x')} }} }}")
        else:
            acc_decl.append(f"  {at}* const sj_acc{k} = {gp};")
    # small tables into shared memory (the cuda target only: the host has no per-block memory)
    tab_decl, tab_copy = "", []
    if target == "cuda":
        tabs = [(i, f"P.p[{i}]", _size(sh) * np.dtype(dt).itemsize) for i, (sh, dt, impl) in zip(L.i_bc, bav)
                if impl is None]
        for j, c in enumerate(L.consts):
            sh, dt = (c[3], c[4]) if c[0] == "dyn" else (c[2].shape, c[2].dtype)
            tabs.append((base_const + j, f"SJ_CONST({j})", _size(tuple(sh)) * np.dtype(dt).itemsize))
        tab_decl, tab_copy = _stage_tables(fdefs, tabs)
    L.staged = len(tab_copy)
    acc_decl += tab_copy
    if tab_copy or any("__shared__" in a for a in acc_decl):
        acc_decl.append("  __syncthreads();")
        acc_flush.insert(0, "  __syncthreads();")
    if spawn is not None:
        prep.append(f"for (i64 i = g; i < {spawn.capacity}; i += s) SJ_SEQ[i] = (u32)i;")
    for nm, lines_ in (("prep", prep), ("post", post)):
        if lines_:
            flags |= 1 if nm == "prep" else 2
            src.append(f"extern \"C\" __global__ void sj_{nm}(const SjParams P, const i64 n_items) {{\n"
                       "  const i64 g = (i64)blockIdx.x * blockDim.x + threadIdx.x, s = (i64)gridDim.x * blockDim.x;\n  "
                       + "\n  ".join(lines_) + "\n}")
    if spawn is not None:
        flags |= 4         # grid is not capped by N (children)
    L.flags = flags
    if em.n_brk:
        src.append(BRK)
    if tab_decl:
        src.append(tab_decl)
    src += fdefs

    # ---------------------------------------------------------------- main kernel
    ywrite = "\n".join(
        (f"        (({ctype(dt)}*)P.p[{L.r_ys[j]}])[t] = y{j};" if sh == () else
         f"        for (int e = 0; e < {_size(sh)}; ++e) (({ctype(dt)}*)P.p[{L.r_ys[j]}])[t * {_size(sh)} + e] = y{j}[e];")
        for j, (sh, dt, _) in enumerate(yav))
    lb = f"__launch_bounds__({block})"
    # ONE item per claim, warp-aggregated: the needy lanes of a warp take one contiguous run between them, so
    # the counter sees one atomic per warp instead of one per lane.  Claiming a RUN of items per lane was
    # measured and removed: it leaves the last claims holding several items each, which is a tail the other
    # lanes wait for (2080 Ti, all 18 benchmarks/vs_jax cells: one item per claim is 1.46x faster on
    # random_walk's forward pass, 1.14-1.18x on transport, uniform and ode, and never slower).
    claim = """if (my_left == 0) {
          const unsigned m = __activemask();
          const int leader = __ffs(m) - 1;
          const unsigned rank = __popc(m & ((1u << lane) - 1u));
          u64 base = 0;
          if (lane == leader) base = atomicAdd((unsigned long long*)sj_ctr, (unsigned long long)__popc(m));
          my_next = __shfl_sync(m, base, leader) + rank; my_left = 1;
        }"""
    xdecl = "".join(svar("x", i, spc) for i, spc in enumerate(pav)) if spawn is not None else ""
    nlim = ""
    if spawn is None:
        fetch = f"""        {claim}
        t = my_next; --my_left;
        if (t >= (u64)n_items) break;"""
        if p.nact is not None:
            # n_active: the launch's item count, read from its broadcast operand.  Items in [n_active, n_items) do
            # not run: their per-item results are written as zero.
            nlim = (f"  const i64 sj_na = (i64)sj_ld({tptr(L.i_bc[p.nact], np.int32)});\n"
                    "  const i64 sj_nlim = sj_na < 0 ? 0 : (sj_na < n_items ? sj_na : n_items);\n")
            yzero = "\n".join(
                (f"          (({ctype(dt)}*)P.p[{L.r_ys[j]}])[t] = {ctype(dt)}{{}};" if sh == () else
                 f"          for (int e = 0; e < {_size(sh)}; ++e) (({ctype(dt)}*)P.p[{L.r_ys[j]}])[t * {_size(sh)} + e] = "
                 f"{ctype(dt)}{{}};")
                for j, (sh, dt, _) in enumerate(yav))
            fetch += f"""
        if (t >= (u64)sj_nlim) {{
{yzero}
          continue;
        }}"""
        finish = f"""{ywrite}"""
    else:
        if p.nact is not None:
            raise CodegenError("n_active with spawn is not supported (internal error)")
        Q = int(spawn.capacity)
        load_p = "\n".join(
            (f"            x{i} = sj_ld({tptr(L.i_pi[i], dt)} + t);" if sh == () else
             f"            for (int e = 0; e < {_size(sh)}; ++e) x{i}[e] = sj_ld({tptr(L.i_pi[i], dt)} + t * {_size(sh)} + e);")
            for i, (sh, dt, _) in enumerate(pav))
        load_q = "\n".join(
            (f"            x{i} = (({ctype(dt)}*)P.p[{L.r_q[i]}])[sl];" if sh == () else
             f"            for (int e = 0; e < {_size(sh)}; ++e) x{i}[e] = (({ctype(dt)}*)P.p[{L.r_q[i]}])[(i64)sl * {_size(sh)} + e];")
            for i, (sh, dt, _) in enumerate(pav))
        tr = f"y{p.trunc}" if p.trunc is not None else "false"
        # the recording primal (_reverse_spawn.rec_primal): the item's leaf `me` <- its queue slot / -1 - its index
        rec_q = "" if p.rec is None else f"\n            x{p.rec} = (i32)sl;"
        rec_p = "" if p.rec is None else f"\n            x{p.rec} = (i32)(-1 - (i64)t);"
        fetch = f"""        child = false;
        if ((i32)SJ_LDV(&SJ_Q(2)) > 0) {{                     // pop a child first (bounds the queue)
          if ((i32)atomicAdd(&SJ_Q(2), 0xffffffffu) > 0) {{
            const u32 h = atomicAdd(&SJ_Q(1), 1u);
            const u32 sl = h % {Q}u;
            while (SJ_LDV(&SJ_SEQ[sl]) != h + 1u) __nanosleep(32);
            __threadfence();
{load_q}{rec_q}
            __threadfence();
            atomicExch(&SJ_SEQ[sl], h + {Q}u);
            atomicSub(&SJ_Q(3), 1u);
            child = true;
          }} else {{
            atomicAdd(&SJ_Q(2), 1u);
          }}
        }}
        if (!child) {{
          {claim}
          t = my_next;
          if (t < (u64)n_items) {{
            --my_left;
            atomicAdd(&SJ_Q(4), 1u);
{load_p}{rec_p}
          }} else {{
            if (SJ_LDV(&SJ_Q(4)) == 0u && (i32)SJ_LDV(&SJ_Q(2)) <= 0) break;
            __nanosleep(200);
            continue;
          }}
        }}"""
        finish = f"""        if (child) {{ if ({tr}) ++n_ctr; }} else {{
{ywrite}
        }}
        __threadfence();
        atomicSub(&SJ_Q(4), 1u);"""
    main = f"""
extern "C" __global__ void {lb} sj_main(const SjParams P, const i64 n_items) {{
  u64* const sj_ctr = (u64*)P.p[{L.r_ctr}];
  const int lane = threadIdx.x & 31;
{chr(10).join(acc_decl)}
{nlim}{decls}{xdecl}  u64 my_next = 0; u32 my_left = 0;
  u64 t = 0; i64 item = 0; bool active = false, done = false, child = false; i32 n_ctr = 0;
  while (true) {{
    if (!active) {{
{fetch}
      item = (i64)t;
      done = false;
      {start}
      active = true;
    }}
    if (!done) {{
      {advance}
    }}
    if (done) {{
{finish}
      active = false;
    }}
  }}
  if (n_ctr) atomicAdd(&SJ_STATS[2], n_ctr);
{chr(10).join(acc_flush)}
}}
"""
    src.append(main)
    L.prims = sorted(em.prims)
    return "\n".join(src), L


# the carry-word ("wide") fixed-point accumulator: a bin is an int64 low word (two's complement, wrapping) plus an
# i32 high word, the net signed carry out of the low word.  atomicAdd returns the old low word o: adding q wrapped iff
# q > 0 and o + q < o (carry +1) or q < 0 and o + q > o (carry -1).  The exact total is low + carry * 2^64 (a sum of
# +-1 per add, so independent of the order of the adds: bitwise deterministic).  sj_wide_f64 rounds it to double
# once (correctly: a 128-bit magnitude rounded to 53 bits); with carry 0 it is (double)(i64)low, today's value.
WIDE = r"""
__device__ __forceinline__ void sj_wadd(u64* a, i32* c, i64 q) {
  const i64 o = (i64)atomicAdd((unsigned long long*)a, (unsigned long long)q);
  const i64 r = (i64)((u64)o + (u64)q);
  if (q > 0 && r < o) atomicAdd(c, 1); else if (q < 0 && r > o) atomicAdd(c, -1); }
__device__ __noinline__ double sj_wide_big(u64 lo, i32 c) {
  u64 h = (u64)((i64)c - (i64)(lo >> 63)), l = lo;          // the 128-bit two's complement h:l
  const bool neg = (i64)h < 0;
  if (neg) { l = ~l + 1ull; h = ~h + (l == 0ull ? 1ull : 0ull); }
  double m;
  int e = 0;
  if (h == 0ull) { m = (double)l; } else {
    e = 64 - __clzll((long long)h);                          // h's bits: the magnitude has 64 + e
    const u64 t = (e == 64 ? h : ((h << (64 - e)) | (l >> e)));   // its top 64 bits (bit 63 set)
    const u64 st = (e == 64 ? l : (l << (64 - e)));          // the bits below them: sticky
    m = (double)(t | (st != 0ull ? 1ull : 0ull));            // one rounding to 53 bits (bit 0 only as sticky)
  }
  m = ldexp(m, e);
  return neg ? -m : m; }
__device__ __forceinline__ double sj_wide_f64(u64 lo, i32 c) { return c == 0 ? (double)(i64)lo : sj_wide_big(lo, c); }
"""
# the exact f32 sum (TSpec.exact).  An f32 v = m * 2^(p - 149) (m < 2^24: the mantissa with its implicit bit; p =
# biased exponent - 1, 0 for a subnormal) is an integer below 2^277 in units of 2^-149.  A bin keeps the sum of those
# integers in DENSE_WORDS = 5 signed int64 words, word j of weight 2^(63 j) (carry-save: adjacent words overlap by one
# bit): an emit adds the signed pieces of m << (p % 63) below 2^63 into words p / 63 and p / 63 + 1 with atomicAdd.  A
# word whose add overflows int64 (read from the old value: the total of that word is then off by 2^64 = 2 x 2^63) adds
# +-2 into the next word.  Each step keeps sum(word_j 2^(63 j)) + the carry in flight, so once every add has run the
# words represent the exact total, whatever the order of the adds (the words themselves may differ; sj_dread
# canonicalises them): bitwise deterministic, no scale, no floor, no cap.  A carry is needed only when a word drifts by
# 2^63, never because the total changes sign (a two's-complement bin would ripple through every upper word at each zero
# crossing: measured 1.20x lumen's SK time).  A carry out of the top word (word 4, weight 2^252: only once a bin's
# running sum passes about 2^315 units = 2^166, i.e. 2^38 emits near FLT_MAX) is detected from the old value too and
# returns true: the caller flags the bin (NaN, counted), so a bin is exact or NaN, never silently wrong.  (Only in that
# regime can the flag depend on the order: a running sum that passes 2^166 and comes back.)
# sj_dread sums the words into a two's-complement 384-bit integer, takes its magnitude, its top 64 bits plus a sticky
# bit, and rounds once to 24 bits (nearest, ties to even): the correctly rounded f32 of the exact total (an
# f32-subnormal total is below 2^24 units, so exact).  tallymap/backends/xla.py (_xemit, _xread) keeps the same integer
# as 32-bit digits in int64 words (EXACT_WORDS) and rounds it the same way, so the backends agree bitwise.
DENSE = r"""
__device__ __noinline__ bool sj_dwrap(u64* a, int j, i64 c) {       // the rare carry c = +-2 into word j; true: lost
  for (; j < 5; ++j) { const i64 o = (i64)atomicAdd((unsigned long long*)&a[j], (unsigned long long)c);
    const i64 r = (i64)((u64)o + (u64)c);
    if (!((c > 0 && r < o) || (c < 0 && r > o))) return false; }
  return true; }
__device__ __forceinline__ bool sj_dadd(u64* a, int j, i64 q) {     // word j += q; an int64 overflow carries +-2
  const i64 o = (i64)atomicAdd((unsigned long long*)&a[j], (unsigned long long)q);
  const i64 r = (i64)((u64)o + (u64)q);
  if (q > 0 && r < o) return sj_dwrap(a, j + 1, 2);
  if (q < 0 && r > o) return sj_dwrap(a, j + 1, -2);
  return false; }
__device__ __forceinline__ bool sj_demit(u64* a, float v) {         // true: flag the bin (non-finite, or a carry lost)
  const u32 b = __float_as_uint(v), e = (b >> 23) & 255u;
  if (e == 255u) return true;
  const u64 m = (u64)(b & 0x7fffffu) | (e ? 0x800000ull : 0ull);
  if (m == 0ull) return false;                                      // +-0 adds nothing
  const u32 p = e ? e - 1u : 0u, j = p / 63u, s = p - 63u * j;
  const i64 lo = (i64)((m << s) & 0x7fffffffffffffffull), hi = s >= 40u ? (i64)(m >> (63u - s)) : 0ll;
  const bool neg = (b >> 31) != 0u;
  bool lost = false;
  if (lo) lost = sj_dadd(a, (int)j, neg ? -lo : lo);
  if (hi) lost = sj_dadd(a, (int)j + 1, neg ? -hi : hi) || lost;
  return lost; }
__device__ __noinline__ float sj_dread(const u64* a, bool* bad) {
  u64 d[6] = {0ull, 0ull, 0ull, 0ull, 0ull, 0ull};                  // sum_j word_j 2^(63 j), two's complement
  for (int j = 0; j < 5; ++j) {
    const i64 w = (i64)a[j];
    const int sh = 63 * j, l = sh >> 6, b = sh & 63;
    const u64 ext = w < 0 ? ~0ull : 0ull;
    u64 x[2] = {(u64)w << b, b ? (u64)(w >> (64 - b)) : ext};
    u64 c = 0ull;
    for (int k = l; k < 6; ++k) {
      const u64 y = k - l < 2 ? x[k - l] : ext, t = d[k] + y, u = t + c;
      c = (t < y ? 1ull : 0ull) + (u < t ? 1ull : 0ull);
      d[k] = u; } }
  const bool neg = (i64)d[5] < 0;
  if (neg) {                                                        // the magnitude: ~d + 1
    u64 k = 1ull;
    for (int j = 0; j < 6; ++j) { d[j] = ~d[j] + k; k = (k && d[j] == 0ull) ? 1ull : 0ull; } }
  int t = 5;
  while (t >= 0 && d[t] == 0ull) --t;
  if (t < 0) return 0.0f;
  const int s = __clzll((long long)d[t]);
  const u64 d1 = t >= 1 ? d[t - 1] : 0ull;
  const u64 top = s ? (d[t] << s) | (d1 >> (64 - s)) : d[t];       // the top 64 bits (bit 63 set)
  bool st = s ? (d1 << s) != 0ull : d1 != 0ull;                     // sticky: any bit below them
  for (int j = 0; j < t - 1; ++j) st = st || d[j] != 0ull;
  u64 q = top >> 40;
  const u64 r = top & 0xffffffffffull, half = 0x8000000000ull;
  if (r > half || (r == half && (st || (q & 1ull)))) ++q;
  const float o = (float)ldexp((double)q, 64 * t + 40 - s - 149);
  if (!(fabs((double)o) <= 3.4028234663852886e38)) *bad = true;
  return neg ? -o : o; }
"""
# integer output types: [lo, hi) of the values a fixed-point bin converts to without overflow
_IRANGE = {"i8": ("-128.0", "128.0"), "u8": ("0.0", "256.0"), "i16": ("-32768.0", "32768.0"), "u16": ("0.0", "65536.0"),
           "i32": ("-2147483648.0", "2147483648.0"), "u32": ("0.0", "4294967296.0"),
           "i64": ("-9223372036854775808.0", "9223372036854775808.0"), "u64": ("0.0", "18446744073709551616.0")}


def _scratch(s):
    return np.int64 if s.fixed_point or s.exact else None


def _scratch_words(s):
    """u32 words of a sum's scratch buffer: exact: per bin DENSE_WORDS int64 words, then one flag bit per bin (DENSE);
    fixed point: per bin a u64 low word, an i32 carry word and a u32 flag (WIDE)."""
    n = int(np.prod(s.lead, dtype=np.int64)) * s.S
    return 2 * DENSE_WORDS * n + (n + 31) // 32 if s.exact else 4 * n


def _acc_t(s):
    if s.fixed_point or s.exact:
        return "u64"
    ot = ctype(s.dtype)
    return "u64" if ot in ("i64", "u64") else ot


def _emit_rule(em, p, ins):
    t, idx, val = ins
    s = p.tallies[t.slot]
    if s.kind == "discard":
        return t
    S = s.S
    nl = int(np.prod(s.lead, dtype=np.int64))
    lead = tuple(s.lead)
    I = tuple(idx.shape)
    for _, B in em.lane_masks:          # (inside a while loop with a batched predicate: op_table._batched_body)
        if lead[:len(B)] != B:
            raise CodegenError(f"an emit into a tally with lead dims {lead} inside a while loop whose predicate has "
                               f"shape {B} (internal error: JAX batches every carry of such a loop)")

    def live(li):                        # the emit of lane index li runs only where every enclosing predicate holds
        c = " && ".join(elem(pm, list(li[:len(B)])) for pm, B in em.lane_masks)
        return f"if ({c}) " if c else ""

    def body(ii):
        fi = em.fresh("ei")
        em.line(f"const i64 {fi} = (i64)({elem(idx, ii)});")
        em.open(f"if ((u64){fi} >= {S}ull)")          # out of range (negative too): dropped, counted
        em.line("atomicAdd(&SJ_STATS[4], 1);")
        em.close("} else {")
        em.ind += 1
        if nl <= 32:
            for l in range(nl):
                li = unravel(l, lead) if lead else []
                em.line(f"{live(li)}sj_emit{t.slot}(P, sj_acc{t.slot}, {l * S} + {fi}, "
                        f"{elem(val, list(li) + list(ii))});")
        else:
            lv = em.fresh("l")
            em.open(f"for (int {lv} = 0; {lv} < {nl}; ++{lv})")
            li = unravel(lv, lead)
            em.line(f"{live(li)}sj_emit{t.slot}(P, sj_acc{t.slot}, (i64){lv} * {S} + {fi}, {elem(val, li + list(ii))});")
            em.close()
        em.close()
    em.each(I, body)
    return t


def _spawn_rule(em, p, ins, pav):
    q, when, leaves = ins[0], ins[1], ins[2:]
    if em.lane_masks:
        raise CodegenError("a spawn inside a while loop with a batched predicate is not supported")
    Q = int(p.spawn.capacity)
    old, tk, sl = em.fresh("qo"), em.fresh("qt"), em.fresh("qs")
    em.open(f"if ({elem(when, [])})")
    em.line(f"const u32 {old} = atomicAdd(&SJ_Q(3), 1u);")
    em.open(f"if ({old} >= {Q}u)")
    em.line("atomicSub(&SJ_Q(3), 1u); atomicAdd(&SJ_STATS[1], 1);")
    em.close("} else {")
    em.ind += 1
    em.line("atomicAdd(&SJ_Q(4), 1u); atomicAdd(&SJ_STATS[0], 1);")
    em.line(f"const u32 {tk} = atomicAdd(&SJ_Q(0), 1u);")
    em.line(f"const u32 {sl} = {tk} % {Q}u;")
    em.line(f"while (SJ_LDV(&SJ_SEQ[{sl}]) != {tk}) __nanosleep(32);")
    if len(leaves) > len(pav) or any(not np.issubdtype(np.dtype(dt), np.inexact) for _, dt, _ in pav[len(leaves):]):
        raise CodegenError(f"a spawn with {len(leaves)} child leaves in a program whose queue has {len(pav)} "
                           "(internal error)")
    for i, (sh, dt, _) in enumerate(pav):
        # (missing trailing leaves: the tangents of a spawn JAX bound without calling its jvp rule, i.e. zero;
        # written as zero, since the queue slot is reused -- see xla._check_tail)
        x = leaves[i] if i < len(leaves) else None
        n = _size(sh)
        qp = f"(({ctype(dt)}*)SJ_QBUF({i}))"
        if x is None:
            em.line(f"for (i64 z_ = 0; z_ < {n}; ++z_) {qp}[(i64){sl} * {n} + z_] = ({ctype(dt)})0;")
        elif sh == ():
            em.line(f"{qp}[{sl}] = {elem(x, [])};")
        else:
            em.each(sh, lambda idx, x=x, qp=qp, n=n, sh=sh: em.line(
                f"{qp}[(i64){sl} * {n} + {_ravel(idx, sh)}] = {elem(x, idx)};"))
    em.line("__threadfence();")
    em.line(f"atomicExch(&SJ_SEQ[{sl}], {tk} + 1u);")
    em.line("atomicAdd(&SJ_Q(2), 1u);")
    em.close()
    em.close()
    return q


def _ravel(idx, shape):
    from .codegen import ravel
    return ravel(idx, shape)


def _claim_rule(em, p, ins, claims):
    """a Records one-pass slot claim (_tally.claim_p): the next slot of the output's raw buffer from its global u64
    counter, -1 when `when` is false or the buffer is full.  One atomic per record; the counter keeps the true total
    (the layout is canonicalised after the launch, _primitive._canon, from the per-item counts and the key tallies)."""
    t, when, _k = ins
    if em.lane_masks:
        raise CodegenError("t.append inside a while loop with a batched predicate is not supported")
    s = p.tallies[t.slot]
    sl = em.fresh("as")
    if s.kind == "discard":
        em.line(f"const i32 {sl} = -1;")
        return [t, Local(sl, (), np.int32)]
    j = claims.setdefault(t.slot, len(claims))
    em.line(f"i32 {sl} = -1;")
    em.open(f"if ({elem(when, [])})")
    em.line(f"const u64 o_{sl} = atomicAdd((unsigned long long*)&SJ_APPC[{j}], 1ull);")
    em.line(f"if (o_{sl} < {int(s.S)}ull) {sl} = (i32)o_{sl};")
    em.close()
    return [t, Local(sl, (), np.int32)]
