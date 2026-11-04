"""CUDA source for one tallymap program: the WHOLE item function is one device function (its loops are C loops),
run by persistent threads that claim items with warp-aggregated atomics.  Emits are atomics into the output
(small outputs privatised in shared memory); spawned children go through a bounded MPMC ticket queue.

Buffers (one by-value struct of pointers): [per-item inputs] [broadcast inputs] [hoisted constants]
| results: [sum tallies] [scratch accumulators (fixed64 / f64)] [claim counter u64] [stats i32[5]]
           [spawn: queue leaves, seq, qctl] [per-item results].
N (the item count) is a kernel ARGUMENT: one cubin serves every N.
"""
import numpy as np

from .codegen import CodegenError, Emitter, Local, Scalar, Table, VTally, _size, ctype, elem, unravel
from .op_table import phys

PRELUDE = r"""
typedef signed char i8; typedef unsigned char u8; typedef short i16; typedef unsigned short u16;
typedef int i32; typedef unsigned int u32; typedef long long i64; typedef unsigned long long u64;
#define SJ_MAXP 504
struct SjParams { void* p[SJ_MAXP]; };
#ifndef TALLYMAP_APPROX_DIVSQRT   // cuda_math='ieee': correctly rounded (the portable reference)
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

template <class T> __device__ __forceinline__ T sj_ld(const T* p) { return __ldg(p); }
template <> __device__ __forceinline__ bool sj_ld<bool>(const bool* p) { return __ldg((const u8*)p) != 0; }
// lax.max/min: NaN-propagating
template <class T> __device__ __forceinline__ T sj_fmax(T a, T b) { return (a >= b || a != a) ? a : b; }
template <class T> __device__ __forceinline__ T sj_fmin(T a, T b) { return (a <= b || a != a) ? a : b; }
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
__device__ __forceinline__ bool sj_inb(i64 x, i64 hi) { return x >= 0 && x <= hi; }
__device__ __forceinline__ float sj_maximum(float a, float b) { return (a != a || b != b) ? __int_as_float(0x7fc00000) : (a > b ? a : (b > a ? b : (signbit(a) ? b : a))); }
__device__ __forceinline__ float sj_minimum(float a, float b) { return (a != a || b != b) ? __int_as_float(0x7fc00000) : (a < b ? a : (b < a ? b : (signbit(a) ? a : b))); }

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

// float min/max atomics
__device__ __forceinline__ void sj_atomic_max_f32(float* a, float v) {
  if (v != v) { atomicExch((u32*)a, __float_as_uint(v)); return; }
  if (v >= 0.0f) atomicMax((i32*)a, __float_as_int(v)); else atomicMin((u32*)a, __float_as_uint(v)); }
#define SJ_LDV(p) (*((volatile u32*)(p)))
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


def _split(cj):
    """If the item has exactly ONE top-level while loop, split it into pre / (cond, body) / post so the kernel can
    refill a lane with a new item at STEP granularity (a lane whose item finished does not idle until the whole
    warp's items finish; measurements: docs/design.md section 1b).  -> None (whole-item mode) or dict of closed
    jaxprs + wiring."""
    import jax
    from jax.extend.core import Literal

    from ..._jaxpr import bind, is_dropvar, lit, while_parts
    jx = cj.jaxpr
    ws = [i for i, e in enumerate(jx.eqns) if e.primitive.name == "while"]
    if len(ws) != 1:
        return None
    w = ws[0]
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
                nc=nc, nb=nb, invars=list(jx.invars), constvars=consts)


BLOCK = 128          # threads per block (measured: 64 / 128 / 256 within 1%)
PRIV_BYTES = 4096    # shared-memory budget for privatised sum tallies per block


def privatised(tallies, budget=PRIV_BYTES):
    """indices of the sum tallies the kernel keeps in shared memory (in order, while they fit in `budget`)."""
    out = []
    for k, s in enumerate(tallies):
        if s.kind != "sum":
            continue
        nbytes = int(np.prod(s.lead, dtype=np.int64)) * s.S * (8 if _acc_t(s) in ("u64", "double") else 4)
        if 0 < nbytes <= budget:
            budget -= nbytes
            out.append(k)
    return out


def build_source(p, *, block=BLOCK, split=True, inplace=True):
    """-> (CUDA source, Layout).  p: tallymap._api.ItemProgram.  Its jaxpr is first rewritten twice: the foreign
    primitives a backend has to lower itself (_foreign.expand_late, e.g. lineax's linear_solve, which may only be
    expanded now that every autodiff rule has been applied) and then the JAX sugar (canon.py)."""
    import dataclasses

    from ... import _foreign
    from .canon import canonical
    p = dataclasses.replace(p, cj=canonical(_foreign.expand_late(p.cj)))
    em = Emitter()
    em.where = p.where
    em.inplace = inplace
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

    outs = [k for k, s in enumerate(p.tallies) if s.kind == "sum"]
    L.n_out = len(outs)
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
    L.cjs = {}
    fdefs = []

    def function(name, cj, ins, out_names=None):
        """emit `cj` as a device function taking the base params + (decl, Val) inputs; non-tally results are
        written to reference params o<j> (or the given names).  -> (results, out (shape, dtype, impl) or None)"""
        L.cjs[name] = cj
        em.lines, em.ind = [], 1
        res = em.run_top(cj, [v for _, v in ins], name)
        odecl, ospec = [], []
        for j, (v, r) in enumerate(zip(cj.jaxpr.outvars, res)):
            if isinstance(r, VTally):
                ospec.append(None)
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
        return f"  {ctype(dt)} {prefix}{j}{'' if sh == () else f'[{max(1, _size(sh))}]'};\n"

    sp = _split(p.cj) if split else None
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
                    raise CodegenError(f"loop constant of kind {type(x).__name__} cannot cross the step split")
                return None, x, None
            sh, dt = tuple(x.shape), x.dtype
            return _decl(name, sh, dt), _val(name, sh, dt, x.key_impl), (x.name if isinstance(x, Local) else x.text)
        pc = [as_param(val_of(v), f"c{j}") for j, v in enumerate(cc)]
        pb = [as_param(val_of(v), f"b{j}") for j, v in enumerate(bc)]
        pk = [as_param(x, f"w{j}") for j, x in enumerate(kvals)]
        function("cond", sp["cond"], [(d, x) for d, x, _ in pc + pk])
        decls += "  bool sj_pr;\n"
        cargs = [a for _, _, a in pc + pk if a is not None]
        bargs = [a for _, _, a in pb + pk if a is not None]
        bres, bspec = function("body", sp["body"], [(d, x) for d, x, _ in pb + pk])
        nargs, upd = [], []
        for j, spc in enumerate(bspec):
            if spc is None:
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
    r = nin
    L.r_out = {}
    for k in outs:
        L.r_out[k] = r
        r += 1
    L.r_scr = {}
    for k in outs:
        if _scratch(p.tallies[k]) is not None:
            L.r_scr[k] = r
            r += 1
    L.r_ctr, L.r_stats = r, r + 1
    r += 2
    if spawn is not None:
        L.r_q = list(range(r, r + nP))
        r += nP
        L.r_seq, L.r_qctl = r, r + 1
        r += 2
    L.n_zero = r - nin
    L.r_ys = list(range(r, r + len(yav)))
    r += len(yav)
    L.n_buf = r
    if r > 504:
        raise CodegenError(f"too many buffers ({r} > 504)")
    L.pav, L.yav = pav, yav

    src = [PRELUDE, f"#define SJ_CONST(j) (P.p[{base_const} + (j)])", f"#define SJ_STATS ((i32*)P.p[{L.r_stats}])"]
    if spawn is not None:
        src.append(f"#define SJ_QCTL ((u32*)P.p[{L.r_qctl}])\n#define SJ_SEQ ((u32*)P.p[{L.r_seq}])\n"
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
        if s.exact:
            gp = f"((u64*)P.p[{L.r_scr[k]}])"
            sc = float(s.scale)
            src.append(f"__device__ __forceinline__ void sj_emit{k}(const SjParams& P, u64* a, i64 i, {ot} v) {{ "
                       f"const double x = (double)v * {sc!r}; if (!(fabs(x) < 9.2e18)) {{ atomicAdd(&SJ_STATS[3], 1); "
                       f"return; }} atomicAdd((unsigned long long*)&a[i], (unsigned long long)__double2ll_rn(x)); }}")
            post.append(f"for (i64 i = g; i < {tot}; i += s) {dst}[i] = ({ot})(((double)(i64){gp}[i]) * {1.0 / sc!r});")
            op = lambda a, i, v: f"atomicAdd((unsigned long long*)&{a}[{i}], (unsigned long long){v});"
        elif ot in ("i64", "u64"):
            gp = f"((u64*)P.p[{L.r_out[k]}])"
            src.append(f"__device__ __forceinline__ void sj_emit{k}(const SjParams& P, u64* a, i64 i, {ot} v) {{ "
                       f"atomicAdd((unsigned long long*)&a[i], (unsigned long long)v); }}")
            op = lambda a, i, v: f"atomicAdd((unsigned long long*)&{a}[{i}], (unsigned long long){v});"
        elif ot in ("float", "double", "i32", "u32"):
            gp = dst
            src.append(f"__device__ __forceinline__ void sj_emit{k}(const SjParams& P, {ot}* a, i64 i, {ot} v) {{ "
                       f"atomicAdd(&a[i], v); }}")
            op = lambda a, i, v: f"atomicAdd(&{a}[{i}], {v});"
        else:
            raise CodegenError(f"sum output of dtype {s.dtype} is not supported")
        if k in priv:
            acc_decl += [f"  __shared__ {at} sj_sh{k}[{tot}];",
                         f"  for (int i = threadIdx.x; i < {tot}; i += blockDim.x) sj_sh{k}[i] = ({at})0;",
                         f"  {at}* const sj_acc{k} = sj_sh{k};"]
            acc_flush.append(f"  for (int i = threadIdx.x; i < {tot}; i += blockDim.x) {{ const {at} x = sj_sh{k}[i]; "
                             f"if (x != ({at})0) {{ {at}* const g = {gp}; {op('g', 'i', 'x')} }} }}")
        else:
            acc_decl.append(f"  {at}* const sj_acc{k} = {gp};")
    if any("__shared__" in a for a in acc_decl):
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
    if spawn is None:
        fetch = f"""        {claim}
        t = my_next; --my_left;
        if (t >= (u64)n_items) break;"""
        finish = f"""{ywrite}"""
    else:
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
        fetch = f"""        child = false;
        if ((i32)SJ_LDV(&SJ_QCTL[2]) > 0) {{                     // pop a child first (bounds the queue)
          if ((i32)atomicAdd(&SJ_QCTL[2], 0xffffffffu) > 0) {{
            const u32 h = atomicAdd(&SJ_QCTL[1], 1u);
            const u32 sl = h % {Q}u;
            while (SJ_LDV(&SJ_SEQ[sl]) != h + 1u) __nanosleep(32);
            __threadfence();
{load_q}
            __threadfence();
            atomicExch(&SJ_SEQ[sl], h + {Q}u);
            atomicSub(&SJ_QCTL[3], 1u);
            child = true;
          }} else {{
            atomicAdd(&SJ_QCTL[2], 1u);
          }}
        }}
        if (!child) {{
          {claim}
          t = my_next;
          if (t < (u64)n_items) {{
            --my_left;
            atomicAdd(&SJ_QCTL[4], 1u);
{load_p}
          }} else {{
            if (SJ_LDV(&SJ_QCTL[4]) == 0u && (i32)SJ_LDV(&SJ_QCTL[2]) <= 0) break;
            __nanosleep(200);
            continue;
          }}
        }}"""
        finish = f"""        if (child) {{ if ({tr}) ++n_ctr; }} else {{
{ywrite}
        }}
        __threadfence();
        atomicSub(&SJ_QCTL[4], 1u);"""
    main = f"""
extern "C" __global__ void {lb} sj_main(const SjParams P, const i64 n_items) {{
  u64* const sj_ctr = (u64*)P.p[{L.r_ctr}];
  const int lane = threadIdx.x & 31;
{chr(10).join(acc_decl)}
{decls}{xdecl}  u64 my_next = 0; u32 my_left = 0;
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


def _scratch(s):
    return np.int64 if s.exact else None


def _acc_t(s):
    if s.exact:
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
                em.line(f"sj_emit{t.slot}(P, sj_acc{t.slot}, {l * S} + {fi}, {elem(val, list(li) + list(ii))});")
        else:
            lv = em.fresh("l")
            em.open(f"for (int {lv} = 0; {lv} < {nl}; ++{lv})")
            em.line(f"sj_emit{t.slot}(P, sj_acc{t.slot}, (i64){lv} * {S} + {fi}, {elem(val, unravel(lv, lead) + list(ii))});")
            em.close()
        em.close()
    em.each(I, body)
    return t


def _spawn_rule(em, p, ins, pav):
    q, when, leaves = ins[0], ins[1], ins[2:]
    Q = int(p.spawn.capacity)
    old, tk, sl = em.fresh("qo"), em.fresh("qt"), em.fresh("qs")
    em.open(f"if ({elem(when, [])})")
    em.line(f"const u32 {old} = atomicAdd(&SJ_QCTL[3], 1u);")
    em.open(f"if ({old} >= {Q}u)")
    em.line("atomicSub(&SJ_QCTL[3], 1u); atomicAdd(&SJ_STATS[1], 1);")
    em.close("} else {")
    em.ind += 1
    em.line("atomicAdd(&SJ_QCTL[4], 1u); atomicAdd(&SJ_STATS[0], 1);")
    em.line(f"const u32 {tk} = atomicAdd(&SJ_QCTL[0], 1u);")
    em.line(f"const u32 {sl} = {tk} % {Q}u;")
    em.line(f"while (SJ_LDV(&SJ_SEQ[{sl}]) != {tk}) __nanosleep(32);")
    for i, ((sh, dt, _), x) in enumerate(zip(pav, leaves)):
        n = _size(sh)
        qp = f"(({ctype(dt)}*)SJ_QBUF({i}))"
        if sh == ():
            em.line(f"{qp}[{sl}] = {elem(x, [])};")
        else:
            em.each(sh, lambda idx, x=x, qp=qp, n=n, sh=sh: em.line(
                f"{qp}[(i64){sl} * {n} + {_ravel(idx, sh)}] = {elem(x, idx)};"))
    em.line("__threadfence();")
    em.line(f"atomicExch(&SJ_SEQ[{sl}], {tk} + 1u);")
    em.line("atomicAdd(&SJ_QCTL[2], 1u);")
    em.close()
    em.close()
    return q


def _ravel(idx, shape):
    from .codegen import ravel
    return ravel(idx, shape)
