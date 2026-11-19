"""Scalar jaxpr -> CUDA C: the values and the evaluator of the cuda emitter.

The user's per-item functions are traced UN-vmapped, so every jaxpr value is a
scalar or a small fixed-shape array.  Each value becomes one of

  Scalar  a C scalar expression (a variable or a literal)
  Local   a per-thread C array `T name[n]` (row-major)
  Table   a strided window into a global-memory buffer (`__ldg` loads)
  View    a lazy index remapping of another value (slice / broadcast /
          reshape / transpose / dynamic_slice ...): no copy
  Lazy    elements computed on demand from the index (iota, concatenate, pad)

and each jaxpr equation is emitted as straight-line C (elementwise math with
explicit round-to-nearest intrinsics, no FMA contraction), C loops / ifs for
while / cond / scan, address arithmetic + loads for gathers from tables.

Float semantics: +, -, * round to nearest with no FMA contraction (f32 and f64).  f32 division, sqrt and rsqrt follow
the cuda_math option (kernel/__init__.py): 'approx', the DEFAULT, is the div.full / sqrt.approx / rsqrt.approx PTX
XLA:GPU emits; 'ieee' rounds them correctly (f64 always does, and so does the CPU target).  Transcendentals call the
CUDA math library (libdevice; libm on the CPU).  Results agree with XLA only statistically (see docs/validation.md),
never bitwise by design.
"""

import re

import jax
import numpy as np
from jax.extend.core import ClosedJaxpr, Literal

from ..._errors import ad_advice
from ..._jaxpr import call_jaxpr, is_dropvar, located
from . import ranges
from .ranges import eqn_ranges

UNROLL = 32          # arrays with <= UNROLL elements are emitted as straight-line code
MAX_LOCAL = 1 << 14  # largest per-thread local array we agree to materialise
# TEST-ONLY (tests patch it, like _primitive.FUSE_JVP): None, or a byte value every UNINITIALISED per-thread
# declaration (Emitter.declare, kernel.build_source's carry / state / result variables) is filled with.  A program
# that never reads an uninitialised value gives bitwise-equal results for FILL = 0x00 and 0xFF (tests/reverse.py).
FILL = None
# TEST-ONLY A/B control of the aliasing analysis: False never updates a per-item array in place (every
# dynamic_update_slice and scatter copies; the step schedule copies every loop carry).  Part of the kernel key.
INPLACE = True
# TEST-ONLY A/B control: False never fuses a functional update under a select into one guarded update
# (guarded_updates).  Part of the kernel key.
GUARDED = True


def fill_decl(n):
    """the statement that fills the just-declared variable `n` with the FILL byte ('' when FILL is None)."""
    if FILL is None:
        return ""
    return (f" {{ unsigned char* sj_f = (unsigned char*)&{n}; for (int sj_e = 0; sj_e < (int)sizeof({n}); ++sj_e)"
            f" sj_f[sj_e] = {int(FILL) & 0xFF}; }}")
_RESERVED = frozenset(f"{p}{b}" for p in "iu" for b in (8, 16, 32, 64))   # the integer typedefs of the kernel template


class CodegenError(NotImplementedError):
    """Raised for a jaxpr construct the cuda emitter cannot lower."""


# ----------------------------------------------------------------------------- dtypes
_CT = {np.dtype(np.float32): "float", np.dtype(np.float64): "double",
       np.dtype(np.int8): "i8", np.dtype(np.int16): "i16", np.dtype(np.int32): "i32",
       np.dtype(np.int64): "i64", np.dtype(np.uint8): "u8", np.dtype(np.uint16): "u16",
       np.dtype(np.uint32): "u32", np.dtype(np.uint64): "u64", np.dtype(np.bool_): "bool",
       np.dtype(np.complex64): "sj_c64", np.dtype(np.complex128): "sj_c128"}
_UT = {"i8": "u8", "i16": "u16", "i32": "u32", "i64": "u64",
       "u8": "u8", "u16": "u16", "u32": "u32", "u64": "u64"}
_ST = {"u8": "i8", "u16": "i16", "u32": "i32", "u64": "i64",
       "i8": "i8", "i16": "i16", "i32": "i32", "i64": "i64"}


def is_key_dtype(dt):
    try:
        return jax.dtypes.issubdtype(dt, jax.dtypes.prng_key)
    except Exception:
        return False


_FLOAT0 = np.dtype([("float0", "V")])


def ctype(dt):
    dt = np.dtype(dt)
    if dt == _FLOAT0:            # jvp tangent of an integer value: carries no data; represented as a bool false
        return "bool"
    if dt not in _CT:
        raise CodegenError(f"dtype {dt} is not supported by the kernel backend")
    return _CT[dt]


def _isf(dt):
    return np.dtype(dt).kind == "f"


def _isint(dt):
    return np.dtype(dt).kind in "iu"


def _bits(dt):
    return np.dtype(dt).itemsize * 8


def lit(val, dt):
    """C literal for a scalar of numpy dtype dt (bit exact)."""
    dt = np.dtype(dt)
    if dt == _FLOAT0:
        return "false"
    v = np.asarray(val).astype(dt)[()]
    if dt == np.bool_:
        return "true" if bool(v) else "false"
    if dt.kind == "c":
        ft = np.float32 if dt == np.complex64 else np.float64
        return f"sj_mk{'c64' if dt == np.complex64 else 'c128'}({lit(np.real(v), ft)}, {lit(np.imag(v), ft)})"
    if dt.kind == "f":
        if dt == np.float32:
            u = int(np.asarray(v, np.float32).view(np.uint32))
            if not np.isfinite(v):
                return f"__int_as_float(0x{u:08x})"
            return f"{float(v).hex()}f"
        u = int(np.asarray(v, np.float64).view(np.uint64))
        if not np.isfinite(v):
            return f"__longlong_as_double(0x{u:016x}ll)"
        return float(v).hex()
    ct = ctype(dt)
    iv = int(v)
    if dt.kind == "u":
        return f"(({ct}){iv}ull)"
    if iv == np.iinfo(dt).min:
        return f"(({ct})0x{iv & ((1 << _bits(dt)) - 1):x}ull)"
    return f"(({ct}){iv}ll)"


# ----------------------------------------------------------------------------- index exprs
def ia(*xs):
    """Sum of index expressions (ints fold)."""
    c, terms = 0, []
    for x in xs:
        if isinstance(x, (int, np.integer)):
            c += int(x)
        else:
            terms.append(x)
    if not terms:
        return c
    s = " + ".join(f"({t})" for t in terms)
    return s if c == 0 else f"{s} + {c}"


def im(a, b):
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) * int(b)
    if isinstance(a, (int, np.integer)):
        a, b = b, a
    if b == 0:
        return 0
    if b == 1:
        return a
    return f"({a})*{b}"


def _row_major(shape):
    st, acc = [], 1
    for n in reversed(shape):
        st.append(acc)
        acc *= int(n)
    return tuple(reversed(st))


def ravel(idx, shape):
    return ia(*[im(i, s) for i, s in zip(idx, _row_major(shape))])


def unravel(flat, shape):
    if isinstance(flat, (int, np.integer)):
        return [int(x) for x in np.unravel_index(int(flat), shape)] if shape else []
    out, total = [], _size(shape)
    for s, n in zip(_row_major(shape), shape):
        e = flat if s == 1 else f"(({flat}) / {s})"
        out.append(e if s * n == total else f"(({e}) % {n})")
    return out


def _size(shape):
    return int(np.prod(shape)) if len(shape) else 1


# ----------------------------------------------------------------------------- values
class Val:
    shape = ()
    dtype = None
    cval = None       # compile-time known scalar (numpy) or None
    key_impl = None   # PRNG impl when this value is a typed key array (physical repr)

    @property
    def size(self):
        return _size(self.shape)


class Scalar(Val):
    def __init__(self, text, dtype, cval=None):
        self.text, self.dtype, self.shape, self.cval = text, np.dtype(dtype), (), cval


class Local(Val):
    def __init__(self, name, shape, dtype):
        self.name, self.shape, self.dtype = name, tuple(shape), np.dtype(dtype)


class Table(Val):
    """Global buffer window: elem(idx) = ptr[off + sum(idx*strides)]."""

    def __init__(self, ptr, shape, dtype, off=0, strides=None):
        self.ptr, self.shape, self.dtype = ptr, tuple(shape), np.dtype(dtype)
        self.off = off
        self.strides = tuple(strides) if strides is not None else _row_major(shape)


class View(Val):
    def __init__(self, src, shape, fn, dtype=None):
        self.src, self.shape, self.fn = src, tuple(shape), fn
        self.dtype = np.dtype(dtype) if dtype is not None else src.dtype
        self.key_impl = src.key_impl


class VTally(Val):
    """a virtual tally (tallymap): no data, emits write straight to the output accumulator of `slot`."""

    def __init__(self, slot, shape, dtype):
        self.slot, self.shape, self.dtype = slot, tuple(shape), np.dtype(dtype)


class Lazy(Val):
    """An array whose elements are C expressions built on demand (fused into the consumer).  cost = the number of
    lazy nodes one element's expression inlines; a value used twice inside another lazy expression is counted
    twice, so a chain that reuses its input doubles per step and is materialised once it passes LAZY_MAX_COST."""
    def __init__(self, shape, dtype, fn, cval=None, cost=1):
        self.shape, self.dtype, self.fn, self.cval, self.cost = tuple(shape), np.dtype(dtype), fn, cval, cost


# inlined lazy nodes per element; above it a lazy result is computed into a local array.
# MEASURED (LUCiD recon item, 2080 Ti, forward; compile s / run s):  256: 1543 / 9.04   128: 1209 / 8.15
# 64: 839 / 7.24   32: 834 / 6.87   16: 1557 / 6.86.  85% of compile time is NVRTC's C++ -> PTX front end, and long
# inlined expressions are what it chokes on.  Results bitwise identical at every cap; perf net unchanged.
LAZY_MAX_COST = 32


# A lazy value (Lazy, or a View of one) is re-inlined into EVERY consumer, so one used k times is emitted k times.
# Rule: a lazy value used more than once in its jaxpr is materialised when one element's C expression is longer than
# MAT_MULTI_CHARS characters (rendered at index 0); a one-use or short one stays lazy, as before.  Measured on the
# GDML geometry kernel (prototype TM_MAT_MULTI=200): 543 kB of source for one equation; 13.0 -> 7.5 MB in all.
MAT_MULTI_CHARS = 200


def lazy_cost(v):
    if isinstance(v, Lazy):
        return v.cost
    if isinstance(v, View):
        return lazy_cost(v.src)
    return 0


def vrng(v):
    """the integer range (lo, hi) every element of v is known to lie in (ranges.py), or None.  Unsigned, bool and
    narrow integer dtypes give at least their own range; a View reads its source's."""
    from . import ranges
    dt = getattr(v, "dtype", None)
    if dt is None or isinstance(v, VTally) or not ranges._is_int(dt) or is_key_dtype(dt) or v.key_impl is not None:
        return None
    if not ranges.ENABLE:
        return None
    if v.cval is not None:
        return ranges.lit_range(v.cval, dt)
    r = getattr(v, "rng", None)
    if r is None and isinstance(v, View) and np.dtype(v.src.dtype) == np.dtype(dt):
        r = vrng(v.src)
    return ranges.clip(ranges.full(r, dt), dt)


def elem(v, idx):
    """C expression of v[idx] (idx: list of int / str index expressions)."""
    if isinstance(v, Scalar):
        return v.text
    if isinstance(v, Local):
        return f"{v.name}[{ravel(idx, v.shape)}]" if v.shape else v.name
    if isinstance(v, Table):
        o = ia(v.off, *[im(i, s) for i, s in zip(idx, v.strides)])
        if v.dtype == np.bool_:
            return f"(sj_ld((const u8*){v.ptr} + ({o})) != 0)"
        return f"sj_ld({v.ptr} + ({o}))"
    if isinstance(v, View):
        return elem(v.src, v.fn(list(idx)))
    if isinstance(v, Lazy):
        return v.fn(list(idx))
    if isinstance(v, VTally):
        raise CodegenError("a tally is read (tallies are write-only)")
    raise TypeError(v)


# ----------------------------------------------------------------------------- wide-multiply chains
# NVVM (the front end of NVRTC and nvcc) is EXPONENTIAL in the depth of a serial dependency chain of 64-bit integer
# multiplies (the 64x64->128 product, spelled natively or in 32-bit limbs) inside one function: on the identical
# arithmetic 0.7 s at depth 12, 127 s at 16, > 300 s at 20 (DTRAX's MixMax reseed is 17 links).  The same multiplies
# without the dependency, or a chain of 32-bit multiplies, are free.  So every scalar whose chain depth (the longest
# path to it counting one per wide multiply) would pass MULCHAIN_K is passed through `sj_brk` (kernel.PRELUDE: an
# empty asm with the value tied in and out, zero instructions, and the front end cannot see through it) and its
# depth restarts at 0.  sj_brk(x) == x, so the bits cannot change.  Measured (wavefront-state 41140a0): cut every
# 1, 4 or 8 links the compile is 0.2 s at depth 16, 24 and 48; every 12 it is 2.2 s; every 16 it is 125.6 s;
# identical SASS (928 == 928 instructions).
MULCHAIN_K = 8

# primitives whose rule may return its operand array updated in place (op_table._may_overwrite)
_INPLACE_PRIMS = frozenset({"dynamic_update_slice", "scatter", "scatter-add", "scatter_add", "scatter-mul",
                            "scatter_mul", "scatter-min", "scatter_min", "scatter-max", "scatter_max"})


def _wide_mul(eqn):
    """1 if `eqn` is one link of a wide-multiply chain (an integer multiply with a >= 64-bit result)."""
    n = eqn.primitive.name
    if n not in ("mul", "mulhi", "integer_pow") or (n == "integer_pow" and int(eqn.params.get("y", 0)) < 2):
        return 0
    return int(any(not is_dropvar(v) and _isint(v.aval.dtype) and _bits(v.aval.dtype) >= 64 for v in eqn.outvars))


def mul_chain(jaxpr, memo, cuts=None):
    """walk `jaxpr` in emission order giving every variable its wide-multiply chain depth; a scalar whose depth
    passes MULCHAIN_K is cut (added to `cuts`) and restarts at 0.  A sub-jaxpr (cond branch, loop body, call)
    adds its input-to-output depth with its own cuts applied (the emitter cuts it when it emits it).
    -> the deepest output depth of `jaxpr`.  memo: id(sub-jaxpr) -> its output depth (one build)."""
    from ..._jaxpr import subjaxprs
    d = {}
    for eqn in jaxpr.eqns:
        w = _wide_mul(eqn)
        for sj in subjaxprs(eqn):
            k = id(sj.jaxpr)
            if k not in memo:
                memo[k] = (sj.jaxpr, mul_chain(sj.jaxpr, memo))       # (the jaxpr is held: its id stays unique)
            w = max(w, memo[k][1])
        dep = max([d.get(a, 0) for a in eqn.invars if not isinstance(a, Literal)] or [0]) + w
        for v in eqn.outvars:
            if is_dropvar(v):
                continue
            if dep > MULCHAIN_K and v.aval.shape == ():
                d[v] = 0
                if cuts is not None:
                    cuts.add(v)
            else:
                d[v] = dep
    return max([d.get(v, 0) for v in jaxpr.outvars if not isinstance(v, Literal)] or [0])


# ----------------------------------------------------------------------------- guarded functional updates
# primitives whose result is the operand with some elements replaced (the guard composes with every one of them)
_GUARDABLE = frozenset({"dynamic_update_slice", "scatter", "scatter-add", "scatter_add", "scatter-mul", "scatter_mul",
                        "scatter-min", "scatter_min", "scatter-max", "scatter_max"})


_CALLS = ("pjit", "jit", "closed_call", "core_call")


def _select_of(e, depth=0):
    """(p, case0, case1) of a two-case select_n with a bool predicate that `e` is, or that makes the single result of
    the call `e` (jnp.where is a pjit), as variables of `e`'s own jaxpr; a predicate broadcast inside the call maps to
    its scalar operand.  None otherwise."""
    if e.primitive.name == "select_n":
        if len(e.invars) != 3 or isinstance(e.invars[0], Literal) or np.dtype(e.invars[0].aval.dtype) != np.bool_:
            return None
        return tuple(e.invars)
    if e.primitive.name not in _CALLS or len(e.outvars) != 1 or depth > 8:
        return None
    sub = call_jaxpr(e)
    if sub is None or isinstance(sub.jaxpr.outvars[0], Literal):
        return None
    j = sub.jaxpr
    inner = next((q for q in j.eqns if j.outvars[0] in q.outvars), None)
    s = None if inner is None else _select_of(inner, depth + 1)
    if s is None:
        return None
    prod = {v: q for q in j.eqns for v in q.outvars}

    def outer(v, bcast=False):
        if isinstance(v, Literal):
            return None
        if v in j.invars:
            w = e.invars[j.invars.index(v)]
            return None if isinstance(w, Literal) else w
        q = prod.get(v)
        if bcast and q is not None and q.primitive.name == "broadcast_in_dim" and q.invars[0].aval.shape == ():
            return outer(q.invars[0])
        return None
    m = (outer(s[0], bcast=True), outer(s[1]), outer(s[2]))
    return None if any(v is None for v in m) else m


def _guard_target(e, x, depth=0):
    """the update equation that makes `e`'s single result from operand x: e itself (a scatter / dynamic_update_slice
    of x), or, through a call (jnp's x.at[i].set is a pjit), the one inside it that makes the call's only result
    from the parameter x is passed as.  None when there is no such equation."""
    if len(e.outvars) != 1 or depth > 8:
        return None
    if e.primitive.name in _GUARDABLE:
        return e if e.invars[0] is x else None
    sub = call_jaxpr(e) if e.primitive.name in _CALLS else None
    if sub is None or x not in e.invars or list(e.invars).count(x) != 1:
        return None
    j = sub.jaxpr
    xi = j.invars[list(e.invars).index(x)]
    o = j.outvars[0]
    if isinstance(o, Literal):
        return None
    inner = next((q for q in j.eqns if o in q.outvars), None)
    return None if inner is None else _guard_target(inner, xi, depth + 1)


def guarded_updates(eqns, uses):
    """find `y = update(x, ...)` (scatter / dynamic_update_slice) whose ONLY use is `select_n(p, x, y)` or
    `select_n(p, y, x)` with a bool p: jnp.where(p, x.at[i].set(v), x), a stack push under a condition.  The pair is
    emitted as one update of x guarded by p (Emitter.run_jaxpr), instead of a copy of x, the update and an
    elementwise select over the whole array.  -> (the equations with each such update moved to just before its
    select, so the select's operands are all computed when it is emitted; {id(update eqn): (select eqn, k, target)},
    k the select case the update is, target the update equation itself (inside a call for jnp's x.at[i].set)).
    Moving it is valid: its only consumer is the select, and its operands come earlier."""
    prod = {}
    for e in eqns:
        for v in e.outvars:
            if not is_dropvar(v):
                prod[v] = e
    pairs = {}
    for e2 in eqns:
        sel = _select_of(e2)
        if sel is None:
            continue
        p = sel[0]
        for k in (1, 2):
            y, x = sel[k], sel[3 - k]
            if isinstance(y, Literal) or isinstance(x, Literal) or uses.get(y, 0) != 1 or y is x:
                continue
            e1 = prod.get(y)
            tgt = _guard_target(e1, x) if e1 is not None and id(e1) not in pairs and y is not p else None
            if tgt is None:
                continue
            pairs[id(e1)] = (e2, k, tgt, p)
            break
    if not pairs:
        return eqns, {}
    before = {id(e2): e1 for e1 in eqns if id(e1) in pairs for e2 in [pairs[id(e1)][0]]}
    out = []
    for e in eqns:
        if id(e) in pairs:
            continue
        if id(e) in before:
            out.append(before[id(e)])
        out.append(e)
    return out, pairs


# ----------------------------------------------------------------------------- emitter
class Emitter:
    def __init__(self):
        self.lines = []
        self.ind = 1
        self.uid = 0
        self.consts = []          # hoisted non-scalar constants (numpy arrays)
        self._const_ids = {}
        self.prims = set()        # primitive names seen (for the report)
        self.hooks = {}           # tallymap primitive name -> rule (set by the kernel builder)
        self._scopes = []         # active run_jaxpr scopes: (env, remaining uses) -- liveness for in-place updates
        self._frozen = [set()]    # names of Local arrays that must never be mutated (loop consts, function params)
        self.where = ""           # the item function (error messages)
        self.inplace = INPLACE    # False (test-only codegen.INPLACE): never update an array in place
        self.guarded = GUARDED    # False (test-only codegen.GUARDED): never fuse a guarded update
        self.store_guard = None   # the C condition the update rule being emitted stores under (guarded_updates)
        self.store_guard_eqn = None   # id() of the update equation it belongs to
        self._mulmemo = {}        # mul_chain: id(jaxpr) -> (jaxpr, output chain depth)
        self.n_brk = 0            # chain cuts emitted (sj_brk)
        self.lane_masks = []      # enclosing while loops with a per-lane predicate: (mask Local, predicate shape)

    # ------------------------------------------------------------ text helpers
    def fresh(self, p="v"):
        """A new identifier.  The counter is shared by every prefix, so a name can land on one of the kernel's own
        typedefs (i8 / i16 / i32 / i64, u8 ... u64): skip those, or the kernel does not compile."""
        self.uid += 1
        while f"{p}{self.uid}" in _RESERVED:
            self.uid += 1
        return f"{p}{self.uid}"

    def line(self, s):
        self.lines.append("  " * self.ind + s)

    def open(self, s):
        self.line(s + " {")
        self.ind += 1

    def close(self, s="}"):
        self.ind -= 1
        self.line(s)

    # ------------------------------------------------------------ materialisation
    def declare(self, shape, dtype, p="v"):
        """Uninitialized Local / Scalar variable (for outputs assigned in blocks)."""
        ct = ctype(dtype)
        n = self.fresh(p)
        if shape == ():
            self.line(f"{ct} {n};{fill_decl(n)}")
            return Local(n, (), dtype)
        if _size(shape) > MAX_LOCAL:
            raise CodegenError(f"per-item array of shape {shape} is too large for a local array")
        self.line(f"{ct} {n}[{max(1, _size(shape))}];{fill_decl(n)}")   # zero-size arrays get one (never read) slot
        return Local(n, shape, dtype)

    def fill(self, dst, fn):
        """dst[idx] = fn(idx) for all idx (dst: Local)."""
        if getattr(dst, "undef", False):
            dst.undef = False              # (written: defined from here on)
        shape = dst.shape
        if shape == ():
            self.line(f"{dst.name} = {fn([])};")
            return
        n = _size(shape)
        if n <= UNROLL:
            for f in range(n):
                idx = unravel(f, shape)
                self.line(f"{dst.name}[{f}] = {fn(idx)};")
            return
        vs = []
        for d, s in enumerate(shape):
            iv = self.fresh("i")
            vs.append(iv)
            self.open(f"for (int {iv} = 0; {iv} < {s}; ++{iv})")
        self.line(f"{dst.name}[{ravel(vs, shape)}] = {fn(vs)};")
        for _ in shape:
            self.close()

    def each(self, shape, body):
        """Emit body(idx) for every index of shape (unrolled when small, else C loops)."""
        shape = tuple(shape)
        if shape == ():
            body([])
            return
        n = _size(shape)
        if n <= UNROLL:
            for f in range(n):
                body(unravel(f, shape))
            return
        vs = []
        for s in shape:
            iv = self.fresh("i")
            vs.append(iv)
            self.open(f"for (int {iv} = 0; {iv} < {s}; ++{iv})")
        body(vs)
        for _ in shape:
            self.close()

    def mat_copy(self, v, p="m"):
        """A fresh MUTABLE copy of v (Local; shape () -> a scalar Local variable)."""
        if v.shape == ():
            n = self.fresh(p)
            self.line(f"{ctype(v.dtype)} {n} = {elem(v, [])};")
            out = Local(n, (), v.dtype)
        else:
            out = self.declare(v.shape, v.dtype, p)
            self.assign(out, v)
        out.key_impl = v.key_impl
        return out

    def assign(self, dst, v):
        if getattr(v, "undef", False):     # an UNDEFINED value (tallymap_uninit, never written): a copy of it is
            return                         # undefined too, so it costs nothing (no loads of garbage, no extra array)
        self.fill(dst, lambda idx: elem(v, idx))

    def mat(self, v, p="v"):
        """Materialise v into a Scalar variable or a Local array."""
        if isinstance(v, (Local,)):
            return v
        if isinstance(v, Scalar):
            return v
        if v.shape == ():
            n = self.fresh(p)
            self.line(f"const {ctype(v.dtype)} {n} = {elem(v, [])};")
            out = Scalar(n, v.dtype, v.cval)
            out.key_impl = v.key_impl
            return out
        d = self.declare(v.shape, v.dtype, p)
        d.key_impl = v.key_impl
        self.assign(d, v)
        return d

    def mat_multi(self, v, uses):
        """v, or v materialised when it is a lazy value used more than once with a long expression (MAT_MULTI_CHARS)"""
        if uses < 2 or not isinstance(v, (Lazy, View)) or not 0 < v.size <= MAX_LOCAL or lazy_cost(v) == 0:
            return v
        saved = (self.lines, self.ind, self.uid)
        try:
            self.lines = []
            n = len(elem(v, [0] * len(v.shape)))
        except Exception:
            return v
        finally:
            self.lines, self.ind, self.uid = saved
        return self.mat(v) if n > MAT_MULTI_CHARS else v

    def brk(self, v):
        """the same scalar through the chain-cut barrier sj_brk (see MULCHAIN_K); other values unchanged."""
        if v.shape != () or v.cval is not None or isinstance(v, VTally) or v.dtype is None or \
                np.dtype(v.dtype).kind not in "iuf" or np.dtype(v.dtype) == _FLOAT0:
            return v
        n = self.fresh("b")
        self.line(f"const {ctype(v.dtype)} {n} = sj_brk({elem(v, [])});")
        self.n_brk += 1
        out = Scalar(n, v.dtype)
        out.key_impl = v.key_impl
        return out

    def computed(self, shape, dtype, fn, p="v"):
        """New value whose elements are fn(idx); scalars become a const variable."""
        ct = ctype(dtype)
        if shape == ():
            n = self.fresh(p)
            self.line(f"const {ct} {n} = {fn([])};")
            return Scalar(n, dtype)
        d = self.declare(shape, dtype, p)
        self.fill(d, fn)
        return d

    # ------------------------------------------------------------ constants
    def dyn_const(self, fn, k, aval):
        """A top-level constant of function `fn` (index k in its ClosedJaxpr.consts) passed as a kernel input at
        call time (closed-over arrays and tracers become explicit table inputs)."""
        shape, dt = tuple(aval.shape), np.dtype(aval.dtype)
        if is_key_dtype(dt):
            raise CodegenError("closed-over typed PRNG key constants are not supported")
        j = len(self.consts)
        self.consts.append(("dyn", fn, k, shape, dt))
        cast = "u8" if dt == np.bool_ else ctype(dt)
        return Table(f"((const {cast}*)SJ_CONST({j}))", shape, dt)

    def const_val(self, x, aval=None):
        if isinstance(x, jax.core.Tracer):
            from ..._errors import CLOSED_TRACER
            raise CodegenError("a traced value is closed over inside a nested call; close over it at the top "
                               "level of the item function or pass it as an argument." + CLOSED_TRACER)
        a = np.asarray(x)
        if is_key_dtype(getattr(x, "dtype", None)):
            raise CodegenError("closed-over typed PRNG key constants are not supported")
        if a.dtype == np.dtype("O"):
            raise CodegenError(f"unsupported constant {type(x)}")
        if np.dtype(a.dtype) == _FLOAT0 or (aval is not None and np.dtype(aval.dtype) == _FLOAT0):
            if a.ndim == 0:
                return Scalar("false", _FLOAT0, cval=False)
            a = np.zeros(a.shape, np.bool_)
            x = a
        if a.ndim == 0:
            dt = a.dtype if aval is None else aval.dtype
            return Scalar(lit(a, dt), dt, cval=np.asarray(a, dt)[()])
        key = id(x)
        if key not in self._const_ids:
            self._const_ids[key] = len(self.consts)
            self.consts.append(("static", x, a))
        j = self._const_ids[key]
        ct = ctype(a.dtype)
        cast = "u8" if a.dtype == np.bool_ else ct
        return Table(f"((const {cast}*)SJ_CONST({j}))", a.shape, a.dtype)

    # ------------------------------------------------------------ jaxpr evaluation
    def freeze(self, vals):
        """push a frame of never-mutable storages (loop consts / pass-through carries / function params): every
        Local array a value of `vals` reads -- the value itself, or, for a lazy value, the storages its elements read
        (a loop constant that is a View of a per-item array is re-read every iteration: so is that array).  Kept as
        the C text each value reads ("*": unrenderable, everything is frozen)."""
        frame = []
        for v in vals:
            if isinstance(v, Local):
                frame.append(v.name)
            elif not isinstance(v, (Table, VTally)) and v is not None:
                saved = (self.lines, self.ind, self.uid)
                try:
                    self.lines = []
                    frame.append(elem(v, [f"_q{k}" for k in range(len(v.shape))]))
                except Exception:
                    frame.append("*")
                finally:
                    self.lines, self.ind, self.uid = saved
        self._frozen.append(frame)

    def unfreeze(self):
        self._frozen.pop()

    def can_mutate(self, val):
        """May the array `val` be updated in place?  Only if it is a mutable per-item Local (not a loop const /
        function parameter) and no LIVE value of any active scope reads its storage (checked on the C text every
        live value would read: Views and Lazy expressions included).  The equation's own reads of it are the
        caller's (op_table._may_overwrite)."""
        if not self.inplace or not self._scopes or not isinstance(val, Local) or val.shape == ():
            return False
        import re
        pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(val.name) + r"(?![A-Za-z0-9_])")
        if any(t == "*" or pat.search(t) for f in self._frozen for t in f):
            return False
        env = self._scopes[-1][0]          # the current scope's bindings (a captured reader is rebound there)
        readers = []                       # live lazy values of THIS scope whose elements read val's storage
        for depth, (env_, rem_) in enumerate(self._scopes):
            for v, x in env_.items():
                if rem_.get(v, 0) <= 0 or isinstance(x, (Table, VTally)):
                    continue
                if isinstance(x, Local):
                    if x.name == val.name:
                        return False
                    continue
                try:
                    saved = (self.lines, self.ind, self.uid)
                    self.lines = []
                    t = elem(x, [f"_q{k}" for k in range(len(x.shape))])
                    self.lines, self.ind, self.uid = saved
                except Exception:
                    self.lines, self.ind, self.uid = saved
                    return False
                if pat.search(t):
                    if depth != len(self._scopes) - 1:
                        return False       # (a value of an enclosing scope: its C scope is not this one)
                    readers.append((v, x))
        if not readers:
            return True
        # A live lazy value reads the array (x = a[j], used after a = a.at[k].set(...)): updated in place, the array
        # would change x.  Rather than copying the whole array, capture the reader's elements NOW, before the update
        # is emitted, when it is smaller than the array: a scalar read costs one register instead of two full-array
        # copies per update (a 512-element carry read this way was copied twice per loop step).
        if sum(max(1, x.size) for _, x in readers) >= val.size:
            return False
        for v, x in readers:
            env[v] = self.mat_copy(x, "r") if x.shape == () else self.mat(x)
        return True

    def elementwise_self(self, dst, x):
        """may the lazy value `x` be written into `dst`'s storage in place?  Yes when x's element expression reads
        that storage only at its own index: `dst[i] = f(.., dst[i])` then gives every element the value a copy
        through a temporary would (no element depends on another element of dst).  Placement only."""
        if (isinstance(x, (Table, VTally, Local)) or x is None or not isinstance(dst, Local)
                or tuple(getattr(x, "shape", ())) != tuple(dst.shape) or np.dtype(x.dtype) != np.dtype(dst.dtype)):
            return False
        sym = [f"_q{k}" for k in range(len(dst.shape))]
        saved = (self.lines, self.ind, self.uid)
        try:
            self.lines = []
            t, own = elem(x, sym), elem(dst, sym)
        except Exception:
            return False
        finally:
            self.lines, self.ind, self.uid = saved
        if own not in t:
            return False
        import re
        return re.search(r"(?<![A-Za-z0-9_])" + re.escape(dst.name) + r"(?![A-Za-z0-9_])", t.replace(own, "")) is None

    def run_top(self, cj, args, fn, mutable=()):
        """Run a top-level ClosedJaxpr: concrete scalar consts become literals, every other const (arrays, and
        tracers closed over from an enclosing trace) is a dynamic kernel input.  mutable: names of argument Locals
        the function may update in place (mutable reference parameters; kernel.py's step-schedule body)."""
        self.freeze([a for a in args if not (isinstance(a, Local) and a.name in mutable)])
        consts = []
        for k, (c, v) in enumerate(zip(cj.consts, cj.jaxpr.constvars)):
            if not isinstance(c, jax.core.Tracer) and np.ndim(c) == 0 and not is_key_dtype(v.aval.dtype):
                consts.append(self.const_val(c, v.aval))
            else:
                consts.append(self.dyn_const(fn, k, v.aval))
        try:
            return self.run_jaxpr(cj.jaxpr, consts, args)
        finally:
            self.unfreeze()

    def run_closed(self, cj, args):
        if isinstance(cj, ClosedJaxpr):
            return self.run_jaxpr(cj.jaxpr, [self.const_val(c, v.aval) for c, v in zip(cj.consts, cj.jaxpr.constvars)],
                                  args)
        return self.run_jaxpr(cj, [], args)

    def run_jaxpr(self, jaxpr, consts, args):
        env = {}

        def read(a):
            if isinstance(a, Literal):
                return self.const_val(a.val, a.aval)
            return env[a]

        assert len(consts) == len(jaxpr.constvars), (len(consts), len(jaxpr.constvars))
        assert len(args) == len(jaxpr.invars), (len(args), len(jaxpr.invars))
        for v, x in zip(jaxpr.constvars, consts):
            env[v] = x
        for v, x in zip(jaxpr.invars, args):
            env[v] = x
        uses = {}
        for eqn in jaxpr.eqns:
            for a in eqn.invars:
                if not isinstance(a, Literal):
                    uses[a] = uses.get(a, 0) + 1
        for a in jaxpr.outvars:
            if not isinstance(a, Literal):
                uses[a] = uses.get(a, 0) + 1
        rem = dict(uses)
        cuts = set()
        mul_chain(jaxpr, self._mulmemo, cuts)
        eqns, fused = guarded_updates(jaxpr.eqns, uses) if self.guarded else (jaxpr.eqns, {})
        skip = set()
        self._scopes.append((env, rem))
        try:
            for eqn in eqns:
                if id(eqn) in skip:
                    continue
                pair = fused.get(id(eqn))
                guard = None
                if pair is not None:
                    e2, k, tgt, pv = pair
                    guard = self._guard_text(read(pv), k)
                    if guard is not None:          # evaluated now, before anything the update emits
                        gv = self.fresh("gd")
                        self.line(f"const bool {gv} = {guard};")
                        guard = gv
                ins = [read(a) for a in eqn.invars]
                for a in eqn.invars:
                    if not isinstance(a, Literal):
                        rem[a] -= 1
                if guard is not None:
                    # x.at[i].set(v) under jnp.where(p, ., x) (or where(p, x, .)): ONE update, guarded by p, of x's
                    # storage (in place when x is dead after the select, else of a copy).  Either way its result is
                    # the select's value: the select is consumed here (its operands' uses too) and skipped
                    for a in e2.invars:
                        if not isinstance(a, Literal) and a is not eqn.outvars[0]:
                            rem[a] -= 1
                    skip.add(id(e2))
                # jaxpr-local use counts of this equation's results (rules may fuse one-use values lazily;
                # 0 = only a nested-call result, consumers unknown here)
                self.out_uses = [uses.get(v, 0) if not is_dropvar(v) else 0 for v in eqn.outvars]
                if guard is None:
                    outs = self.eqn(eqn, ins)
                else:
                    saved_g = (self.store_guard, self.store_guard_eqn)
                    self.store_guard, self.store_guard_eqn = guard, id(tgt)
                    try:
                        outs = self.eqn(eqn, ins)
                        if self.store_guard is not None:    # (the update rule takes it: _take_guard)
                            raise CodegenError("internal: a guarded update was not emitted")
                    finally:
                        self.store_guard, self.store_guard_eqn = saved_g
                if not eqn.primitive.multiple_results:
                    outs = [outs]
                rr = eqn_ranges(eqn, [vrng(x) for x in ins], [getattr(x, "dtype", None) for x in ins], nested=False)
                ovars = [e2.outvars[0]] if guard is not None else eqn.outvars
                for v, x, r in zip(ovars, outs, rr):
                    if not is_dropvar(v):
                        x = self.brk(x) if v in cuts else self.mat_multi(x, uses.get(v, 0))
                        if not isinstance(x, VTally):
                            # an operand updated in place (scatter / dynamic_update_slice) gets r, even None; one
                            # returned unchanged (a folded select) keeps its own, tighter fact; a new value gets r
                            if any(x is i for i in ins) and eqn.primitive.name in _INPLACE_PRIMS:
                                x.rng = r
                            elif r is not None:
                                x.rng = ranges.meet(getattr(x, "rng", None), r) if any(x is i for i in ins) else r
                        env[v] = x
            return [read(a) for a in jaxpr.outvars]
        finally:
            self._scopes.pop()

    def _guard_text(self, pred, k):
        """the C condition under which a guarded update stores: the select's predicate when the update is its case
        k == 2 (picked where it is true), negated for k == 1; None when the predicate is not one value for every
        element (an elementwise mask: the update is not fused)."""
        if pred.shape != () and not isinstance(pred, (View, Lazy, Scalar)):
            return None
        saved = (self.lines, self.ind, self.uid)
        try:
            self.lines = []
            t = elem(pred, [f"_q{j}" for j in range(len(pred.shape))])
        except Exception:
            return None
        finally:
            self.lines, self.ind, self.uid = saved
        if re.search(r"(?<![A-Za-z0-9_])_q[0-9]+(?![A-Za-z0-9_])", t):
            return None
        return f"({t})" if k == 2 else f"(!({t}))"

    def eqn(self, eqn, ins):
        try:
            return self._eqn(eqn, ins)
        except CodegenError as e:
            raise located(e, eqn, f"item function {self.where}" if self.where else "") from None
        except Exception as e:
            better = ad_advice(e, f"item function {self.where}")
            if better is None:
                raise
            raise located(better, eqn, f"item function {self.where}") from None

    def _eqn(self, eqn, ins):
        name = eqn.primitive.name
        self.prims.add(name)
        if eqn.effects:
            raise CodegenError(f"primitive {name!r} has side effects {eqn.effects}; effects are not allowed "
                                "inside the item function on any backend.")
        from . import op_table
        rule = self.hooks.get(name) or op_table.RULES.get(name)
        if name not in op_table.TALLY_AWARE and name not in self.hooks and any(isinstance(v, VTally) for v in ins):
            raise CodegenError(f"primitive {name!r} reads a tally")
        if rule is None:
            raise CodegenError(f"primitive {name!r} is not supported by the tallymap kernel backend "
                                f"(params: {sorted(eqn.params)}).")
        return rule(self, eqn, ins, **eqn.params)
