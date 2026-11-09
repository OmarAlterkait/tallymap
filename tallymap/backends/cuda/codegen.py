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

Float semantics: IEEE-754 round-to-nearest +,-,*,/,sqrt, no FMA contraction; transcendentals call the CUDA
math library (libdevice).  Results agree with XLA only statistically (see docs/validation.md), never bitwise by design.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import ClosedJaxpr, Literal

from ..._errors import ad_advice
from ..._jaxpr import is_dropvar, located

UNROLL = 32          # arrays with <= UNROLL elements are emitted as straight-line code
MAX_LOCAL = 1 << 14  # largest per-thread local array we agree to materialize
# TEST-ONLY (tests patch it, like _primitive.FUSE_JVP): None, or a byte value every UNINITIALISED per-thread
# declaration (Emitter.declare, kernel.build_source's carry / state / result variables) is filled with.  A program
# that never reads an uninitialised value gives bitwise-equal results for FILL = 0x00 and 0xFF (tests/reverse.py).
FILL = None


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
        raise CodegenError(f"dtype {dt} is not supported by the cuda backend")
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


# ----------------------------------------------------------------------------- constant folding
FOLD_MIN = 16        # constant-folded results of at least this many elements become pooled constant tables
_NOFOLD = ("custom_lin",)


def _uniform(a):
    """every element has the same bits (zeros, broadcast scalars)"""
    b = np.ascontiguousarray(a).reshape(-1)
    if b.size == 0:
        return True
    u = b.view(np.uint8).reshape(b.size, -1)
    return bool(np.all(u == u[0]))


def _with_cnp(v, a):
    try:
        if v.shape == tuple(np.shape(a)):
            v.cnp = a
    except Exception:
        pass
    return v


# ----------------------------------------------------------------------------- emitter
class Emitter:
    def __init__(self):
        self.lines = []
        self.ind = 1
        self.uid = 0
        self.consts = []          # hoisted non-scalar constants (numpy arrays)
        self._const_ids = {}
        self.prims = set()        # primitive names seen (for the report)
        self.cf_depth = 0         # control-flow nesting depth (while/cond/scan bodies)
        self.hooks = {}           # tallymap primitive name -> rule (set by the kernel builder)
        self._scopes = []         # active run_jaxpr scopes: (env, remaining uses) -- liveness for in-place updates
        self._frozen = [set()]    # names of Local arrays that must never be mutated (loop consts, function params)
        self.where = ""           # the item function (error messages)
        self.inplace = True       # False (private _cuda_inplace=False): never update an array in place
        self._mulmemo = {}        # mul_chain: id(jaxpr) -> (jaxpr, output chain depth)
        self.n_brk = 0            # chain cuts emitted (sj_brk)

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

    # ------------------------------------------------------------ materialization
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
        """Materialize v into a Scalar variable or a Local array."""
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
        t = Table(f"((const {cast}*)SJ_CONST({j}))", a.shape, a.dtype)
        t.cnp = a
        return t

    # ------------------------------------------------------------ jaxpr evaluation
    def freeze(self, vals):
        """push a frame of never-mutable Local arrays (loop consts / pass-through carries / function params)."""
        self._frozen.append({v.name for v in vals if isinstance(v, Local) and v.shape != ()})

    def unfreeze(self):
        self._frozen.pop()

    def can_mutate(self, var, val):
        """May the array `val` (bound to jaxpr var `var` of the CURRENT scope) be updated in place?  Only if it is
        a mutable per-item Local (not a loop const / function parameter / compile-time-known value), `var` has no
        remaining use in this scope, and no LIVE value of any active scope reads its storage (checked on the C
        text every live value would read: Views and Lazy expressions included)."""
        if not self.inplace or not self._scopes or not isinstance(val, Local) or val.shape == () or \
                isinstance(var, Literal) or hasattr(val, "cnp"):
            return False
        if any(val.name in f for f in self._frozen):
            return False
        env, rem = self._scopes[-1]
        if rem.get(var, 0) != 0:
            return False
        import re
        pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(val.name) + r"(?![A-Za-z0-9_])")
        for env_, rem_ in self._scopes:
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
                    return False
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
        the function may update in place (mutable reference parameters; kernel.py's step-split body)."""
        self.freeze([a for a in args if not (isinstance(a, Local) and a.name in mutable)])
        consts = []
        for k, (c, v) in enumerate(zip(cj.consts, cj.jaxpr.constvars)):
            if not isinstance(c, jax.core.Tracer) and np.ndim(c) == 0 and not is_key_dtype(v.aval.dtype):
                consts.append(self.const_val(c, v.aval))
            else:
                dc = self.dyn_const(fn, k, v.aval)
                if not isinstance(c, jax.core.Tracer) and not is_key_dtype(v.aval.dtype):
                    try:
                        dc.cnp = np.asarray(c)          # known value: foldable downstream (and in the kernel key)
                    except Exception:
                        pass
                consts.append(dc)
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
        self._scopes.append((env, rem))
        try:
            for eqn in jaxpr.eqns:
                ins = [read(a) for a in eqn.invars]
                for a in eqn.invars:
                    if not isinstance(a, Literal):
                        rem[a] -= 1
                self.cur_invars = eqn.invars
                # jaxpr-local use counts of this equation's results (rules may fuse one-use values lazily;
                # 0 = only a nested-call result, consumers unknown here)
                self.out_uses = [uses.get(v, 0) if not is_dropvar(v) else 0 for v in eqn.outvars]
                outs = self.eqn(eqn, ins)
                if not eqn.primitive.multiple_results:
                    outs = [outs]
                for v, x in zip(eqn.outvars, outs):
                    if not is_dropvar(v):
                        env[v] = self.brk(x) if v in cuts else self.mat_multi(x, uses.get(v, 0))
            return [read(a) for a in jaxpr.outvars]
        finally:
            self._scopes.pop()

    # ------------------------------------------------------------ constant folding of arrays
    def fold_table(self, a):
        """Hoist a compile-time-known array into a per-dtype pooled read-only constant buffer (one kernel buffer
        per dtype, however many folded arrays)."""
        a = np.ascontiguousarray(a)
        dt = a.dtype
        if not hasattr(self, "_pools"):
            self._pools = {}
        pool = self._pools.get(dt)
        if pool is None:
            pool = self._pools[dt] = [len(self.consts), [], 0]
            self.consts.append(("static", None, np.zeros((1,), dt)))
        off = pool[2]
        pool[1].append(a.ravel())
        pool[2] += a.size
        cast = "u8" if dt == np.bool_ else ctype(dt)
        t = Table(f"((const {cast}*)SJ_CONST({pool[0]}))", a.shape, dt, off=off)
        t.cnp = a
        return t

    def flush_pools(self):
        """write each pooled constant buffer into self.consts, concatenated ONCE (re-concatenating at every
        fold_table was quadratic in the pool).  kernel.build_source calls it before reading self.consts."""
        for pool in getattr(self, "_pools", {}).values():
            self.consts[pool[0]] = ("static", None, np.concatenate(pool[1]))

    def _try_fold(self, eqn, ins):
        """All inputs compile-time known -> evaluate the primitive eagerly ON THE DEFAULT DEVICE (XLA's own kernels,
        so the bits are what XLA computes) and hoist results with >= FOLD_MIN elements as constant tables.
        XLA folds/fuses these constant-only subgraphs; per item they would otherwise be recomputed every step
        into per-thread local arrays (PROD: 15+ 255-element quantile grids per step).  -> list of Vals or None"""
        if eqn.effects or eqn.primitive.name.startswith("random") or eqn.primitive.name in _NOFOLD:
            return None
        vals = []
        for x in ins:
            if getattr(x, "key_impl", None) is not None:
                return None
            if hasattr(x, "cnp"):
                vals.append(x.cnp)
            elif isinstance(x, Scalar) and x.cval is not None and np.dtype(x.dtype) != _FLOAT0:
                vals.append(np.asarray(x.cval, x.dtype))
            else:
                return None
        for v in eqn.outvars:
            dt = getattr(v.aval, "dtype", None)
            if dt is None or is_key_dtype(dt) or np.dtype(dt) == _FLOAT0 or not hasattr(v.aval, "shape"):
                return None
        try:
            with jax.ensure_compile_time_eval():
                outs = eqn.primitive.bind(*[jnp.asarray(v) for v in vals], **eqn.params)
            outs = outs if eqn.primitive.multiple_results else [outs]
            outs = [np.asarray(o) for o in outs]
        except Exception:
            return None
        # UNIFORM arrays (zeros / broadcast scalars: tape buffers, zero cotangents) are never hoisted: the normal
        # rule gives a literal (Lazy) value, while a pooled table turns every use into a global load (and every
        # mutable copy of it into an O(size) load loop) and bloats the pool (measured: 33 MB and 80 MB adjoint
        # pools of zeros on two engine workloads).  They stay known (cnp) for downstream folding.
        fold = [o.size >= FOLD_MIN and not _uniform(o) for o in outs]
        if not any(fold):
            return outs      # small or uniform: emitted normally, but known (cnp) for downstream folding
        return [self.fold_table(o) if f else o for o, f in zip(outs, fold)]

    def eqn(self, eqn, ins):
        try:
            return self._eqn_folded(eqn, ins)
        except CodegenError as e:
            raise located(e, eqn, f"item function {self.where}" if self.where else "") from None
        except Exception as e:
            better = ad_advice(e, f"item function {self.where}")
            if better is None:
                raise
            raise located(better, eqn, f"item function {self.where}") from None

    def _eqn_folded(self, eqn, ins):
        folded = self._try_fold(eqn, ins)
        if folded is not None and any(isinstance(f, Val) for f in folded):
            # mixed: small outputs still go through the rule (only multi-result primitives)
            if eqn.primitive.multiple_results:
                if not all(isinstance(f, Val) for f in folded):
                    r = self._eqn(eqn, ins)
                    folded = [f if isinstance(f, Val) else _with_cnp(rv, f) for f, rv in zip(folded, r)]
                return folded
            return folded[0]
        r = self._eqn(eqn, ins)
        if folded is not None and not isinstance(r, VTally):
            if eqn.primitive.multiple_results:
                r = [_with_cnp(rv, f) for rv, f in zip(r, folded)]
            else:
                r = _with_cnp(r, folded[0])
        return r

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
            raise CodegenError(f"primitive {name!r} is not supported by the tallymap cuda backend "
                                f"(params: {sorted(eqn.params)}).")
        return rule(self, eqn, ins, **eqn.params)
