"""xla backend: the portable backend.  Plain JAX (XLA runs it), any platform.

A masked lockstep interpreter of the item jaxpr over a batch of P items: every value is (array, batched?), control
flow runs while ANY active item of the batch needs it, and emits / spawns / per-item writes are masked by the
active items.  Unlike jax.vmap, tallies are never per-item: an emit is a masked scatter-add into ONE shared
accumulator, so memory is O(P * state + outputs), not O(P * outputs).  Items are processed in batches of P
(xla_batch_size); with spawn, children are queued in a ring buffer and run in later batches (children first).
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend.core import Literal, jaxpr_as_fun

from .._api import EXACT_WORDS
from .._errors import ad_advice
from .._jaxpr import bind, call_jaxpr, cond_branches, lit, located, scan_parts, while_parts
from .._tally import KK

BLOCK = 1 << 18
# f32 sum tallies: R partial accumulators (lane p adds into row p % R), summed by jnp.sum at the end.  One
# accumulator takes every item's every emit in series (XLA:CPU's scatter-add is a serial loop), so its rounding
# error grows with the number of adds per bin: 85x plain JAX's on a 3.6e3-per-bin tally (round4/xla, Q1).  R rows
# cut the adds per bin R-fold; the rows cost R x the tally memory, so R shrinks to stay within ROWS_ELEMS.
ROWS, ROWS_ELEMS = 64, 1 << 24
# f32 sum tallies keep subnormals: a GPU's f32 scatter-add is an atomic add that flushes subnormal inputs and results
# to zero (measured on a 2080 Ti, jax 0.6.2: XLA's own .at[].add of 4 x 1.4e-44 is 0).  An add of |v| >= 2^-102 never
# meets a subnormal when every value in its accumulator is a sum of such values (they are all multiples of 2^-125),
# so the accumulator takes only those; a smaller v (subnormals, zeros) is added, times 2^100 (exact, and normal), into
# one more row of the tally, LOW, whose values are multiples of 2^-49 (never subnormal).  The readout adds
# LOW * 2^-100 (exact) to the sum of the rows: one more rounding, a reassociation of the same adds.
SPLIT, LOW, UNLOW = np.float32(2.0 ** -102), np.float32(2.0 ** 100), np.float32(2.0 ** -100)


def _low(spec):
    """an f32 non-fixed-point sum tally: one more row, for its adds below SPLIT (scaled by LOW)."""
    return (spec.kind == "sum" and not spec.fixed_point and not _exact(spec)
            and np.dtype(spec.dtype) == np.dtype(np.float32))


class VT:
    """a virtual tally value (no data): slot index."""
    __slots__ = ("slot",)

    def __init__(self, slot):
        self.slot = slot


class B:
    """an interpreter value: array x (with a leading batch axis iff b)."""
    __slots__ = ("x", "b")

    def __init__(self, x, b):
        self.x, self.b = x, b


def _bc(v, P):
    return v.x if v.b else jnp.broadcast_to(v.x, (P,) + jnp.shape(v.x))


class _St:
    def __init__(self, p, P, accs, queue):
        self.p, self.P = p, P
        self.accs = accs                # slot -> array
        self.queue = queue              # None | dict(leaves, head, tail, spawned, dropped)
        self.ovf = jnp.int32(0)         # fixed-point overflow count
        self.bad = {k: jnp.zeros(a.shape[:-1] + (p.tallies[k].S,), bool)                           # flagged bins
                    for k, a in accs.items() if p.tallies[k].fixed_point or _exact(p.tallies[k])}
        # digit-word tallies: values added since the last carry normalisation (_xnorm): into the tally (int64 words)
        # or into each bin (int32 words: [S])
        self.nadd = {k: jnp.zeros((p.tallies[k].S,) if _perbin(p.tallies[k]) else (), _xfmt(p.tallies[k])[2])
                     for k in accs if _words(p.tallies[k])}
        self.drops = jnp.int32(0)       # out-of-range (dropped) emit indices
        self.mask = None
        self.lanes = []                 # enclosing while loops with a batched predicate: their [P, *B] masks
        # Records one-pass claim counters (_claim), one per output, keyed by its key tally (the claim token)
        self.ctr = {k: jnp.int32(0) for k, s in enumerate(p.tallies) if s.kind == "sum" and s.name.endswith(KK)}

    # ------------------------------------------------------------- state threaded through lax control flow
    def pack(self):
        q = self.queue
        return (tuple(self.accs[k] for k in sorted(self.accs)), tuple(self.bad[k] for k in sorted(self.bad)), self.ovf,
                None if q is None else (tuple(q["leaves"]), q["head"], q["tail"], q["spawned"], q["dropped"]),
                self.drops, tuple(self.nadd[k] for k in sorted(self.nadd)),
                tuple(self.ctr[k] for k in sorted(self.ctr)))

    def unpack(self, s):
        a, bad, self.ovf, q, self.drops, na, ct = s
        self.ctr = dict(zip(sorted(self.ctr), ct))
        self.accs = dict(zip(sorted(self.accs), a))
        self.bad = dict(zip(sorted(self.bad), bad))
        self.nadd = dict(zip(sorted(self.nadd), na))
        if q is not None:
            self.queue = dict(leaves=list(q[0]), head=q[1], tail=q[2], spawned=q[3], dropped=q[4])


def _fixed(v, scale):
    """-> (int64 fixed-point values, representable mask); round-half-even like __double2ll_rn (x64 only: without it
    a fixed-point tally keeps int32 words, _femit)."""
    x = v.astype(jnp.float64) * scale
    ok = jnp.abs(x) < 9.2e18
    return jnp.where(ok, jnp.round(x), 0).astype(jnp.int64), ok


def _exact(s):
    """an exact f32 sum tally (TSpec.exact): a user's Sum(fixed_point=True) of float32, its tangents, and the reverse
    cotangent tallies.  With jax_enable_x64 it keeps int64 words of 32-bit digits, without it the same integer in
    int32 words of 16-bit digits (_xfmt): exact, order-independent and bitwise the x64 and kernel results either way."""
    return bool(s.exact)


def _fixed32(s):
    """a fixed-point (not exact) sum tally without jax_enable_x64: its int64 fixed-point integers are kept as 16-bit
    digits in FIXED_WORDS32 int32 words (_femit, _fread32), the same integers and the same readout as _fixed / _wide."""
    return s.kind == "sum" and s.fixed_point and not s.exact and not jax.config.jax_enable_x64


def _words(s):
    """a sum tally kept as digit words (scatter-added pieces, carry-normalised by _xnorm)."""
    return s.kind == "sum" and (_exact(s) or _fixed32(s))


def _perbin(s):
    """a digit-word tally in int32 words: normalised by its bins' value counts (XNORM32), not the tally's."""
    return _words(s) and (_fixed32(s) or not jax.config.jax_enable_x64)


# without x64 an exact tally keeps its integer (units of 2^-149) as 16-bit digits in EXACT_WORDS32 int32 words (a value
# m << p, m < 2^24: m << (p % 16) < 2^39 is cut into 3 pieces below 2^16 at words p / 16 .. p / 16 + 2 <= 17).  Word 18
# has weight 2^288 and takes only carries; it is flagged at 2^30: the same threshold (2^318 units) as the int64 format's
# top word (2^256 x 2^62).  A fixed-point tally's integers (|q| < 9.2e18 < 2^63: 3 pieces at words <= 5) take
# FIXED_WORDS32 words; its top word (weight 2^112) is flagged at 2^30 (the x64 format's pair flags a
# word at 2^62, _wnorm).  A word below the top stays within 2^16 + 2^15 + XNORM32 2^16 < 2^31 in
# magnitude: each bin counts the values added into it since the last carry step (_St.nadd: a value adds one piece to a
# word at most), and the tally takes a carry step (_xnorm) before a bin could pass XNORM32 (per bin: a bin's count,
# not the tally's, so a large tally whose bins each take few values takes one rarely, if ever).  20 x 4 B per bin
# (19 words + the count).
EXACT_WORDS32 = 19
FIXED_WORDS32 = 8
XNORM32 = (1 << 15) - 4


def _xfmt(s=None):
    """-> (words per bin, digit bits, word dtype, pieces per value, normalisation period, flag threshold) of a digit
    word tally s (_words; None: an exact one)."""
    if s is not None and _fixed32(s):
        return FIXED_WORDS32, 16, jnp.int32, 3, XNORM32, 1 << 30
    if jax.config.jax_enable_x64:
        return EXACT_WORDS, 32, jnp.int64, 2, XNORM, 1 << 62
    return EXACT_WORDS32, 16, jnp.int32, 3, XNORM32, 1 << 30


def _shl(x, k):
    """uint32 x << k for int32 k >= 0 (0 for k >= 32)."""
    return jnp.where(k >= 32, jnp.uint32(0), x << jnp.clip(k, 0, 31).astype(jnp.uint32))


def _shr(x, k):
    """uint32 x >> k for int32 k >= 0 (0 for k >= 32)."""
    return jnp.where(k >= 32, jnp.uint32(0), x >> jnp.clip(k, 0, 31).astype(jnp.uint32))


def _blen(x):
    """bit length of uint32 x (0 for 0)."""
    return 32 - lax.clz(x).astype(jnp.int32)


FIXED32_FLOATS = (np.dtype(jnp.float16), np.dtype(jnp.bfloat16))


def _scale_exp(s):
    """a fixed-point tally's scale as 2^e (without x64 only a power of two in [2^-600, 2^600] is supported: the int32
    format multiplies and divides by it exactly, as float64 does)."""
    m, e = np.frexp(float(s.scale))
    dt = np.dtype(s.dtype)
    ok_dt = dt in FIXED32_FLOATS or (np.issubdtype(dt, np.integer) and dt.itemsize <= 4)
    if m != 0.5 or not -600 <= e - 1 <= 600 or not ok_dt:
        raise NotImplementedError(
            f"tallymap: the fixed-point sum {s.name!r} (dtype {dt}, scale {s.scale!r}) without jax_enable_x64 on the "
            "xla backend: only a power-of-two scale in [2^-600, 2^600] and a float16 / bfloat16 / integer (<= 32-bit) "
            "dtype are kept in int32 words; enable x64, use the kernel backend, or a power-of-two scale")
    return int(e - 1)


TQ = 9200000000000000000                              # _fixed's bound: |v * scale| < 9.2e18 (exactly this integer)


def _femit(v, s):
    """_xemit for a fixed-point tally without x64 (_fixed32): values [..., M] of s.dtype -> (word offsets [..., 3 M],
    signed 16-bit digit pieces [..., 3 M], representable mask [..., M]): the digits of round(v * scale) (half to even),
    the integer _fixed makes; a value with |v * scale| >= 9.2e18 (or non-finite) is not representable."""
    es = _scale_exp(s)
    dt = np.dtype(s.dtype)
    if np.issubdtype(dt, np.integer):
        x = v.astype(dt)
        if np.issubdtype(dt, np.signedinteger):
            xi = x.astype(jnp.int32)
            neg = xi < 0
            mv = lax.bitcast_convert_type(jnp.where(neg, -xi, xi), jnp.uint32)     # (|-2^31| = 2^31 as uint32)
        else:
            mv, neg = x.astype(jnp.uint32), jnp.zeros(x.shape, bool)
        ev, fin = jnp.zeros(x.shape, jnp.int32), jnp.ones(x.shape, bool)
    else:
        b = lax.bitcast_convert_type(v.astype(dt).astype(jnp.float32), jnp.uint32)  # (exact)
        e = ((b >> 23) & 255).astype(jnp.int32)
        mv = (b & jnp.uint32(0x7FFFFF)) | jnp.where(e > 0, jnp.uint32(0x800000), jnp.uint32(0))
        ev, fin, neg = jnp.maximum(e, 1) - 150, e < 255, (b >> 31) != 0
    sh = ev + es                                      # |v * scale| = mv 2^sh
    # sh >= 0: the integer mv << sh, below 9.2e18 (as hi:lo words) or not representable
    hi = jnp.where(sh >= 32, _shl(mv, sh - 32), _shr(mv, 32 - sh))
    lo = _shl(mv, sh)
    th, tl = jnp.uint32(TQ >> 32), jnp.uint32(TQ & 0xFFFFFFFF)
    fits = (mv == 0) | ((_blen(mv) + sh <= 63) & ((hi < th) | ((hi == th) & (lo < tl))))
    # sh < 0: mv / 2^k rounded half to even (below 2^32)
    k = -sh
    qq = _shr(mv, k)
    rem = mv - _shl(qq, k)
    half = _shl(jnp.uint32(1), k - 1)
    up = (k <= 32) & ((rem > half) | ((rem == half) & ((qq & 1) != 0)))
    qq = qq + up.astype(jnp.uint32)
    pos = sh >= 0
    j = jnp.where(pos, sh >> 4, 0)
    s15 = jnp.where(pos, sh & 15, 0)
    ylo = jnp.where(pos, _shl(mv, s15), qq)           # the integer = (yhi:ylo) << 16 j, yhi < 2^16
    yhi = jnp.where(pos, _shr(mv, 32 - s15), jnp.uint32(0))
    ok = fin & jnp.where(pos, fits, True)
    sg = jnp.where(neg, -1, 1).astype(jnp.int32)
    m16 = jnp.uint32(0xFFFF)
    pc = [ylo & m16, ylo >> 16, yhi]
    return (jnp.concatenate([j + i for i in range(3)], axis=-1),
            jnp.concatenate([sg * p.astype(jnp.int32) for p in pc], axis=-1), ok)


def _xemit(v):
    """f32 values [..., M] -> (word index offsets [..., K M], signed digit pieces [..., K M], finite mask [..., M]):
    the digit pieces (K of them, _xfmt) of each value's integer in units of 2^-149 (the kernel backend adds the same
    integer: kernel.DENSE)."""
    if jax.config.jax_enable_x64:
        b = lax.bitcast_convert_type(v.astype(jnp.float32), jnp.uint32).astype(jnp.int64)
        e = (b >> 23) & 255
        m = (b & 0x7FFFFF) | jnp.where(e > 0, 0x800000, 0)
        p = jnp.maximum(e - 1, 0)
        x = m << (p & 31)
        sg = jnp.where((b >> 31) != 0, -1, 1)
        j = p >> 5
        return (jnp.concatenate([j, j + 1], axis=-1),
                jnp.concatenate([sg * (x & 0xFFFFFFFF), sg * (x >> 32)], axis=-1), e < 255)
    b = lax.bitcast_convert_type(v.astype(jnp.float32), jnp.int32)
    e = (b >> 23) & 255
    m = (b & 0x7FFFFF) | jnp.where(e > 0, 0x800000, 0)
    p = jnp.maximum(e - 1, 0)
    s = (p & 15).astype(jnp.int32)
    mu = m.astype(jnp.uint32)
    x = mu << s.astype(jnp.uint32)                    # m << s mod 2^32; the bits from 32 up: m >> (32 - s)
    sg = jnp.where(b < 0, -1, 1)
    j = p >> 4
    pc = [(x & jnp.uint32(0xFFFF)).astype(jnp.int32), (x >> 16).astype(jnp.int32), _shr(mu, 32 - s).astype(jnp.int32)]
    return (jnp.concatenate([j, j + 1, j + 2], axis=-1), jnp.concatenate([sg * q for q in pc], axis=-1), e < 255)


# an exact tally's words take pieces below 2^D (_xemit) by scatter-add, which cannot carry: they are carry-normalised
# (_xnorm) before XNORM (int64 words: values into the tally; XNORM32 for int32: values into one bin) more pieces could
# have reached one word since the last time (a value adds at most one piece to a word), so every word but the top stays
# below 2^D + XNORM 2^D in magnitude (2^62; int32 words, after _xnorm's carry step: 2^16 + 2^15 + XNORM32 2^16 <
# 2^31) and none wraps; the top word only grows with the total, and one at or past the flag threshold (_xfmt) flags
# its bin (as a carry out of the top word does on the kernel backend): a bin is exact or NaN, never wrong
XNORM = 1 << 29


FUSED_WORDS = 1 << 20     # words: a digit-word tally this small takes its carry step fused (in a lax.cond), a larger
                          # one in place (_xnorm_loop)


def _xnorm(a, S, s=None):
    """a SMALL digit-word tally's words [*lead, W * S] (at most FUSED_WORDS: a larger one takes _xnorm_loop, in place)
    -> (the same integers, words 0..W-2 below 2^D (int64 words) or within [-2^15, 2^16 + 2^15) (int32 words), the bins
    whose top word is at or past the flag threshold (_xfmt)).  int64 words (bin-major: bin i's words at i W ..): the
    full carry chain.  int32 words (word-major: word j of every bin at j S ..): one carry step (word j keeps its low 16
    bits and takes word j-1's signed high part, below 2^15 in magnitude): enough for the headroom (XNORM32); the
    readout (_xscan) runs the full chain."""
    W, D, dt, _, _, lim = _xfmt(s)
    if dt == jnp.int32:
        m = dt((1 << D) - 1)
        w = a.reshape(a.shape[:-1] + (W, S))
        hi = jnp.concatenate([jnp.zeros_like(w[..., :1, :]), w[..., :W - 1, :] >> D], -2)
        w = jnp.concatenate([w[..., :W - 1, :] & m, w[..., W - 1:, :]], -2) + hi
        top = w[..., W - 1, :]
        return w.reshape(a.shape), (top >= lim) | (top <= -lim)
    w = a.reshape(a.shape[:-1] + (S, W))
    c = jnp.zeros(w.shape[:-1], dt)
    out = []
    for j in range(W - 1):
        t = w[..., j] + c
        out.append(t & dt((1 << D) - 1))
        c = t >> D
    top = w[..., W - 1] + c
    out.append(top)
    return jnp.stack(out, axis=-1).reshape(a.shape), (top >= lim) | (top <= -lim)


def _xnorm_loop(big, a, bad, ovf, n, fresh, S, s):
    """_xnorm when `big`, in place, for a large tally: -> (a, bad, ovf, n) as the lax.cond(big, carry step, identity)
    of the small-tally path gives them, but as ONE loop of W iterations (0 when not big) that updates a word row per
    iteration.  (XLA copies the operands of a lax.cond whose branch returns them unchanged: the accumulator, each
    time the cond is taken that way; measured 2 accumulators per tally held under jvp, round9/cellbytes NOTES.)
    int32 words: one carry step from the top word down (word j keeps its low 16 bits -- the top word all of them --
    plus word j-1's signed high part; word 0 takes none): _xnorm's step.  int64 words: the full carry chain from word 0
    up (the carry is loop state).  The top word's flag (the threshold of _xfmt) is taken at the iteration that writes
    its final value; the value count n becomes fresh(n) at the first iteration."""
    W, D, dt, _, _, lim = _xfmt(s)
    m = dt((1 << D) - 1)
    lead = a.shape[:-1]
    i32 = dt == jnp.int32
    w = a.reshape(lead + ((W, S) if i32 else (S, W)))
    ax = w.ndim - (2 if i32 else 1)

    def row(w, j):
        return lax.dynamic_index_in_dim(w, j, axis=ax, keepdims=False)

    def body(i, c):
        w, bad, ovf, n, cy = c
        if i32:                                       # top down: row j from rows j and j - 1 (both not yet updated)
            j = W - 1 - i
            cur, prev = row(w, j), jnp.where(j > 0, row(w, jnp.maximum(j - 1, 0)), dt(0))
            new = jnp.where(j == W - 1, cur, cur & m) + (prev >> D)
            fin = i == 0                              # (the top word's final value)
        else:                                         # bottom up with the carry cy
            j = i
            t = row(w, j) + cy
            new = jnp.where(j == W - 1, t, t & m)
            cy = t >> D
            fin = i == W - 1
        w = lax.dynamic_update_index_in_dim(w, new, j, axis=ax)
        f = fin & ((new >= lim) | (new <= -lim)) & ~bad
        return (w, bad | f, ovf + jnp.sum(f.astype(jnp.int32), dtype=jnp.int32),
                jnp.where(i == 0, fresh(n), n), cy)
    cy0 = jnp.zeros(() if i32 else lead + (S,), dt)
    w, bad, ovf, n, _ = lax.fori_loop(0, jnp.where(big, W, 0), body, (w, bad, ovf, n, cy0))
    return w.reshape(a.shape), bad, ovf, n


def _xread(a, s):
    """an exact tally's words [*lead, W * S] -> (its f32 output, the bins that cannot be read: a word at or past
    the flag threshold, or a total past f32's range).  The rounding of the kernel backend's sj_dread (kernel.DENSE):
    carry-normalise to digits, take the magnitude, round it once to 24 bits (nearest, ties to even), scale."""
    if not jax.config.jax_enable_x64:
        return _xread32(a, s)
    W = EXACT_WORDS
    w = a.reshape(a.shape[:-1] + (s.S, W))
    bad = jnp.any((w >= (1 << 62)) | (w <= -(1 << 62)), axis=-1)
    M32 = jnp.int64(0xFFFFFFFF)
    d, c = [], jnp.zeros(w.shape[:-1], jnp.int64)
    for j in range(W):
        t = w[..., j] + c
        d.append(t & M32)
        c = t >> 32
    neg = c < 0
    k = jnp.ones_like(c)
    nd = []
    for j in range(W):
        t = (~d[j] & M32) + k
        nd.append(t & M32)
        k = t >> 32
    d = [jnp.where(neg, x, y) for x, y in zip(nd, d)] + [jnp.where(neg, ~c + k, c)]
    D = jnp.stack(d, axis=-1).astype(jnp.uint64)                       # [..., W + 1] digits, magnitude
    nz = D != 0
    t = W - jnp.argmax(nz[..., ::-1], axis=-1)                          # the top nonzero digit (W + 1: none)
    t = jnp.where(jnp.any(nz, axis=-1), t, -1).astype(jnp.int32)
    get = lambda i: jnp.where(i >= 0, jnp.take_along_axis(D, jnp.clip(i, 0, W)[..., None], axis=-1)[..., 0],
                              jnp.uint64(0))
    dt, d1, d2 = get(t), get(t - 1), get(t - 2)
    s_ = (lax.clz(dt.astype(jnp.uint32)) % 32).astype(jnp.uint64)       # (dt == 0 only for an all-zero bin)
    h = (dt << 32) | d1
    top = jnp.where(s_ > 0, (h << s_) | (d2 >> (32 - s_)), h)
    st = jnp.where(s_ > 0, ((d2 << s_) & jnp.uint64(0xFFFFFFFF)) != 0, d2 != 0)
    below = jnp.arange(W + 1) < (t - 2)[..., None]
    st = st | jnp.any(nz & below, axis=-1)
    q = top >> 40
    r, half = top & jnp.uint64(0xFFFFFFFFFF), jnp.uint64(0x8000000000)
    q = q + ((r > half) | ((r == half) & (st | ((q & 1) != 0)))).astype(jnp.uint64)
    o = jnp.ldexp(q.astype(jnp.float64), 32 * (t - 1) - s_.astype(jnp.int32) - 149 + 40).astype(jnp.float32)
    # a total below 2^24 units is exact and its f32 bits ARE the integer: no float64 -> float32 conversion, which
    # XLA:CPU flushes to zero for a subnormal result (2^-149 read as 0; as _xread32)
    small = lax.bitcast_convert_type(D[..., 0].astype(jnp.uint32), jnp.float32)
    o = jnp.where((t <= 0) & (D[..., 0] < (1 << 24)), small, o)
    o = jnp.where(t < 0, jnp.float32(0), jnp.where(neg, -o, o))
    return o, bad | ~jnp.isfinite(o)



def _xscan(a, s):
    """int32 digit words [*lead, W * S] (word-major) -> (the magnitude's top 9 bytes g[0..8] from its top nonzero byte
    down (uint32 each), any nonzero byte below them, the sign, the top nonzero byte's index (-1: none), the magnitude's
    low 24 bits, the bins whose top word is at or past 2^30).  Two loops over the words with per-bin state: the carry
    chain to 16-bit digits (bottom up, kept as uint16 rows) and the lowest nonzero digit, then the magnitude's bytes
    top down (a negative total's digit j: 0 below the lowest nonzero digit L, 2^16 - d_L at L, ~d_j above)."""
    W, S = _xfmt(s)[0], s.S
    w = a.reshape(a.shape[:-1] + (W, S))
    m16, u32 = jnp.int32(0xFFFF), jnp.uint32
    sh = w.shape[:-2] + (S,)

    def up(j, c_):
        c, L, D = c_
        x = lax.dynamic_index_in_dim(w, j, axis=-2, keepdims=False) + c
        d = x & m16
        D = lax.dynamic_update_index_in_dim(D, d.astype(jnp.uint16), j, axis=-2)
        return x >> 16, jnp.where((L == W - 1) & (d != 0), j, L), D
    c, L, D = lax.fori_loop(0, W - 1, up, (jnp.zeros(sh, jnp.int32), jnp.full(sh, W - 1, jnp.int32),
                                          jnp.zeros(w.shape[:-2] + (W - 1, S), jnp.uint16)))
    top = w[..., W - 1, :] + c
    bad = (top >= (1 << 30)) | (top <= -(1 << 30))
    neg = top < 0
    mt = jnp.where(neg, ~top + (L == W - 1).astype(jnp.int32), top)      # (below 2^30 when not flagged)

    def dig(j):                                       # the magnitude's 16-bit digit j (0 <= j <= W)
        d = lax.dynamic_index_in_dim(D, jnp.clip(j, 0, W - 2), axis=-2, keepdims=False).astype(jnp.int32)
        e = jnp.where(neg, jnp.where(j < L, 0, jnp.where(j == L, (0x10000 - d) & m16, ~d & m16)), d)
        return jnp.where(j < W - 1, e, jnp.where(j == W - 1, mt & m16, (mt >> 16) & m16))

    def down(r, st_):
        A, B, C, n, t, st = st_
        i = 2 * (W + 1) - 1 - r
        b = ((dig(i >> 1) >> (8 * (i & 1))) & 255).astype(u32)
        on = (n > 0) | (b != 0)
        t = jnp.where((n == 0) & (b != 0), i, t)
        A = A | jnp.where(on & (n < 4), b << (8 * (3 - jnp.clip(n, 0, 3))).astype(u32), u32(0))
        B = B | jnp.where(on & (n >= 4) & (n < 8), b << (8 * (7 - jnp.clip(n, 4, 7))).astype(u32), u32(0))
        C = jnp.where(on & (n == 8), b, C)
        st = st | (on & (n >= 9) & (b != 0))
        return A, B, C, jnp.where(on, n + 1, n), t, st
    z = jnp.zeros(sh, u32)
    A, B, C, _, t, st = lax.fori_loop(0, 2 * (W + 1), down, (z, z, z, jnp.zeros(sh, jnp.int32),
                                                             jnp.full(sh, -1, jnp.int32), jnp.zeros(sh, bool)))
    g = [(A >> (24 - 8 * i)) & u32(255) for i in range(4)] + [(B >> (24 - 8 * i)) & u32(255) for i in range(4)] + [C]
    low24 = (dig(0) | ((dig(1) & 255) << 16)).astype(u32)
    return g, st, neg, t, low24, bad


def _xread32(a, s):
    """_xread for the int32 format (no x64): 16-bit digits, read a byte at a time; the same integer and rounding."""
    g, st9, neg, t, low24, bad = _xscan(a, s)
    dt, d1, d2, d3, d4 = g[:5]
    s_ = (lax.clz(dt) - 24).astype(jnp.uint32)                          # [0, 7] (8 only for an all-zero bin)
    h = (dt << 24) | (d1 << 16) | (d2 << 8) | d3
    top = (h << s_) | (d4 >> (8 - s_))                                  # the top 32 bits, msb set
    st = (((d4 << s_) & jnp.uint32(255)) != 0) | st9 | ((g[5] | g[6] | g[7] | g[8]) != 0)
    q = top >> 8
    r, half = top & jnp.uint32(255), jnp.uint32(128)
    q = q + ((r > half) | ((r == half) & (st | ((q & 1) != 0)))).astype(jnp.uint32)
    E = 8 * t - s_.astype(jnp.int32) - 165                              # o = q 2^E
    pow2 = lambda e: lax.bitcast_convert_type((e + 127) << 23, jnp.float32)     # 2^e, |e| <= 126
    E1 = jnp.clip(E, -100, 100)
    o = (q.astype(jnp.int32).astype(jnp.float32) * pow2(E1)) * pow2(E - E1)     # (exact, then one rounding)
    # a total below 2^24 units is exact and its f32 bits ARE the integer (subnormal or the first binade): no
    # multiply, so a platform that flushes subnormal arithmetic results keeps it
    small = lax.bitcast_convert_type(low24.astype(jnp.int32), jnp.float32)
    o = jnp.where(t <= 2, small, o)
    o = jnp.where(t < 0, jnp.float32(0), jnp.where(neg, -o, o))
    return o, bad | ~jnp.isfinite(o)


def _fread32(a, s):
    """_wide for a fixed-point tally without x64 (_fixed32) -> (its output, the bins that do not fit s.dtype): the exact
    integer total rounded once to float64's 53 bits (nearest, ties to even: _wide's int64 / float64 conversion), times
    2^-e (scale 2^e: exact, as float64's division), then converted as float64 -> s.dtype is (a float: rounded once more,
    nearest even, subnormals kept, past the range inf; an integer: truncated, saturated at the type's range)."""
    u32 = jnp.uint32
    g, st9, neg, t, _, bad = _xscan(a, s)
    zero = t < 0
    s_ = jnp.minimum(lax.clz(g[0]).astype(jnp.int32) - 24, 7)             # [0, 7]
    su = s_.astype(u32)
    h = (g[0] << 24) | (g[1] << 16) | (g[2] << 8) | g[3]
    lw = (g[4] << 24) | (g[5] << 16) | (g[6] << 8) | g[7]
    H = (h << su) | _shr(lw, 32 - s_)                                     # the top 64 bits H:L, msb set
    L = (lw << su) | (g[8] >> (8 - su))
    st = (((g[8] << su) & u32(255)) != 0) | st9
    Mh, Ml = H >> 11, (H << 21) | (L >> 11)                               # the top 53 bits (21 + 32)
    r = L & u32(0x7FF)
    up = (r > 0x400) | ((r == 0x400) & (st | ((Ml & 1) != 0)))
    Ml2 = Ml + up.astype(u32)
    Mh = Mh + (Ml2 < Ml).astype(u32)
    Ml = Ml2
    ov = Mh >= (1 << 21)                                                  # (rounded up to 2^53)
    Mh, Ml = jnp.where(ov, u32(1 << 20), Mh), jnp.where(ov, u32(0), Ml)
    E = 8 * (t - 7) - s_ + 11 + ov.astype(jnp.int32) - _scale_exp(s)     # the output value: M 2^E, M in [2^52, 2^53)
    dt = np.dtype(s.dtype)
    if dt in FIXED32_FLOATS:
        fi = jnp.finfo(dt)
        p, emin, emax = int(fi.nmant) + 1, int(fi.minexp), int(fi.maxexp) - 1
        eb = E + 52
        pp = jnp.where(eb >= emin, p, p - (emin - eb))                    # the bits the target keeps at eb
        rr = jnp.clip(53 - pp, 33, 60)                                    # (53 - p >= 42)
        q = _shr(Mh, rr - 32)
        rb = (_shr(Mh, rr - 33) & 1) != 0
        stk = ((Mh & (_shl(u32(1), rr - 33) - 1)) != 0) | (Ml != 0)
        q = q + (rb & (stk | ((q & 1) != 0))).astype(u32)
        ex = E + rr                                                       # the output: q 2^ex, exactly a dt value
        n = _blen(q)
        fe = ex + n - 1
        nrm = (jnp.clip(fe + 127, 0, 255).astype(u32) << 23) | (_shl(q, 24 - n) & u32(0x7FFFFF))
        bits = jnp.where(fe >= -126, nrm, _shl(q, ex + 149))             # (an f32 subnormal: bf16's)
        bits = jnp.where(fe > emax, u32(0x7F800000), bits)
        bits = jnp.where(zero | (q == 0), u32(0), bits) | jnp.where(neg, u32(0x80000000), u32(0))
        o = lax.bitcast_convert_type(bits, jnp.float32).astype(dt)       # (exact)
        return o, bad | ~jnp.isfinite(o)
    k = jnp.clip(-E, 0, 63)                                               # the value: M 2^-k
    Il = jnp.where(k >= 32, _shr(Mh, k - 32), _shl(Mh, 32 - k) | _shr(Ml, k))
    Ih = jnp.where(k >= 32, u32(0), _shr(Mh, k)) | (E > 0).astype(u32)   # nonzero: the integer part >= 2^32
    frac = jnp.where(k >= 32, ((Mh & (_shl(u32(1), k - 32) - 1)) | Ml) != 0, (Ml & (_shl(u32(1), k) - 1)) != 0)
    Il, Ih, frac = jnp.where(zero, u32(0), Il), jnp.where(zero, u32(0), Ih), frac & ~zero
    nb = 8 * dt.itemsize
    if np.issubdtype(dt, np.signedinteger):
        lim = u32(1 << (nb - 1))
        big = jnp.where(neg, (Ih != 0) | (Il > lim) | ((Il == lim) & frac), (Ih != 0) | (Il >= lim))
        vi = lax.bitcast_convert_type(Il, jnp.int32)
        vi = jnp.where(neg, -vi, vi)
        vi = jnp.where(big, jnp.where(neg, jnp.int32(-(1 << (nb - 1))), jnp.int32((1 << (nb - 1)) - 1)), vi)
        return vi.astype(dt), bad | big
    mx = (1 << nb) - 1
    big = jnp.where(neg, (Ih != 0) | (Il != 0) | frac, (Ih != 0) | (Il > u32(mx)))
    vu = jnp.where(neg, u32(0), jnp.where(big, u32(mx), Il))
    return vu.astype(dt), bad | big


PAIR_LIM = 1 << 62      # a fixed-point pair's high word at or past this flags its bin (no wrap: a chunk adds < 2^61)


def _wnorm(a, bad, ovf, f, S, lead):
    """the wrap guard of a fixed-point pair accumulator a [*lead, 2S] (_wide) at the bins f a chunk is about to add
    into (f [M]: bin indices, S = dropped), -> (bad, ovf).  A chunk adds at most XNORM values, each below 2^32 to a
    low word and below 2^31 in magnitude to a high word, so a word below PAIR_LIM = 2^62 before the chunk cannot wrap
    in it; a touched bin with a word at or past PAIR_LIM is flagged (NaN; ovf counts the chunk's values into a newly
    flagged bin).  So a bin is exact or NaN, never wrapped: exact up to at least 2^30 emits (2^62 / 2^32), NaN past
    where its low word reaches 2^62.  The bins are only read (a gather of M values per row), never written: a carry
    step that writes them back before the add held a copy of the accumulator under jvp (68 B per cell against 38 at
    2^23 cells, x64, 46202160; this guard 42, 46202877)."""
    w = a.reshape(tuple(lead) + (2, S))
    ok = f < S
    fi = jnp.clip(f, 0, S - 1)
    lo, hi = jnp.take(w[..., 0, :], fi, axis=-1), jnp.take(w[..., 1, :], fi, axis=-1)
    top = ((lo >= PAIR_LIM) | (hi >= PAIR_LIM) | (hi <= -PAIR_LIM)) & ok
    ovf = ovf + jnp.sum((top & ~jnp.take(bad, fi, axis=-1)).astype(jnp.int32), dtype=jnp.int32)
    return bad.at[..., jnp.where(ok, f, S)].max(top, mode="drop"), ovf


def _wide(a, s):
    """a fixed-point tally's accumulator [*lead, 2S] (the sums of the values' low 32 bits, unsigned, and of their
    high 32 bits, signed) -> (its output: the exact total / scale, in s.dtype; the bins that do not fit s.dtype).
    The kernel backend's carry word made explicit, and no int64 word wraps (_wnorm flags a bin whose word could
    pass 2^63 in the next chunk of at most XNORM values: exact up to at least 2^30 emits per bin, then NaN).
    A total that fits int64 converts as before; a larger one is hi * 2^32 + lo rounded once (|hi| < 2^53)."""
    S = s.S
    lo, hi = a[..., :S], a[..., S:]
    hi = hi + (lo >> 32)
    lo = lo & jnp.int64(0xFFFFFFFF)
    fits = (hi >= -(1 << 31)) & (hi < (1 << 31))
    f64 = jnp.float64
    t = jnp.where(fits, ((hi << 32) + lo).astype(f64), hi.astype(f64) * 2.0 ** 32 + lo.astype(f64))
    v = t / s.scale
    if jnp.issubdtype(s.dtype, jnp.floating):
        o = v.astype(s.dtype)
        return o, ~jnp.isfinite(o)
    if jnp.issubdtype(s.dtype, jnp.integer):
        ii = jnp.iinfo(s.dtype)
        return v.astype(s.dtype), ~((v >= float(ii.min)) & (v < float(ii.max) + 1.0))
    return v.astype(s.dtype), jnp.zeros(v.shape, bool)


def _rows(spec):
    """partial rows of a float, non-fixed-point sum tally (see ROWS); 1 for every other tally."""
    if spec.kind != "sum" or spec.fixed_point or _exact(spec) or not jnp.issubdtype(spec.dtype, jnp.floating):
        return 1
    n = int(np.prod(spec.lead, dtype=np.int64)) * spec.S
    return int(max(1, min(ROWS, ROWS_ELEMS // max(n, 1))))


def _emit(st, t, idx, val):
    spec = st.p.tallies[t.slot]
    if spec.kind == "discard":
        return t
    P = st.P
    S = spec.S
    L = tuple(spec.lead)
    i = _bc(idx, P)                                   # [P, *I]
    v = _bc(val, P)                                   # [P, *L, *I]
    for lm in st.lanes:     # inside a loop with a batched predicate (_while): a lane whose predicate is false
        nb = lm.ndim - 1    # emits nothing (adding zero), the "select" of a tally carry
        if tuple(L[:nb]) != tuple(lm.shape[1:]):
            raise NotImplementedError(f"tallymap: an emit into a tally with lead dims {L} inside a while loop whose "
                                      f"predicate has shape {lm.shape[1:]} (internal error)")
        v = jnp.where(lm.reshape(lm.shape + (1,) * (v.ndim - lm.ndim)), v, jnp.zeros((), v.dtype))
    m = jnp.broadcast_to(st.mask.reshape((P,) + (1,) * (i.ndim - 1)), i.shape)
    live = m & (i >= 0) & (i < S)
    st.drops = st.drops + jnp.sum((m & ~live).astype(jnp.int32), dtype=jnp.int32)   # out of range: dropped
    R = _rows(spec)
    if R > 1:                                         # lane p -> partial row p % R; dropped stays out of range
        row = jnp.broadcast_to((jnp.arange(P, dtype=jnp.int32) % R).reshape((P,) + (1,) * (i.ndim - 1)), i.shape)
        f = jnp.where(live, row * S + i, R * S).reshape(-1).astype(jnp.int32)
    else:
        f = jnp.where(live, i, S).reshape(-1).astype(jnp.int32)
    nl = len(L)
    v = jnp.moveaxis(v, 0, nl).reshape(L + (-1,))      # [*L, P*I]
    a = st.accs[t.slot]
    if _words(spec):                                  # (R == 1) the K digit pieces of each value (_xemit, _femit)
        W, _, wdt, K, XN, _ = _xfmt(spec)
        dt32 = wdt == jnp.int32
        k = t.slot
        M = int(f.shape[-1])
        CH = XN                                       # (an emit of more values runs in chunks, normalised between)
        for c0 in range(0, max(M, 1), CH):
            fc, vc = f[c0:c0 + CH], v[..., c0:c0 + CH]
            # this chunk adds at most M2 pieces into one word (a value's K pieces go to K words; a word takes low
            # pieces of some values and high pieces of others): normalise first if XN could be passed
            M2 = int(fc.shape[-1])
            if M2 == 0:
                continue
            if _perbin(spec):                         # per bin: the counts after this chunk, and the largest
                old = st.nadd[k]
                cnt = old.at[fc].add(1, mode="drop")
                big = jnp.max(jnp.where(fc < S, cnt[jnp.clip(fc, 0, S - 1)], 0)) > XN
                fresh = lambda n_, old=old: n_ - old                                        # noqa: E731
            else:
                cnt = st.nadd[k] + M2
                big = cnt > XN
                fresh = lambda n_, M2=M2: jnp.asarray(M2, n_.dtype)                        # noqa: E731
            if a.size > FUSED_WORDS:      # in place: a loop of W carry-step rows, run 0 times when not needed
                a, st.bad[k], st.ovf, st.nadd[k] = _xnorm_loop(big, a, st.bad[k], st.ovf, cnt, fresh, S, spec)
            else:                         # (a small tally: lax.cond, whose identity branch copies its operands)

                def norm(op, fresh=fresh):
                    a_, bad_, ovf_, n_ = op
                    a_, top = _xnorm(a_, S, spec)
                    return (a_, bad_ | top, ovf_ + jnp.sum((top & ~bad_).astype(jnp.int32), dtype=jnp.int32),
                            fresh(n_))
                a, st.bad[k], st.ovf, st.nadd[k] = lax.cond(big, norm, lambda op: op, (a, st.bad[k], st.ovf, cnt))
            jo, q, ok = _xemit(vc) if _exact(spec) else _femit(vc, spec)    # [*L, K P*I] x2, [*L, P*I]
            live = jnp.broadcast_to(fc < S, ok.shape)
            st.ovf = st.ovf + jnp.sum((~ok & live).astype(jnp.int32), dtype=jnp.int32)
            st.bad[k] = st.bad[k].at[..., fc].max(~ok, mode="drop")
            fK = jnp.concatenate([fc] * K)
            fb = jnp.where(jnp.concatenate([ok] * K, axis=-1) & (fK < S), fK, S)     # (dropped: bin S, out of range)
            # (word, bin) as two indices into the [W, S] (int32 words, word-major) or [S, W] (int64, bin-major) view:
            # no flat index W * S, which passes int32 (and jax's int32 index arithmetic) above 2^31 / W bins
            n = int(np.prod(a.shape[:-1], dtype=np.int64))
            fb, jo, q = fb.reshape(n, -1), jo.reshape(n, -1), q.reshape(n, -1)
            r = jnp.arange(n, dtype=jnp.int32)[:, None]
            if dt32:
                a = a.reshape(n, W, S).at[r, jo, fb].add(q, mode="drop").reshape(a.shape)
            else:
                a = a.reshape(n, S, W).at[r, fb, jo].add(q, mode="drop").reshape(a.shape)
        st.accs[k] = a
        return t
    if spec.fixed_point:
        v, ok = _fixed(v, spec.scale)
        live = jnp.broadcast_to(f < S, ok.shape)
        st.ovf = st.ovf + jnp.sum((~ok & live).astype(jnp.int32), dtype=jnp.int32)
        st.bad[t.slot] = st.bad[t.slot].at[..., f].max(~ok, mode="drop")        # a bin that dropped an emit
        # the wide accumulator (_wide): the low 32 bits of each value into row 0 of its [2, S] view, the signed high
        # 32 into row 1 (two indices: no flat index past 2S, which passes int32 above 2^30 bins), in chunks of at
        # most XNORM values, each chunk's bins checked first (_wnorm): a word that could wrap flags its bin
        k = t.slot
        M = int(f.shape[-1])
        for c0 in range(0, M, XNORM):
            fc, vc = f[c0:c0 + XNORM], v[..., c0:c0 + XNORM]
            st.bad[k], st.ovf = _wnorm(a, st.bad[k], st.ovf, fc, S, L)
            hl = jnp.concatenate([jnp.zeros(fc.shape, jnp.int32), jnp.ones(fc.shape, jnp.int32)])
            a = a.reshape(tuple(L) + (2, S)).at[..., hl, jnp.concatenate([fc, fc])].add(
                jnp.concatenate([vc & jnp.int64(0xFFFFFFFF), vc >> 32], axis=-1), mode="drop").reshape(a.shape)
        st.accs[k] = a
        return t
    v = v.astype(a.dtype)
    if _low(spec):                                    # SPLIT: a small value goes, scaled, into the LOW row
        RS = R * S
        small = jnp.abs(v) < SPLIT                    # (NaN: False)
        # (row, bin) into the [R + 1, S] view (row R: LOW): no flat index past R S + S, which passes int32 above
        # 2^30 bins (R = 1 there).  Dropped: bin S, out of range.  (f < RS: R S <= ROWS_ELEMS when R > 1)
        row = jnp.where(small, R, f // S) if R > 1 else jnp.where(small, 1, 0).astype(jnp.int32)
        col = jnp.broadcast_to(jnp.where(f < RS, f % S if R > 1 else f, S), small.shape)     # ([*L, P*I])
        row = jnp.broadcast_to(row, small.shape)
        q = jnp.where(small, v * LOW, v)
        n = int(np.prod(L, dtype=np.int64))
        r = jnp.arange(n, dtype=jnp.int32)[:, None]
        st.accs[t.slot] = a.reshape(n, R + 1, S).at[r, row.reshape(n, -1), col.reshape(n, -1)].add(
            q.reshape(n, -1), mode="drop").reshape(a.shape)
        return t
    st.accs[t.slot] = a.at[..., f].add(v, mode="drop")
    return t


def _check_tail(qleaves, leaves):
    """a spawn in a jvp program may carry only its PRIMAL leaves: JAX does not call spawn_p's jvp rule (which appends
    the tangents of the float leaves) when every tangent reaching it is a symbolic zero.  The missing trailing leaves
    are those tangents, i.e. zero, and must be written as zero (a queue slot is reused)."""
    if len(leaves) > len(qleaves) or any(not jnp.issubdtype(ql.dtype, jnp.inexact) for ql in qleaves[len(leaves):]):
        raise TypeError(f"tallymap: a spawn with {len(leaves)} child leaves in a program whose queue has "
                        f"{len(qleaves)} (internal error)")


def _spawn(st, t, when, leaves):
    if st.lanes:
        raise NotImplementedError("tallymap: a spawn inside a while loop with a batched predicate is not supported")
    q = st.queue
    P = st.P
    Q = st.p.spawn.capacity
    w = _bc(when, P) & st.mask
    pos = q["tail"] + jnp.cumsum(w.astype(jnp.int32), dtype=jnp.int32) - 1
    ok = w & (pos - q["head"] < Q)
    slot = jnp.where(ok, pos % Q, Q)
    _check_tail(q["leaves"], leaves)
    q["leaves"] = [ql.at[slot].set(_bc(leaves[i], P).astype(ql.dtype) if i < len(leaves) else
                                   jnp.zeros((P,) + ql.shape[1:], ql.dtype), mode="drop")
                   for i, ql in enumerate(q["leaves"])]
    n_ok = jnp.sum(ok.astype(jnp.int32), dtype=jnp.int32)
    q["tail"] = q["tail"] + n_ok
    q["spawned"] = q["spawned"] + n_ok
    q["dropped"] = q["dropped"] + jnp.sum((w & ~ok).astype(jnp.int32), dtype=jnp.int32)
    return t


def _claim(st, t, when):
    """a Records one-pass slot claim: the batch's claiming lanes take the next slots of the output's counter in lane
    order (masked lockstep: deterministic here; the layout is canonicalised after the launch either way,
    _primitive._canon).  slot -1: not claimed, or past the capacity (the raw buffer is full)."""
    if st.lanes:
        raise NotImplementedError("tallymap: t.append inside a while loop with a batched predicate is not supported")
    if st.p.tallies[t.slot].kind == "discard":
        return [t, B(jnp.full((st.P,), -1, jnp.int32), True)]
    w = _bc(when, st.P) & st.mask
    pos = st.ctr[t.slot] + jnp.cumsum(w.astype(jnp.int32), dtype=jnp.int32) - 1
    st.ctr[t.slot] = st.ctr[t.slot] + jnp.sum(w.astype(jnp.int32), dtype=jnp.int32)
    return [t, B(jnp.where(w & (pos < st.p.tallies[t.slot].S), pos, -1).astype(jnp.int32), True)]


def _ev(st, jaxpr, consts, args):
    env = {}

    def read(v):
        return B(lit(v), False) if isinstance(v, Literal) else env[v]

    for v, c in zip(jaxpr.constvars, consts):
        env[v] = B(c, False)
    for v, a in zip(jaxpr.invars, args):
        env[v] = a
    for e in jaxpr.eqns:
        ins = [read(v) for v in e.invars]
        name = e.primitive.name
        if name == "tallymap_emit":
            outs = [_emit(st, ins[0], ins[1], ins[2])]
        elif name == "tallymap_spawn":
            outs = [_spawn(st, ins[0], ins[1], ins[2:])]
        elif name == "tallymap_claim":
            outs = _claim(st, ins[0], ins[1])
        elif call_jaxpr(e) is not None:
            s = call_jaxpr(e)
            outs = _ev(st, s.jaxpr, s.consts, ins)
        elif name == "while":
            outs = _while(st, e, ins)
        elif name == "cond":
            outs = _cond(st, e, ins)
        elif name == "scan":
            outs = _scan(st, e, ins)
        else:
            if e.effects:
                raise located(NotImplementedError(
                    f"tallymap: primitive {name!r} has side effects {e.effects}; effects (debug.print, callbacks "
                    "with effects) are not supported inside tallymap.vmap."), e,
                    f"item function {st.p.where}")
            if any(isinstance(a, VT) for a in ins):
                raise located(TypeError(f"tallymap: {name} reads a tally"), e, f"item function {st.p.where}")
            try:
                if not any(a.b for a in ins):
                    outs = [B(x, False) for x in bind(e, [a.x for a in ins])]
                else:
                    r = jax.vmap(lambda *xs: bind(e, xs),
                                 in_axes=[0 if a.b else None for a in ins])(*[a.x for a in ins])
                    outs = [B(x, True) for x in r]
            except Exception as ex:
                better = ad_advice(ex, f"item function {st.p.where}")
                if better is None:
                    raise
                raise located(better, e, f"item function {st.p.where}") from None

        for v, o in zip(e.outvars, outs):
            env[v] = o
    return [read(v) for v in jaxpr.outvars]


def _sel(m, new, old):
    """per-item select for batched values (VT passes through)."""
    if isinstance(old, VT):
        return old
    mm = m.reshape(m.shape + (1,) * (new.ndim - m.ndim))
    return jnp.where(mm, new, old)


def _while(st, e, ins):
    """A predicate of shape B (batched: jax.vmap of a loop whose condition reads a batched value) follows JAX's while
    contract: the loop runs while ANY entry holds; an entry whose predicate is false keeps its carries and emits
    nothing.  The predicate is then [P, *B] and selects every carry ([P, *B, ...]) per entry."""
    w = while_parts(e)
    nc, nb = w.nc, w.nb
    cc, bc, init = ins[:nc], ins[nc:nc + nb], ins[nc + nb:]
    P = st.P
    isvt = [isinstance(a, VT) for a in init]
    arr0 = [_bc(a, P) for a, v in zip(init, isvt) if not v]
    outer = st.mask

    def carries(arr):
        it = iter(arr)
        return [a if v else B(next(it), True) for a, v in zip(init, isvt)]

    def pred(arr):
        r = _bc(_ev(st, w.cond.jaxpr, w.cond.consts, cc + carries(arr))[0], P)
        return r & outer.reshape(outer.shape + (1,) * (r.ndim - 1))

    def cond_f(s):
        return jnp.any(s[1])

    def body_f(s):
        arr, m, packed = s
        st.unpack(packed)
        saved = st.mask
        st.mask = m if m.ndim == 1 else jnp.any(m.reshape(P, -1), axis=1)
        if m.ndim > 1:
            st.lanes.append(m)
        out = _ev(st, w.body.jaxpr, w.body.consts, bc + carries(arr))
        if m.ndim > 1:
            st.lanes.pop()
        st.mask = saved
        new = [_sel(m, _bc(o, P), a) for o, a in zip([o for o, v in zip(out, isvt) if not v], arr)]
        return new, pred(new), st.pack()
    arr, _, packed = lax.while_loop(cond_f, body_f, (arr0, pred(arr0), st.pack()))
    st.unpack(packed)
    return carries(arr)


def _cond(st, e, ins):
    """Each branch is interpreted ONCE, into a jaxpr (jax.make_jaxpr); lax.cond / lax.switch then bind that jaxpr,
    so a cond nested d deep costs d interpretations, not 3^d (an interpretation per tally probe, per shape probe and
    per lax trace).  Which outputs are virtual tallies, and the shapes of the others, come from the same trace."""
    idx, ops = ins[0], ins[1:]
    brs = cond_branches(e)
    n = len(brs)
    P = st.P
    ii = jnp.clip(idx.x, 0, n - 1)

    def trace(k, mask):
        vt = {}

        def f(packed):
            st.unpack(packed)
            saved = st.mask
            st.mask = mask
            out = _ev(st, brs[k].jaxpr, brs[k].consts, ops)
            st.mask = saved
            vt.update({j: o for j, o in enumerate(out) if isinstance(o, VT)})
            return [None if isinstance(o, VT) else _bc(o, P) for o in out], st.pack()
        saved = st.pack()
        cj, shapes = jax.make_jaxpr(f, return_shape=True)(saved)
        st.unpack(saved)
        tree = jax.tree_util.tree_structure(shapes)
        run = lambda packed: jax.tree_util.tree_unflatten(
            tree, jaxpr_as_fun(cj)(*jax.tree_util.tree_leaves(packed)))
        return run, vt, shapes[0]
    masks = [st.mask if not idx.b else st.mask & (ii == k) for k in range(n)]
    traced = [trace(k, masks[k]) for k in range(n)]
    # outputs that are a virtual tally in ANY branch stay virtual: an inert (discard) tally may meet a filler value
    # in another branch (JAX's cond partial eval pads a residual that only one branch produces with zeros)
    vts = [None] * len(e.outvars)
    for _, vt, _ in traced:
        for j, o in vt.items():
            vts[j] = vts[j] or o
    keep = [j for j in range(len(vts)) if vts[j] is None]

    def branch(k):
        run = traced[k][0]

        def g(packed):
            out, packed = run(packed)
            return [out[j] for j in keep], packed
        return g
    if not idx.b:
        res, packed = lax.switch(ii, [branch(k) for k in range(n)], st.pack())
        st.unpack(packed)
    else:
        outs = []
        for k in range(n):
            r0 = [traced[k][2][j] for j in keep]
            skip = lambda packed, r0=r0: ([jnp.zeros(a.shape, a.dtype) for a in r0], packed)
            r, packed = lax.cond(jnp.any(masks[k]), branch(k), skip, st.pack())     # rare branches cost nothing
            st.unpack(packed)
            outs.append(r)
        res = [_select_n(ii, [o[j] for o in outs]) for j in range(len(outs[0]))]
    it = iter(res)
    return [vt if vt is not None else B(next(it), True) for vt in vts]


def _select_n(ii, cases):
    shp = cases[0].shape
    w = ii.reshape(ii.shape + (1,) * (len(shp) - 1))
    w = jnp.broadcast_to(w, shp)
    return lax.select_n(w, *cases) if len(cases) > 1 else cases[0]


def _scan(st, e, ins):
    sc = scan_parts(e)
    nc, nk = sc.nconsts, sc.ncarry
    consts, init, xs = ins[:nc], ins[nc:nc + nk], ins[nc + nk:]
    P = st.P
    isvt = [isinstance(a, VT) for a in init]
    arr0 = [_bc(a, P) for a, v in zip(init, isvt) if not v]
    # (an inert discard tally may be a scanned input / stacked output: a residual of reverse mode; it stays virtual)
    xvt = [isinstance(a, VT) for a in xs]
    xr = [a for a in xs if not isinstance(a, VT)]
    xsb = [a.b for a in xr]
    xarr = [jnp.moveaxis(a.x, 1, 0) if a.b else a.x for a in xr]     # scanned axis first
    yvt = {}

    def body(s, x):
        arr, packed = s
        st.unpack(packed)
        it = iter(arr)
        car = [a if v else B(next(it), True) for a, v in zip(init, isvt)]
        xi = iter(zip(x, xsb))
        xin = [a if v else B(*next(xi)) for a, v in zip(xs, xvt)]
        out = _ev(st, sc.body.jaxpr, sc.body.consts, consts + car + xin)
        new = [_bc(o, P) for o, v in zip(out[:nk], isvt) if not v]
        yvt.update((j, o) for j, o in enumerate(out[nk:]) if isinstance(o, VT))
        return (new, st.pack()), [_bc(o, P) for o in out[nk:] if not isinstance(o, VT)]
    (arr, packed), ys = lax.scan(body, (arr0, st.pack()), xarr, length=sc.length, reverse=sc.reverse)
    st.unpack(packed)
    it = iter(arr)
    car = [a if v else B(next(it), True) for a, v in zip(init, isvt)]
    yi = iter(ys)
    return car + [yvt[j] if j in yvt else B(jnp.moveaxis(next(yi), 0, 1), True) for j in range(len(e.outvars) - nk)]


# ------------------------------------------------------------------------------------------- driver
def run(p, pi, bc, xla_batch_size=None, **_):
    """-> [sum tally arrays] + [stats int32[5]] + [per-item results [N, ...]].  p.nact (n_active): items
    i >= bc[p.nact] are masked off like the padding of the last batch (no emits, per-item results zero)."""
    N = int(pi[0].shape[0])
    nlim = N if p.nact is None else jnp.clip(jnp.asarray(bc[p.nact], jnp.int32), 0, N)
    nT = p.nT
    ya = p.y_avals()
    accs = {}
    for k, s in enumerate(p.tallies):
        if s.kind == "sum":
            dt = _xfmt(s)[2] if _words(s) else jnp.int64 if s.fixed_point else s.dtype
            if _fixed32(s):
                _scale_exp(s)                       # (raises: a scale or dtype the int32 format cannot keep)
            w = _xfmt(s)[0] if _words(s) else 2 if s.fixed_point else _rows(s) + _low(s)
            accs[k] = jnp.zeros(tuple(s.lead) + (w * s.S,), dt)
    ys0 = [jnp.zeros((N,) + tuple(a.shape), a.dtype) for a in ya]
    P = int(min(max(N, 1), xla_batch_size or BLOCK))
    cj = p.cj
    queue = None
    if p.spawn is not None:
        Q = int(p.spawn.capacity)
        pav = [v.aval for v in cj.jaxpr.invars[nT:nT + p.nP]]
        queue = dict(leaves=[lax.full((Q,) + tuple(a.shape), 0, a.dtype) for a in pav], head=jnp.int32(0),
                     tail=jnp.int32(0), spawned=jnp.int32(0), dropped=jnp.int32(0))
    st = _St(p, P, accs, queue)
    lanes = jnp.arange(P, dtype=jnp.int32)

    def block_fn(ids, mask, lane_pi, ys, primary):
        st.mask = mask
        args = [VT(k) for k in range(nT)] + [B(x, True) for x in lane_pi] + [B(x, False) for x in bc]
        out = _ev(st, cj.jaxpr, cj.consts, args)
        yv = [_bc(o, P) for o in out[nT:]]
        wid = jnp.where(primary, ids, N)
        ys = [y.at[wid].set(v.astype(y.dtype), mode="drop") for y, v in zip(ys, yv)]
        ctr = jnp.int32(0)
        if p.trunc is not None:
            tr = yv[p.trunc].astype(bool).reshape(P, -1).any(axis=1)     # a laned program (jacrev) flags per lane
            ctr = jnp.sum((tr & mask & ~primary).astype(jnp.int32), dtype=jnp.int32)
        return ys, ctr

    if N == 0 and queue is None:
        return _finish(p, accs, None, ys0, st)
    if queue is None:
        nblk = -(-N // P)

        def body(c, s):
            packed, ys = s
            st.unpack(packed)
            ids = c * P + lanes
            mask = ids < nlim
            lane_pi = [jnp.take(x, jnp.clip(ids, 0, N - 1), axis=0, mode="clip") for x in pi]
            ys, _ = block_fn(ids, mask, lane_pi, ys, mask)
            return st.pack(), ys
        if nblk == 1:
            packed, ys = body(0, (st.pack(), ys0))
        else:
            packed, ys = lax.fori_loop(0, nblk, body, (st.pack(), ys0))
        st.unpack(packed)
        return _finish(p, st.accs, None, ys, st)

    # spawn: children (queued) first, then primaries
    def cond_s(s):
        nxt, packed, ys, ctr = s
        return (nxt < N) | (packed[3][2] > packed[3][1])

    def body_s(s):
        nxt, packed, ys, ctr = s
        st.unpack(packed)
        q = st.queue
        kq = jnp.minimum(q["tail"] - q["head"], P)
        is_child = lanes < kq
        pid = nxt + lanes - kq
        primary = ~is_child & (pid < N)
        mask = is_child | primary
        qs = (q["head"] + lanes) % Q
        take = lambda a, i: jnp.take(a, i, axis=0, mode="clip")
        if N > 0:
            lane_pi = [jnp.where(_lm(is_child, x), take(ql, qs), take(x, jnp.clip(pid, 0, max(N - 1, 0))))
                       for ql, x in zip(q["leaves"], pi)]
        else:
            lane_pi = [take(ql, qs) for ql in q["leaves"]]
        if p.rec is not None:           # the recording primal (_reverse_spawn.rec_primal): `me` <- slot / -1 - index
            lane_pi[p.rec] = jnp.where(is_child, qs, -1 - pid).astype(jnp.int32)
        q["head"] = q["head"] + kq
        ys, c = block_fn(pid, mask, lane_pi, ys, primary)
        return jnp.minimum(nxt + P - kq, N), st.pack(), ys, ctr + c
    _, packed, ys, ctr = lax.while_loop(cond_s, body_s, (jnp.int32(0), st.pack(), ys0, jnp.int32(0)))
    st.unpack(packed)
    q = st.queue
    return _finish(p, st.accs, (q["spawned"], q["dropped"], ctr), ys, st) + (q["leaves"] if p.rec is not None else [])


def _lm(m, x):
    return m.reshape(m.shape + (1,) * (x.ndim - 1))


# the readout of a digit-word or fixed-point tally (_xread, _fread32, _wide) holds per-bin temporaries (the carry
# chain's digit rows and per-bin state: about 70 B per bin for the int32 words, more for the int64 ones), on top of the
# accumulator: a tally above READ_CHUNK bins is read READ_CHUNK bins at a time (a loop; each bin's readout depends on
# its own words only, so the result is the same bits), so they are O(READ_CHUNK), not O(bins)
READ_CHUNK = 1 << 20


def _chunked(read, a, s):
    """read(a, s) -> (out [*lead, S], flags [*lead, S]) of the accumulator a [*lead, w * S] (w words per bin: word-major
    [w, S] for the int32 digit words and the int64 wide pair, bin-major [S, w] for the int64 digit words), READ_CHUNK
    bins at a time (the last chunk overlaps the one before it: its bins are read twice, to the same bits)."""
    S, CH = s.S, READ_CHUNK
    if S <= CH:
        return read(a, s)
    lead = a.shape[:-1]
    w = a.shape[-1] // S
    bin_major = _words(s) and not _perbin(s)          # (x64 exact: _xread's [S, W] layout)
    v = a.reshape(lead + ((S, w) if bin_major else (w, S)))
    sc = dataclasses.replace(s, S=CH)

    def piece(i):
        st = jnp.minimum(i * CH, S - CH)
        x = lax.dynamic_slice_in_dim(v, st, CH, axis=v.ndim - (2 if bin_major else 1))
        return st, read(x.reshape(lead + (w * CH,)), sc)
    o_av, f_av = jax.eval_shape(lambda x: read(x, sc), jax.ShapeDtypeStruct(lead + (w * CH,), a.dtype))

    def body(i, c):
        o, f = c
        st, (oc, fc) = piece(i)
        return (lax.dynamic_update_slice_in_dim(o, oc, st, axis=o.ndim - 1),
                lax.dynamic_update_slice_in_dim(f, fc, st, axis=f.ndim - 1))
    init = (jnp.zeros(lead + (S,), o_av.dtype), jnp.zeros(lead + (S,), f_av.dtype))
    return lax.fori_loop(0, -(-S // CH), body, init)


def _finish(p, accs, sp, ys, st):
    out = []
    for k, s in enumerate(p.tallies):
        if s.kind == "sum":
            a = accs[k]
            R = _rows(s)
            if _low(s):                             # the rows' sum + LOW * 2^-100 (exact: LOW holds multiples of
                a = a.reshape(tuple(s.lead) + (R + 1, s.S))    # 2^-49, so the product is a multiple of 2^-149)
                lo = a[..., R, :]
                a = (jnp.sum(a[..., :R, :], axis=-2) if R > 1 else a[..., 0, :]) + lo * UNLOW
            elif R > 1:
                a = jnp.sum(a.reshape(tuple(s.lead) + (R, s.S)), axis=-2)
            if s.fixed_point or _exact(s):          # a bin that dropped an emit is NaN (as on cuda)
                rd = _xread if _exact(s) else _fread32 if _fixed32(s) else _wide
                a, big = _chunked(rd, a, s)
                st.ovf = st.ovf + jnp.sum((big & ~st.bad[k]).astype(jnp.int32), dtype=jnp.int32)
                if jnp.issubdtype(s.dtype, jnp.floating):
                    a = jnp.where(st.bad[k] | big, jnp.asarray(np.nan, s.dtype), a)
            out.append(a)
    sp = sp if sp is not None else (jnp.int32(0),) * 3
    return out + [jnp.stack([jnp.asarray(x, jnp.int32) for x in (*sp, st.ovf, st.drops)])] + list(ys)
