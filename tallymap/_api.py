"""The public API (vmap, Sum, Queue, Result) and the ItemProgram: trace the item once, hand ONE primitive to JAX.

    user fn(tally, *args) -> tally | (tally, y)
      |  make_jaxpr + tally linearity check + hoist closed-over tracers as explicit operands
      |  retrace: cap top-level while loops at max_steps (a per-item `truncated` output)
      v
    ItemProgram (item jaxpr + layout)  ->  bind_call(*per_item, *broadcast, node=Node(prog))      (_primitive.py)

Every transformation is a rule of that primitive that builds a new ItemProgram by applying jax.jvp / jax.vjp /
jax.vmap to an ITEM function (jvp_prog here, vjp_prog in _reverse.py, the rest in _primitive.py), so the rules
compose to any order.
"""
import collections
import dataclasses
import os
import sys
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.core import Tracer
from jax.extend.core import ClosedJaxpr, jaxpr_as_fun

from . import _diagnostics, _errors
from ._jaxpr import digest, host_callbacks, item_name, located, subjaxprs
from ._rewrite import check_tallies, isfloat, retrace
from ._tally import KI, KK, LANED, RSEP, Tally, emit_p, rkey

MAX_ITEMS = 2 ** 31 - 1
_HERE = os.path.dirname(os.path.abspath(__file__))
_SKIP = []


def _call_site(fallback=None):
    """(filename, lineno) of the user's call into tallymap: the innermost frame that is neither tallymap's nor
    JAX's own (under jax.jit the mapped function is called by JAX's tracing machinery, not by the user).  A
    backend warning is raised when the program is LOWERED, which is nowhere near the user's code, so the call
    site travels with the program (Node.site) and the warning is attributed to it."""
    if not _SKIP:
        _SKIP.extend((_HERE, os.path.dirname(os.path.abspath(jax.__file__))))
    f = sys._getframe(1)
    while f is not None:
        path = os.path.abspath(f.f_code.co_filename)
        if not any(path == d or path.startswith(d + os.sep) for d in _SKIP):
            return (f.f_code.co_filename, f.f_lineno)
        f = f.f_back
    return fallback


# ------------------------------------------------------------------------------------------- user specs
@dataclasses.dataclass(frozen=True)
class Sum:
    """A commutative (sum) output.

    fixed_point=True with dtype float32 (the default) is EXACT: each bin is the correctly rounded float32 of the exact
    sum of every finite f32 value emitted into it, bitwise independent of the order of the emits (run to run, AD mode,
    backend), with no scale, no floor and no cap (`scale` is not used).  Every finite f32 is an integer multiple of
    2^-149 below 2^128, and a bin keeps that integer sum (5 signed int64 words on the kernel backend).  A bin is
    exact or NaN: a non-finite emit, a running sum past the accumulator's top word (about 2^166: 2^38 emits near
    float32's maximum) and a total beyond float32's range each make it NaN, counted in
    stats['fixed_point_overflow'].  Its jvp / jacfwd tangents, and reverse-mode cotangents of float32 broadcast
    operands, are exact too.  Device memory on the kernel backend: 44 B per bin (a 40 B accumulator + the output).
    The xla backend keeps an exact sum in int64 words with jax_enable_x64 and in int32 words of 16-bit digits without
    it: the same integer and the same bits either way (and the kernel backend's).  docs/float_semantics.md item 4.

    fixed_point=True with another dtype accumulates round(v * scale) in int64 fixed point: bitwise
    order-independent, each value rounded to a multiple of 1/scale.  Without jax_enable_x64 the xla backend keeps the
    same integers in int32 words (bitwise the x64 result) for a power-of-two scale in [2^-600, 2^600] and a float16,
    bfloat16 or <= 32-bit integer dtype; another scale or dtype raises there (enable x64, or the kernel backend).
    |v * scale| >= 9.2e18 cannot be represented: such an emit is dropped, counted in
    Result.stats['fixed_point_overflow'], and the BIN it was emitted to becomes NaN (every kernel flags its own bins: a
    primal bin by the primal's emits, a jvp / jacfwd tangent bin by that tangent's emits, on every path).  A BIN's sum
    does not wrap: it keeps a signed high word (the net carry out of its int64 word), so its total is exact past
    2^63 / scale; a total that does not fit `dtype` is counted in 'fixed_point_overflow' too (a float bin is NaN).
    A jvp / vjp tangent tally of this output uses the same scale."""
    shape: Any
    dtype: Any = jnp.float32
    fixed_point: bool = False
    scale: float = 2.0 ** 32


@dataclasses.dataclass(frozen=True)
class Queue:
    """Spawn queue of `capacity` pending children (ring buffer).  A child that does not fit is dropped and counted
    in stats['children_dropped'].  The capacity is a tallymap resource limit, so a dropped child makes every float
    tally of the call NaN (never a silently truncated shower; integer tallies cannot be NaN, so they are left as they
    are: check stats['children_dropped']).  Reverse mode (the tree adjoint) records every child of the call, so it
    needs capacity >= the TOTAL number of children; a larger tree gives a NaN gradient (never a wrong one).
    Sizing: README.md, "Sizing the Queue"."""
    capacity: int


@dataclasses.dataclass(frozen=True, init=False)
class Records:
    """A RECORDS output: each item appends records with ``t = t.append(name, {field: value, ...}, when=...)``.

        tallymap.Records({"sensor": jnp.int32, "t": jnp.float32, "w": jnp.float32}, capacity=1 << 20)
        tallymap.Records(1 << 20, {...})            # (the same: capacity first)
        tallymap.Records({...})                     # eager only: capacity from the counted records

    Result.outputs[name] is {field: array[capacity]}: the concatenation over items, in item order, of each item's
    records in the order it appended them, i.e. record k of item i is at slot offset[i] + k (offset = the exclusive
    prefix sum of the per-item counts).  Slots past the last record are zero.  Result.stats['appended'][name] is
    the number of records (the true total, also when it exceeds the capacity).  The layout is deterministic: it does
    not depend on the backend, on the order the items ran in or on the run.

    Fields are scalars of a numeric dtype.  Float fields carry tangents and cotangents (the pathwise derivative of
    each record's value); whether a record exists is a discrete decision and carries no derivative; integer fields
    have none.  A float field stores each value exactly, subnormals included, except that -0.0 is stored as +0.0
    (on the xla backend on a CPU, XLA:CPU flushes subnormals to zero in its own arithmetic, so they are stored as
    0 there; docs/float_semantics.md item 6).

    capacity: the number of slots (static: it is the outputs' shape).  None: the eager call's own count, rounded up
    to a power of two (a traced call -- jax.jit, a vmap of the call -- needs an int).  A call that appends more than
    `capacity` records is a resource-limit violation like a full Queue: every float output of the call (records and
    sums) is NaN, every integer record field is -1, and stats['records_dropped'][name] counts the records that did
    not fit.  Overflow is deterministic: it is decided by the true total, and the poisoned outputs do not depend on
    which records reached the buffer.  Sizing: stats['appended'][name] is exact also on overflow (the sum of the
    per-item counts of the same launch, an int32 scalar on the device), so a caller can rerun at the next power of
    two above it.

    Implementation.  ONE PASS (the default, with a capacity and without spawn; _records.py): each append takes the
    next slot of a raw buffer (one atomic counter per output) and writes its fields and its key (item, k) there; the
    same launch then moves every record to offset[i] + k (offset from that launch's own per-item counts), so the
    layout is exactly count-then-fill's.  A jvp writes each record's tangent to the same slot in the same launch.  A
    derivative that replays the items in another launch (grad, vjp, linearize, jax.checkpoint, an unfused jvp) runs
    the program's fill twin with the primal's per-item counts as its record: record k of item i at offset[i] + k,
    written only while k < count; a replay whose count differs (not bitwise) makes that derivative NaN.
    COUNT-THEN-FILL (spawn, capacity=None): the call runs the items twice, a count pass (the per-item record counts;
    every other output is discarded) and the fill pass, which writes record k of item i at offset[i] + k.  The fill
    writes an item's records only while k < its counted number, and an item whose fill appends a different number of
    records than its count (the two passes did not replay bitwise) poisons the call like an overflow:
    stats['records_mismatch'][name] counts those items (always 0 in one pass).  With spawn the items are the spawn
    tree's nodes in breadth-first order (the roots in item order, then each generation by (parent, spawn ordinal)):
    the count pass runs as the tree adjoint's recording primal and each child receives its node id through the spawn.
    Both implementations give the same outputs bitwise (tests/records.py).  Not supported (raises): t.append inside
    jax.vmap within an item (its lanes would need lane-dependent slots, which tallymap's emits do not support)."""
    fields: tuple          # ((name, np.dtype), ...), sorted by name
    capacity: Any = None

    def __init__(self, *args, fields=None, capacity=None):
        for a in args:
            if isinstance(a, dict):
                if fields is not None:
                    raise TypeError("tallymap.Records: fields given twice")
                fields = a
            elif a is None or isinstance(a, (int, np.integer)):
                capacity = a
            else:
                raise TypeError(f"tallymap.Records(capacity, fields): unexpected argument {a!r}")
        if not isinstance(fields, dict) or not fields:
            raise TypeError("tallymap.Records needs fields: a non-empty dict {field name: dtype}")
        fl = []
        for f, dt in sorted(fields.items()):
            dt = np.dtype(dt)
            if not isinstance(f, str) or not (np.issubdtype(dt, np.number)):
                raise TypeError(f"tallymap.Records: field {f!r}: a str name and a numeric dtype, got {dt} (store a "
                                "bool as an integer)")
            fl.append((f, dt))
        if capacity is not None:
            if isinstance(capacity, bool) or not isinstance(capacity, (int, np.integer)) or not (
                    1 <= int(capacity) <= MAX_ITEMS):
                raise ValueError(f"tallymap.Records: capacity must be an int in [1, 2^31 - 1] or None, got "
                                 f"{capacity!r}")
            capacity = int(capacity)
        object.__setattr__(self, "fields", tuple(fl))
        object.__setattr__(self, "capacity", capacity)


class Result(NamedTuple):
    """What a tallymap.vmap function returns.  When a tallymap resource limit intervened (max_steps, the spawn
    Queue capacity) every float output is NaN; a fixed_point=True bin an unrepresentable emit was dropped from (a
    non-finite one, for float32) is NaN.
    The counters in `stats` say which limit it was."""
    outputs: dict        # name -> summed output (float: NaN when a resource limit intervened)
    per_item: Any        # pytree of [N, ...] per-item results (None if the item returns only its tally)
    truncated: Any       # bool[N]: the item hit max_steps in a top-level while loop
    stats: dict          # n_truncated, dropped_emits, fixed_point_overflow; with spawn also spawned /
                         # children_dropped / children_truncated


# the stats output of every backend call (int32[len(STATS)]), slot by slot: ST[name] is its index.  The kernel
# backend's C writes the same slots by number (SJ_STATS[i], backends/kernel/kernel.py).
STATS = ("spawned", "children_dropped", "children_truncated", "fixed_point_overflow", "dropped_emits")
ST = {n: i for i, n in enumerate(STATS)}


class UnsupportedError(NotImplementedError):
    """A backend cannot run this program.  backend='auto' raises it too (the kernel backend never falls back to the
    xla backend: pass backend='xla' for that)."""


# ------------------------------------------------------------------------------------------- programs
@dataclasses.dataclass(frozen=True, eq=False)
class TSpec:
    name: str
    S: int                 # flat size
    dtype: Any
    lead: tuple = ()       # extra leading dims (dual tangent lanes)
    fixed_point: bool = False
    kind: str = "sum"      # 'sum' | 'spawn' | 'discard'
    scale: float = 2.0 ** 32
    exact: bool = False    # float32 sum, exact and order-independent: the integer sum of every finite f32 value (in
                           # units of 2^-149), rounded once to f32 at readout; no scale, no floor, no cap.  A user's
                           # Sum(fixed_point=True) of dtype float32, its jvp / jacfwd tangents, and the reverse
                           # program's cotangent tallies (vjp_prog).  The kernel backend keeps DENSE_WORDS int64 words
                           # per bin, the xla backend EXACT_WORDS int64 words with jax_enable_x64, EXACT_WORDS32 int32
                           # words without it (xla._xfmt)
    slots: bool = False    # indexed by the recording primal's queue slot (ItemProgram.rec): under a sharded jit it is
                           # renumbered across devices like the queue ring (_primitive._global_ring), not summed


# an f32 value is m * 2^(p - 149), m < 2^24, p = max(biased exponent - 1, 0) <= 253: an integer below 2^277 in units of
# 2^-149.  The kernel backend keeps a bin's integer sum in DENSE_WORDS signed int64 words of weight 2^(63 j) (an int64
# overflow of a word carries +-2 into the next: kernel.DENSE); the xla backend in EXACT_WORDS int64 words of 32-bit
# digits (no carries: scatter-add)
EXACT_WORDS = 9
DENSE_WORDS = 5


@dataclasses.dataclass(frozen=True, eq=False)
class ItemProgram:
    """invars: [tallies] + [per-item leaves (per-item avals)] + [broadcast leaves];
    outvars: [tallies (same slots)] + [per-item results]."""
    cj: ClosedJaxpr
    tallies: tuple
    nP: int
    nB: int
    trunc: Any = None      # index of the `truncated` flag among the per-item results
    spawn: Any = None      # Queue or None
    tag: str = "primal"
    where: str = ""        # the user's item function ('qualname (file:line)'), for error messages
    nact: Any = None       # n_active: index among the broadcast operands of the int32 ACTIVE ITEM COUNT.  It is the
                           # launch's item count: items i >= it do not run (no emits, per-item results zero).  The
                           # item jaxpr does not read it; every derived program keeps it at the same index.
    rec: Any = None        # spawn, the tree adjoint's recording primal (_reverse_spawn.rec_primal): index of the int32
                           # per-item leaf `me`, which the backend sets to each item's queue slot (a child) or -1 - its
                           # index (a root) when it starts; the queue leaves are returned after the per-item results.
    app: Any = None        # a ONE-PASS Records program (_records.py): (index of the per-item operand cid, ((output
                           # name, capacity, index of its per-item count result), ...)).  Every launch is canonicalised
                           # (_records.canon); a replay in another launch runs its fill twin (_records.twin).

    @property
    def nT(self):
        return len(self.tallies)

    @property
    def nY(self):
        return len(self.cj.jaxpr.outvars) - self.nT

    def y_avals(self):
        return [v.aval for v in self.cj.jaxpr.outvars[self.nT:]]

    def in_avals(self):
        return [v.aval for v in self.cj.jaxpr.invars]

    # the backend outputs of a program: [out_tallies (the sums)] + [stats int32[len(STATS)]] + [per-item results]
    @property
    def out_tallies(self):
        """indices of the tallies a backend returns, in output order: the sums."""
        return [k for k, s in enumerate(self.tallies) if s.kind == "sum"]

    @property
    def float_outs(self):
        """positions among out_tallies of the float ones (the outputs that have tangents and cotangents)."""
        return [j for j, k in enumerate(self.out_tallies) if isfloat(self.tallies[k].dtype)]

    @property
    def float_ys(self):
        """indices of the float per-item results."""
        return [j for j, a in enumerate(self.y_avals()) if isfloat(a.dtype)]


def _sds(a):
    return jax.ShapeDtypeStruct(a.shape, a.dtype)


def _tally_aval(s):
    if s.kind == "spawn":
        return jax.ShapeDtypeStruct((), s.dtype)
    return jax.ShapeDtypeStruct(tuple(s.lead) + (s.S,), s.dtype)


def _mk(fn, avals):
    """trace fn to a ClosedJaxpr.  Every ItemProgram is built through here, so a failure gets
    _errors.ad_advice, which explains the failures an item makes likely (a custom_vjp, which JAX cannot
    forward-differentiate; a leaked tracer)."""
    try:
        return jax.make_jaxpr(fn)(*avals)
    except Exception as e:
        raise (_errors.ad_advice(e, "item program") or e) from None


# ------------------------------------------------------------------------------------------- transforms
def jvp_prog(p, pert_pi, pert_bc):
    f = jaxpr_as_fun(p.cj)
    nT, nP, nB = p.nT, p.nP, p.nB
    # discarded float tallies get (discarded) tangent tallies too: the jvp of an emit into them is an emit, and it
    # must land in a tally slot (derived programs -- lin / T -- discard their primal tallies)
    tf = [i for i, s in enumerate(p.tallies) if s.kind in ("sum", "discard") and isfloat(s.dtype)]
    yf = p.float_ys
    ip = [i for i in range(nP) if pert_pi[i]]
    ib = [i for i in range(nB) if pert_bc[i]]

    def J(*a):
        t, td = list(a[:nT]), list(a[nT:nT + len(tf)])
        o = nT + len(tf)                 # layout: [t, t'] [pi, pi'] [bc, bc']
        pi, pid = list(a[o:o + nP]), list(a[o + nP:o + nP + len(ip)])
        o += nP + len(ip)
        bc, bcd = list(a[o:o + nB]), list(a[o + nB:])

        def g(ts, ps, bs):
            tt, pp, bb = list(t), list(pi), list(bc)
            for i, v in zip(tf, ts):
                tt[i] = v
            for i, v in zip(ip, ps):
                pp[i] = v
            for i, v in zip(ib, bs):
                bb[i] = v
            return f(*tt, *pp, *bb)
        out, dout = jax.jvp(g, ([t[i] for i in tf], [pi[i] for i in ip], [bc[i] for i in ib]), (td, pid, bcd))
        dt = [dout[i] for i in tf]
        if p.trunc is not None:      # (max_steps) a truncated item poisons the tangent sums itself: _poison_tally
            bad = jnp.any(out[nT + p.trunc])          # (a dual-lanes program flags per lane: lanes share control flow)
            dt = [_poison_tally(x, bad) if p.tallies[i].kind == "sum" else x for i, x in zip(tf, dt)]
        return list(out[:nT]) + dt + list(out[nT:]) + [dout[nT + j] for j in yf]
    av = p.in_avals()
    tav, pav, bav = av[:nT], av[nT:nT + nP], av[nT + nP:]
    avals = tav + [tav[i] for i in tf] + pav + [pav[i] for i in ip] + bav + [bav[i] for i in ib]
    from ._reverse import forward_tangents  # (here: _reverse imports this module)
    with forward_tangents():        # (sparse table reads of an adjoint program: see _reverse.tread)
        cj = _mk(J, [_sds(a) for a in avals])
    tl = list(p.tallies) + [dataclasses.replace(p.tallies[i], name=p.tallies[i].name + "'") for i in tf]
    return ItemProgram(cj, tuple(tl), nP + len(ip), nB + len(ib), p.trunc, p.spawn, p.tag + "+jvp", where=p.where,
                       nact=p.nact, app=p.app)


def _poison_tally(t, bad):
    """the tally t with NaN emitted into every bin when the item's `bad` (its truncation flag) is set.  The primal
    outputs of a truncated call are poisoned by _poison_primal, whose tangent is the identity (so that jax.grad needs
    no forward launch); the tangent sums are poisoned here, inside the tangent / fused / dual-lane kernel, by the
    items that were truncated (per item).  A fixed-point sum flags the
    NaN emits as overflows (stats['fixed_point_overflow'] then counts them too)."""
    nan = jnp.full(t.shape[:-1], np.nan, t.dtype)
    return lax.cond(bad, lambda t: lax.fori_loop(0, t.shape[-1], lambda i, t: emit_p.bind(t, i, nan), t),
                    lambda t: t, t)


# (the item VJP program, vjp_prog, lives with the reverse-mode internals in _reverse.py)


# ------------------------------------------------------------------------------------------- dispatch
# Item programs already built, least recently used first: (structure of the user's traced item, the identity of its
# constants and host callbacks, the build's static configuration) -> (that traced item, its ItemProgram).  An eager
# call still traces the user's function (only that trace sees the values it closes over now), but a trace equal to
# an earlier one reuses that ItemProgram: no re-trace of the rewrite, and every cache downstream (the eager jit, the
# kernel, JAX's own) sees the same program object instead of a fresh copy per call.  Equal means the same structural
# digest (literals by their bytes, nested constants by their bytes), the same host-callback objects, and constants
# that are the same objects or small arrays of the same dtype, shape and BYTES (an array made inside the item is a
# new object at every trace).  A program closing over a live trace's values is never cached.
_PROGS = collections.OrderedDict()
PROG_CACHE = 64


def _spec_key(s):
    return tuple(getattr(s, f.name) for f in dataclasses.fields(s))


def _prog_key(cj, tspecs, mode, caps, M, max_steps, spawn, split, nact, where):
    return (digest(cj.jaxpr), len(cj.consts),
            tuple(_spec_key(s) for s in tspecs), mode, None if caps is None else tuple(sorted(caps.items())), M,
            max_steps, None if spawn is None else int(spawn.capacity), split, nact, where,
            bool(jax.config.jax_enable_x64))


PROG_CONST_BYTES = 1 << 16    # constants up to this size are compared by value; larger ones only by identity


def _same_const(x, y):
    if x is y:
        return True
    try:
        if (type(x) is not type(y) or tuple(x.shape) != tuple(y.shape) or x.dtype != y.dtype
                or not isinstance(x.dtype, np.dtype) or x.dtype == np.dtype(object)
                or x.size * x.dtype.itemsize > PROG_CONST_BYTES):
            return False
        return np.asarray(x).tobytes() == np.asarray(y).tobytes()
    except Exception:                   # (an opaque constant: identity only)
        return False


def _same_objects(a, b):
    """the cached trace `a` can stand for the new trace `b` (equal digests): the same host-callback objects, and
    each constant the same object or an equal small array."""
    ca, cb = host_callbacks(a.jaxpr), host_callbacks(b.jaxpr)
    return (len(a.consts) == len(b.consts) and all(_same_const(x, y) for x, y in zip(a.consts, b.consts))
            and len(ca) == len(cb) and all(x is y for x, y in zip(ca, cb)))


_WARNED = set()


# ------------------------------------------------------------------------------------------- vmap
_OPTS = {                  # the public options; measurement switches are test-only module constants
    "cuda_math": "kernel backend on a CUDA GPU, f32 div/sqrt: 'approx' (default) | 'ieee' | 'fast' (opt-in: also FMA "
                 "contraction and fast division by a divisor kept from zero; statistical) (the CPU: IEEE)",
    "schedule": "kernel backend: 'item' (default: a thread runs whole items) | 'step' (refill per loop step)",
    "min_blocks_per_sm": "kernel backend on a CUDA GPU: None (default) | N: __launch_bounds__(block, N), registers "
                         "capped for N resident blocks per SM (may spill; outputs bitwise unchanged; ignored on CPU "
                         "and xla)",
    "xla_batch_size": "xla backend: items per batch",
    "rev_checkpoints": "reverse mode: checkpoint buffer size C (default 'auto': from the loop state's size)",
    "rev_stats": "reverse through spawn: report the tree adjoint's stats to the host (costs jit's fast path)",
}
BACKENDS = ("auto", "xla")


def vmap(fn=None, *, outputs, in_axes=0, max_steps=None, spawn=None, backend="auto", **opts):
    """Map `fn(tally, *args) -> tally | (tally, y)` over the leading axis of the mapped args.

    Meaning: jax.vmap(fn) over the items followed by a sum of their tallies.  Every JAX transformation works.

    outputs:   {name: tallymap.Sum(shape, dtype=float32, fixed_point=False) | tallymap.Records(fields, capacity)}
    in_axes:   0 (mapped) or None (broadcast, e.g. tables) per positional arg, like jax.vmap
    max_steps: cap on every top-level while_loop.  None (the default) is exactly jax.vmap: no cap, and an item
               that does not terminate does not terminate.  Setting it is the recommended safety net; an item that
               hits the cap is flagged in Result.truncated and POISONS every float tally of the call (see below)
    spawn:     tallymap.Queue(capacity) to allow tally.spawn(child)
    backend:   'auto' (default): the kernel backend, compiled for the platform JAX places the computation on (a
               CUDA GPU or the CPU: jax.default_device / device_put / jit decide, never tallymap); a program it
               cannot compile raises UnsupportedError; other platforms run the xla backend with a one-time warning
               | 'xla': the plain-JAX reference interpreter
    opts:      cuda_math          cuda f32 div/sqrt: 'approx' (default; XLA:GPU's own PTX) | 'ieee' (correctly
                                  rounded and portable) | 'fast' (opt-in: approx plus FMA
                                  contraction of multiply-adds and div.approx for a divisor proven at least 2^-126
                                  in magnitude; results agree only statistically, never bitwise).
               schedule           cuda: 'item' (default: a thread runs one whole item, then claims the next) |
                                  'step' (the step schedule: a thread is refilled with a new item at loop-STEP
                                  granularity, so a thread whose item ended does not idle until its warp's longest
                                  item ends).  'step' needs an item with exactly one top-level while loop and
                                  raises UnsupportedError otherwise; derived kernels (jvp, lanes, an adjoint) use it
                                  wherever they split.  Which is faster depends on the workload: measure.
               min_blocks_per_sm  cuda: None (default: __launch_bounds__(block) alone) | N (opt-in): the kernel's
                                  __launch_bounds__(block, N): ptxas caps registers so N blocks fit one SM (more
                                  resident warps to hide latency; it may spill to local memory).  Register
                                  allocation does not change arithmetic: outputs are bitwise those of the default.
                                  Ignored on the CPU target and the xla backend.  Measure: it helps latency-bound
                                  kernels and can hurt when the spills land in the hot loop.
               xla_batch_size     xla backend: items per batch (default 2^18; bounds its memory).
               rev_checkpoints    reverse mode: states per checkpoint buffer C (default 'auto': 16 for a loop step
                                  of fewer than 1000 equations, else the largest of 4, 8, ..., 64 whose 3C+1
                                  (single-loop adjoint: 2C+1) states fit 64 KiB; _reverse.auto_checkpoints).  A loop
                                  of K <= C steps needs no replay; O(K) work for K <= C^2.
               rev_stats          reverse through spawn: report the tree adjoint's stats to the host
                                  (tallymap.diagnostics()['reverse']).  Off by default: a host callback takes jax.jit
                                  off its C++ fast path.  A failed check makes the gradient NaN either way.
               Any other name raises (there are no private options: measurement switches are test-only module
               constants, e.g. codegen.FILL, _primitive.LANES; negative controls live in the tests).
    The returned function takes the mapped/broadcast args, plus optionally n_active=<traced int>: only items
    i < n_active run (fixed capacity N, varying active count: no retrace / recompile).  n_active is the launch's
    item count, not part of the item program: items i >= n_active do not run, so they emit nothing, never count
    as truncated, and their per-item results are zero.

    Resource limits.  The result is exactly jax.vmap(fn) over the items followed by a sum, and deviates only where
    a tallymap resource limit intervenes: max_steps, or the spawn Queue capacity.  When one of them does, every
    float tally of the call is NaN, so a truncated or dropped item is never silently missing from a sum.  Which
    limit it was is in Result.stats: n_truncated (and children_truncated) for max_steps, children_dropped for the
    Queue.
    Integer tallies cannot be NaN, so they are left as they are and their counter is the only signal.  Per-item
    outputs are not poisoned: they carry their own per-item flag (Result.truncated) in the same Result.  Gradients
    of a poisoned tally are NaN, and nothing else about them changes.  There is no opt-out: a deliberate step
    budget belongs in the loop condition ((alive) & (step < K)), which is exact JAX semantics.

    An out-of-range emit index (negative or >= the output size) is DROPPED, exactly as JAX's own scatter drops
    out-of-bounds updates, so this is not a deviation and does not poison.  Dropped emits are counted in
    stats['dropped_emits']: every dropped index of the call, over every tally, so an off-by-one binning shows up
    there.  It counts the EMITS the launch executed, so a transformation that emits more than once per index (a
    fused jvp also emits into the tangent tally) counts each of them.
    """
    if fn is None:
        return lambda f: vmap(f, outputs=outputs, in_axes=in_axes, max_steps=max_steps, spawn=spawn,
                              backend=backend, **opts)
    bad = sorted(set(opts) - set(_OPTS))
    if bad:
        raise ValueError(f"tallymap: unknown option(s) {bad}; options are "
                         + ", ".join(_OPTS))
    if opts.get("schedule", "item") not in ("item", "step"):
        raise ValueError(f"tallymap: schedule must be 'item' or 'step', got {opts['schedule']!r}")
    mb = opts.get("min_blocks_per_sm")
    if mb is not None and (isinstance(mb, bool) or not isinstance(mb, (int, np.integer)) or not 1 <= int(mb) <= 32):
        raise ValueError(f"tallymap: min_blocks_per_sm must be None or an integer in 1..32, got {mb!r}")
    if backend == "cuda":
        raise ValueError("tallymap: backend='cuda' was removed in 0.7: use backend='auto' (the default), which runs "
                         "the compiled kernel on the platform JAX places the computation on (a CUDA GPU or the CPU)")
    if backend not in BACKENDS:
        raise ValueError(f"tallymap: unknown backend {backend!r}; backends are {', '.join(BACKENDS)}")
    if spawn is not None and not isinstance(spawn, Queue):
        raise TypeError("tallymap: spawn must be tallymap.Queue(capacity)")
    outs_spec = {k: (v if isinstance(v, (Sum, Records)) else Sum(v)) for k, v in outputs.items()}
    rspec = {k: v for k, v in outs_spec.items() if isinstance(v, Records)}
    if any(RSEP in k for k in outs_spec):
        raise ValueError("tallymap: an output name must not contain the character \\x1f")
    names = sorted(k for k in outs_spec if k not in rspec)      # the Sum outputs
    rnames = sorted(rspec)                                      # the Records outputs
    where = item_name(fn)
    decl = _call_site()          # where tm.vmap was written: the fallback when the call itself is all JAX frames

    def mapped(*args, n_active=None):
        axes = in_axes if isinstance(in_axes, (tuple, list)) else (in_axes,) * len(args)
        if len(axes) != len(args) or any(a not in (0, None) for a in axes):
            raise ValueError("tallymap: in_axes must be 0/None or a tuple of 0/None per positional argument")
        margs = tuple(a for a, x in zip(args, axes) if x == 0)
        bargs = tuple(a for a, x in zip(args, axes) if x is None)
        pi, ptree = jax.tree_util.tree_flatten(margs)
        ctree = jax.tree_util.tree_structure(margs[0] if len(margs) == 1 else margs)    # a spawned child
        bc, btree = jax.tree_util.tree_flatten(bargs)
        pi = [jnp.asarray(x) for x in pi]
        bc = [jnp.asarray(x) for x in bc]
        nmp, nbu = len(pi), len(bc)            # user leaves
        if n_active is not None:               # fixed (bucketed) capacity N, traced active count: no retrace.  The
            if spawn is not None:              # count is a broadcast operand the item does not read (ItemProgram.nact)
                raise ValueError("tallymap: n_active with spawn is not supported")
            bc = bc + [jnp.asarray(n_active, jnp.int32)]
        if not pi:
            raise ValueError("tallymap: at least one argument must be mapped (in_axes=0)")
        Ns = {int(x.shape[0]) if x.ndim else -1 for x in pi}
        if len(Ns) != 1 or -1 in Ns:
            raise ValueError(f"tallymap: mapped arguments must share one leading axis length, got {sorted(Ns)}")
        N = Ns.pop()
        if N > MAX_ITEMS:
            raise ValueError(f"tallymap: N={N} items exceeds 2^31-1; split the call (and fold the chunk index into "
                             "your keys)")
        shapes = {n: (outs_spec[n].shape,) if isinstance(outs_spec[n].shape, int) else tuple(outs_spec[n].shape)
                  for n in names}
        # a fixed_point float32 Sum is the exact sum (TSpec.exact); fixed point at `scale` for the other dtypes
        ex = {n: bool(outs_spec[n].fixed_point) and np.dtype(outs_spec[n].dtype) == np.float32 for n in names}
        sspecs = [TSpec(n, int(np.prod(shapes[n], dtype=np.int64)), np.dtype(outs_spec[n].dtype), (),
                        bool(outs_spec[n].fixed_point) and not ex[n], "sum", float(outs_spec[n].scale), exact=ex[n])
                  for n in names]
        pav = [jax.ShapeDtypeStruct(x.shape[1:], x.dtype) for x in pi]
        bav = [_sds(x) for x in bc]
        nb0 = len(bc)
        meta = {}

        def build(tspecs, tshapes, mode=None, caps=None, M=0):
            """trace the user's item into an ItemProgram.  mode None: the plain program.  With Records outputs:
            'count': the tallies are the Sums, discarded (and the Queue); per-item results = each Records output's
            count.  'fill': one hidden int32 per-item operand, the item's canonical node id `cid` (a root: its index;
            a spawned child: its parent's first child + its spawn ordinal, carried by the spawn), and hidden
            broadcast tables after the user's ([first child per node, with spawn] + [offset, count] per Records
            output, indexed by cid); the tallies are the Sums, the field tallies and the mismatch tally; per-item
            results = the user's, then [truncated], then the item's POISON flag (ItemProgram.trunc: its fill count
            differs from its count, or it was truncated), so every derived kernel poisons itself when its own replay
            disagrees with the count."""
            nT = len(tspecs)
            tn = [s.name for s in tspecs if s.kind != "spawn"]
            fill = mode == "fill"
            nh = int(mode in ("fill", "one"))
            nbx = (int(spawn is not None) + 2 * len(rnames)) if fill else 0
            rinfo = (mode, tuple((r, (caps[r], rspec[r].fields)) for r in rnames)) if mode else None
            ys0 = int(spawn is not None) if fill else 0      # first offset/count table

            def item(*flat):
                ta = {n: flat[i] for i, n in enumerate(tn)}
                rec, cnts = None, []
                if mode == "count":
                    rec = {r: (jnp.int32(0),) for r in rnames}
                elif mode == "one":                 # (one pass: the record key's cid, the item index)
                    rec = {r: (jnp.int32(0), flat[nT + nmp]) for r in rnames}
                elif fill:
                    cid = flat[nT + nmp]
                    tabs = flat[nT + nmp + nh + nb0:nT + nmp + nh + nb0 + nbx]
                    cnts = [tabs[ys0 + 2 * j + 1][cid] for j in range(len(rnames))]
                    rec = {r: (jnp.int32(0), tabs[ys0 + 2 * j][cid], cnts[j]) for j, r in enumerate(rnames)}
                    if spawn is not None:
                        rec[RSEP] = (tabs[0][cid], jnp.int32(0))
                t = Tally(ta, tshapes, flat[nT - 1] if spawn is not None else None,
                          ctree if spawn is not None else None, rec, rinfo)
                m = jax.tree_util.tree_unflatten(ptree, list(flat[nT:nT + nmp]))
                b = jax.tree_util.tree_unflatten(btree, list(flat[nT + nmp + nh:nT + nmp + nh + nbu]))
                mi, bi = iter(m), iter(b)
                uargs = [next(mi) if x == 0 else next(bi) for x in axes]
                r = fn(t, *uargs)
                if isinstance(r, Tally):
                    t2, y = r, None
                elif isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], Tally):
                    t2, y = r
                else:
                    raise TypeError("tallymap: the item function must return its Tally or (Tally, per_item_result)")
                ks = []
                for rn in rnames:
                    if rn not in t2._r or jnp.ndim(t2._r[rn][0]) != 0:
                        raise TypeError(LANED)
                    ks.append(t2._r[rn][0])
                a2 = dict(t2._a)
                extra = []
                if fill:                            # the item's count mismatch: counted per output, and its flag
                    mis = jnp.stack([k != c for k, c in zip(ks, cnts)])
                    a2[MISMATCH] = emit_p.bind(a2[MISMATCH], jnp.arange(len(rnames), dtype=jnp.int32),
                                               mis.astype(jnp.int32))
                    extra = [jnp.any(mis)]
                tl = [a2[n] for n in tn] + ([t2._q] if spawn is not None else [])
                yl, meta["ytree"] = jax.tree_util.tree_flatten(y)
                if mode == "count":                 # (the user's per-item results are not computed)
                    return tl + ks
                if mode == "one":                   # (the counts: the per-item results after the user's)
                    return tl + [jnp.asarray(v) for v in yl] + ks
                return tl + [jnp.asarray(v) for v in yl] + extra
            tav = [_tally_aval(s) for s in tspecs]
            hav = [jax.ShapeDtypeStruct((), jnp.int32)] * nh
            xav = [jax.ShapeDtypeStruct((M,), jnp.int32)] * nbx
            cj = _mk(item, tav + pav + hav + bav + xav)
            check_tallies(cj, [s.kind for s in tspecs], name=f"item function {where}")
            # hoist closed-over tracers (outer jit/grad/jvp/vmap values) as explicit broadcast operands
            hidx = [k for k, c in enumerate(cj.consts) if isinstance(c, Tracer)]
            ckey = None
            if not hidx:                    # (a program closing over a live trace's values is never reused)
                ckey = _prog_key(cj, tspecs, mode, caps, M, max_steps, spawn, (nmp, nh, nb0, nbx),
                                 n_active is not None, where)
                hit = _PROGS.get(ckey)
                if hit is not None and _same_objects(hit[0], cj):
                    _PROGS.move_to_end(ckey)
                    _diagnostics.record_program(hit[1])
                    return hit[1], []
            hoisted = [cj.consts[k] for k in hidx]
            npi, nbt = nmp + nh, nb0 + nbx
            cap1 = int(max_steps is not None)

            def item2(*a):
                consts = list(cj.consts)
                for k, x in zip(hidx, a[nT + npi + nbt:]):
                    consts[k] = x
                outs = list(retrace(ClosedJaxpr(cj.jaxpr, consts), cap=max_steps, n_tally=nT)(*a[:nT + npi + nbt]))
                if fill:                            # [.., mismatch, (truncated)] -> [.., (truncated), poison flag]
                    mis = outs.pop(len(outs) - 1 - cap1)
                    outs.append(mis | outs[-1] if cap1 else mis)
                return outs
            cj2 = _mk(item2, tav + pav + hav + bav + xav + [_sds(h) for h in hoisted])
            if spawn is not None:
                _check_spawn(cj2, pav + hav, where)
            nY = len(cj2.jaxpr.outvars) - nT
            trunc = nY - 1 if (fill or (mode in (None, "one") and max_steps is not None)) else None
            app = None
            if mode == "one":
                nyc = nY - cap1 - len(rnames)
                app = (nmp, tuple((r, caps[r], nyc + j) for j, r in enumerate(rnames)))
            prog = ItemProgram(cj2, tuple(tspecs), npi, nbt + len(hoisted), trunc, spawn, where=where,
                               nact=nb0 - 1 if n_active is not None else None, app=app)
            if ckey is not None:
                _PROGS[ckey] = (cj, prog)
                while len(_PROGS) > PROG_CACHE:
                    _PROGS.popitem(last=False)
            _diagnostics.record_program(prog)
            return prog, hoisted

        from ._primitive import bind_call, make_node  # (here: _primitive imports this module)
        site = _call_site(decl)

        def call(prog, pi_, bc_):
            return bind_call(*pi_, *bc_, node=make_node(prog, backend, opts, site))
        if rnames:
            return _records(build, call, sspecs, shapes, names, rnames, rspec, pi, bc, max_steps, meta, N, spawn)
        tspecs = list(sspecs)
        if spawn is not None:
            tspecs.append(TSpec("__spawn__", 1, np.dtype(np.int32), (), False, "spawn"))
        prog, hoisted = build(tspecs, shapes)
        flat = call(prog, pi, bc + hoisted)
        nt = len(prog.out_tallies)
        tvals, st, ys = flat[:nt], flat[nt], flat[nt + 1:]
        res_out = {n: tvals[i].reshape(shapes[n]) for i, n in enumerate(names)}
        stats = {n: st[ST[n]] for n in ("fixed_point_overflow", "dropped_emits")}
        if spawn is not None:
            stats.update((n, st[ST[n]]) for n in ("spawned", "children_dropped", "children_truncated"))
        if max_steps is not None:
            trunc = ys[-1]
            ys = ys[:-1]
        else:
            trunc = jnp.zeros((N,), bool)
        stats["n_truncated"] = jnp.sum(trunc, dtype=jnp.int32)
        # (a truncated call: the derived kernels poison their own tangents / cotangents, so the primal's poison
        # has an identity tangent; the tree adjoint (spawn) keeps the poison that differentiates)
        res_out = (_poison if spawn is not None else _poison_primal)(res_out, _limits(stats, st, spawn, max_steps))
        per_item = jax.tree_util.tree_unflatten(meta["ytree"], ys)
        return Result(res_out, per_item, trunc, stats)
    mapped.__wrapped__ = fn
    return mapped


# ------------------------------------------------------------------------------------------- Records
_COUNTS_HOOK = None     # test-only: f(name, counts) -> counts handed to a fill (count-then-fill's fill pass; one pass:
                        # the fill twin of a derived launch) -- the mismatch negative control
MISMATCH = rkey("", "mismatch")     # the fill's hidden int32 tally: items whose fill count != count, per output
ONE_PASS = True         # test-only: False runs every Records call as count-then-fill (step 1's implementation)


def _sg(x):
    return lax.stop_gradient(x) if isfloat(x.dtype) else x


def _records_one(build, call, sspecs, shapes, names, rnames, rspec, pi, bc, max_steps, meta, N):
    """a call with Records outputs in ONE pass (see Records): one launch of the one-pass program, canonicalised by
    the primitive itself (_records.canon), so its outputs, and every derivative's, are in the count-then-fill
    layout."""
    R = len(rnames)
    caps = {r: rspec[r].capacity for r in rnames}
    rspecs = [TSpec(rkey(r, f), caps[r], dt) for r in rnames for f, dt in rspec[r].fields]
    kspecs = [TSpec(rkey(r, x), caps[r], np.dtype(np.int32)) for r in rnames for x in (KI, KK)]
    fspecs = list(sspecs) + rspecs + kspecs
    fshapes = dict(shapes, **{s.name: (s.S,) for s in rspecs + kspecs})
    prog, h = build(fspecs, fshapes, "one", caps)
    flat = call(prog, list(pi) + [jnp.arange(N, dtype=jnp.int32)], list(bc) + h)
    nt = len(prog.out_tallies)
    tvals, st, ys = flat[:nt], flat[nt], list(flat[nt + 1:])
    trunc = ys.pop() if max_steps is not None else jnp.zeros((N,), bool)
    counts = ys[len(ys) - R:]
    ys = ys[:len(ys) - R]
    tv = {s.name: v for s, v in zip([fspecs[k] for k in prog.out_tallies], tvals)}
    stats = {n: st[ST[n]] for n in ("fixed_point_overflow", "dropped_emits")}
    stats["n_truncated"] = jnp.sum(trunc, dtype=jnp.int32)
    totals = {r: jnp.sum(c, dtype=jnp.int32) for r, c in zip(rnames, counts)}
    stats["appended"] = totals
    stats["records_dropped"] = {r: jnp.maximum(totals[r] - caps[r], 0) for r in rnames}
    stats["records_mismatch"] = {r: jnp.int32(0) for r in rnames}
    lim = stats["n_truncated"] > 0 if max_steps is not None else None
    ovf = jnp.bool_(False)              # (a multiply that carries the poison into derivatives)
    for r in rnames:
        ovf = ovf | (totals[r] > caps[r])
    out = {n: tv[n].reshape(shapes[n]) for n in names}
    out.update((s.name, tv[s.name]) for s in rspecs)
    out = _poison(_poison_primal(out, lim), ovf)
    bad = ovf if lim is None else lim | ovf
    res = {n: out[n] for n in names}
    for r in rnames:
        res[r] = {f: out[rkey(r, f)] if isfloat(dt) else jnp.where(bad, jnp.asarray(-1, dt), out[rkey(r, f)])
                  for f, dt in rspec[r].fields}
    per_item = jax.tree_util.tree_unflatten(meta["ytree"], ys)
    return Result(res, per_item, trunc, stats)


def _count_recorder(p, R):
    """the count program of a spawning call, run as the tree adjoint's recording primal (_reverse_spawn.rec_primal:
    each item's (me, par, ord, gen), the queue ring returned), plus one int32 tally of the Queue's capacity per
    Records output into which a CHILD emits its count at its queue slot `me` (a child's per-item results are
    discarded).  Tallies: [the discarded Sums] + [the counts] + [the queue]."""
    from ._reverse_spawn import rec_primal  # (here: _reverse_spawn imports this module)
    q = rec_primal(p)
    Q, ns = int(q.spawn.capacity), q.nT - 1
    f = jaxpr_as_fun(q.cj)
    cspec = [TSpec(f"{RSEP}count{j}", Q, np.dtype(np.int32), slots=True) for j in range(R)]

    def item(*a):
        sums, cnts, sp, rest = a[:ns], a[ns:ns + R], a[ns + R], a[ns + R + 1:]
        outs = f(*sums, sp, *rest)
        me = rest[q.rec]
        cnts = [emit_p.bind(c, jnp.where(me >= 0, me, -1), k) for c, k in zip(cnts, outs[q.nT:q.nT + R])]
        return list(outs[:ns]) + cnts + [outs[ns]] + list(outs[q.nT:])
    tl = tuple(q.tallies[:ns]) + tuple(cspec) + (q.tallies[ns],)
    cj = _mk(item, [_tally_aval(s) for s in tl] + [_sds(a) for a in q.in_avals()[q.nT:]])
    check_tallies(cj, [s.kind for s in tl], name=f"records count recorder ({q.where})")
    return dataclasses.replace(q, cj=cj, tallies=tl, tag=q.tag + "+count")


def _tree_order(N, Q, par, ordn, gen, spawned):
    """canonical node ids of a spawn tree from the recording primal's ring: roots 0..N-1 (item order), then each
    generation in turn, its children ordered by (parent id, spawn ordinal) -- so a parent's children have consecutive
    ids, first[parent] + ordinal.  Deterministic: it reads only the tree, never the queue slots' order.
    -> (cid per queue slot (-1: unused), first child id per node [N + Q])."""
    M = N + Q
    slot = jnp.arange(Q, dtype=jnp.int32)
    valid = slot < spawned
    pcid = lambda cs: jnp.where(par >= 0, cs[jnp.clip(par, 0, Q - 1)], -1 - par)

    def body(c):
        g, base, cs, _ = c
        m = valid & (gen == g)
        pc = jnp.where(m, pcid(cs), M)
        i1 = jnp.argsort(jnp.where(m, ordn, 0), stable=True)
        i2 = i1[jnp.argsort(pc[i1], stable=True)]
        rank = jnp.zeros((Q,), jnp.int32).at[i2].set(slot)
        n = jnp.sum(m, dtype=jnp.int32)                 # (int32 under x64 too: jnp.sum widens int32 to int64)
        return g + 1, base + n, jnp.where(m, base + rank, cs), n
    c = lax.while_loop(lambda c: c[3] > 0, body,
                       (jnp.int32(1), jnp.int32(N), jnp.full((Q,), -1, jnp.int32), jnp.int32(1)))
    cs = c[2]
    first = jnp.full((M,), M, jnp.int32).at[jnp.where(valid, pcid(cs), M)].min(jnp.where(valid, cs, M), mode="drop")
    return cs, jnp.where(first == M, 0, first)


def _records(build, call, sspecs, shapes, names, rnames, rspec, pi, bc, max_steps, meta, N, spawn):
    """a call with Records outputs: count pass, canonical offsets, fill pass (see Records).  Both passes are
    ordinary tallymap programs, so every transformation applies to the fill unchanged; the count pass is not
    differentiated (its operands are stop_gradient'ed: counts are integers)."""
    if ONE_PASS and spawn is None and all(rspec[r].capacity is not None for r in rnames):
        return _records_one(build, call, sspecs, shapes, names, rnames, rspec, pi, bc, max_steps, meta, N)
    R = len(rnames)
    qspec = [TSpec("__spawn__", 1, np.dtype(np.int32), (), False, "spawn")] if spawn is not None else []
    # 1. count: every Sum discarded, the user's per-item results dropped; per-item results = the counts
    cprog, ch = build([dataclasses.replace(s, kind="discard") for s in sspecs] + qspec, shapes, "count",
                      {r: 0 for r in rnames})
    psg, bsg = [_sg(x) for x in pi], [_sg(x) for x in bc + ch]
    if spawn is None:
        M, tabs0 = N, []
        cflat = call(cprog, psg, bsg)
        cys = cflat[len(cprog.out_tallies) + 1:]
        counts = {r: cys[j] for j, r in enumerate(rnames)}
    else:                       # node ids in the canonical tree order; a child's count from its queue slot
        Q = int(spawn.capacity)
        M = N + Q
        rp = _count_recorder(cprog, R)
        cflat = call(rp, psg + [jnp.zeros((N,), jnp.int32)] * 4, bsg)
        nt = len(rp.out_tallies)
        ring = cflat[nt + 1 + rp.nY:]
        cs, first = _tree_order(N, Q, ring[rp.rec + 1], ring[rp.rec + 2], ring[rp.rec + 3],
                                cflat[nt][ST["spawned"]])
        dest = jnp.where(cs >= 0, cs, M)
        counts = {r: jnp.zeros((M,), jnp.int32).at[:N].set(cflat[nt + 1 + j]).at[dest].set(cflat[j], mode="drop")
                  for j, r in enumerate(rnames)}
        tabs0 = [first]
    if _COUNTS_HOOK is not None:
        counts = {r: _COUNTS_HOOK(r, c) for r, c in counts.items()}
    totals = {r: jnp.sum(c, dtype=jnp.int32) for r, c in counts.items()}
    caps = {}
    for r in rnames:
        C = rspec[r].capacity
        if C is None:
            try:
                tot = int(totals[r])
            except (jax.errors.ConcretizationTypeError, jax.errors.TracerIntegerConversionError):
                raise ValueError(f"tallymap: Records output {r!r} has no capacity, and this call is traced (jax.jit, "
                                 "a jax.vmap of the call, ...), so its number of records is not known when the "
                                 "outputs' shape is fixed: pass tallymap.Records(fields, capacity=...)") from None
            C = 1 << max(0, (tot - 1).bit_length())
        caps[r] = C
    offs = {r: (jnp.cumsum(c, dtype=jnp.int32) - c).astype(jnp.int32) for r, c in counts.items()}
    # 2. fill: the Sums, one Sum of the capacity per Records field (written once per slot: 0 + v = v), the mismatch
    # counter; hidden operands: the node id (roots: the item index), [first child,] offset and count tables
    rspecs = [TSpec(rkey(r, f), caps[r], dt) for r in rnames for f, dt in rspec[r].fields]
    fspecs = list(sspecs) + rspecs + [TSpec(MISMATCH, R, np.dtype(np.int32))] + qspec
    fshapes = dict(shapes, **{s.name: (s.S,) for s in rspecs}, **{MISMATCH: (R,)})
    fprog, fh = build(fspecs, fshapes, "fill", caps, M)
    tabs = tabs0 + [x for r in rnames for x in (offs[r], counts[r])]
    flat = call(fprog, list(pi) + [jnp.arange(N, dtype=jnp.int32)], list(bc) + tabs + fh)
    nt = len(fprog.out_tallies)
    tvals, st, ys = flat[:nt], flat[nt], list(flat[nt + 1:])
    ys.pop()                                             # the poison flag (the derived kernels read it)
    trunc = ys.pop() if max_steps is not None else jnp.zeros((N,), bool)
    tv = {s.name: v for s, v in zip([fspecs[k] for k in fprog.out_tallies], tvals)}
    stats = {n: st[ST[n]] for n in ("fixed_point_overflow", "dropped_emits")}
    if spawn is not None:
        stats.update((n, st[ST[n]]) for n in ("spawned", "children_dropped", "children_truncated"))
    stats["n_truncated"] = jnp.sum(trunc, dtype=jnp.int32)
    stats["appended"] = totals
    stats["records_dropped"] = {r: jnp.maximum(totals[r] - caps[r], 0) for r in rnames}
    stats["records_mismatch"] = {r: tv[MISMATCH][j] for j, r in enumerate(rnames)}
    lim = jnp.any(tv[MISMATCH] > 0)     # poisoned by the derived kernels themselves too (ItemProgram.trunc)
    if max_steps is not None:
        lim = lim | (stats["n_truncated"] > 0)
    ovf = jnp.bool_(False)              # from the count pass: a multiply that carries the poison into derivatives
    for r in rnames:
        ovf = ovf | (totals[r] > caps[r])
    if spawn is not None:               # (as a spawning call without Records: the poison that differentiates)
        ovf = ovf | lim | (stats["children_dropped"] > 0)
        if max_steps is not None:
            ovf = ovf | (stats["children_truncated"] > 0)
    out = {n: tv[n].reshape(shapes[n]) for n in names}
    out.update((s.name, tv[s.name]) for s in rspecs)
    out = _poison(_poison_primal(out, lim), ovf)
    bad = lim | ovf
    res = {n: out[n] for n in names}
    for r in rnames:
        res[r] = {f: out[rkey(r, f)] if isfloat(dt) else jnp.where(bad, jnp.asarray(-1, dt), out[rkey(r, f)])
                  for f, dt in rspec[r].fields}
    per_item = jax.tree_util.tree_unflatten(meta["ytree"], ys)
    return Result(res, per_item, trunc, stats)


def _limits(stats, st, spawn, max_steps):
    """-> a traced bool: did a tallymap RESOURCE LIMIT intervene in this call?  Either the spawn Queue was too small
    for a child, or an item (or a spawned child) hit max_steps."""
    flags = []
    if spawn is not None:
        flags.append(st[ST["children_dropped"]] > 0)       # a child that did not fit the Queue
    if max_steps is not None:
        flags.append(stats["n_truncated"] > 0)             # an item hit max_steps
        if spawn is not None:
            flags.append(st[ST["children_truncated"]] > 0)  # a spawned child hit max_steps
    if not flags:
        return None
    bad = flags[0]
    for f in flags[1:]:
        bad = bad | f
    return bad


def _poison(outs, bad):
    """Every float output becomes NaN when a resource limit intervened: a truncated or dropped item is never
    silently missing from a sum.  The counters in Result.stats stay the signal; integer tallies cannot be NaN.
    The primal is a select, so a call that hit no limit is bitwise unchanged (a multiply by 1.0 is not: XLA:CPU
    flushes a subnormal product to zero); the tangent is unchanged too, or times NaN when poisoned, which carries the
    poison into tangents and cotangents (a NaN gradient, never a wrong one)."""
    if bad is None:
        return outs
    poison = jnp.where(bad, jnp.float32(np.nan), jnp.float32(1.0))
    return {n: _psel(v, poison.astype(v.dtype)) if isfloat(v.dtype) else v for n, v in outs.items()}


def _sel(x, f):
    """x where the poison f is 1, NaN where it is NaN: bitwise x (subnormals included) or NaN."""
    return jnp.where(jnp.isnan(f), f, x)


@jax.custom_jvp
def _psel(x, f):
    return _sel(x, f)


# the tangent: tangent * NaN where poisoned, else the tangent itself (bitwise; a select of two linear functions of the
# tangent, so it transposes to the cotangent)
_psel.defjvp(lambda pr, tg: (_psel(*pr), jnp.where(jnp.isnan(pr[1]), tg[0] * pr[1], tg[0])))


@jax.custom_jvp
def _pmul(x, f):
    return _sel(x, f)


_pmul.defjvp(lambda pr, tg: (_pmul(*pr), tg[0]))


def _poison_primal(outs, bad):
    """_poison whose tangent is the identity: the tangent / adjoint kernels poison themselves (jvp_prog's
    _poison_tally; _primitive._vjp_call with the adjoint's own truncation flags), so neither jax.jvp's tangent nor
    jax.grad's cotangent reads this call's primal outputs, and a gradient is ONE launch (lumen T6b)."""
    if bad is None:
        return outs
    poison = jnp.where(bad, jnp.float32(np.nan), jnp.float32(1.0))
    return {n: _pmul(v, poison.astype(v.dtype)) if isfloat(v.dtype) else v for n, v in outs.items()}


def _check_spawn(cj, pav, where=""):
    want = [(tuple(a.shape), a.dtype) for a in pav]     # not np.dtype: a typed PRNG key

    def walk(j):
        for e in j.eqns:
            if e.primitive.name == "tallymap_spawn":
                got = [(tuple(v.aval.shape), v.aval.dtype) for v in e.invars[2:]]
                if got != want:
                    raise located(TypeError(f"tallymap: spawned child leaves {got} do not match the mapped inputs "
                                            f"{want}"), e, f"item function {where}")
            for s in subjaxprs(e):
                walk(s.jaxpr)
    walk(cj.jaxpr)
