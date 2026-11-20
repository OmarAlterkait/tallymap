"""Device memory of a call's outputs: tallymap.output_memory (slope + fixed), call_memory, cell_bytes, max_cells.

A caller that sizes its outputs from the free device memory (lumen budgets its readout blocks this way) needs the bytes
one cell of an output holds while a call runs, including the accumulator, the copies XLA keeps in its loops and the
readout's temporaries.  The numbers are a model of the implementation: the structural part (accumulator words, value
counts, flags, the output, the tangent tallies of a jvp, a cotangent) is computed from the same constants the backends
use (kernel.DENSE_WORDS, xla._xfmt, xla._rows, xla.READ_CHUNK); what XLA's buffer assignment adds on top is a
measured constant per platform (job ids below; round9/cellbytes NOTES).  Every number is an upper bound of the
measured peak device memory of a call (tests/cellbytes.py checks it, GPU and CPU, forward / jvp / grad / vjp).
"""


import math
from typing import NamedTuple

import jax
import numpy as np

MAX_CELLS = (1 << 31) - 1            # every backend indexes an output's cells with int32 (tally indices, emits)
DERIVATIVES = ("forward", "jvp", "vjp", "grad")

# XLA's additions to the structural bytes of a digit-word tally (xla backend; exact float32 and, without x64,
# fixed point): (extra bytes per cell, readout bytes per cell of a chunk of READ_CHUNK cells, readout bytes per cell
# of a smaller output, read in one piece).  Measured on tallymap a15bd4b / 5d7ebe8, jax 0.6.2 and 0.12: GPU (RTX 2080
# Ti) 46124514 and 46130141, CPU (milano) 46124515 and 46130142 (runtime peak; on the GPU also the compiled memory
# analysis): the slope between 2^22 and 2^24 cells, the values at 2^23 and 2^24, and 2^19 cells.
_XWORD = {("gpu", False, "exact"): (10, 154, 154), ("gpu", False, "fixed"): (6, 70, 70),
          ("gpu", True, "exact"): (20, 48, 140), ("cpu", False, "exact"): (12, 150, 150),
          ("cpu", False, "fixed"): (12, 88, 88), ("cpu", True, "exact"): (64, 165, 165)}
# a GPU call with an int64-pair accumulator (the kernel backend's fixed point; the xla backend's x64 fixed point) held
# 16 B per cell more at run time than its compiled analysis, except under jvp (46130141: 34-36 against 20); the xla
# backend's int32-word fixed point too, under grad (46130818: 64 against 48)
_GPU_PAIR = 16
# a launch's own fixed device memory, whatever its cell count (stats, per-launch buffers, the item loop's XLA
# temporaries, the GPU allocator's rounding): the largest excess of measured peak memory over the slope, 2^20 to 2^26
# cells on the GPU, 2^20 to 2^28 on the CPU (fit 46142828 / 46142829; lumen SK and SBND 46142910-14)
_LAUNCH = {("auto", "gpu"): 96 << 20, ("auto", "cpu"): 16 << 20, ("xla", "gpu"): 16 << 20, ("xla", "cpu"): 16 << 20}


def _platform(platform):
    p = platform or jax.default_backend()
    if p not in ("cpu", "gpu", "cuda", "rocm", "tpu"):
        raise ValueError(f"tallymap: platform must be 'gpu' or 'cpu' (or None: JAX's default), got {platform!r}")
    return "cpu" if p == "cpu" else "gpu"


def _tallies(output):
    """the tallies one output keeps: [(dtype, kind)]; kind "exact" | "fixed" | "sum"."""
    from ._api import Records, Sum
    if isinstance(output, Records):
        fl = [(np.dtype(dt), "sum") for _, dt in output.fields]
        return fl + [(np.dtype(np.int32), "sum")] * 2          # (the one-pass keys: item, ordinal)
    if not isinstance(output, Sum):
        output = Sum(output)
    dt = np.dtype(output.dtype)
    kind = ("exact" if dt == np.float32 else "fixed") if output.fixed_point else "sum"
    return [(dt, kind)]


def _size(output):
    from ._api import Records, Sum
    if isinstance(output, Records):
        return output.capacity
    shape = (output.shape if isinstance(output, Sum) else output)
    shape = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)
    return int(np.prod(shape, dtype=np.int64))


def _isfloat(dt):
    return np.issubdtype(dt, np.inexact)


def _kernel(dt, kind):
    """(bytes per cell held during a launch, its output's bytes): the kernel backend allocates the scratch and the
    output of every tally for the launch (kernel._scratch_words, Layout.r_out)."""
    from ._api import DENSE_WORDS
    out = dt.itemsize
    if kind == "exact":
        return 8 * DENSE_WORDS + 1 / 8 + out, out          # int64 words + one flag bit per cell
    if kind == "fixed":
        return 16 + out, out                                  # u64 low word, i32 carry word, u32 flag
    return out, out                                           # (accumulates in its output)


def _xla(dt, kind, S, plat, x64):
    """(bytes per cell, fixed bytes of the readout (one tally's at a time), fixed bytes held with the tally, output
    bytes per cell) of one tally on the xla backend with S cells."""
    from .backends import xla as X
    out = dt.itemsize
    if kind == "exact" or (kind == "fixed" and not x64):
        W = X.EXACT_WORDS32 if kind == "exact" and not x64 else X.FIXED_WORDS32 if kind == "fixed" else X.EXACT_WORDS
        wb = 8 if x64 and kind == "exact" else 4
        nadd = 4 if not (x64 and kind == "exact") else 0       # per-bin value counts (int32 words, _St.nadd)
        e, rc, r1 = _XWORD[(plat, x64, kind)]
        read = min(S, X.READ_CHUNK) * max(rc, r1)              # (non-decreasing in S)
        return W * wb + 1 + max(nadd, out + 1) + e, read, 0, out     # (+ 1: the flags, _St.bad)
    if kind == "fixed":                                       # x64: the int64 low / high pair (_wide)
        return 16 + 1 + out + 1, 0, 0, out
    if not _isfloat(dt):
        return out, 0, 0, out                                 # (one row, read out as it is)
    from types import SimpleNamespace
    spec = SimpleNamespace(kind="sum", fixed_point=False, exact=False, dtype=dt, lead=(), S=S)
    R, low = X._rows(spec), int(dt == np.float32)             # partial rows (R S <= ROWS_ELEMS when R > 1), LOW row
    if R + low == 1:
        return out, 0, 0, out
    # the rows and their sum (the output): measured up to twice the rows held (46124514, 46124515, 46130141/2: XLA
    # keeps the partial rows and the summed rows; the GPU also at run time beyond its compiled analysis).  The
    # partial rows past the first are fewer than ROWS_ELEMS elements: a fixed term (total non-decreasing in S)
    return 2 * (1 + low) * out + out, 0, 2 * out * max(X.ROWS_ELEMS - S, 0), out     # ((R - 1) S <= ROWS_ELEMS - S)


def _model(output, backend, derivative, S, plat, x64):
    """(bytes per cell, readout bytes (one tally at a time), fixed bytes held) of one output with S cells."""
    from ._api import Records
    rec = isinstance(output, Records)
    per, read, held, outs, cts, words, pair = 0.0, 0, 0, 0, 0, False, False
    for dt, kind in _tallies(output):
        if backend == "xla":
            b, r, h, o = _xla(dt, kind, S, plat, x64)
            words = words or r > 0
            pair = pair or (kind == "fixed" and (x64 or derivative == "grad"))   # (fixed32: under grad, 46130818)
        else:
            (b, o), r, h = _kernel(dt, kind), 0, 0
            pair = pair or kind == "fixed"
        n = 2 if derivative == "jvp" and _isfloat(dt) else 1      # (a jvp: the tangent tally of a float output)
        per += n * b
        read = max(read, r)                                   # (the readouts run one tally at a time)
        held += n * h
        outs += n * o
        cts += o if _isfloat(dt) else 0
    if derivative == "jvp" and words:
        per += outs                                           # (xla digit words: both outputs held at the readouts)
    if derivative == "vjp":
        per += cts                                            # (the cotangent, an input, is live during the forward)
    if derivative == "grad" and not rec:                      # the reverse launch holds the output and its cotangent
        per = (max(per, (2 if plat == "gpu" else 1) * (outs + cts)) if backend == "auto" else
               per + (cts if not words else 0))
    if pair and plat == "gpu" and derivative != "jvp":
        per += _GPU_PAIR
    if rec:
        # _records.canon: the raw fields and keys, their canonical copies and the positions (int32).  Measured up to
        # 2x the raw tallies on the kernel backend (jvp, GPU) and 1.6x on the xla backend (jax 0.6.2, GPU; jax 0.12
        # holds less): 46130141/2.  Reverse mode reads the cotangents in place (0103ac3): grad as the forward, vjp + the
        # cotangents (46142828/9)
        k = 2 if backend == "auto" else 1.6
        per, held = per * k + 4, held * k
    return per, read, held


def _total(output, backend, derivative, S, plat, x64):
    per, read, held = _model(output, backend, derivative, S, plat, x64)
    return per * S + read + held + _LAUNCH[(backend, plat)]


class OutputMemory(NamedTuple):
    """Device memory of one output in a call: at most per_cell * cells + fixed bytes (tallymap.output_memory)."""
    per_cell: float      # bytes per cell
    fixed: int           # bytes whatever the cell count (up to the given one): readout chunk, partial rows, launch

    def total(self, cells):
        return self.per_cell * cells + self.fixed


def _args(output, backend, derivative, cells, platform, x64, who):
    plat = _platform(platform)
    x64 = bool(jax.config.jax_enable_x64) if x64 is None else bool(x64)
    d = _derivative(derivative)
    if backend not in ("auto", "xla"):
        raise ValueError(f"tallymap.{who}: backend must be 'auto' or 'xla', got {backend!r}")
    S = _size(output) if cells is None else int(cells)
    if S is None:
        raise ValueError(f"tallymap.{who}: a Records output with capacity=None needs cells=")
    if not 1 <= S <= MAX_CELLS:
        raise ValueError(f"tallymap.{who}: cells must be in [1, 2^31 - 1], got {S}")
    return d, S, plat, x64


def output_memory(output, backend="auto", derivative=None, *, cells=None, platform=None, x64=None):
    """Device memory one output of a tallymap.vmap call holds while the call runs, as a slope and a fixed term:
    OutputMemory(per_cell, fixed), the call's peak memory for this output at most per_cell * cells + fixed for every
    cell count up to `cells`.  It covers the output's tallies (and a jvp's tangent tallies, a vjp / grad's output
    cotangent), their accumulators and flags, the copies XLA keeps in its loops, the readout's temporaries, a Records
    output's fields, one-pass keys and canonical copies, and the launch's own fixed buffers.  Upper bounds of measured
    peak memory (tests/cellbytes.py; GPU and CPU, jax 0.6.2 and 0.12, to 2^28 cells).

        m = tm.output_memory(tm.Sum((n,), jnp.float32, fixed_point=True), "auto", "jvp")
        m.per_cell, m.fixed, m.total(n)

    output      a tallymap.Sum (exact: fixed_point=True float32; fixed point; plain; any dtype), a tallymap.Records
                (per slot), or a shape (a plain float32 Sum).
    backend     'auto' (the kernel backend: CUDA GPU or CPU) | 'xla'.
    derivative  None / 'forward' | 'jvp' (one tangent direction; jax.jacfwd / jax.vmap of a jvp over T directions
                holds T tangent tallies) | 'vjp' (the output, and its cotangent as an input) | 'grad' (a loss that reads
                the output: the forward launch runs; a loss LINEAR in the output needs no forward launch and holds
                about its cotangent only).
    cells       the largest cell count the bound must hold for (default: the output's own size).
    platform    'gpu' | 'cpu' (default: JAX's default backend); the kernel backend's slope is the same on both.
    x64         jax_enable_x64 (default: the current setting).

    Per output: a call's outputs add up (an upper bound; their readouts do not overlap, and the launch's fixed term
    is counted once per output).  NOT included: the items' own memory -- their inputs, per-item results and
    per-item state (on the xla backend xla_batch_size items at a time; on the kernel backend, CUDA's per-thread local
    memory, which the driver reserves outside JAX's pool) -- and the caller's own arrays and arithmetic on the
    outputs.  Measured slopes, B per cell (GPU / CPU), exact float32 Sum, x64 off: kernel 44.125 forward, 88.25 jvp,
    48.125 vjp, 44.125 grad (both); xla 87-91 / 94 forward, 173 / 196 jvp (fits to 2^26 / 2^28 cells)."""
    d, S, plat, x64 = _args(output, backend, derivative, cells, platform, x64, "output_memory")
    per, read, held = _model(output, backend, d, S, plat, x64)
    return OutputMemory(float(per), int(math.ceil(read + held + _LAUNCH[(backend, plat)])))


def cell_bytes(output, backend="auto", derivative=None, *, cells=None, platform=None, x64=None):
    """Device memory per cell of one output at `cells` cells (default: its own size), fixed term included:
    output_memory(...).total(cells) / cells.  The arguments as tallymap.output_memory.

        tm.cell_bytes(tm.Sum((n,), jnp.float32, fixed_point=True), "xla", "jvp")     # exact sum, xla, under jvp"""
    d, S, plat, x64 = _args(output, backend, derivative, cells, platform, x64, "cell_bytes")
    return _total(output, backend, d, S, plat, x64) / S



def call_memory(outputs, backend="auto", derivative=None, *, platform=None, x64=None):
    """Device memory, in bytes, that the outputs of ONE tallymap.vmap call hold while it runs: the sum over outputs of
    per_cell * cells (each at its own size: a Sum's shape, a Records capacity) and of the partial rows, plus the
    largest readout (the readouts run one tally at a time) and the launch's fixed buffers once.  outputs: the
    outputs= dict of tallymap.vmap (or a list of Sum / Records).  The other arguments as tallymap.output_memory; the
    items' own memory is not included.  Measured on lumen (kernel backend, exact sums, 2080 Ti, 46142910 /
    46142912): SK 40.7M cells forward 1.81 GB, jvp 3.61 GB; xla backend 3.64 / 7.11 GB."""
    outs = list(outputs.values()) if isinstance(outputs, dict) else list(outputs)
    tot, rmax = 0.0, 0.0
    for o in outs:
        d, S, plat, x64_ = _args(o, backend, derivative, None, platform, x64, "call_memory")
        per, read, held = _model(o, backend, d, S, plat, x64_)
        tot += per * S + held
        rmax = max(rmax, read)
    plat = _platform(platform)
    return int(math.ceil(tot + rmax + _LAUNCH[(backend, plat)]))

def _derivative(d):
    d = "forward" if d is None else d
    if d not in DERIVATIVES:
        raise ValueError(f"tallymap: derivative must be None or one of {DERIVATIVES}, got {d!r}")
    return d


def max_cells(output, backend="auto", derivative=None, *, memory=None, platform=None, x64=None):
    """The largest cell count of one output in a call: 2^31 - 1 on every backend (cells are indexed with int32: a
    larger Sum raises at tallymap.vmap), and with memory= (bytes) also the largest count whose
    output_memory(...).total(count) fits it.  The other arguments as tallymap.output_memory; the output's own size is
    ignored (only its dtype, kind and fields matter)."""
    d, _, plat, x64 = _args(output, backend, derivative, 1, platform, x64, "max_cells")
    if memory is None:
        return MAX_CELLS
    tot = lambda S: _total(output, backend, d, S, plat, x64)          # noqa: E731  (non-decreasing in S)
    if tot(1) > memory:
        return 0
    lo, hi = 1, MAX_CELLS
    if tot(hi) <= memory:
        return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if tot(mid) <= memory else (lo, mid)
    return lo


__all__ = ["output_memory", "OutputMemory", "call_memory", "cell_bytes", "max_cells", "MAX_CELLS"]
