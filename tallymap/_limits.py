"""Resource limits: the one overflow rule of every path that consumes a tallymap resource, and the counts it reads.

Counts.  A count of a call (records appended, children spawned) can pass 2^31 although no capacity can (every
capacity indexes int32 slots: Records capacity, Queue capacity, a Sum's cells <= 2^31 - 1).  A count is kept EXACT as
a Count, two uint32 words (lo, hi) of an integer below 2^64, whatever jax_enable_x64 says, built from the backends'
int32 per-item counts and stats words; a capacity is exceeded when the count is > it (exceeds).  The count a call
REPORTS (stats) is report(n): int64 and exact with jax_enable_x64; without it int32, exact up to 2^31 - 1 and
saturated there (it never wraps: a saturated count exceeds every capacity, so its call is poisoned).  Offsets into a
capacity-C buffer (record k of item i at offset[i] + k) only matter below C, so they are clipped at C (offsets).

The spawn Queue is a ring of Q slots.  A forward call needs only its PENDING children to fit: a child that finds the
ring full is dropped (stats['children_dropped']).  A path that reads the spawn TREE back from the ring (the tree
adjoint's record, _reverse_spawn.tree_vjp; Records + spawn, whose count pass is that record, _api._records) needs
every child of the call in it: a call with more than Q children in all has reused a slot and lost a node of the tree,
even when it dropped none.  queue_lost is that rule, for both kinds of consumer.
"""
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

I32MAX = (1 << 31) - 1
_U = jnp.uint32


class Count(NamedTuple):
    """an exact nonnegative integer below 2^64: lo + 2^32 hi (uint32 arrays of one shape)."""
    lo: Any
    hi: Any


def _u32(x):
    """int32 / uint32 words -> uint32, bit for bit (an int32 counter that passed 2^31 once reads as its uint32)."""
    x = jnp.asarray(x)
    if x.dtype == jnp.uint32:
        return x
    return lax.bitcast_convert_type(x.astype(jnp.int32), jnp.uint32)


def count(x):
    """a counter word (int32 / uint32: a stats entry, a per-item count) as a Count."""
    lo = _u32(x)
    return Count(lo, jnp.zeros_like(lo))


def add(a, b):
    lo = a.lo + b.lo
    return Count(lo, a.hi + b.hi + (lo < a.lo).astype(_U))


def sub(a, b):
    """a - b for a >= b."""
    lo = a.lo - b.lo
    return Count(lo, a.hi - b.hi - (a.lo < b.lo).astype(_U))


def prefix(c, axis=-1):
    """the INCLUSIVE prefix sums of the counters c (int32 / uint32 words, each below 2^32) along axis, exact: Count.
    (The uint32 prefix sum wraps at most once per element, exactly where it decreases; hi counts those wraps.)"""
    u = _u32(c)
    lo = jnp.cumsum(u, axis=axis, dtype=_U)
    prev = lax.slice_in_dim(jnp.concatenate([jnp.zeros_like(lax.slice_in_dim(lo, 0, 1, axis=axis)), lo], axis=axis),
                            0, lo.shape[axis], axis=axis)
    return Count(lo, jnp.cumsum((lo < prev).astype(_U), axis=axis, dtype=_U))


def total(c, axis=-1):
    """the exact sum of the counters c along axis: Count."""
    if c.shape[axis] == 0:
        z = jnp.zeros(c.shape[:axis % c.ndim] + c.shape[axis % c.ndim + 1:], _U)
        return Count(z, z)
    p = prefix(c, axis)
    last = lambda x: lax.index_in_dim(x, x.shape[axis] - 1, axis=axis, keepdims=False)    # noqa: E731
    return Count(last(p.lo), last(p.hi))


def exceeds(n, cap):
    """n > cap (cap: an int in [0, 2^32)): the resource limit cap was passed."""
    return (n.hi > 0) | (n.lo > _U(int(cap)))


def clip(n, cap):
    """min(n, cap) as int32 (cap: an int in [0, 2^31))."""
    return jnp.where(exceeds(n, cap), jnp.int32(int(cap)), lax.bitcast_convert_type(n.lo, jnp.int32))


def offsets(c, cap, axis=-1):
    """the EXCLUSIVE prefix sums of the counts c along axis, clipped at cap (int32; exact below cap)."""
    return clip(sub(prefix(c, axis), count(c)), cap)


def report(n):
    """the count n as a call reports it (stats): int64 and exact with jax_enable_x64, else int32 saturated at
    2^31 - 1 (exact up to it, never wrapped)."""
    if jax.config.jax_enable_x64:
        return (n.hi.astype(jnp.int64) << 32) | n.lo.astype(jnp.int64)
    return clip(n, I32MAX)


def excess(n, cap):
    """max(n - cap, 0) as a Count (cap: an int in [0, 2^32)): what did not fit."""
    over = exceeds(n, cap)
    c = Count(jnp.full(n.lo.shape, int(cap), _U), jnp.zeros(n.lo.shape, _U))
    return sub(Count(jnp.where(over, n.lo, c.lo), jnp.where(over, n.hi, c.hi)), c)


def limit(n, cap):
    """THE rule for a resource of capacity cap used n times (a Count): -> (exceeded: a traced bool, the reported
    demand report(n), the reported excess report(max(n - cap, 0)))."""
    return exceeds(n, cap), report(n), report(excess(n, cap))


def queue_lost(stats, Q, record):
    """-> a traced bool: the spawn Queue of capacity Q lost a child for this consumer.  stats: a launch's stats
    vector (int32[len(_api.STATS)]).  record=False (a forward call): a child was dropped.  record=True (a path that
    reads the tree from the ring): a child was dropped or the call had more than Q children in all (a slot was
    reused; the spawned counter is read as a uint32, so a count that passed 2^31 still exceeds Q)."""
    from ._api import ST  # (here: _api imports this module)
    bad = stats[ST["children_dropped"]] != 0
    if record:
        bad = bad | exceeds(count(stats[ST["spawned"]]), Q)
    return bad
