"""The two primitives an item function may use, and the Tally object that carries them.

    emit_p (tally[*L, S], idx[*I] int32, val[*L, *I]) -> tally'      tally'[..., idx] += val; OOB/negative dropped
                                                                     (as JAX's scatter does, and counted in
                                                                     stats['dropped_emits'])
    spawn_p(q:int32[], when:bool[], *child_leaves)    -> q'           push a child item (only inside tallymap.vmap)

Both take the value they "modify" as an operand and return its successor, so the user threads the Tally through
their loops like a PRNG key.  This data dependency is what keeps emits alive through every JAX transformation
(no effects system, no DCE hazards), and it gives emit_p a plain dense meaning: outside tallymap.vmap, a Tally is just
an array and jax.vmap(item)(xs) followed by a sum over items IS the semantics (tm.reference computes it).
"""
import jax
import jax.numpy as jnp
from jax import lax
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import ad, mlir

from ._compat import register_batcher
from ._rewrite import isfloat


def _zero_like(x):
    return ad.Zero(jax.typeof(x).to_tangent_aval())


# ----------------------------------------------------------------------------------------------- emit_p
emit_p = Primitive("tallymap_emit")


def _flat(t, idx, val):
    S = t.shape[-1]
    L = t.shape[:-1]
    f = jnp.reshape(idx, (-1,)).astype(jnp.int32)
    f = jnp.where((f >= 0) & (f < S), f, S)          # out of range (incl. negative) -> dropped
    v = jnp.reshape(val, L + (-1,))
    return f, v


def _emit_dense(t, idx, val):
    f, v = _flat(t, idx, val)
    return t.at[..., f].add(v.astype(t.dtype), mode="drop")


def _emit_abstract(t, idx, val):
    L = t.shape[:-1]
    if tuple(val.shape) != tuple(L) + tuple(idx.shape):
        raise TypeError(f"tallymap: t.add(...): value shape {val.shape} != tally lead {L} + index shape {idx.shape}")
    if not jnp.issubdtype(idx.dtype, jnp.integer):
        raise TypeError(f"tallymap: t.add(...): index dtype {idx.dtype} is not an integer")
    return ShapedArray(t.shape, t.dtype)


emit_p.def_impl(_emit_dense)
emit_p.def_abstract_eval(_emit_abstract)
mlir.register_lowering(emit_p, mlir.lower_fun(_emit_dense, multiple_results=False))


def _emit_jvp(primals, tangents):
    t, idx, val = primals
    dt, _, dv = tangents
    out = emit_p.bind(t, idx, val)
    if not isfloat(t.dtype) or (type(dt) is ad.Zero and type(dv) is ad.Zero):
        return out, _zero_like(out)
    dt = ad.instantiate_zeros(dt)
    if type(dv) is ad.Zero:
        return out, dt
    return out, emit_p.bind(dt, idx, dv)


def _emit_transpose(ct, t, idx, val):
    if type(ct) is ad.Zero:
        return [None, None, None]
    cv = None
    if ad.is_undefined_primal(val):
        S = ct.shape[-1]
        f = jnp.reshape(idx, (-1,)).astype(jnp.int32)
        ok = (f >= 0) & (f < S)
        g = jnp.where(ok, ct[..., jnp.clip(f, 0, S - 1)], jnp.zeros((), ct.dtype))
        cv = jnp.reshape(g, ct.shape[:-1] + tuple(idx.shape))
    return [ct if ad.is_undefined_primal(t) else None, None, cv]


def _emit_batch(args, dims):
    t, idx, val = args
    dt, di, dv = dims
    n = [a.shape[d] for a, d in zip(args, dims) if d is not None][0]
    mv = lambda a, d: jnp.moveaxis(a, d, 0) if d is not None else jnp.broadcast_to(a, (n,) + a.shape)
    if di is not None:
        # per-lane indices only occur in plain-JAX reference evaluation (jax.vmap of an item function):
        # fall back to the dense scatter there.  Backends never see this form (it would fail their tally check).
        return jax.vmap(_emit_dense)(mv(t, dt), mv(idx, di), mv(val, dv)), 0
    return emit_p.bind(mv(t, dt), idx, mv(val, dv)), 0


ad.primitive_jvps[emit_p] = _emit_jvp
ad.primitive_transposes[emit_p] = _emit_transpose
register_batcher(emit_p, _emit_batch)


# ----------------------------------------------------------------------------------------------- spawn_p
spawn_p = Primitive("tallymap_spawn")


def _spawn_abstract(q, when, *leaves):
    if q.shape != () or when.shape != ():
        raise TypeError("tallymap spawn: the queue token and `when` must be scalars (spawn under jax.vmap inside an "
                        "item is not supported)")
    return ShapedArray((), q.dtype)


def _spawn_impl(*a, **k):
    raise NotImplementedError("tallymap.spawn only runs inside tallymap.vmap (it has no plain-JAX meaning)")


spawn_p.def_impl(_spawn_impl)
spawn_p.def_abstract_eval(_spawn_abstract)
mlir.register_lowering(spawn_p, lambda ctx, *a, **k: _spawn_impl())


def _spawn_jvp(primals, tangents):
    q, when, *c = primals
    ct = [ad.instantiate_zeros(t) for x, t in zip(c, tangents[2:]) if isfloat(x.dtype)]
    out = spawn_p.bind(q, when, *c, *ct)       # child = (primal leaves, tangent leaves of the float leaves)
    return out, _zero_like(out)


def _spawn_batch(args, dims):
    q, when, *c = args
    if dims[0] is not None or dims[1] is not None:
        raise NotImplementedError("tallymap spawn: `when` depends on a vmapped value")
    c = [jnp.moveaxis(x, d, 0) if d is not None else x for x, d in zip(c, dims[2:])]
    return spawn_p.bind(q, when, *c), None


ad.primitive_jvps[spawn_p] = _spawn_jvp
register_batcher(spawn_p, _spawn_batch)


# ----------------------------------------------------------------------------------------------- claim_p
# claim_p(tok:int32[C], when:bool[], k:int32[], r=j) -> (tok', slot:int32[])   (a Records output's one-pass append)
# slot = the next free slot of the launch's raw record buffer of capacity C (an atomic counter per output), or -1
# when `when` is False or the buffer is full.  tok is the output's hidden key tally (it orders the claim like an emit:
# the backends treat claim_p as a tally operation); k is the item's append ordinal and r the output's index, which only
# the fill twin reads (_records.twin: claim -> off + k).  No plain-JAX meaning, like spawn.
claim_p = Primitive("tallymap_claim")
claim_p.multiple_results = True


def _claim_abstract(tok, when, k, *, r):
    if when.shape != () or k.shape != ():
        raise TypeError(LANED)
    return [ShapedArray(tok.shape, tok.dtype), ShapedArray((), jnp.int32)]


def _claim_impl(*a, **k):
    raise NotImplementedError("tallymap: t.append's slot claim only runs inside tallymap.vmap")


claim_p.def_impl(_claim_impl)
claim_p.def_abstract_eval(_claim_abstract)
mlir.register_lowering(claim_p, lambda ctx, *a, **k: _claim_impl())


def _claim_jvp(primals, tangents, *, r):
    outs = claim_p.bind(*primals, r=r)
    return outs, [_zero_like(o) for o in outs]      # (integer outputs: the key tally and the slot)


def _claim_batch(args, dims, *, r):
    raise TypeError(LANED)


ad.primitive_jvps[claim_p] = _claim_jvp
register_batcher(claim_p, _claim_batch)


# ----------------------------------------------------------------------------------------------- Tally
# A Records output (tallymap.Records).  ONE PASS (mode 'one', the default; _records.py): t.append claims a raw slot
# (claim_p) and, when it got one, emits each field, the item's node id cid and k there; the launch's raw layout is
# canonicalised by the primitive (_records.canon).  COUNT-THEN-FILL (spawn, capacity=None): tallymap.vmap traces the
# item twice.  In the COUNT pass t.append only advances the item's append ordinal k; in the FILL pass it also emits
# each field into an ordinary
# Sum tally of the launch's capacity at slot off + k (off: the item's first slot, the exclusive prefix sum of the
# counts), guarded by k < cnt (the item's count from the count pass) so that a fill that disagrees with its count
# never writes into the next item's slots.  k, off and cnt are int32 scalars carried in the Tally (pytree leaves),
# so they follow the user's loops exactly like the tally.  tm.reference has a third mode, REF: each item appends into
# its own dense buffer at index k (the plain-JAX meaning, independent of the offsets).
RSEP = "\x1f"
LANED = ("tallymap: t.append inside jax.vmap within an item (e.g. a vmapped while loop) is not supported: the "
         "records of the lanes would share one append ordinal.  Append from the item's own, unbatched code.")


def rkey(name, field):
    """the tally key of a Records output's field."""
    return f"{name}{RSEP}{field}"


KI, KK = RSEP + "i", RSEP + "k"     # a one-pass Records output's hidden key tallies: each record's item (node id) and k


@jax.tree_util.register_pytree_node_class
class Tally:
    """Write-only commutative outputs of one item.  Thread it through loops: ``t = t.add(name, idx, val)``.

    Every operation returns a NEW Tally; using an old one again (forking) or dropping one is rejected by
    tallymap.vmap (it would silently disagree with the dense meaning)."""

    def __init__(self, arrays, shapes, q=None, child_tree=None, rec=None, rinfo=None):
        self._a = dict(arrays)           # name -> flat array [*lead, S]
        self._shapes = dict(shapes)      # name -> user shape
        self._q = q                      # spawn token (int32 scalar) or None
        self._child_tree = child_tree    # treedef of the per-item inputs (spawn) or None
        self._r = dict(rec or {})        # Records name -> (k,) | (k, off, cnt): int32 scalars
        self._ri = rinfo                 # static: (mode, {name: (capacity, ((field, dtype), ...))}) as a tuple

    def tree_flatten(self):
        names = tuple(sorted(self._a))
        rn = tuple(sorted(self._r))
        ch = (tuple(self._a[n] for n in names) + ((self._q,) if self._q is not None else ())
              + tuple(x for r in rn for x in self._r[r]))
        return ch, (names, tuple(self._shapes[n] for n in names), self._q is not None, self._child_tree,
                    tuple((r, len(self._r[r])) for r in rn), self._ri)

    @classmethod
    def tree_unflatten(cls, aux, ch):
        names, shapes, has_q, tree, rl, ri = aux
        n = len(names)
        q = ch[n] if has_q else None
        o = n + int(has_q)
        rec = {}
        for r, m in rl:
            rec[r] = tuple(ch[o:o + m])
            o += m
        return cls(dict(zip(names, ch[:n])), dict(zip(names, shapes)), q, tree, rec, ri)

    def _with(self, name=None, arr=None, q=None, rec=None):
        a = dict(self._a)
        if name is not None:
            a[name] = arr
        return Tally(a, self._shapes, self._q if q is None else q, self._child_tree,
                     self._r if rec is None else rec, self._ri)

    def _records(self):
        return dict(self._ri[1]) if self._ri else {}

    def add(self, name, index, value):
        """tally[name][index] += value.  `index`: int array (flat index for 1-D outputs) or a tuple of int arrays
        (one per output dimension); any shape, value broadcast to it.  Out-of-range or negative indices are
        DROPPED, as JAX's own scatter drops out-of-bounds updates, and counted in Result.stats['dropped_emits']."""
        if name in self._records():
            raise KeyError(f"tallymap: output {name!r} is a tallymap.Records output: write it with "
                           f"t.append({name!r}, {{field: value, ...}})")
        if name not in self._a or RSEP in name:
            raise KeyError(f"tallymap: unknown output {name!r} (outputs: "
                           f"{sorted([n for n in self._a if RSEP not in n] + list(self._records()))})")
        shape = self._shapes[name]
        t = self._a[name]
        if isinstance(index, tuple):
            if len(index) != len(shape):
                raise IndexError(f"tallymap: output {name!r} has {len(shape)} dims, index has {len(index)}")
            idx = [jnp.asarray(i) for i in index]
            bs = jnp.broadcast_shapes(*[i.shape for i in idx])
            flat, ok = jnp.zeros(bs, jnp.int32), jnp.ones(bs, bool)
            for i, n in zip(idx, shape):
                _check_int(i)
                i = jnp.broadcast_to(i, bs).astype(jnp.int32)
                ok = ok & (i >= 0) & (i < n)
                flat = flat * n + jnp.clip(i, 0, n - 1)
            flat = jnp.where(ok, flat, -1)
        else:
            if len(shape) > 1:
                raise IndexError(f"tallymap: output {name!r} has shape {shape}; index it with a tuple")
            flat = jnp.asarray(index)
            _check_int(flat)
            flat = flat.astype(jnp.int32)
        if (not jnp.issubdtype(t.dtype, jnp.inexact) and not isinstance(value, (int, float))
                and jnp.issubdtype(jnp.asarray(value).dtype, jnp.inexact)):
            raise TypeError(f"tallymap: t.add({name!r}, ...): a float value added to an integer output")
        val = jnp.broadcast_to(jnp.asarray(value), t.shape[:-1] + flat.shape)
        return self._with(name, emit_p.bind(t, flat, val.astype(t.dtype)))

    def spawn(self, child, when=True):
        """Append a new item whose per-item inputs are `child` (same structure as the mapped arguments).
        It runs the same function; its own per-item result is discarded (emit what you need)."""
        if self._q is None:
            raise ValueError("tallymap: spawn needs tallymap.vmap(..., spawn=tallymap.Queue(capacity))")
        leaves, tree = jax.tree_util.tree_flatten(child)
        if tree != self._child_tree:
            raise TypeError(f"tallymap: spawned child structure {tree} != mapped inputs {self._child_tree}")
        w = jnp.asarray(when)
        if w.shape != () or w.dtype != jnp.bool_:
            raise TypeError("tallymap: spawn `when` must be a scalar bool")
        leaves = [jnp.asarray(x) for x in leaves]
        if RSEP in self._r:     # (Records fill pass) the child's canonical node id: its parent's first child + ordinal
            first, s = self._r[RSEP]
            rec = {**self._r, RSEP: (first, s + w.astype(jnp.int32))}
            return self._with(q=spawn_p.bind(self._q, w, *leaves, first + s), rec=rec)
        return self._with(q=spawn_p.bind(self._q, w, *leaves))

    def append(self, name, values, when=True):
        """Append one record to the tallymap.Records output `name`: `values` = {field: scalar} with exactly the
        declared fields; nothing is appended when `when` (a scalar bool) is False.  The output holds the records of
        every item in (item index, append order) order (see tallymap.Records)."""
        recs = self._records()
        if name not in recs:
            raise KeyError(f"tallymap: {name!r} is not a tallymap.Records output (Records outputs: {sorted(recs)})"
                           + ("; write a Sum output with t.add" if name in self._a else ""))
        mode = self._ri[0]
        C, fields = recs[name]
        if not isinstance(values, dict) or set(values) != {f for f, _ in fields}:
            got = sorted(values) if isinstance(values, dict) else type(values).__name__
            raise TypeError(f"tallymap: t.append({name!r}, ...) takes a dict of exactly the declared fields "
                            f"{sorted(f for f, _ in fields)}, got {got}")
        w = jnp.asarray(when)
        if w.shape != () or w.dtype != jnp.bool_:
            raise TypeError("tallymap: t.append `when` must be a scalar bool")
        st = self._r[name]
        k = st[0]
        if jnp.ndim(k) != 0:
            raise TypeError(LANED)
        vals = []
        for f, dt in fields:
            v = values[f]
            if (not jnp.issubdtype(dt, jnp.inexact) and not isinstance(v, (int, float))
                    and jnp.issubdtype(jnp.asarray(v).dtype, jnp.inexact)):
                raise TypeError(f"tallymap: t.append({name!r}, ...): a float value for the integer field {f!r}")
            v = jnp.asarray(v)
            if v.shape != ():
                raise TypeError(f"tallymap: t.append({name!r}, ...): field {f!r} must be a scalar, got shape "
                                f"{v.shape}")
            vals.append(v.astype(dt))
        k2 = k + w.astype(jnp.int32)
        if mode == "count":
            return self._with(rec={**self._r, name: (k2,)})
        keys = [rkey(name, f) for f, _ in fields]
        a = dict(self._a)
        if mode == "one":                       # one pass: a raw slot, the record and its key (cid, k) written there
            cid = st[1]
            j = [n for n, _ in self._ri[1]].index(name)
            tok, slot = claim_p.bind(a[rkey(name, KK)], w, k, r=j)
            ok = w & (slot >= 0) & (slot < C)
            keys += [rkey(name, KI), rkey(name, KK)]
            vals += [cid, k]

            def put1(ts):
                return tuple(emit_p.bind(t, slot, v) for t, v in zip(ts, vals))
            ts = lax.cond(ok, put1, lambda ts: ts, tuple(a[key] for key in keys[:-1]) + (tok,))
            a.update(zip(keys, ts))
            return Tally(a, self._shapes, self._q, self._child_tree, {**self._r, name: (k2, cid)}, self._ri)
        if mode == "ref":                       # the item's own buffer, slot k (a too-small buffer drops)
            for key, v in zip(keys, vals):
                a[key] = emit_p.bind(a[key], jnp.where(w, k, -1), v)
            return Tally(a, self._shapes, self._q, self._child_tree, {**self._r, name: (k2,)}, self._ri)
        _, off, cnt = st
        j = off + k
        ok = w & (k < cnt) & (j < C)            # the fill guard: never past the item's counted slots or the capacity

        def put(ts):
            return tuple(emit_p.bind(t, j, v) for t, v in zip(ts, vals))
        ts = lax.cond(ok, put, lambda ts: ts, tuple(a[key] for key in keys))
        a.update(zip(keys, ts))
        return Tally(a, self._shapes, self._q, self._child_tree, {**self._r, name: (k2, off, cnt)}, self._ri)


def _check_int(i):
    if not jnp.issubdtype(jnp.asarray(i).dtype, jnp.integer):
        raise TypeError(f"tallymap: t.add(...): the index must be an integer array, got {jnp.asarray(i).dtype}")

