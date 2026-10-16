"""The two primitives an item function may use, and the Tally object that carries them.

    emit_p (tally[*L, S], idx[*I] int32, val[*L, *I]) -> tally'      tally'[..., idx] += val; OOB/negative dropped
                                                                     (as JAX's scatter does, and counted in
                                                                     stats['dropped_emits'])
    spawn_p(q:int32[], when:bool[], *child_leaves)    -> q'           push a child item (only inside tallymap.vmap)

Both take the value they "modify" as an operand and return its successor, so the user threads the Tally through
their loops like a PRNG key.  This data dependency is what keeps emits alive through every JAX transformation
(no effects system, no DCE hazards), and it gives emit_p a plain dense meaning: outside tallymap.vmap, a Tally is just
an array and jax.vmap(item)(xs) followed by a sum over items IS the reference semantics.
"""
import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import ad, mlir

from ._compat import register_batcher

FLOAT0 = jax.dtypes.float0


def _isfloat(dt):
    return jnp.issubdtype(dt, jnp.inexact)


def _zero_like(x):
    return ad.Zero(jax.typeof(x).to_tangent_aval())     # (= Zero.from_primal_value on 0.5.3; that is gone on 0.11)


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
        raise TypeError(f"tallymap emit: value shape {val.shape} != tally lead {L} + index shape {idx.shape}")
    if not jnp.issubdtype(idx.dtype, jnp.integer):
        raise TypeError(f"tallymap emit: index dtype {idx.dtype} is not an integer")
    return ShapedArray(t.shape, t.dtype)


emit_p.def_impl(_emit_dense)
emit_p.def_abstract_eval(_emit_abstract)
mlir.register_lowering(emit_p, mlir.lower_fun(_emit_dense, multiple_results=False))


def _emit_jvp(primals, tangents):
    t, idx, val = primals
    dt, _, dv = tangents
    out = emit_p.bind(t, idx, val)
    if not _isfloat(t.dtype) or (type(dt) is ad.Zero and type(dv) is ad.Zero):
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
    ct = [ad.instantiate_zeros(t) for x, t in zip(c, tangents[2:]) if _isfloat(x.dtype)]
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


# ----------------------------------------------------------------------------------------------- Tally
@jax.tree_util.register_pytree_node_class
class Tally:
    """Write-only commutative outputs of one item.  Thread it through loops: ``t = t.add(name, idx, val)``.

    Every operation returns a NEW Tally; using an old one again (forking) or dropping one is rejected by
    tallymap.vmap (it would silently disagree with the dense meaning)."""

    def __init__(self, arrays, shapes, q=None, child_tree=None):
        self._a = dict(arrays)           # name -> flat array [*lead, S]
        self._shapes = dict(shapes)      # name -> user shape
        self._q = q                      # spawn token (int32 scalar) or None
        self._child_tree = child_tree    # treedef of the per-item inputs (spawn) or None

    def tree_flatten(self):
        names = tuple(sorted(self._a))
        ch = tuple(self._a[n] for n in names) + ((self._q,) if self._q is not None else ())
        return ch, (names, tuple(self._shapes[n] for n in names), self._q is not None, self._child_tree)

    @classmethod
    def tree_unflatten(cls, aux, ch):
        names, shapes, has_q, tree = aux
        q = ch[-1] if has_q else None
        return cls(dict(zip(names, ch[:len(names)])), dict(zip(names, shapes)), q, tree)

    def _with(self, name=None, arr=None, q=None):
        a = dict(self._a)
        if name is not None:
            a[name] = arr
        return Tally(a, self._shapes, self._q if q is None else q, self._child_tree)

    def add(self, name, index, value):
        """tally[name][index] += value.  `index`: int array (flat index for 1-D outputs) or a tuple of int arrays
        (one per output dimension); any shape, value broadcast to it.  Out-of-range or negative indices are
        DROPPED, as JAX's own scatter drops out-of-bounds updates, and counted in Result.stats['dropped_emits']."""
        if name not in self._a:
            raise KeyError(f"tallymap: unknown output {name!r} (outputs: {sorted(self._a)})")
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
            raise TypeError(f"tallymap: float value emitted into integer output {name!r}")
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
        return self._with(q=spawn_p.bind(self._q, w, *[jnp.asarray(x) for x in leaves]))


def _check_int(i):
    if not jnp.issubdtype(jnp.asarray(i).dtype, jnp.integer):
        raise TypeError(f"tallymap: emit index must be an integer array, got {jnp.asarray(i).dtype}")

