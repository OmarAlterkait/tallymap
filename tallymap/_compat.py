"""The JAX-version shims, in one place (docs/maintenance.md: "when X breaks, do Y").

Every use of a JAX interface that changed between the tested versions (0.6.2 - 0.11.2), or that JAX marks as legacy, is
routed through this module, so a JAX upgrade that breaks one of them breaks HERE, with one obvious fix.
"""
import jax
from jax.interpreters import batching

NM = None                       # batching: an unmapped dimension (batching.not_mapped: None up to 0.9, gone from 0.10)

# ------------------------------------------------------------------------------------------- version policy
# the CPU suite passes on every one of these (docs/maintenance.md, "Version policy"); the floor is also
# pyproject's jax>=0.6.2 requirement
TESTED_JAX = ("0.6.2", "0.7.2", "0.8.3", "0.9.2", "0.10.2", "0.11.2")


def _vtuple(v):
    out = []
    for part in str(v).split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit()) if part[:1].isdigit() else ""
        if not digits:
            break
        out.append(int(digits))
    return tuple(out + [0] * (3 - len(out)))


def _check_jax_version(version=None, warn=True):
    """'newer' | 'ok' for a JAX version against TESTED_JAX: 'newer' above the newest tested version, 'ok'
    otherwise.  Only 'newer' warns (once, at import).  A version below the newest tested one is either tested or
    between two tested ones, which is best effort and not worth a warning; anything below the floor is already
    refused by the jax>=0.6.2 requirement."""
    import warnings
    v = _vtuple(version or jax.__version__)
    status = "newer" if v > _vtuple(TESTED_JAX[-1]) else "ok"
    if warn and status == "newer":
        warnings.warn(f"tallymap: JAX {version or jax.__version__} is newer than the newest tested version "
                      f"{TESTED_JAX[-1]} (tested: {', '.join(TESTED_JAX)}); run tests/run_all.py "
                      "(docs/maintenance.md) before relying on it", stacklevel=3)
    return status


# ------------------------------------------------------------------------------------------- batching rules
def register_batcher(prim, rule):
    """Register rule(args, dims, **params) -> (outs, out_dims) as the batching rule of `prim`.

    `batching.primitive_batchers` is a "backwards compat shim" on JAX 0.11 (slated for deletion): it wraps the rule
    into `fancy_primitive_batchers` (which takes a leading AxisData) and binds the primitive unchanged when no operand
    is batched.  We register that wrapper in `fancy_primitive_batchers` ourselves (present on every tested version)."""
    fancy = batching.fancy_primitive_batchers

    def wrapped(axis_data, vals, dims, **params):
        del axis_data
        if all(d is NM for d in dims):
            o = prim.bind(*vals, **params)
            return (o, [NM] * len(o)) if prim.multiple_results else (o, NM)
        return rule(vals, dims, **params)
    fancy[prim] = wrapped


# ------------------------------------------------------------------------------------------- sharding
def requires_devices(prim):
    """mark `prim` as lowered with the concrete device assignment, as JAX marks custom_partitioning itself: its
    lowering rule binds a custom_partitioning call, which needs the devices, and JAX finds that out only from this
    set (jax._src.dispatch.prim_requires_devices_during_lowering; private, so feature-detected)."""
    try:
        from jax._src import dispatch
        dispatch.prim_requires_devices_during_lowering.add(prim)
    except (ImportError, AttributeError):
        pass


# ------------------------------------------------------------------------------------------- jaxpr types
def drop_var(aval):
    """a jaxpr DropVar of `aval` (jax.extend.core from 0.10.2; jax.core up to 0.10.2, removed there in 0.11)."""
    try:
        from jax.extend.core import DropVar
    except ImportError:
        from jax.core import DropVar
    return DropVar(aval)


# ------------------------------------------------------------------------------------------- typed PRNG keys
def key_impl(dtype):
    """the PRNGImpl of a typed-key dtype: `dtype._impl` (the one private attribute left; present on every tested
    version)."""
    return dtype._impl


def _vmap_n(n, f):
    for _ in range(n):
        f = jax.vmap(f)
    return f


def _vmap_bcast(shape1, shape2, f):
    """jax.random's iterated vmap of a binary key op with numpy-style broadcasting (as JAX's own impl rules)."""
    import jax.numpy as jnp
    n1, n2 = len(shape1), len(shape2)
    if n1 == n2 == 0:
        return f
    if n1 == 0:
        return lambda x, y: _vmap_n(n2, lambda y_: f(x, y_))(y)
    if n2 == 0:
        return lambda x, y: _vmap_n(n1, lambda x_: f(x_, y))(x)
    for s1, s2 in reversed(list(zip(shape1, shape2))):
        if s1 == s2:
            f = jax.vmap(f, out_axes=0)
        else:
            def sq(g, left):
                def h(x, y):
                    if left:
                        return jax.vmap(g, in_axes=(None, 0), out_axes=0)(jnp.squeeze(x, axis=0), y)
                    return jax.vmap(g, in_axes=(0, None), out_axes=0)(x, jnp.squeeze(y, axis=0))
                return h
            f = sq(f, s1 == 1)
    return f


def random_bits(impl, keys, keys_ndim, bit_width, shape):
    """random_bits_p on the key DATA `keys` (uint32[*keys_shape, *impl.key_shape]) through the impl's public method."""
    return _vmap_n(keys_ndim, lambda k: impl.random_bits(k, bit_width, shape))(keys)


def random_split(impl, keys, keys_ndim, shape):
    return _vmap_n(keys_ndim, lambda k: impl.split(k, shape))(keys)


def random_fold_in(impl, keys, msgs, keys_shape):
    import numpy as np
    return _vmap_bcast(tuple(keys_shape), np.shape(msgs), impl.fold_in)(keys, msgs)


def random_seed(impl, seeds):
    import numpy as np
    return _vmap_n(np.ndim(seeds), impl.seed)(seeds)
