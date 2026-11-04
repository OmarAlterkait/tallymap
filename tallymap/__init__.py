"""tallymap: jax.vmap for divergent per-item loops whose results are sums ("tallies").

    import tallymap as tm

    @tm.vmap(outputs={"Q": tm.Sum((n_sens,))}, in_axes=(0, None), max_steps=64)
    def transport(t, photon, tables):          # ONE item; t is its Tally
        ...  t = t.add("Q", sensor, w)  ...     # thread t through loops like a PRNG key
        return t, final_time                    # or just `t`
    res = transport(photons, tables)            # res.outputs["Q"], res.per_item, res.truncated, res.stats

The meaning of a tm.vmap function is jax.vmap(item) followed by a sum over items; every JAX transformation works on
it.  See README.md and docs/.
"""
from . import _primitive  # noqa: F401  (registers tallymap_call / tallymap_jvp and their rules)
from ._api import Queue, Result, Sum, UnsupportedError, vmap
from ._compare import reference
from ._compat import TESTED_JAX, _check_jax_version
from ._diagnostics import assert_arms_differ, diagnostics
from ._rewrite import TallyError
from ._tally import Tally


def supported_primitives():
    """-> [dict(name, how)]: every primitive the cuda backend compiles, and how ('rule' | 'canonicalised' |
    'tally').  Anything else raises UnsupportedError on cuda; the lockstep backend runs every JAX primitive.
    The same table: docs/primitives.md."""
    from .backends.cuda.census import supported_primitives as f
    return f()


__version__ = "0.5.0"
_check_jax_version()       # one warning at import, only when JAX is newer than the newest tested version
__all__ = ["vmap", "Sum", "Queue", "Result", "Tally", "reference", "diagnostics",
           "assert_arms_differ", "supported_primitives", "TallyError", "UnsupportedError", "TESTED_JAX"]
