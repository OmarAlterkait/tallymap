"""Error messages that name the cause and the fix, for the failures a JAX program makes likely inside an item.

`ad_advice` is asked by `_api._mk` (every item program is traced there) and by both backends when building or running
a program raised; `holds_tracer` is asked by the lowering rule in `_primitive.py` after a lowering failed.
"""
import jax

CUSTOM_VJP = (
    " The item contains a custom_vjp, which JAX cannot forward-differentiate and which tallymap's reverse mode "
    "also has to probe in forward mode. Inside an item write the function without it: tallymap differentiates "
    "the item's own loops itself, forward and reverse (or give the function a custom_jvp instead)."
)


CLOSED_TRACER = (
    " The program closes over a value from a trace that has already finished (a leaked tracer). Inside a tallymap "
    "item this almost always means a jax.custom_jvp / custom_vjp function (or its rule) closes over a value "
    "computed inside the item. JAX cannot differentiate that once the function is staged - plain "
    "jax.jit(jax.vmap(f)) fails the same way - and tallymap always stages the item. Pass the value to the custom "
    "function as an explicit argument (a nondiff_argnums argument if it must not be differentiated), or wrap the "
    "function with jax.closure_convert."
)


def holds_tracer(cj, _seen=None):
    """-> True if a ClosedJaxpr (or a sub-jaxpr) holds a tracer among its constants.  Asked only AFTER lowering a
    program failed: closed-over values of the live jax.jit around the call are hoisted into operands before
    lowering, so a tracer still here is a stale one, and it is what broke the lowering."""
    from ._jaxpr import subjaxprs
    _seen = set() if _seen is None else _seen
    if id(cj) in _seen:
        return False
    _seen.add(id(cj))
    if any(isinstance(c, jax.core.Tracer) for c in getattr(cj, "consts", ())):
        return True
    return any(holds_tracer(s, _seen) for e in getattr(cj, "jaxpr", cj).eqns for s in subjaxprs(e))


def ad_advice(e, what):
    """-> an exception to raise in place of `e` when building or running an item program failed for a reason
    tallymap can explain, or None.  Called from _api._mk (every program is traced there) and from both backends."""
    s = f"{type(e).__name__}: {e}"
    # a tracer reached lowering: the program closes over a value of a finished trace (diagnosed where it fails,
    # never in advance: a closed-over tracer of the LIVE jax.jit around the call is legitimate)
    if (isinstance(e, (jax.errors.UnexpectedTracerError, jax.errors.TracerArrayConversionError))
            or ("No constant handler for type" in s and "Tracer" in s)):
        from ._api import UnsupportedError
        return UnsupportedError(f"tallymap: cannot build the {what} of this item.{CLOSED_TRACER}\n{s}")
    if "custom_vjp" in s:
        from ._api import UnsupportedError
        return UnsupportedError(f"tallymap: cannot build the {what} of this item.{CUSTOM_VJP}\n{s}")
    return None
