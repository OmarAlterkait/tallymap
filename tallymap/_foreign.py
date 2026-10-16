"""Foreign primitives: the Equinox ecosystem's own primitives, expanded into JAX primitives at the item trace.

Equinox, Lineax, Optimistix, Diffrax and Quadax all build on `equinox.internal.while_loop`, which wraps the
predicate of every loop in `unvmap_any` and masks every carry update with `select_if_vmap`, so that ONE loop can
serve a whole `jax.vmap` batch.  Those are real JAX primitives, so without this module a tallymap item that calls
`diffrax.diffeqsolve` fails with `UnsupportedError: primitive 'unvmap_any' is not supported`.

An item is UN-batched by construction: tallymap maps one item per lane and the item function sees one trajectory.
That is exactly the case these primitives are written for, and each one documents what it then means:

| primitive                    | its own `impl`, which is what this module traces  | inside an item |
|---|---|---|
| `unvmap_any` / `_all`        | `jnp.any` / `jnp.all`                            | the reduction itself |
| `unvmap_max`                 | `jnp.max`                                        | the reduction itself |
| `select_if_vmap(p, x, y)`    | `x`                                              | un-batched `pred` is always True |
| `maybe_set(p, xs, x, i)`     | `xs.at[i].set(select_if_vmap(p, x, xs[i]))`      | the buffer write |
| `nonbatchable`               | identity                                         | identity |
| `nontraceable`               | identity                                         | identity |
| `nondifferentiable_backward` | identity                                         | identity (see below) |

The expansion is the primitive's OWN `impl`, traced with `jax.make_jaxpr` on the equation's concrete avals: this
module never reimplements any of their semantics and does not import Equinox.  `impl` is the right thing to trace
because every one of these primitives registers its XLA lowering as `mlir.lower_fun(impl)`, i.e. the impl IS the
lowering; the primitive exists only to carry a batching / autodiff rule.

`expand` runs ONCE, on the item jaxpr, in `_api.py` - before tallymap's own `jvp` / reverse programs and before
either backend.  Two consequences are the point of doing it there rather than in a backend:

* `nondifferentiable_backward` is Equinox's guard against reverse-differentiating a `lax.while_loop`, which JAX
  cannot do.  tallymap replaces every while loop in the item by a checkpointed replay loop that it CAN
  reverse-differentiate (`_reverse.py`), so the guard would reject a gradient that is available.  Expanding it to
  the identity it already is in the forward pass removes the guard; `tests/equinox_compat.py` checks the gradient
  it releases against `jacfwd`.
* `maybe_set`'s transpose rule assumes each buffer location is written at most once, which holds for Equinox's own
  loop but not for tallymap's replay of it.  Expanding `maybe_set` into a plain scatter means JAX's own scatter
  transpose is used instead.

## Why Lineax's `linear_solve` is expanded LATE instead

`expand` must only be used on a primitive whose autodiff rules are the autodiff rules of its impl.  Lineax's
`linear_solve` is not such a primitive: its impl reads the factorisation out of the solver `state`, while its jvp
rule asserts that `state` has NO tangent and reconstructs the derivative from the OPERATOR through the implicit
function theorem (`lineax/_solve.py:150-175`).  Differentiating its impl therefore loses the whole dependence on
the matrix and returns a tangent of zero - MEASURED: a `lineax.LU` solve inside an item gave a gradient of 0
instead of -0.056 when `linear_solve` was expanded early.

So `expand_late` is a second table, called by the cuda emitter on the program it is about to emit
(`backends/cuda/kernel.py`), after tallymap has already built its jvp and reverse programs with JAX's own rules.
By then every remaining `linear_solve` is a primal solve, and expanding it is a lowering step and nothing more.
The lockstep backend does not need it at all: XLA runs the primitive itself.

`vprim` is in the same table for the same reason.  Equinox builds a primitive's batching rule by vmapping its
other rules, and does that by wrapping the call in a second primitive, `_vprim_p`, whose impl is
`jax.vmap(the wrapped primitive's impl)`.  It appears when tallymap batches an item (the dual tangent lanes of
`jacfwd`, a batch folded into the item axis), and like `linear_solve` it may only be expanded once the rules it
exists to carry have been used.

Primitives neither table expands are listed in `UNSUPPORTED`, with the advice their error message carries.
"""
from ._jaxpr import source_summary
from ._sugar import Sugar, rewrite

# Primitive names expanded into their own impl.  Name-based: no Equinox import, and a program without them is
# untouched.  Keep every entry justified in the table above.
NAMES = (
    "unvmap_all", "unvmap_any", "unvmap_max",          # equinox/_unvmap.py
    "select_if_vmap", "maybe_set",                     # equinox/internal/_loop/common.py
    "nonbatchable", "nontraceable", "nondifferentiable_backward",   # equinox/internal/_nontraceable.py
)
# Expanded only when a backend is about to lower the program, never before autodiff (see the module docstring).
LATE_NAMES = NAMES + (
    "linear_solve",                                    # lineax/_solve.py (an equinox "vprim")
    "vprim",                                           # equinox/internal/_primitive.py: a vprim under vmap
)

# name -> what to tell the user, appended to the UnsupportedError.  These are foreign primitives whose impl is
# NOT a safe expansion inside an item.
UNSUPPORTED = {
    "noinline": "equinox.internal.noinline keeps a function out of the enclosing program, which a single fused "
                "kernel cannot do: call the function directly instead.",
    "announce_jaxpr": "equinox.debug.announce_transform prints from inside the program; a kernel cannot print. "
                      "Remove it, or run the item under backend='lockstep'.",
    "quadax_leibniz_tangent": "quadax's differentiable-limits adjoint is not supported inside an item: pass "
                              "fixed integration limits, or differentiate the limits outside tallymap.vmap.",
    "eval_jaxpr": "this comes from diffrax's DirectAdjoint (or another adjoint that re-evaluates a jaxpr in its "
                  "backward pass). Inside an item pass adjoint=diffrax.ForwardMode(): tallymap differentiates "
                  "the solver's own loop itself. See docs/equinox.md.",
}

# Equinox's runtime error checks (equinox.error_if, and every library check built on it) become a host callback
# under the default EQX_ON_ERROR=raise.  A kernel cannot call the host.
_EQX_CALLBACK = (
    "This is an equinox.error_if runtime check, which calls back to the host under the default "
    "EQX_ON_ERROR=raise. A tallymap item runs inside one kernel and cannot call the host. Set the environment "
    "variable EQX_ON_ERROR=nan before importing equinox (the check then writes NaN instead of raising), or pass "
    "throw=False to the solver. See docs/equinox.md."
)


def _applies(eqn):
    impl = getattr(eqn.primitive, "impl", None)
    if impl is None:                                      # pragma: no cover (all of NAMES define an impl)
        return False
    # a primitive whose impl is JAX's default dispatcher would expand into itself
    return getattr(impl, "__module__", "") != "jax._src.dispatch"


def _expand(eqn, ins):
    """trace the primitive's own impl on this equation's operands."""
    outs = eqn.primitive.impl(*ins, **eqn.params)
    if not eqn.primitive.multiple_results:
        outs = [outs]
    return list(outs)


_TABLE = Sugar("foreign primitive", NAMES, _applies, _expand)
_LATE = Sugar("foreign primitive (lowering)", LATE_NAMES, _applies, _expand)


def expand(cj):
    """-> cj with every foreign primitive whose autodiff rules are its impl's replaced by that impl (the same
    object when there is none).  Called at the item trace."""
    return rewrite(cj, _TABLE)


def expand_late(cj):
    """-> cj with `LATE_NAMES` expanded too.  Called by a backend on the program it is about to lower, after
    every autodiff rule has already been applied."""
    return rewrite(cj, _LATE)


def advice(eqn):
    """-> a sentence to append to a backend's error message for this equation, or ''."""
    name = eqn.primitive.name
    if name in UNSUPPORTED:
        return " " + UNSUPPORTED[name]
    if eqn.params.get("callback") is not None or eqn.effects:
        src = source_summary(eqn)
        if "equinox" in src or any("equinox" in str(e) for e in eqn.effects):
            return " " + _EQX_CALLBACK
    return ""


def lanes_unsafe(cj):
    """-> True when this program must NOT be split into dual tangent lanes.

    tallymap's dual lanes vmap the item, which turns a foreign primitive into a BATCHED one: equinox wraps it in
    `_vprim_p` and batches by vmapping its rules.  MEASURED on an L40S: a `lineax.CG` solve inside an item - a
    vprim with a data-dependent while loop inside it - then gives NaN in every lane but the first under
    `jacfwd` on the cuda backend, while the SAME conjugate-gradient iteration written by hand is right in every
    lane, and so is a `lineax.LU` solve (a vprim with no loop).  Until that is understood, a program with a
    batched vprim runs one lane at a time, which is MEASURED to be right on both backends.  The cost is
    performance, and only for `jacfwd` of an item that calls a Lineax or Optimistix solver.
    """
    from ._jaxpr import contains
    return contains(cj.jaxpr, ("vprim",))


def has_foreign(cj):
    """does this jaxpr still hold a foreign primitive this module expands?  (a test hook)"""
    return _TABLE.maybe(cj.jaxpr)


CUSTOM_VJP = (
    " The item contains a custom_vjp, which JAX cannot forward-differentiate and which tallymap's reverse mode "
    "also has to probe in forward mode. If it is diffrax's adjoint, pass adjoint=diffrax.ForwardMode(): that "
    "makes the solver's loop a plain lax.while_loop, which tallymap differentiates itself, forward and reverse. "
    "See docs/equinox.md."
)


def ad_advice(e, what):
    """-> an exception to raise in place of `e` when building or running an item program failed for a reason
    tallymap can explain, or None.  Called from _api._mk (every program is traced there) and from both backends."""
    s = f"{type(e).__name__}: {e}"
    if "custom_vjp" in s:
        from ._api import UnsupportedError
        return UnsupportedError(f"tallymap: cannot build the {what} of this item.{CUSTOM_VJP}\n{s}")
    return None
