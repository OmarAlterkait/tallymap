"""What the kernel backend supports (one emitter, both targets): the table behind
tallymap.supported_primitives() and docs/primitives.md.

NOT_IN_JAX_EXTEND: op-table names that jax.extend.core.primitives does not export, and why they are still right.
tests check  RULES | SUGAR  is a subset of  jax.extend.core.primitives names | NOT_IN_JAX_EXTEND  (catches a JAX rename,
which would otherwise show up only as an UnsupportedError in some user's program).
"""

NOT_IN_JAX_EXTEND = {
    # call-like / control primitives that JAX does not re-export
    "pjit": "jit's primitive name on JAX < 0.7", "jit": "jit's primitive name on JAX >= 0.7",
    "core_call": "jax.core.call", "closed_call": "jax.core.closed_call (not re-exported on 0.11)",
    "checkpoint": "jax.checkpoint (older name)", "remat": "jax.checkpoint (older name)",
    "remat2": "jax.checkpoint (0.5.3 name; re-exported only on newer JAX)",
    "platform_index": "lax.platform_dependent", "optimization_barrier": "lax.optimization_barrier",
    # array / RNG internals
    "copy_p": "lax copy (older name)", "empty": "lax.empty (older name)", "empty2": "lax.empty (0.5.3)",
    "expand_dims": "lax.expand_dims as a primitive (some versions)", "split": "lax.split (not re-exported)",
    "stack": "lax.stack (0.11 primitive)", "unstack": "lax.unstack (0.11 primitive)",
    "mulhi": "lax.mul_hi (internal)", "iota_2x32_shape": "threefry counter iota (internal to jax.random)",
    "random_clone": "jax.random.clone", "random_key_data": "jax.random.key_data", "random_unwrap": "key data",
    "random_wrap": "jax.random.wrap_key_data",
    "scatter_add": "alias of scatter-add", "scatter_mul": "alias of scatter-mul", "scatter_min": "alias of scatter-min",
    "scatter_max": "alias of scatter-max",
    # primitives that exist only on some tested versions
    "log2": "a primitive from JAX 0.6 (jnp.log2 is log(x)/log(2) on 0.5.3)",
    "one_minus_square": "a primitive from JAX 0.10",
    # tallymap's own
    "tallymap_uninit": "an uninitialised reverse-mode checkpoint buffer (_reverse.uninit_p)",
}


def supported_primitives():
    """-> [dict(name, how)] for every primitive the kernel backend compiles, sorted by name.
    how: 'rule' (one emission rule in op_table.py), 'canonicalised' (rewritten into rule primitives by canon.py),
    'tally' (tallymap's own emit / spawn).  Anything else raises tm.UnsupportedError on cuda (the xla backend runs every
    JAX primitive)."""
    from . import canon, op_table
    names = set(op_table.RULES) | set(canon.SUGAR) | {"tallymap_emit", "tallymap_spawn"}
    rows = []
    for n in sorted(names):
        how = "tally" if n in ("tallymap_emit", "tallymap_spawn") else "canonicalised" if n in canon.SUGAR else "rule"
        rows.append(dict(name=n, how=how))
    return rows


def primitives_table():
    """docs/primitives.md (markdown), generated from supported_primitives()."""
    rows = supported_primitives()
    out = ["# Primitives the kernel backend compiles", "",
           "Generated from `tallymap.supported_primitives()` by",
           "`tallymap.backends.kernel.census.primitives_table()` (a test checks that this file is current).",
           "`how`: `rule` = one emission rule in `backends/kernel/op_table.py`; `canonicalised` = rewritten into rule",
           "primitives by `backends/kernel/canon.py` before emission; `tally` = tallymap's own emit / spawn.",
           "Any other primitive raises `tm.UnsupportedError` on cuda; the xla backend runs every JAX",
           "primitive.", "",
           f"{len(rows)} names ({sum(r['how'] == 'rule' for r in rows)} rules, "
           f"{sum(r['how'] == 'canonicalised' for r in rows)} canonicalised).", "",
           "| primitive | how |", "|---|---|"]
    out += [f"| `{r['name']}` | {r['how']} |" for r in rows]
    return "\n".join(out) + "\n"
