"""Reverse mode: the item VJP program, reverse-differentiable while loops, sparse table cotangents.

JAX refuses to reverse-differentiate lax.while_loop (unknown trip count).  _rewrite.retrace(rev=...) replaces every
while loop (recursively) by a CHECKPOINTED LOOP (checkpointed_while): a custom_vjp loop whose forward stores only
(consts, initial carry, trip count K, the first C states) and whose VJP re-creates any other state it needs from the
initial carry:

  Per item, a HEAD buffer of C+1 states, a checkpoint buffer of C states and a segment tape of C states (C: option
    rev_checkpoints, by default chosen from the state's size, see auto_checkpoints; a state is the carries the step
    or predicate READS, _rewrite._live_carries).  The forward pass of the loop counts the steps K and stores the
    states before steps 0..C-1 in the head (slot C is an unconditional scratch sink for k >= C).  The backward
    pass then:
      K <= C: sweeps the head directly, newest first: K step VJPs and NO replay.
      K >  C: replays the loop once, storing every m-th state with a stride of m = max(C, ceil(K/C)) (so at most C
              checkpoints, every segment tape full); then walks the segments from the last one, refilling a
              segment's (<= m) states from its checkpoint into the tape (the first C states: straight from the
              head) and applying the VJP of one body step to each, newest first.
    Work: K body steps + K step-VJPs for K <= C; ~3K + K VJPs for K <= C^2 (O(K)); K^2/(2C^2) + 3K beyond.  Memory
    3C+1 states per item.  No bound on K is needed in advance.  (The original O(K^2) replay-from-the-start scheme
    was 3.4-13x slower everywhere it was measured and was removed; a counting pass that stores nothing -- which
    makes every K <= C item replay its whole loop once more -- was 1.7x slower on a photon-transport gradient,
    lumen SK, and 1.4x on SBND.)

Only ACTIVE float carries are differentiated in a step VJP: a carry is active if it can depend on a perturbed
operand of the loop (a fixpoint over the body, probed with JAX's own symbolic zeros).  Tally carries get a
pass-through cotangent (emit is additive), so no dense per-item cotangent of an output is ever formed.

Sparse table cotangents: a perturbed broadcast operand with >= sparse_min elements that is read only by
gather / dynamic_slice / slice (through pjit, loops, cond and scan) never gets a dense per-item cotangent.  Each
read is rewritten into a custom_vjp `tread` that threads a cotangent token (a virtual tally) linearly through
the item: the transpose of a read is an emit of the read's cotangent at the flat indices it read, i.e. an
atomic scatter-add into ONE shared accumulator in the kernel.  A table with any other use keeps the dense
per-item cotangent for ALL its uses (decided by a trace-time pass; never silently dropped).

The item VJP is again a plain JAX function with forward-only loops that every backend runs.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.core import ShapedArray as _SA
from jax.custom_derivatives import SymbolicZero
from jax.extend.core import Primitive
from jax.interpreters import ad as _jad
from jax.interpreters import batching as _batching
from jax.interpreters import mlir as _mlir

from ._rewrite import isfloat
from ._tally import emit_p

FLOAT0 = jax.dtypes.float0
LAST = {}          # what the last vjp_prog / checkpointed_while built (instrument checks in tests and benchmarks)


@dataclasses.dataclass(frozen=True)
class RevOpts:
    checkpoints: int = 0       # C: states per checkpoint buffer (public option rev_checkpoints); 0 = auto_checkpoints
    sparse_min: int = 32       # tables with >= this many elements get sparse (emitted) cotangents (private)


def rev_opts(opts=None):
    """RevOpts from the tallymap.vmap option rev_checkpoints (public: an int >= 1, or 'auto' / None) and
    _rev_sparse_min (private)."""
    if isinstance(opts, RevOpts):
        return opts
    opts = dict(opts or {})
    c = opts.get("rev_checkpoints")
    auto = c is None or (isinstance(c, str) and c == "auto")
    r = RevOpts(checkpoints=0 if auto else int(c), sparse_min=int(opts.get("_rev_sparse_min", 32)))
    if not auto and r.checkpoints < 1:
        raise ValueError("tallymap: rev_checkpoints must be >= 1 (or 'auto' / None)")
    return r


# The automatic C (rev_checkpoints unset).  A state of fewer than AUTO_BIG_STATE bytes (the carries the step reads)
# keeps C = AUTO_C_SMALL; a larger one gets the largest power of two in [AUTO_C_MIN, AUTO_C_MAX] whose 3C+1 states
# (head C+1, checkpoints C, segment tape C) fit AUTO_CKPT_BYTES.
#   MEASURED on an L40S (lumen/perf 762da14, 830c071).  lumen's photon transport (states of 306-318 B; per-photon
#   trip counts K: SK mean 7.1 / warp(32)-max over mean 2.29, SBND 15.0 / 2.35, ANNIE 12.4 / 3.83, longest 1263):
#   the adjoint runs whole items per warp, a warp pays its LONGEST item, and an item of K <= C steps needs no
#   replay, so C must cover the warp maximum (16 to 60 steps); C = 64 against 16: SK 0.685 against 0.878 s, ANNIE
#   5.22 against 8.21 s.  The benchmarks/vs_jax problems (states of 8-25 B; mean K 4.5 to 252): C = 64 was 1.5x to
#   3x SLOWER than 16 at 2^19 items (random_walk 2.53 against 0.92 ms, transport 3.86 against 1.81).  The emitter
#   writes arrays of <= 32 elements (codegen.UNROLL) as straight-line code that ptxas can keep in registers: a
#   scalar's C = 16 buffers can, C >= 32 cannot, while a 28-carry state lives in local memory at any C.
#   AUTO_BIG_STATE = 128 B lies in the measured gap (25 .. 306 B): nothing in between was measured.
#   KNOWN LIMIT: AUTO_CKPT_BYTES counts LOGICAL state bytes and ignores the device.  Per-item memory is local
#   memory, which the CUDA driver reserves for EVERY resident thread outside JAX's pool (bytes/thread x max
#   threads/SM x SMs); the right rule sizes C from the compiled frame (tm.diagnostics()["kernels"][i]["resources"])
#   against free device memory.  Not done yet.
AUTO_BIG_STATE = 128
AUTO_C_SMALL = 16
AUTO_C_MIN, AUTO_C_MAX = 4, 64
AUTO_CKPT_BYTES = 65536          # of LOGICAL state bytes: lumen (306 B per state) gets C = 64
LEAN_C = 16                      # C of an adjoint that is itself differentiated (second order): _primitive._lean


def state_bytes(xs):
    """bytes of one loop state (the stored carries), as stored in a checkpoint buffer slot"""
    return int(sum(int(np.prod(jnp.shape(_phys(x)), dtype=np.int64)) * np.dtype(_dt(_phys(x))).itemsize for x in xs))


def auto_checkpoints(sb):
    """C for a loop state of sb bytes (see AUTO_BIG_STATE): AUTO_C_SMALL below AUTO_BIG_STATE bytes, else the largest
    C <= AUTO_C_MAX whose 3C+1 states fit AUTO_CKPT_BYTES, at least AUTO_C_MIN."""
    C = AUTO_C_SMALL if sb < AUTO_BIG_STATE else AUTO_C_MAX
    while C > AUTO_C_MIN and (3 * C + 1) * sb > AUTO_CKPT_BYTES:
        C //= 2
    return C


# ------------------------------------------------------------------------------------------- replay consistency
REPLAY = {"calls": 0, "mismatch": 0, "last": None}


def _replay_cb(n):
    import warnings
    n = int(n)
    if not n:                  # (under vmap the cond becomes a select and every member calls back)
        return
    REPLAY["calls"] += 1
    REPLAY["last"] = n
    REPLAY["mismatch"] += n
    msg = (f"tallymap: replay mismatch in {n} item(s): the adjoint kernel's replay did not reproduce the forward "
           "kernel's trajectory (fingerprint of the final loop carries differs); the cotangents of this call are NaN.  "
           "Forward and adjoint must use identical math options.")
    warnings.warn(msg, stacklevel=2)


def check_replay(mm, outs, count=False):
    """mm: per-item int32 mismatch flags of an adjoint kernel; outs: its cotangent outputs.  Any mismatch -> every
    float cotangent of the call is NaN: never silently wrong, and free.  count (option rev_stats=True): also report
    to the host (warning + tallymap.diagnostics()['reverse']) -- NOT the default: ANY host callback in the jitted
    program takes jax.jit off its C++ dispatch path (measured: a photon-transport jit(grad) wall 247 -> 119 ms, a
    shower tree adjoint 1632 -> 229 ms without it; toy walk 41.5 -> 2.3 ms, also when the callback sits in an
    untaken lax.cond)."""
    n = jnp.sum(mm.astype(jnp.int32))
    if count:
        lax.cond(n > 0, lambda m: jax.debug.callback(_replay_cb, m), lambda m: None, n)
    return [jnp.where(n > 0, jnp.asarray(np.nan, o.dtype), o) if isfloat(o.dtype) else o for o in outs]


def replay_mismatch():
    """{calls, mismatch, last}: adjoint calls that MISMATCHED so far in this process, their total mismatching items,
    the last such call's count (clean calls are not counted: they never reach the host)."""
    return dict(REPLAY)


class _Static:
    """pytree node carrying static python data through custom_vjp residuals."""

    def __init__(self, v):
        self.v = v


jax.tree_util.register_pytree_node(_Static, lambda s: ((), s.v), lambda v, _: _Static(v))


def _dt(x):
    return x.dtype if hasattr(x, "dtype") else jnp.result_type(x)


def _is_key(x):
    return jax.dtypes.issubdtype(_dt(x), jax.dtypes.prng_key)


def _phys(x):
    return jax.random.key_data(x) if _is_key(x) else x


def _unphys(d, like):
    return jax.random.wrap_key_data(d, impl=jax.random.key_impl(like)) if _is_key(like) else d


# An UNINITIALISED per-item buffer (a checkpoint buffer: every slot is written before it is read, by construction).
# The cuda backend declares it without a fill; everywhere else it is zeros.  MEASURED (lumen/perf d6cb9fc): zero-
# filling 3C+1 states per item (C = 64) cost the cheap-step benchmarks/vs_jax gradients up to 5x at 2^19 items
# (random_walk 0.80 -> 3.98 ms) -- more than their whole step.  tests/reverse.py gates it: the cuda per-item
# cotangents are bitwise equal with every uninitialised declaration filled with 0x00 and with 0xFF.
uninit_p = Primitive("tallymap_uninit")


def _uninit_impl(*, shape, dtype):
    return jnp.zeros(shape, dtype)


uninit_p.def_impl(_uninit_impl)
uninit_p.def_abstract_eval(lambda *, shape, dtype: _SA(shape, dtype))
_mlir.register_lowering(uninit_p, _mlir.lower_fun(_uninit_impl, multiple_results=False))


def _uninit_jvp(primals, tangents, *, shape, dtype):
    out = uninit_p.bind(shape=shape, dtype=dtype)
    return out, _jad.Zero(_SA(shape, dtype).to_tangent_aval())


_jad.primitive_jvps[uninit_p] = _uninit_jvp


def _uninit_batch(args, dims, *, shape, dtype):         # (no operands: never batched)
    return uninit_p.bind(shape=shape, dtype=dtype), None


_batching.primitive_batchers[uninit_p] = _uninit_batch


def _buf_new(C, xs):
    return [uninit_p.bind(shape=(C,) + tuple(jnp.shape(_phys(x))), dtype=np.dtype(_dt(_phys(x)))) for x in xs]


def _buf_set(buf, i, xs):
    return [lax.dynamic_update_index_in_dim(b, jnp.asarray(_phys(x), b.dtype), i, 0) for b, x in zip(buf, xs)]


def _buf_get(buf, i, likes):
    return [_unphys(lax.dynamic_index_in_dim(b, i, 0, keepdims=False), l) for b, l in zip(buf, likes)]


# ------------------------------------------------------------------------------------------- activity
def _active(bfn, bc, init, pb, pc, cand, w_idx):
    """float state carries (indices in `cand`) whose value can depend on a perturbed body const / initial carry
    / table token: fixpoint of 'output tangent is not a symbolic zero' over one body step (probed under
    jax.vjp's linearization, which handles the custom_vjp loops and table reads inside)."""
    fb = [i for i, x in enumerate(bc) if pb[i] and isfloat(_dt(x))]
    act = {j for j in cand if pc[j]}
    while True:
        rec = {}

        @jax.custom_jvp
        def probe(xs):
            return xs

        def _rule(p, t):
            rec["nz"] = [not isinstance(ti, SymbolicZero) for ti in t[0]]
            return p[0], [jnp.zeros_like(x) if isinstance(ti, SymbolicZero) else ti for x, ti in zip(p[0], t[0])]
        probe.defjvp(_rule, symbolic_zeros=True)
        a = sorted(act)

        def h(bc_, init_):
            def g(bv, cv, wv):
                b, c = list(bc_), list(init_)
                for i, v in zip(fb, bv):
                    b[i] = v
                for j, v in zip(a, cv):
                    c[j] = v
                for j, v in zip(w_idx, wv):
                    c[j] = v
                out = bfn(b, c)
                return probe([out[j] for j in cand])
            jax.vjp(g, [bc_[i] for i in fb], [init_[j] for j in a], [init_[j] for j in w_idx])
            return 0
        jax.eval_shape(h, list(bc), list(init))
        nz = {j for j, z in zip(cand, rec.get("nz", [False] * len(cand))) if z}
        if nz <= act:
            return a
        act |= nz


# ------------------------------------------------------------------------------------------- checkpointed_while
def checkpointed_while(cfn, bfn, cc, bc, init, kinds, opts=None, bflags=None, live=None):
    """cfn(cc, carry) -> bool, bfn(bc, carry) -> carry (lists).
    kinds[j]: None/False = state, True/'o' = output Tally (pass-through cotangent), 'w' = table cotangent token
    (its cotangent is threaded through the backward sweep).  bflags[i]: body const i is a tally (never
    differentiated: a closed-over tally is a discarded one).  live[j]: does the body or the predicate READ carry j
    (_rewrite._live_carries; None: all do)?  Only read state carries are stored: one that is never read is restored
    as its initial value, which no replayed step and no step VJP can tell apart."""
    o = rev_opts(opts)
    btal = list(bflags) if bflags is not None else [False] * len(bc)
    kinds = ["o" if k is True else (k or None) for k in kinds]
    n = len(init)
    s_idx = [j for j in range(n) if kinds[j] is None]
    f_idx = [j for j in s_idx if isfloat(_dt(init[j]))]
    o_idx = [j for j in range(n) if kinds[j] == "o" and isfloat(_dt(init[j]))]
    w_idx = [j for j in range(n) if kinds[j] == "w"]

    @jax.custom_vjp
    def loop(cc, bc, init):
        return lax.while_loop(lambda c: cfn(cc, c), lambda c: bfn(bc, c), init)

    l_idx = [j for j in s_idx if live is None or live[j]]       # the stored state: the carries the step reads
    sb = state_bytes([init[j] for j in l_idx])
    C = int(o.checkpoints) if o.checkpoints else auto_checkpoints(sb)

    def fwd(cc, bc, init):
        pb = tuple(bool(x.perturbed) and not t for x, t in zip(bc, btal))
        pc = tuple(bool(x.perturbed) for x in init)
        cc, bc, init = [x.value for x in cc], [x.value for x in bc], [x.value for x in init]
        # the forward counts the steps and stores the states before steps 0..C-1 in the HEAD (slot C: a scratch
        # sink, so the store is unconditional and in place): an item of K <= C steps then needs no replay at all
        def b_(s):
            k, c, h = s
            return k + 1, bfn(bc, c), _buf_set(h, jnp.minimum(k, C), [c[j] for j in l_idx])
        k, out, head = lax.while_loop(lambda s: cfn(cc, s[1]), b_,
                                      (jnp.int32(0), init, _buf_new(C + 1, [init[j] for j in l_idx])))
        act = tuple(_active(bfn, bc, init, pb, pc, f_idx, w_idx))
        LAST.setdefault("loops", []).append(dict(checkpoints=C, auto=not o.checkpoints, n_state=len(s_idx),
                                                 n_live=len(l_idx), n_float=len(f_idx), n_active=len(act),
                                                 n_tok=len(w_idx), state_bytes=sb, ckpt_bytes=(3 * C + 1) * sb))
        return out, (cc, bc, init, k, head, _Static((pb, pc, act)))

    def bwd(res, ct):
        cc, bc, init, K, head, st = res
        pb, pc, act = st.v
        act = list(act)
        fb = [i for i, x in enumerate(bc) if pb[i] and isfloat(_dt(x))]

        def inst(c, x):
            return jnp.zeros(jnp.shape(x), _dt(x)) if c is None or isinstance(c, SymbolicZero) else c
        ct = list(ct)
        ct_o = [inst(ct[j], init[j]) for j in o_idx]
        acc0 = ([inst(ct[j], init[j]) for j in act], [inst(ct[j], init[j]) for j in w_idx],
                [jnp.zeros_like(bc[i]) for i in fb])

        def vjp_step(c, acc):
            ctd, ctw, ctb = acc

            def f(dv, bv, ov, wv):
                cf, b = list(c), list(bc)
                for j, v in zip(act, dv):
                    cf[j] = v
                for j, v in zip(o_idx, ov):
                    cf[j] = v
                for j, v in zip(w_idx, wv):
                    cf[j] = v
                for i, v in zip(fb, bv):
                    b[i] = v
                out = bfn(b, cf)
                return [out[j] for j in act], [out[j] for j in o_idx], [out[j] for j in w_idx]
            _, vjp = jax.vjp(f, [c[j] for j in act], [bc[i] for i in fb], [c[j] for j in o_idx],
                             [c[j] for j in w_idx])
            dc, db, _, dw = vjp((list(ctd), ct_o, list(ctw)))
            return list(dc), list(dw), [a + b for a, b in zip(ctb, db)]

        def adv(c, steps):
            return lax.fori_loop(0, steps, lambda i, c: bfn(bc, c), c)

        lst = list(init)
        likes = [init[j] for j in l_idx]

        def state(c):
            return [c[j] for j in l_idx]

        def restore(vals):
            c = list(init)
            for j, v in zip(l_idx, vals):
                c[j] = v
            return c
        # Checkpoint stride.  C, not ceil(K/C): a SEGMENT TAPE of C slots is allocated once per segment, and
        # allocating it costs C stores whatever the segment's length is, so the sweep's local-memory traffic is
        # ns * C states, not K.  ceil(K/C) makes ns = K for K <= C (one state per segment, C-1 slots wasted):
        # measured on a 2080 Ti at 2^19 items, the transport adjoint (K ~ 16 = C) cost 15.6 ms at stride ceil(K/C)
        # and 5.8 ms at rev_checkpoints=8, i.e. the waste, not the arithmetic, was most of the kernel.  A stride of
        # C fills every tape: ns * C ~ K for every K, and ns <= C still holds (for K > C^2 the stride is ceil(K/C)
        # again and the sub-chunk loop below splits a segment into ceil(m/C) tapes, which are also full).
        m = jnp.maximum(C, (K + C - 1) // C)            # checkpoint stride
        ns = (K + m - 1) // m                           # segments (<= C)
        # checkpoint pass: empty when K <= C (then ns = 1 and the only checkpoint is slot 0 = the initial state)
        def fill1(s, cb):
            c, b = cb
            c = adv(c, m)
            return c, _buf_set(b, s, state(c))
        _, buf1 = lax.fori_loop(1, ns, fill1, (lst, _buf_set(_buf_new(C, likes), 0, state(init))))

        # The FIRST tape (states 0..min(K, C)-1) is the head itself, read-only: for K <= C (ns = nsub = 1) that is
        # the whole sweep, with no replay.  Every other tape is a fresh C-slot buffer refilled from its checkpoint.
        # (The head and the tape are separate buffers: measured on lumen SK, reusing the head as the tape -- a loop
        # carry of the segment loops -- made the emitter copy it, a 28 KB stack frame against 12 KB, 17% slower,
        # and multiplied second-order memory.  ONE reverse loop with a select between two buffers defeats the
        # emitter's in-place updates and is 2-5x slower than these nested loops; L40S toy bench.)
        def seg(sr, acc):
            s = ns - 1 - sr
            L = jnp.minimum(m, K - s * m)
            c0 = restore(_buf_get(buf1, s, likes))
            nsub = (L + C - 1) // C

            def sub(ur, acc):
                u = nsub - 1 - ur
                LL = jnp.minimum(C, L - u * C)
                first = (s == 0) & (u == 0)                         # states 0..LL-1: the head's slots 0..LL-1
                c = adv(c0, u * C)
                b2 = _buf_set(_buf_new(C, likes), 0, state(c))      # (segment tape)

                def fill2(v, cb):
                    c, b = cb
                    c = bfn(bc, c)
                    return c, _buf_set(b, v, state(c))
                _, b2 = lax.fori_loop(1, jnp.where(first, 1, LL), fill2, (c, b2))

                def rv(vr, acc):
                    i = LL - 1 - vr
                    raw = [jnp.where(first, lax.dynamic_index_in_dim(h_, i, 0, keepdims=False),
                                     lax.dynamic_index_in_dim(t_, i, 0, keepdims=False)) for h_, t_ in zip(head, b2)]
                    return vjp_step(restore([_unphys(v, l) for v, l in zip(raw, likes)]), acc)
                return lax.fori_loop(0, LL, rv, acc)
            return lax.fori_loop(0, nsub, sub, acc)
        acc = lax.fori_loop(0, ns, seg, acc0)
        ctd, ctw, ctb = acc
        g_bc = [None] * len(bc)
        for i, v in zip(fb, ctb):
            g_bc[i] = v
        g_init = [None] * n
        for j, v in zip(act, ctd):
            if pc[j]:
                g_init[j] = v
        for j, v in zip(o_idx, ct_o):
            if pc[j]:
                g_init[j] = v
        for j, v in zip(w_idx, ctw):
            g_init[j] = v
        return [None] * len(cc), g_bc, g_init

    loop.defvjp(fwd, bwd, symbolic_zeros=True)
    return loop(list(cc), list(bc), list(init))


# ------------------------------------------------------------------------------------------- table reads
_IOTA = {}


def _iota(shape):
    k = tuple(shape)
    if k not in _IOTA:
        _IOTA[k] = np.arange(int(np.prod(k, dtype=np.int64)), dtype=np.int32).reshape(k)
    return _IOTA[k]


# tlin_p: the LINEAR part of a table read.  (y', tok_dot') = tlin_p(tok_dot, flat): y' = tok_dot[flat] (the read of
# the table perturbation carried by the token), tok_dot' = tok_dot.  Its transpose is ONE emit into the token
# cotangent, threaded linearly: ct_tok_in = emit(ct_tok_out, flat, ct_y).  It only ever exists inside a
# linearization (JAX transposes it away), so no backend sees it.

tlin_p = Primitive("tallymap_tread_lin")
tlin_p.multiple_results = True


def _tlin_impl(tok, flat):
    S = tok.shape[-1]
    f = jnp.where((flat >= 0) & (flat < S), flat, 0)
    return [jnp.where((flat >= 0) & (flat < S), tok[f], jnp.zeros((), tok.dtype)), tok]


tlin_p.def_impl(_tlin_impl)
tlin_p.def_abstract_eval(lambda tok, flat: [_SA(flat.shape, tok.dtype), _SA(tok.shape, tok.dtype)])
_mlir.register_lowering(tlin_p, _mlir.lower_fun(_tlin_impl, multiple_results=True))


def _tlin_transpose(cts, tok, flat):
    cy, ctok = cts
    if type(ctok) is _jad.Zero:
        ctok = jnp.zeros(ctok.aval.shape, ctok.aval.dtype)          # never on the threaded chain
    if type(cy) is not _jad.Zero:
        ctok = emit_p.bind(ctok, flat, cy.astype(ctok.dtype))
    return [ctok, None]


_jad.primitive_transposes[tlin_p] = _tlin_transpose


_FWD = [0]


class forward_tangents:
    """while tracing a FORWARD-mode program (jvp_prog: jvp / lin of any program, e.g. the jvp of an adjoint in second
    order), a table read's token tangent contributes nothing: the primal token is a fresh zero tally that is only
    passed through (its cotangent receives the emits, never the token itself), so its tangent is identically zero
    and tlin_p -- meaningful only inside a linearization that is transposed (vjp_prog) -- is not emitted."""

    def __enter__(self):
        _FWD[0] += 1

    def __exit__(self, *a):
        _FWD[0] -= 1


def tread(eqn, tok, ins):
    """a selection equation (gather / dynamic_slice / slice) reading table ins[0]: returns (outs, tok').
    A custom_jvp: primal = the equation itself (and tok passes through), tangent = the read of the table tangent
    (dense uses only, symbolic zero otherwise) + tlin_p of the token tangent.  The flat indices read are the same
    equation applied to an iota table (out-of-range fill -> index -1 -> dropped by emit)."""
    from ._jaxpr import bind
    p2 = dict(eqn.params)
    if "fill_value" in p2:
        p2["fill_value"] = -1
    iota = _iota(jnp.shape(ins[0]))

    @jax.custom_jvp
    def rd(tok, tab, *rest):
        return bind(eqn, [tab] + list(rest))[0], tok

    def rd_jvp(primals, tangents):
        tok, tab, *rest = primals
        dtok, dtab = tangents[0], tangents[1]
        y = bind(eqn, [tab] + list(rest))[0]
        if isinstance(dtok, SymbolicZero) or _FWD[0]:
            dy, dtok2 = jnp.zeros_like(y), (jnp.zeros_like(tok) if isinstance(dtok, SymbolicZero) else dtok)
        else:
            flat = eqn.primitive.bind(jnp.asarray(iota), *rest, **p2)
            dy, dtok2 = tlin_p.bind(dtok, flat)
        if not isinstance(dtab, SymbolicZero):
            dy = dy + bind(eqn, [dtab] + list(rest))[0]
        return (y, tok), (dy, dtok2)
    rd.defjvp(rd_jvp, symbolic_zeros=True)
    y, tok2 = rd(tok, *ins)
    return [y], tok2


# ------------------------------------------------------------------------------------------- the item VJP
def vjp_prog(p, pert_pi, pert_bc, opts=None):
    """ItemProgram of the item VJP: inputs [discard tallies][ct tallies of perturbed broadcast operands][discard table
    tokens] + [per-item operands, per-item result cotangents, (forward fingerprint)] + [broadcast operands, output
    cotangents]; outputs [same tallies] + [per-item operand cotangents] + [(replay mismatch flag, int32)].
    The fingerprint (ItemProgram.fp) is a hash of the final carries of the top-level loops, computed by the same jaxpr
    ops in the forward kernel and in this replay: a mismatch means the adjoint differentiated a DIFFERENT trajectory."""
    from ._api import ItemProgram, TSpec, _mk, _sds
    from ._rewrite import retrace
    if p.spawn is not None:          # (spawning programs go through the tree adjoint, _reverse_spawn.tree_vjp)
        raise NotImplementedError("tallymap: vjp_prog of a spawning program (internal error: use the tree adjoint)")
    o = rev_opts(opts)
    nT, nP, nB = p.nT, p.nP, p.nB
    ya = p.y_avals()
    tf = [i for i, s in enumerate(p.tallies) if s.kind == "sum" and isfloat(s.dtype)]
    yf = [j for j, a in enumerate(ya) if isfloat(a.dtype)]
    ip = [i for i in range(nP) if pert_pi[i]]
    ib = [i for i in range(nB) if pert_bc[i]]
    av = p.in_avals()
    tav, pav, bav = av[:nT], av[nT:nT + nP], av[nT + nP:]
    FP = int(p.fp is not None)
    size = lambda i: int(np.prod(bav[i].shape, dtype=np.int64))
    flat_aval = lambda i: jax.ShapeDtypeStruct((size(i),), bav[i].dtype)
    big = [i for i in ib if size(i) >= o.sparse_min]
    tables = {nT + nP + i: k for k, i in enumerate(big)}
    dense = set()
    if big:                                   # trace-time pass: which tables have non-read (dense) uses
        rep = {}
        jax.make_jaxpr(retrace(p.cj, rev=o, n_tally=nT, tables=tables, report=rep))(
            *[_sds(a) for a in av], *[flat_aval(i) for i in big])
        dense = {big[k] for k in rep["dense"]}
        if dict(opts or {}).get("_rev_nodense"):               # NEGATIVE CONTROL ONLY: drop the dense uses
            dense = set()
    big = [i for i in big if i not in dense]  # a table with any other use is differentiated densely for ALL uses
    tables = {nT + nP + i: k for k, i in enumerate(big)}
    dib = [i for i in ib if i not in big]
    fr = retrace(p.cj, rev=o, n_tally=nT, tables=tables)
    LAST.clear()
    LAST.update(opts=o, sparse=[(i, size(i)) for i in big], dense=[(i, size(i)) for i in dib],
                loops=[])

    def V(*a):
        t = list(a[:nT])
        k = nT
        tc = list(a[k:k + len(ib)])
        k += len(ib)
        tk = list(a[k:k + len(big)])
        k += len(big)
        pi, cy = list(a[k:k + nP]), list(a[k + nP:k + nP + len(yf)])
        k += nP + len(yf)
        fpf = a[k] if FP else None                  # the FORWARD kernel's replay fingerprint of this item
        k += FP
        bc, ct = list(a[k:k + nB]), list(a[k + nB:])

        def g(ps, bs, ks):
            pp, bb = list(pi), list(bc)
            for i, v in zip(ip, ps):
                pp[i] = v
            for i, v in zip(dib, bs):
                bb[i] = v
            return fr(*t, *pp, *bb, *ks)
        out, vjp = jax.vjp(g, [pi[i] for i in ip], [bc[i] for i in dib], tk)
        nout = len(out) - len(big)
        cts = []
        for j, x in enumerate(out[:nout]):
            if j < nT:          # (TX) discarded float tallies: zero cotangent (not float0)
                cts.append(ct[tf.index(j)] if j in tf else
                           jnp.zeros(x.shape, x.dtype) if isfloat(x.dtype) else np.zeros(x.shape, FLOAT0))
            else:
                q = j - nT
                cts.append(cy[yf.index(q)] if q in yf else np.zeros(x.shape, FLOAT0))
        cts += [tc[ib.index(i)] for i in big]
        dpi, dbc, dtk = vjp(cts)
        tc2 = []
        for q, i in enumerate(ib):
            c = dtk[big.index(i)] if i in big else tc[q]
            if i in dib:
                d = dbc[dib.index(i)]
                c = emit_p.bind(c, jnp.arange(d.size, dtype=jnp.int32), jnp.ravel(d).astype(c.dtype))
            tc2.append(c)
        mm = [(out[nT + p.fp] != fpf).astype(jnp.int32)] if FP else []      # replay != forward trajectory
        return t + tc2 + tk + list(dpi) + mm
    avals = (tav + [flat_aval(i) for i in ib] + [flat_aval(i) for i in big] + pav + [ya[j] for j in yf]
             + ([ya[p.fp]] if FP else []) + bav
             + [jax.ShapeDtypeStruct(tuple(p.tallies[i].lead) + (p.tallies[i].S,), p.tallies[i].dtype) for i in tf])
    cj = _mk(V, [_sds(x) for x in avals])
    tl = (tuple(dataclasses.replace(s, kind="discard") for s in p.tallies)
          + tuple(TSpec(f"ct{i}", size(i), bav[i].dtype) for i in ib)
          + tuple(TSpec(f"tok{i}", size(i), bav[i].dtype, kind="discard") for i in big))
    check_slots(cj, tl, p.tag + "+vjp")             # the cotangent tallies are threaded linearly (or raise)
    return ItemProgram(cj, tl, nP + len(yf) + FP, nB + len(tf), None, None, p.tag + "+vjp", where=p.where)


def check_slots(cj, tl, tag):
    """check_tallies with the program's SLOT LEGEND appended to a refusal: a reverse program's slots are the user's
    tallies, then one cotangent tally per perturbed broadcast operand, then one cotangent token per sparse table,
    which the user never wrote and cannot recognise from an index alone."""
    from ._rewrite import TallyError, check_tallies
    try:
        check_tallies(cj, [s.kind for s in tl], name=tag)
    except TallyError as e:
        legend = ", ".join(f"{k}={s.name}({s.kind},S={s.S})" for k, s in enumerate(tl))
        new = TallyError(f"{e}\n  the {len(tl)} tally slot(s) of {tag}, in order: {legend}")
        new.__dict__.update(e.__dict__)
        raise new from None
