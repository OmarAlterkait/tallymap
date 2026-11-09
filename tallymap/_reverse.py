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
    single: bool = False       # T3: the top-level loop's adjoint is ONE while loop (checkpointed_while_single); set
                               # by vjp_prog only for a program it folded (fold_post)
    probe: tuple = None        # T3: (probe carry, truncation carry) of the folded loop (fold_post)


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


def auto_checkpoints(sb, slots=3):
    """C for a loop state of sb bytes (see AUTO_BIG_STATE): AUTO_C_SMALL below AUTO_BIG_STATE bytes, else the largest
    C <= AUTO_C_MAX whose slots*C+1 states fit AUTO_CKPT_BYTES (checkpointed_while keeps 3C+1, the single-loop
    adjoint 2C+1), at least AUTO_C_MIN."""
    C = AUTO_C_SMALL if sb < AUTO_BIG_STATE else AUTO_C_MAX
    while C > AUTO_C_MIN and (slots * C + 1) * sb > AUTO_CKPT_BYTES:
        C //= 2
    return C


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


# ------------------------------------------------------------------------------------------- T3: single-loop adjoint
# The adjoint of checkpointed_while is a nest of loops (count pass; checkpoint pass; per segment: advance, fill,
# sweep), so the cuda backend runs it whole-item: every phase costs the warp its LONGEST item (lumen: warp max /
# mean trip count 2.3-3.8), and the step is instantiated five times (NVRTC time).  Here the adjoint of an item's ONE
# top-level loop is ONE while loop whose iteration is one MICRO-STEP of a per-lane state machine:
#
#   COUNT  x = c          store x in the head (slot min(k, C))       c = step(x)     until the loop's predicate fails
#   CKPT   x = c          store x as checkpoint j/m when m | j       c = step(x)     (only K > C)
#   ADV    x = c|ckpt     -                                          c = step(x)     (advance u*C into a segment)
#   FILL   x = c|ckpt     store x in tape slot v                     c = step(x)
#   SWEEP  x = tape[i]    -                                          acc = VJP(step at x)(acc)
#
# Every micro-step evaluates the SAME linearized step at its x (jax.vjp(..., has_aux=True): the forward with its
# residuals, the next state as aux), and only SWEEP lanes run the transpose (under a cond).  So lanes in different
# phases execute the same instructions, the step is instantiated once, and the adjoint program has exactly one
# top-level while: the cuda backend's step-granular refill applies.  Work and memory are those of checkpointed_while
# (head of C+1 states, C checkpoints; the head doubles as the tape; stride max(C, ceil(K/C))).  The states are
# advanced by the LINEARIZED step's primal, which is what jax.grad itself evaluates (a custom_jvp rule's primal
# output; JAX requires it to equal the function's).  The item's post-loop code is FOLDED into the loop as one
# extra iteration (fold_post), so the loop's outputs are the item's outputs and the custom_vjp's forward does no
# work (its outputs are dead in vjp_prog).
# MEASURED (lumen/perf ec953da against the same tree with the nested adjoint, L40S, 524,288 photons): grad / value
# 2.66 / 2.78 / 3.31x against 3.99 / 5.78 / 8.46x on SK / SBND / ANNIE; cold first grad 22 / 65 / 34 s against
# 115 / 484 / 190 s.  T3_MIN_EQNS: below it the per-micro-step overhead (bookkeeping, a state select, a buffer load
# and store, the cond) dominates: the toy walk (~30 equations) was 2-50x slower, lumen (16.5k-26k) 8-47% faster;
# the benchmarks/vs_jax problems have 35-190.  The threshold lies in the measured gap.
COUNT, CKPT, ADV, FILL, SWEEP, DONE = range(6)
T3_MIN_EQNS = 1000
SINGLE = "auto"                  # 'auto' (by single_blocker) | 'force' (whenever the shape allows) | 'off' (tests)
LAST_SINGLE = {}                 # what the last vjp_prog decided: used, why not (diagnostics()['reverse']['single'])


def _n_eqns(jaxpr):
    from ._jaxpr import subjaxprs
    return sum(1 + sum(_n_eqns(sj.jaxpr) for sj in subjaxprs(e)) for e in jaxpr.eqns)


def single_blocker(p):
    """-> None if the single-loop adjoint is used for program p, else why not."""
    from ._jaxpr import while_parts
    if SINGLE == "off":
        return "off"
    ws = [e for e in p.cj.jaxpr.eqns if e.primitive.name == "while"]
    if len(ws) != 1:
        return f"{len(ws)} top-level loops"
    n = _n_eqns(while_parts(ws[0]).body.jaxpr)
    if SINGLE != "force" and n < T3_MIN_EQNS:
        return f"a step of {n} equations (< T3_MIN_EQNS = {T3_MIN_EQNS})"
    return None


def _tally_slots(jaxpr, kinds):
    """var -> the item tally it descends from, for the TOP-LEVEL vars of an item jaxpr (check_tallies' walk)."""
    from jax.extend.core import Literal

    from ._rewrite import _walk_eqn
    slot, state, prod = {}, {}, {}
    for k, v in enumerate(jaxpr.invars[:len(kinds)]):
        slot[v], state[v] = k, "live"

    def use(v, what):
        if v in slot:
            state[v] = "used"
    lin = lambda s_: s_ is not None and kinds[s_] != "discard"
    tslots = lambda vs: [None if isinstance(v, Literal) else slot.get(v) for v in vs]
    for e in jaxpr.eqns:
        _walk_eqn(e, slot, state, prod, tslots, use, lin, kinds, "fold_post")
    return slot


def fold_post(p):
    """-> (an ItemProgram equal to p whose post-loop code runs INSIDE its one top-level while loop, as one extra
    iteration after the predicate fails, meta), or None when p is not of that shape (a loop in the pre / post code,
    key-typed results).  The folded program takes one extra per-item operand after p's, the PROBE (float32[1]),
    carried through the loop unchanged: checkpointed_while_single returns the item's truncation flag, as the COUNT
    phase computed it, as the probe's cotangent (the adjoint's primal outputs are dead, so vjp_prog reads it there).
    meta = (probe carry, truncation carry or -1)."""
    from jax.extend.core import Literal, jaxpr_as_fun

    from ._api import _mk, _sds
    from ._jaxpr import bind, contains, is_dropvar, lit, subjaxprs, while_parts
    from ._rewrite import check_tallies
    cj, nT = p.cj, p.nT
    jx = cj.jaxpr
    w = [i for i, e in enumerate(jx.eqns) if e.primitive.name == "while"][0]
    pre_e, we, post_e = jx.eqns[:w], jx.eqns[w], jx.eqns[w + 1:]
    if any(contains(sj.jaxpr, ("while",)) for e in pre_e + post_e for sj in subjaxprs(e)):
        return None
    if any(jax.dtypes.issubdtype(v.aval.dtype, jax.dtypes.prng_key) for v in jx.outvars[nT:]):
        return None
    kinds = [s.kind for s in p.tallies]
    slot = _tally_slots(jx, kinds)
    wp = while_parts(we)
    nc, nb = wp.nc, wp.nb
    wops = list(we.invars)
    nI = len(wops) - nc - nb
    loop_slot = {slot[v]: j for j, v in enumerate(wops[nc + nb:]) if not isinstance(v, Literal) and v in slot}
    cur = {k: jx.invars[k] for k in range(nT)}   # tally -> its var at loop entry
    for e in pre_e:
        for v in e.outvars:
            if v in slot and not is_dropvar(v):
                cur[slot[v]] = v
    extra = [k for k in range(nT) if k not in loop_slot]
    nE = len(extra)
    consts = dict(zip(jx.constvars, cj.consts))
    yav = [v.aval for v in jx.outvars[nT:]]

    def ev(eqns, env):
        rd = lambda v: lit(v) if isinstance(v, Literal) else env[v]
        for e in eqns:
            for v, x in zip(e.outvars, bind(e, [rd(v) for v in e.invars])):
                env[v] = x
        return rd
    nPi = nT + p.nP                              # position of the probe among the folded item's operands
    meta = (nI + nE + len(yav), nI + nE + p.trunc if p.trunc is not None else -1)
    cond_f, body_f = jaxpr_as_fun(wp.cond), jaxpr_as_fun(wp.body)

    def item2(*a):
        a = list(a)
        probe = a.pop(nPi)
        env = dict(consts)
        env.update(zip(jx.invars, a))
        rd = ev(pre_e, env)
        ops = [rd(v) for v in wops]
        cc, bc, init = ops[:nc], ops[nc:nc + nb], ops[nc + nb:]

        def body2(c):
            ci, ce, cy = list(c[:nI]), list(c[nI:nI + nE]), list(c[nI + nE:-2])
            go = cond_f(*cc, *ci)[0]

            def run_post(ci, ce, cy, pz):
                env2 = dict(env)
                for v, x in zip(we.outvars, ci):
                    if not is_dropvar(v):
                        env2[v] = x
                for k, x in zip(extra, ce):
                    env2[cur[k]] = x
                rd2 = ev(post_e, env2)
                outs = [rd2(v) for v in jx.outvars]
                ci2, ce2 = list(ci), list(ce)
                for k in range(nT):
                    if k in loop_slot:
                        ci2[loop_slot[k]] = outs[k]
                    else:
                        ce2[extra.index(k)] = outs[k]
                ys = [jnp.asarray(o, av.dtype) for o, av in zip(outs[nT:], yav)]
                # the post iteration WRITES the probe (the truncation flag, else 0): a carry the body passed through
                # unchanged would be hoisted out of the loop as a constant by JAX >= 0.7 (loop-invariant carry
                # forwarding), and the probe would no longer be the carry the adjoint returns its cotangent for
                pz2 = (jnp.reshape(ys[p.trunc].astype(jnp.float32), (1,)) if p.trunc is not None else
                       jnp.zeros_like(pz))
                return ci2, ce2, ys, pz2
            ci2, ce2, cy2, pz2 = lax.cond(go, lambda ci, ce, cy, pz: (list(body_f(*bc, *ci)), list(ce), list(cy), pz),
                                          run_post, ci, ce, cy, c[-2])
            return list(ci2) + list(ce2) + list(cy2) + [pz2, jnp.logical_not(go)]
        y0 = [jnp.zeros(av.shape, av.dtype) for av in yav]
        res = lax.while_loop(lambda c: jnp.logical_not(c[-1]), body2,
                             list(init) + [env[cur[k]] for k in extra] + y0 + [probe, jnp.bool_(False)])
        ci, ce, cy = res[:nI], res[nI:nI + nE], res[nI + nE:-2]
        return [ci[loop_slot[k]] if k in loop_slot else ce[extra.index(k)] for k in range(nT)] + list(cy)
    avs = [_sds(v.aval) for v in jx.invars]
    avs.insert(nPi, jax.ShapeDtypeStruct((1,), jnp.float32))
    cj2 = _mk(item2, avs)
    w2 = [e for e in cj2.jaxpr.eqns if e.primitive.name == "while"]
    if len(w2) != 1 or len(w2[0].outvars) != meta[0] + 2:      # (meta indexes the folded loop's carries)
        raise AssertionError(f"tallymap: fold_post built {len(w2)} top-level loops / "
                             f"{[len(e.outvars) for e in w2]} carries, want 1 / {meta[0] + 2} (internal error)")
    check_tallies(cj2, kinds, name=f"fold_post({p.where})")
    return dataclasses.replace(p, cj=cj2, nP=p.nP + 1, tag=p.tag + "+fold"), meta


def checkpointed_while_single(cfn, bfn, cc, bc, init, kinds, opts=None, bflags=None, live=None):
    """checkpointed_while whose adjoint is ONE while loop (T3, see the section comment).  Its forward returns the
    INITIAL carry in place of the loop's outputs: only for the top-level loop of a fold_post program inside vjp_prog,
    whose primal outputs nothing reads."""
    o = rev_opts(opts)
    btal = list(bflags) if bflags is not None else [False] * len(bc)
    kinds = ["o" if k is True else (k or None) for k in kinds]
    n = len(init)
    s_idx = [j for j in range(n) if kinds[j] is None]
    f_idx = [j for j in s_idx if isfloat(_dt(init[j]))]
    o_idx = [j for j in range(n) if kinds[j] == "o" and isfloat(_dt(init[j]))]
    w_idx = [j for j in range(n) if kinds[j] == "w"]
    l_idx = [j for j in s_idx if live is None or live[j]]
    pidx, tr_c = o.probe
    if tr_c >= 0 and tr_c not in l_idx:
        raise ValueError("tallymap: the single-loop adjoint cannot see the truncation carry (internal error)")
    tpos = l_idx.index(tr_c) if tr_c >= 0 else None
    sb = state_bytes([init[j] for j in l_idx])
    C = int(o.checkpoints) if o.checkpoints else auto_checkpoints(sb, slots=2)

    @jax.custom_vjp
    def loop(cc, bc, init):
        return lax.while_loop(lambda c: cfn(cc, c), lambda c: bfn(bc, c), init)

    def fwd(cc, bc, init):
        pb = tuple(bool(x.perturbed) and not t for x, t in zip(bc, btal))
        pc = tuple(bool(x.perturbed) for x in init)
        cc, bc, init = [x.value for x in cc], [x.value for x in bc], [x.value for x in init]
        act = tuple(j for j in _active(bfn, bc, init, pb, pc, f_idx, w_idx) if j != pidx)
        LAST.setdefault("loops", []).append(dict(checkpoints=C, auto=not o.checkpoints, n_state=len(s_idx),
                                                 n_live=len(l_idx), n_float=len(f_idx), n_active=len(act),
                                                 n_tok=len(w_idx), state_bytes=sb, ckpt_bytes=(2 * C + 1) * sb,
                                                 single=True))
        return list(init), (cc, bc, init, _Static((pb, pc, act)))   # (dead outputs: see the docstring)

    def bwd(res, ct):
        cc, bc, init, st = res
        pb, pc, act = st.v
        act = list(act)
        fb = [i for i, x in enumerate(bc) if pb[i] and isfloat(_dt(x))]

        def inst(c, x):
            return jnp.zeros(jnp.shape(x), _dt(x)) if c is None or isinstance(c, SymbolicZero) else c
        ct = list(ct)
        # the tally cotangents are built INSIDE the micro-step: a value computed before the machine loop and read in
        # it is per-lane state of the step split, so a symbolic-zero tally cotangent instantiated here would be a
        # tally-sized local array per lane (lumen SK: (88768,), "too large")
        ct_o_of = lambda: [inst(ct[j], init[j]) for j in o_idx]
        acc0 = ([inst(ct[j], init[j]) for j in act], [inst(ct[j], init[j]) for j in w_idx],
                [jnp.zeros_like(bc[i]) for i in fb])
        likes = [init[j] for j in l_idx]

        def restore(vals):
            c = list(init)
            for j, v in zip(l_idx, vals):
                c[j] = v
            return c

        def lin_step(x):
            """the linearized step at state x: (next state, vjp of the differentiated outputs)"""
            c = restore(x)

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
                return ([out[j] for j in act], [out[j] for j in o_idx], [out[j] for j in w_idx]), \
                    [out[j] for j in l_idx]
            _, vjp, y = jax.vjp(f, [c[j] for j in act], [bc[i] for i in fb], [c[j] for j in o_idx],
                                [c[j] for j in w_idx], has_aux=True)
            return y, vjp

        I32 = jnp.int32
        cdiv = lambda a, b: (a + b - 1) // b
        sel = lambda p_, a, b: jnp.where(p_, a, b)

        def body(stt):
            ph, k, K, m, ns, j, s, L, u, na, v, LL, i, ldc, c, B, acc, fin = stt
            sweep = ph == SWEEP
            ld = sweep | ldc
            x = [jnp.where(ld, a, b) for a, b in zip(_buf_get(B, sel(sweep, i, C + 1 + s), likes), c)]
            y, vjp = lin_step(x)

            def dv(a):
                ctd, ctw, ctb = a
                dc, db, _, dw = vjp((list(ctd), ct_o_of(), list(ctw)))
                return list(dc), list(dw), [p_ + q_ for p_, q_ in zip(ctb, db)]
            acc = lax.cond(sweep, dv, lambda a: (list(a[0]), list(a[1]), list(a[2])), acc)
            # the store of x: head slot (COUNT, FILL), checkpoint slot (CKPT), else the scratch slot C
            si = sel(ph == COUNT, jnp.minimum(k, C),
                     sel(ph == CKPT, sel(j % m == 0, C + 1 + j // m, C), sel(ph == FILL, v, C)))
            B = _buf_set(B, si, x)
            # ---- transitions (scalar)
            go = cfn(cc, restore(y))
            k1 = k + 1
            cnt_end = (ph == COUNT) & jnp.logical_not(go)
            if tpos is not None:                 # the probe: the truncation flag COUNT computed
                fin = jnp.where(cnt_end, jnp.reshape(y[tpos].astype(jnp.float32), (1,)), fin)
            K2 = sel(cnt_end, k1, K)
            short = K2 <= C
            m2 = sel(cnt_end, jnp.maximum(C, cdiv(K2, C)), m)
            ns2 = sel(cnt_end, cdiv(K2, m2), ns)
            j1 = j + 1
            ck_end = (ph == CKPT) & (j1 > (ns - 1) * m)
            na1 = na - 1
            ad_end = (ph == ADV) & (na1 == 0)
            v1 = v + 1
            fi_end = (ph == FILL) & (v1 == LL)
            i1 = i - 1
            sw_end = sweep & (i1 < 0)
            # begin a chunk: after the checkpoint pass (segment ns-1, its last chunk), or after a sweep (the previous
            # chunk of this segment, or the last chunk of the previous segment)
            seg_new = ck_end | (sw_end & (u == 0) & (s > 0))
            s_b = sel(ck_end, ns - 1, sel(sw_end & (u == 0), s - 1, s))
            L_b = sel(seg_new, jnp.minimum(m, K - s_b * m), L)
            u_b = sel(seg_new, cdiv(L_b, C) - 1, u - 1)
            begin = ck_end | (sw_end & ((u > 0) | (s > 0)))
            done = sw_end & (u == 0) & (s == 0)
            LL_b = jnp.minimum(C, L_b - u_b * C)
            na_b = u_b * C
            ph2 = sel(ph == COUNT, sel(go, COUNT, sel(short, SWEEP, CKPT)), ph)
            ph2 = sel(begin, sel(na_b > 0, ADV, FILL), ph2)
            ph2 = sel(ad_end, FILL, ph2)
            ph2 = sel(fi_end, SWEEP, ph2)
            ph2 = sel(done, DONE, ph2)
            i2 = sel(cnt_end & short, K2 - 1, sel(fi_end, LL - 1, sel(sweep, i1, i)))
            # the next primal state: y after a primal micro-step; the initial state when the checkpoint pass starts
            # (COUNT -> CKPT); unchanged by a sweep (and replaced by a checkpoint load after `begin`)
            c2 = [sel(cnt_end, a0, sel(sweep, a, b)) for a0, a, b in zip(likes, c, y)]
            return (ph2, sel(ph == COUNT, k1, k), K2, m2, ns2, sel(cnt_end, 0, sel(ph == CKPT, j1, j)),
                    sel(begin, s_b, sel(cnt_end, 0, s)), sel(begin, L_b, L), sel(begin, u_b, sel(cnt_end, 0, u)),
                    sel(begin, na_b, sel(ph == ADV, na1, na)), sel(begin | ad_end, 0, sel(ph == FILL, v1, v)),
                    sel(begin, LL_b, LL), i2, begin, c2, B, acc, fin)

        z = I32(0)
        ph0 = sel(cfn(cc, list(init)), I32(COUNT), I32(DONE))
        st0 = (ph0, z, z, I32(C), I32(1), z, z, z, z, z, z, z, z, jnp.bool_(False), list(likes),
               _buf_new(2 * C + 1, likes), acc0, jnp.zeros((1,), jnp.float32))
        out = lax.while_loop(lambda st_: st_[0] != DONE, body, st0)
        ctd, ctw, ctb = out[-2]
        g_bc = [None] * len(bc)
        for i, v in zip(fb, ctb):
            g_bc[i] = v
        g_init = [None] * n
        for j, v in zip(act, ctd):
            if pc[j]:
                g_init[j] = v
        for j, v in zip(o_idx, ct_o_of()):
            if pc[j]:
                g_init[j] = v
        for j, v in zip(w_idx, ctw):
            g_init[j] = v
        g_init[pidx] = out[-1]          # the probe: what COUNT saw (fold_post)
        return [None] * len(cc), g_bc, g_init

    loop.defvjp(fwd, bwd, symbolic_zeros=True)
    return loop(list(cc), list(bc), list(init))


def _dce(cj):
    """ClosedJaxpr -> the same with its dead code removed (all inputs and outputs kept), INCLUDING inside while
    bodies and cond branches (JAX 0.5.3's dce has no rule for while).  T3: the fold's cond carries every tally, and
    the transpose of a cond returns each tally input's pass-through cotangent as a cond OUTPUT, unused but
    materialised per item by the cuda emitter (lumen SK: an (88768,) local array, "too large")."""
    from jax._src.interpreters import partial_eval as pe
    from jax.extend.core import ClosedJaxpr

    def closed(c):
        # consts -> leading invars (kept: instantiate=True), DCE, then the same leading invars back as constvars with
        # the same consts.  Only convert_constvars_jaxpr / dce_jaxpr / replace / ClosedJaxpr(jaxpr, consts), which
        # JAX 0.5.3 (separate ClosedJaxpr) and 0.11 (ClosedJaxpr = Jaxpr, consts attached) both have;
        # convert_invars_to_constvars is gone in 0.11.
        nc = len(c.consts)
        jd = deep(pe.convert_constvars_jaxpr(c.jaxpr))
        return ClosedJaxpr(jd.replace(constvars=list(jd.invars[:nc]), invars=list(jd.invars[nc:])), list(c.consts))

    def deep(jaxpr):
        eqns = []
        for e in jaxpr.eqns:
            if e.primitive.name == "while":
                e = e.replace(params=dict(e.params, body_jaxpr=closed(e.params["body_jaxpr"]),
                                          cond_jaxpr=closed(e.params["cond_jaxpr"])))
            elif e.primitive.name == "cond":
                e = e.replace(params=dict(e.params, branches=tuple(closed(b) for b in e.params["branches"])))
            eqns.append(e)
        return pe.dce_jaxpr(jaxpr.replace(eqns=eqns), [True] * len(jaxpr.outvars), instantiate=True)[0]
    return closed(cj)


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
    tokens] + [per-item operands, per-item result cotangents] + [broadcast operands, output cotangents]; outputs
    [same tallies] + [per-item operand cotangents] + [(the replay's truncation flag, bool: max_steps only)].
    The truncation flag is the adjoint's OWN (its replay runs the same capped loops as the forward): _vjp_call
    poisons the cotangents with it, so a gradient needs no forward launch."""
    from ._api import ItemProgram, TSpec, _mk, _sds
    from ._rewrite import retrace
    if p.spawn is not None:          # (spawning programs go through the tree adjoint, _reverse_spawn.tree_vjp)
        raise NotImplementedError("tallymap: vjp_prog of a spawning program (internal error: use the tree adjoint)")
    o = rev_opts(opts)
    why = single_blocker(p)
    r = fold_post(p) if why is None else None
    if why is None and r is None:
        why = "a loop in the pre / post code, or key-typed results"
    LAST_SINGLE.clear()
    LAST_SINGLE.update(used=r is not None, why=why)
    pc_ = p
    if r is not None:                # T3: the adjoint retraces the folded program
        pc_, meta = r
        o = dataclasses.replace(o, single=True, probe=meta)
    sh = int(r is not None)          # (the folded program's probe operand follows the per-item operands)
    nT, nP, nB = p.nT, p.nP, p.nB
    ya = p.y_avals()
    tf = [i for i, s in enumerate(p.tallies) if s.kind == "sum" and isfloat(s.dtype)]
    yf = [j for j, a in enumerate(ya) if isfloat(a.dtype)]
    ip = [i for i in range(nP) if pert_pi[i]]
    ib = [i for i in range(nB) if pert_bc[i]]
    av = p.in_avals()
    tav, pav, bav = av[:nT], av[nT:nT + nP], av[nT + nP:]
    size = lambda i: int(np.prod(bav[i].shape, dtype=np.int64))
    flat_aval = lambda i: jax.ShapeDtypeStruct((size(i),), bav[i].dtype)
    big = [i for i in ib if size(i) >= o.sparse_min]
    tables = {nT + nP + sh + i: k for k, i in enumerate(big)}
    dense = set()
    if big:                                   # trace-time pass: which tables have non-read (dense) uses
        rep = {}
        jax.make_jaxpr(retrace(pc_.cj, rev=o, n_tally=nT, tables=tables, report=rep))(
            *[_sds(a) for a in pc_.in_avals()], *[flat_aval(i) for i in big])
        dense = {big[k] for k in rep["dense"]}
        if dict(opts or {}).get("_rev_nodense"):               # NEGATIVE CONTROL ONLY: drop the dense uses
            dense = set()
    big = [i for i in big if i not in dense]  # a table with any other use is differentiated densely for ALL uses
    tables = {nT + nP + sh + i: k for k, i in enumerate(big)}
    dib = [i for i in ib if i not in big]
    fr = retrace(pc_.cj, rev=o, n_tally=nT, tables=tables)
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
        bc, ct = list(a[k:k + nB]), list(a[k + nB:])

        def g(ps, bs, ks, *zz):
            pp, bb = list(pi), list(bc)
            for i, v in zip(ip, ps):
                pp[i] = v
            for i, v in zip(dib, bs):
                bb[i] = v
            return fr(*t, *pp, *zz, *bb, *ks)
        zz = [jnp.zeros((1,), jnp.float32)] if sh else []          # T3: the probe (fold_post)
        out, vjp = jax.vjp(g, [pi[i] for i in ip], [bc[i] for i in dib], tk, *zz)
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
        dpi, dbc, dtk, *dz = vjp(cts)
        tc2 = []
        for q, i in enumerate(ib):
            c = dtk[big.index(i)] if i in big else tc[q]
            if i in dib:
                d = dbc[dib.index(i)]
                c = emit_p.bind(c, jnp.arange(d.size, dtype=jnp.int32), jnp.ravel(d).astype(c.dtype))
            tc2.append(c)
        tr = []
        if p.trunc is not None:  # the replay's truncation flag (T3: from the COUNT phase, through the probe)
            tr = [dz[0][0] > 0.5 if sh else jnp.asarray(out[nT + p.trunc], jnp.bool_)]
        return t + tc2 + tk + list(dpi) + tr
    avals = (tav + [flat_aval(i) for i in ib] + [flat_aval(i) for i in big] + pav + [ya[j] for j in yf] + bav
             + [jax.ShapeDtypeStruct(tuple(p.tallies[i].lead) + (p.tallies[i].S,), p.tallies[i].dtype) for i in tf])
    cj = _mk(V, [_sds(x) for x in avals])
    if sh:
        cj = _dce(cj)
    tl = (tuple(dataclasses.replace(s, kind="discard") for s in p.tallies)
          + tuple(TSpec(f"ct{i}", size(i), bav[i].dtype) for i in ib)
          + tuple(TSpec(f"tok{i}", size(i), bav[i].dtype, kind="discard") for i in big))
    check_slots(cj, tl, p.tag + "+vjp")             # the cotangent tallies are threaded linearly (or raise)
    return ItemProgram(cj, tl, nP + len(yf), nB + len(tf), len(ip) if p.trunc is not None else None, None,
                       p.tag + ("+fold" if sh else "") + "+vjp", where=p.where)


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
