"""Jaxpr utilities shared by the core and the backends.

  (jaxpr structure -- while / cond / scan / call-like params -- is read through tallymap/_jaxpr.py)
  check_tallies(cj, kinds)                 tallies are write-only and used LINEARLY (virtual == dense meaning)
  retrace(cj, cap=, rev=)                  re-trace an item jaxpr: cap top-level while loops at max_steps (adds a
                                           `truncated` output), or (rev=True) replace every while loop by a
                                           reverse-differentiable replay loop (_reverse.checkpointed_while)
"""
import jax.numpy as jnp
from jax import lax
from jax.extend.core import Literal

from ._jaxpr import (
    bind,
    call_jaxpr,
    cond_branches,
    contains,
    is_dropvar,
    lit,
    located,
    scan_parts,
    source_summary,
    subjaxprs,
    while_parts,
    with_source,
)

TALLYMAP_PRIMS = ("tallymap_emit", "tallymap_spawn")


class TallyError(TypeError):
    """A Tally is used in a way the virtual (write-through) implementation cannot honour: read, forked, dropped,
    closed over by a loop, or used differently by two cond branches.  Raised at trace time."""


def isfloat(dt):
    return jnp.issubdtype(dt, jnp.inexact)


# ---------------------------------------------------------------------------------------- tally linearity
def check_tallies(cj, kinds, name="item function"):
    """kinds[i] in {'sum','spawn','discard'} for the first len(kinds) invars (the tally inputs).  name: the item
    function (error messages name it, plus the source location of the offending equation).

    Rules (they make the virtual, write-through implementation equal the dense meaning of emit):
      * a tally value may only be consumed by emit/spawn, passed through control flow, or returned;
      * a non-discard tally value is consumed EXACTLY once (no forks, no dropped emit results);
      * loop tallies must be carries (not consts / scanned inputs); cond branches must agree;
      * the item's tally outputs are the tally inputs' successors, in order.
    """
    n = len(kinds)
    jaxpr = cj.jaxpr
    outs, used = _walk(jaxpr, list(range(n)) + [None] * (len(jaxpr.invars) - n), kinds, name)
    if outs[:n] != list(range(n)) or any(o is not None for o in outs[n:]):
        raise TallyError(f"tallymap: the item function must return its Tally (the successor of the one it was given) "
                         f"and must not return tallies inside the per-item result (item function {name})")


def _walk(jaxpr, in_slots, kinds, where):
    try:
        return _walk1(jaxpr, in_slots, kinds, where)
    except TallyError as e:
        eqn = getattr(e, "_tm_eqn", None)
        if eqn is None or getattr(e, "_tm_located", False):
            raise
        raise located(e, eqn) from None


def _walk1(jaxpr, in_slots, kinds, where):
    slot, state, prod = {}, {}, {}

    def lin(s):
        return s is not None and kinds[s] != "discard"

    for v, s in zip(jaxpr.invars, in_slots):
        if s is not None:
            slot[v] = s
            state[v] = "live"

    def use(v, what):
        if v not in slot:
            return
        if lin(slot[v]) and state[v] == "used":
            raise TallyError(f"tallymap: a Tally value is used twice ({what} in {where}): tallies must be threaded "
                             "linearly (use the Tally returned by add/spawn, exactly once)")
        state[v] = "used"

    def tslots(vs):
        return [None if isinstance(v, Literal) else slot.get(v) for v in vs]

    for eqn in jaxpr.eqns:
        try:
            _walk_eqn(eqn, slot, state, prod, tslots, use, lin, kinds, where)
        except TallyError as e:
            if not hasattr(e, "_tm_eqn"):
                e._tm_eqn = eqn
            raise
    outs = tslots(jaxpr.outvars)
    for v, s in zip(jaxpr.outvars, outs):
        if s is not None:
            use(v, "returned")
    for v, st in state.items():
        if st == "live" and lin(slot[v]) and v not in jaxpr.invars:
            e = TallyError(f"tallymap: the Tally returned by t.add / t.spawn is discarded (in {where}); thread it "
                           "through (a dropped Tally would silently disagree with the dense meaning)")
            if v in prod:
                e._tm_eqn = prod[v]
            raise e
    used = [state.get(v) == "used" for v, s in zip(jaxpr.invars, in_slots) if s is not None]
    return outs, used


def _walk_eqn(eqn, slot, state, prod, tslots, use, lin, kinds, where):
    name = eqn.primitive.name
    ins = tslots(eqn.invars)
    for v in eqn.outvars:
        prod[v] = eqn
    if name in TALLYMAP_PRIMS:
        if ins[0] is None:
            raise TallyError(f"tallymap: {name} on a value that is not this tallymap.vmap's Tally (in {where}); "
                             "nested tallymap.vmap tallies cannot be written from an inner item function")
        if any(s is not None for s in ins[1:]):
            raise TallyError(f"tallymap: a Tally is used as a t.add value / spawned input (in {where})")
        want = "spawn" if name == "tallymap_spawn" else ("sum", "discard")
        if kinds[ins[0]] not in want:
            raise TallyError(f"tallymap: {name} applied to the wrong kind of tally leaf (in {where})")
        use(eqn.invars[0], name)
        slot[eqn.outvars[0]] = ins[0]
        state[eqn.outvars[0]] = "live"
        return
    subs = subjaxprs(eqn)
    if all(s is None for s in ins) and not any(contains(s.jaxpr, TALLYMAP_PRIMS) for s in subs):
        return
    sub = call_jaxpr(eqn)
    if sub is not None:
        try:
            o, u = _walk(sub.jaxpr, ins, kinds, f"{name} in {where}")
        except TallyError as e:
            raise _align(e, eqn, sub, ins) from None
        _consume(eqn.invars, ins, u, use, name)
        _bind_out(eqn.outvars, o, slot, state)
    elif name == "while":
        w = while_parts(eqn)
        nc, nb = w.nc, w.nb
        cs, bs, init = ins[:nc], ins[nc:nc + nb], ins[nc + nb:]
        oc, uc = _walk(w.cond.jaxpr, cs + init, kinds, f"while predicate in {where}")
        ob, ub = _walk(w.body.jaxpr, bs + init, kinds, f"while body in {where}")
        lc, lb = [lin(s) for s in cs if s is not None], [lin(s) for s in bs if s is not None]
        if any(u and l for u, l in zip(uc, lc)) or any(u and l for u, l in zip(ub, lb)):   # (discard tallies may be)
            raise TallyError(f"tallymap: a Tally is closed over (not carried) by a while_loop and consumed inside "
                             f"(in {where}); make it a loop carry and return the emit result")
        ob = ob[:len(init)]
        if ob != init:
            raise TallyError(f"tallymap: a while_loop body must return each Tally carry's successor in the same "
                             f"carry position (in {where})")
        _consume(eqn.invars[nc + nb:], init, [True] * len(init), use, name)
        _bind_out(eqn.outvars, init, slot, state)
    elif name == "scan":
        sc = scan_parts(eqn)
        nc, nk = sc.nconsts, sc.ncarry
        cs, init, xs = ins[:nc], ins[nc:nc + nk], ins[nc + nk:]
        o, u = _walk(sc.body.jaxpr, ins, kinds, f"scan body in {where}")
        lc = [lin(s) for s in cs if s is not None]
        if any(a and l for a, l in zip(u, lc)) or any(s is not None and lin(s) for s in xs):
            raise TallyError(f"tallymap: a Tally is a scan constant / scanned input consumed inside (in {where}); "
                             "make it a carry")
        # (an inert DISCARD tally may be stacked: a reverse-mode residual, e.g. of a loop inside a cond inside a scan)
        if o[:nk] != init or any(lin(s) for s in o[nk:]):
            raise TallyError(f"tallymap: a scan body must return each Tally carry's successor as that carry "
                             f"(and not as a stacked output) (in {where})")
        _consume(eqn.invars[nc:nc + nk], init, [True] * nk, use, name)
        _bind_out(eqn.outvars, o, slot, state)
    elif name == "cond":
        ops = ins[1:]
        if ins[0] is not None:
            raise TallyError(f"tallymap: a Tally used as a cond index (in {where})")
        res = [_walk(b.jaxpr, ops, kinds, f"cond branch {i} in {where}") for i, b in enumerate(cond_branches(eqn))]
        tops = [s for s in ops if s is not None]
        sig = lambda r: ([s if lin(s) else None for s in r[0]], [u for u, s in zip(r[1], tops) if lin(s)])
        if any(sig(r) != sig(res[0]) for r in res[1:]):
            raise TallyError(f"tallymap: cond/switch branches must use each Tally the same way (all t.add and "
                             f"return it, or none touch it) (in {where})")
        _consume(eqn.invars[1:], ops, res[0][1], use, name)
        merged = [next((r[0][j] for r in res if r[0][j] is not None), None) for j in range(len(eqn.outvars))]
        _bind_out(eqn.outvars, merged, slot, state)      # (an inert discard tally may meet a filler value)
    else:
        if any(s is not None for s in ins):
            raise TallyError(f"tallymap: primitive {name!r} reads a Tally (in {where}); tallies are write-only "
                             "(emit with .add, return it, thread it through control flow); "
                             + _which(eqn, ins, kinds, prod))
        raise TallyError(f"tallymap: t.add / t.spawn inside an unsupported higher-order primitive {name!r} "
                         f"(in {where})")


def _which(eqn, ins, kinds, prod):
    """which operand of a refused equation is a tally, which slot and kind it is, and what produced it: on a program
    with many tallies and identically shaped arrays (a cotangent token has the shape of its table) the primitive's
    name alone does not identify the defect."""
    parts = []
    for i, s in enumerate(ins):
        if s is not None:
            v, pe = eqn.invars[i], prod.get(eqn.invars[i])
            src = f", produced by {pe.primitive.name!r} at {source_summary(pe)}" if pe is not None else ""
            parts.append(f"operand {i} ({getattr(v, 'aval', None)}) is tally slot {s} (kind {kinds[s]}){src}")
    return "the tally operand(s): " + "; ".join(parts)


def _align(e, eqn, sub, ins):
    """a refusal raised inside a call-like sub-jaxpr, with the position alignment appended once (outermost call):
    tally slots are carried in BY POSITION, so the equation's operands against the sub-jaxpr's invars are what
    says whether a slot was assigned to the right operand."""
    if getattr(e, "_tm_align", False):
        return e
    n_e, n_s = len(eqn.invars), len(sub.jaxpr.invars)
    extra = "; ".join(f"{k}={eqn.params[k]!r}" for k in ("num_consts", "symbolic_zeros") if k in eqn.params)
    ops = ", ".join(f"[{i} {getattr(v, 'aval', None)} {'slot %d' % ins[i] if ins[i] is not None else '-'}]"
                    for i, v in enumerate(eqn.invars[:40]))
    new = TallyError(f"{e}\n  in {eqn.primitive.name}: {n_e} equation operand(s) vs {n_s} sub-jaxpr invar(s)"
                     + (f" ({extra})" if extra else "")
                     + (" -- THESE DO NOT MATCH, so the tally slots are carried in by the wrong position"
                        if n_e != n_s else "")
                     + f"\n  operand(s) [index aval slot]: {ops}"
                     + ("" if n_e <= 40 else f", ... ({n_e} in all)")
                     + "\n  sub-jaxpr invar aval(s): "
                     + ", ".join(str(getattr(v, "aval", None)) for v in list(sub.jaxpr.invars)[:40]))
    new.__dict__.update(e.__dict__)
    new._tm_align = True
    return new


def _consume(vars_, slots, used, use, name):
    it = iter(used)
    for v, s in zip(vars_, slots):
        if s is not None and next(it):
            use(v, name)


def _bind_out(outvars, slots, slot, state):
    for v, s in zip(outvars, slots):
        if s is not None and not is_dropvar(v):
            slot[v] = s
            state[v] = "live"



# ---------------------------------------------------------------------------------------- re-tracing
SELECT = ("gather", "dynamic_slice", "slice")          # reads that can get a sparse (emitted) cotangent
_CUSTOM = ("custom_jvp_call", "custom_vjp_call")


def emit_p_bind(t, i, v):
    from ._tally import emit_p
    return emit_p.bind(t, i, v)


def retrace(cj, *, cap=None, rev=False, n_tally=0, emit=None, tables=None, report=None, spawn=None, gate=None):
    """-> python function f(*invars) evaluating cj with (cap) every top-level while loop capped at `cap`
    iterations (f returns its outputs + [truncated:bool]), or (rev: True or _reverse.RevOpts) every while loop replaced
    by a reverse-differentiable loop (_reverse.checkpointed_while).  The first n_tally invars are tallies (rev: their
    carries get a pass-through cotangent instead of a dense one).  emit: f(tally, idx, val) -> tally' replaces every
    emit (the tree adjoint records its emits through it).
    spawn: f(token, when, *child) -> token' replaces every spawn (the tree adjoint, _reverse_spawn.py).  With hooks a
    tally value may be a python TUPLE (e.g. (array, hash)): equations it flows into are interpreted (never bound),
    and lax control flow carries it as a pytree.  Only for rev=False (the hooked program is re-traced to a flat jaxpr).
    gate: a traced bool; every top-level while loop runs only while `gate` holds (an inactive item skips its loops
    without a cond around the item; the caller masks its effects).
    tables (rev only): {invar index: token index}.  Reads (gather/dynamic_slice/slice) of these operands become
    _reverse.tread calls threading a cotangent token; f then takes the tokens as extra trailing arguments and returns
    them as extra trailing outputs.  report: a dict that receives 'dense' = token indices of tables that also have
    other uses (they need the dense cotangent for those)."""
    tables = dict(tables or {})
    nt = len(tables)

    def f(*args):
        args = list(args)
        a, toks = (args[:len(args) - nt], args[len(args) - nt:]) if nt else (args, [])
        ctx = {"trunc": jnp.bool_(False), "emit": emit, "spawn": spawn, "gate": gate, "toks": list(toks),
               "tread": nt > 0, "dense": set()}
        flags = [True] * n_tally + [False] * (len(a) - n_tally)
        tags = [tables.get(i) for i in range(len(a))]
        outs, _ = _interp(cj.jaxpr, cj.consts, a, flags, ctx, cap, rev, top=True, tags=tags)
        if report is not None:
            report["dense"] = set(ctx["dense"])
        return list(outs) + ([ctx["trunc"]] if cap is not None else []) + list(ctx["toks"])
    return f


def _needs(jaxpr, rev, top, cap, ctx):
    return ((rev or (cap is not None and top)) and contains(jaxpr, ("while",))) or \
        (ctx["emit"] is not None and contains(jaxpr, ("tallymap_emit",))) or \
        (ctx.get("spawn") is not None and contains(jaxpr, ("tallymap_spawn",))) or \
        (ctx.get("gate") is not None and top and contains(jaxpr, ("while",)))


def _set_tok(ctx, i, v):
    toks = list(ctx["toks"])
    toks[i] = v
    ctx["toks"] = toks


def _interp(jaxpr, consts, args, flags, ctx, cap, rev, top, tags=None):
    env, tf, tg = {}, {}, {}

    def read(v):
        return lit(v) if isinstance(v, Literal) else env[v]

    for v, x in zip(jaxpr.constvars, consts):
        env[v], tf[v] = x, False
    for v, x, fl, t in zip(jaxpr.invars, args, flags, tags or [None] * len(args)):
        env[v], tf[v] = x, fl
        if t is not None:
            tg[v] = t
    for eqn in jaxpr.eqns:
        with with_source(eqn):       # re-traced equations keep pointing at the user's code
            ins = [read(v) for v in eqn.invars]
            fin = [False if isinstance(v, Literal) else tf[v] for v in eqn.invars]
            name = eqn.primitive.name
            oflags = None
            subs = subjaxprs(eqn)
            itags = [None if isinstance(v, Literal) else tg.get(v) for v in eqn.invars]
            tagged = bool(ctx.get("tread")) and any(t is not None for t in itags)
            if tagged and name in SELECT and itags[0] is not None and all(t is None for t in itags[1:]):
                from ._reverse import tread
                outs, tok = tread(eqn, ctx["toks"][itags[0]], ins)
                _set_tok(ctx, itags[0], tok)
            elif tagged and name == "while":
                outs, oflags = _while(eqn, ins, fin, ctx, cap, rev, top, itags)
            elif tagged and name == "cond":
                outs, oflags = _cond(eqn, ins, fin, ctx, cap, rev, top, itags)
            elif tagged and name == "scan":
                outs, oflags = _scan(eqn, ins, fin, ctx, cap, rev, itags)
            elif tagged and name not in _CUSTOM and call_jaxpr(eqn) is not None:
                s = call_jaxpr(eqn)
                outs, oflags = _interp(s.jaxpr, s.consts, ins, fin, ctx, cap, rev, top, itags)
            else:
                if tagged:                     # any other use of a sparse table: it needs a dense cotangent too
                    ctx["dense"].update(t for t in itags if t is not None)
                force = any(isinstance(x, tuple) for x in ins)   # a hooked (tuple-valued) tally flows in: interpret
                if name == "while" and (force or rev or (top and (cap is not None or ctx.get("fp") is not None
                                                                  or ctx.get("gate") is not None))
                                        or any(_needs(s.jaxpr, rev, top, cap, ctx) for s in subs)):
                    outs, oflags = _while(eqn, ins, fin, ctx, cap, rev, top)
                elif name == "tallymap_emit" and ctx["emit"] is not None:
                    outs = [ctx["emit"](*ins)]
                elif name == "tallymap_spawn" and ctx.get("spawn") is not None:
                    outs = [ctx["spawn"](*ins)]
                elif not force and not any(_needs(s.jaxpr, rev, top, cap, ctx) for s in subs):
                    outs = bind(eqn, ins)
                elif call_jaxpr(eqn) is not None:
                    s = call_jaxpr(eqn)
                    outs, oflags = _interp(s.jaxpr, s.consts, ins, fin, ctx, cap, rev, top)
                elif name == "cond":
                    outs, oflags = _cond(eqn, ins, fin, ctx, cap, rev, top)
                elif name == "scan":
                    outs, oflags = _scan(eqn, ins, fin, ctx, cap, rev)
                else:
                    raise located(NotImplementedError(f"tallymap: while loop inside unsupported higher-order primitive "
                                                  f"{name!r}"), eqn)
            if oflags is None:
                oflags = _flags_through(eqn, fin)
            for v, x, fl in zip(eqn.outvars, outs, oflags):
                env[v], tf[v] = x, fl
    if ctx.get("tread"):                   # a table returned (as a carry, branch result, per-item result): dense
        ctx["dense"].update(tg[v] for v in jaxpr.outvars if not isinstance(v, Literal) and v in tg)
    return [read(v) for v in jaxpr.outvars], [False if isinstance(v, Literal) else tf[v] for v in jaxpr.outvars]


def _flags_through(eqn, fin, any_branch=False):
    """tally flags of the outputs of an equation bound unchanged (only control flow passes tallies).  any_branch: a
    cond output is a tally if it is one in ANY branch (the backends' rule: an inert discard tally may meet a filler
    value in another branch); default: branch 0 (the re-tracer's carry classification)."""
    name = eqn.primitive.name
    n = len(eqn.outvars)
    if name in TALLYMAP_PRIMS:
        return [True]
    if not any(fin):
        return [False] * n
    sub = call_jaxpr(eqn)
    if sub is not None:
        return _outflags(sub.jaxpr, fin, any_branch)
    if name == "cond":
        brs = cond_branches(eqn) if any_branch else cond_branches(eqn)[:1]
        per = [_outflags(b.jaxpr, fin[1:], any_branch) for b in brs]
        return [any(f) for f in zip(*per)]
    if name == "scan":
        sc = scan_parts(eqn)
        return fin[sc.nconsts:sc.nconsts + sc.ncarry] + _outflags(sc.body.jaxpr, fin, any_branch)[sc.ncarry:]
    return [False] * n


def tally_flags_out(jaxpr, fin):
    """which outvars of `jaxpr` carry a tally (virtual in the backends), given which invars do (fin): tallies pass only
    through emit / spawn and control flow, and a cond output is one if it is in ANY branch.  (The cuda emitter uses it
    to predict virtual stacked scan outputs.)"""
    return _outflags(jaxpr, fin, any_branch=True)


def _outflags(jaxpr, fin, any_branch=False):
    tf = {v: f for v, f in zip(jaxpr.invars, fin)}
    for e in jaxpr.eqns:
        fi = [False if isinstance(v, Literal) else tf.get(v, False) for v in e.invars]
        if e.primitive.name == "while":
            w = while_parts(e)
            fo = fi[w.nc + w.nb:]
        else:
            fo = _flags_through(e, fi, any_branch)
        for v, f in zip(e.outvars, fo):
            tf[v] = f
    return [False if isinstance(v, Literal) else tf.get(v, False) for v in jaxpr.outvars]


def _sub_fn(cj, nlead_flags, ctx, cap, rev, top, track_trunc=False, tags=None):
    """closed sub-jaxpr -> python function (re-traced recursively).  track_trunc: also return the branch's
    truncated flag (cond branches at top level)."""
    def f(*a):
        if track_trunc:
            saved = ctx["trunc"]
            ctx["trunc"] = jnp.bool_(False)
        outs, fl = _interp(cj.jaxpr, cj.consts, list(a), nlead_flags, ctx, cap, rev, top, tags)
        if track_trunc:
            t = ctx["trunc"]
            ctx["trunc"] = saved
            return outs, t
        return outs
    return f


def _notread(ctx, fn):
    """fn evaluated without table-read rewriting (a loop predicate: no derivative flows through it)."""
    def g(*a):
        saved = ctx.get("tread")
        ctx["tread"] = False
        try:
            return fn(*a)
        finally:
            ctx["tread"] = saved
    return g


def _with_toks(ctx, tids, fn, n):
    """wrap fn(consts, carry) so that the table tokens `tids` are threaded as extra trailing carries."""
    def g(c_, c2):
        saved = ctx["toks"]
        for t, v in zip(tids, c2[n:]):
            _set_tok(ctx, t, v)
        out = list(fn(c_, list(c2[:n])))
        new = [ctx["toks"][t] for t in tids]
        ctx["toks"] = saved
        return out + new
    return g


def _tids(ctx, tags, subs=()):
    """the threaded tokens a sub-computation needs: the table tokens of the tables it reads."""
    return sorted({t for t in tags if t is not None}) if ctx.get("tread") else []


def _while(eqn, ins, fin, ctx, cap, rev, top, itags=None):
    w = while_parts(eqn)
    nc, nb = w.nc, w.nb
    cc, bc, init = ins[:nc], ins[nc:nc + nb], ins[nc + nb:]
    itags = itags or [None] * len(ins)
    ctags, btags, ktags = itags[:nc], itags[nc:nc + nb], itags[nc + nb:]
    ctx["dense"].update(t for t in ktags if t is not None)        # a table as a loop carry: dense
    tflags = fin[nc + nb:]
    # (consts keep their tally flags: a DISCARD tally closed over by a loop -- the adjoint's replay loops -- must
    # still be known as a tally inside, or an inner loop would treat it as state)
    condf = _sub_fn(w.cond, list(fin[:nc]) + tflags, ctx, cap, rev, False, tags=ctags + [None] * len(init))
    bodyf = _sub_fn(w.body, list(fin[nc:nc + nb]) + tflags, ctx, cap, rev, False,
                    tags=btags + [None] * len(init))
    cfn = lambda cc_, c: condf(*cc_, *c)[0]
    bfn = lambda bc_, c: list(bodyf(*bc_, *c))
    g = ctx.get("gate") if top and not rev else None
    if g is not None:
        cfn0 = cfn
        cfn = lambda cc_, c: g & cfn0(cc_, c)
    if cap is not None and top:
        M = int(cap)

        def c2(st):
            return (st[0] < M) & cfn(cc, st[1])

        def b2(st):
            return st[0] + 1, bfn(bc, st[1])
        k, out = lax.while_loop(c2, b2, (jnp.int32(0), list(init)))
        ctx["trunc"] = ctx["trunc"] | ((k >= M) & cfn(cc, out))
        return list(out), tflags
    if not rev:
        return lax.while_loop(lambda c: cfn(cc, c), lambda c: bfn(bc, c), list(init)), tflags
    from ._reverse import checkpointed_while, checkpointed_while_single
    ropt = None if rev is True else rev
    if top and getattr(ropt, "single", False):      # T3: the (fold_post) item's one top-level loop
        checkpointed_while = checkpointed_while_single
    cfn = _notread(ctx, cfn)
    tids = _tids(ctx, btags, [w.body])
    bfl = list(fin[nc:nc + nb])
    live = _live_carries(w, nc, nb, len(init))
    from ._reverse import T3_MIN_EQNS, _n_eqns
    big = _n_eqns(w.body.jaxpr) >= T3_MIN_EQNS            # (auto_checkpoints)
    if not tids:
        return checkpointed_while(cfn, bfn, list(cc), list(bc), list(init), tflags, ropt, bfl, live=live,
                                  big=big), tflags
    n = len(init)
    out = checkpointed_while(lambda cc_, c2: cfn(cc_, c2[:n]), _with_toks(ctx, tids, bfn, n), list(cc), list(bc),
                    list(init) + [ctx["toks"][t] for t in tids], list(tflags) + ["w"] * len(tids),
                    ropt, bfl, live=live + [False] * len(tids), big=big)
    for t, v in zip(tids, out[n:]):
        _set_tok(ctx, t, v)
    return list(out[:n]), tflags


def _live_carries(w, nc, nb, n):
    """per loop carry: does the body or the predicate READ it?  (an operand of any equation, or a result in another
    carry's slot; a carry the body passes through unchanged and nothing else reads is not read).  From the loop's
    own jaxprs, so it costs no trace (lumen/perf measured a re-trace of the body for this at 284 s of a 307 s first
    call on lumen SK: its step has inner loops, each rebuilt and re-traced inside every traced body)."""
    used = [False] * n
    for jx, off, own in ((w.body.jaxpr, nb, True), (w.cond.jaxpr, nc, False)):
        idx = {v: j for j, v in enumerate(jx.invars[off:off + n])}
        for e in jx.eqns:
            for v in e.invars:
                if not isinstance(v, Literal) and v in idx:
                    used[idx[v]] = True
        for pos, v in enumerate(jx.outvars):
            if not isinstance(v, Literal) and v in idx and (not own or idx[v] != pos):
                used[idx[v]] = True
    return used


def _cond(eqn, ins, fin, ctx, cap, rev, top, itags=None):
    idx, ops = ins[0], ins[1:]
    brs = cond_branches(eqn)
    otags = (itags or [None] * len(ins))[1:]
    track = cap is not None
    fns = [_sub_fn(b, fin[1:], ctx, cap, rev, top, track_trunc=track, tags=otags) for b in brs]
    tids = _tids(ctx, otags, brs)
    nops = len(ops)

    def wrap(f):
        def g(*o2):
            saved = ctx["toks"]
            for t, v in zip(tids, o2[nops:]):
                _set_tok(ctx, t, v)
            r = f(*o2[:nops])
            outs, tr = r if track else (r, jnp.bool_(False))
            new = tuple(ctx["toks"][t] for t in tids)
            ctx["toks"] = saved
            return tuple(outs), tr, new
        return g
    outs, t, new = lax.switch(idx, [wrap(f) for f in fns], *ops, *[ctx["toks"][t] for t in tids])
    if track:                      # (only tracked when needed: a stale tracer must never leak across traces)
        ctx["trunc"] = ctx["trunc"] | t
    for tt, v in zip(tids, new):
        _set_tok(ctx, tt, v)
    return list(outs), _outflags(brs[0].jaxpr, fin[1:])


def _scan(eqn, ins, fin, ctx, cap, rev, itags=None):
    sc = scan_parts(eqn)
    nc, nk = sc.nconsts, sc.ncarry
    consts, init, xs = ins[:nc], ins[nc:nc + nk], ins[nc + nk:]
    itags = itags or [None] * len(ins)
    ctx["dense"].update(t for t in itags[nc:] if t is not None)  # a table as a scan carry / scanned input: dense
    body = _sub_fn(sc.body, fin, ctx, cap, rev, False, tags=itags[:nc] + [None] * (len(ins) - nc))
    tids = _tids(ctx, itags[:nc], [sc.body])

    def f(carry, x):
        c, tk = carry
        saved = ctx["toks"]
        for t, v in zip(tids, tk):
            _set_tok(ctx, t, v)
        o = body(*consts, *c, *x)
        new = [ctx["toks"][t] for t in tids]
        ctx["toks"] = saved
        return (list(o[:nk]), new), list(o[nk:])
    (carry, tk), ys = lax.scan(f, (list(init), [ctx["toks"][t] for t in tids]), list(xs), length=sc.length,
                               reverse=sc.reverse, unroll=sc.unroll)
    for t, v in zip(tids, tk):
        _set_tok(ctx, t, v)
    return list(carry) + list(ys), fin[nc:nc + nk] + [False] * len(ys)
