"""A branching-process outbreak as a tallymap spawn tree, against the best plain-JAX program for the same job.

There is no incumbent JAX library for individual-level branching processes, and the README says so.  The
established tools (`epichains`, `ringbp`, `bpmodels`) are R packages that simulate one lineage at a time on the
CPU, so a head-to-head against them would measure R against a GPU and tell you nothing.  The baselines here are
plain-JAX programs for the same model: the obvious one, and the best one.

The model is the one used for early-outbreak risk assessment (Lloyd-Smith et al. 2005 on superspreading;
Hellewell et al. 2020):

    one infected individual i, infected at time t_i, in generation g_i
      individual reproduction number  nu_i = R0 exp(sigma Z_i - sigma^2/2),  Z_i ~ N(0, 1)
      offspring count                 K_i ~ Poisson(nu_i)
      offspring infection time        t_j = t_i + G,  G ~ Weibull(alpha, beta)
      the individual is DETECTED with probability p_det and then isolated, so K_i = 0
    a lineage stops at t > 40 days or at generation 10

Both random draws are explicit transforms of uniforms (Weibull by inversion, Poisson by exponential
interarrivals) rather than library rejection samplers, which keeps every primitive differentiable where it
should be.

  tm         item = one infected individual.  Its offspring loop `spawn`s each child as a new item, and the
             child runs the same function.  Tallies: incidence by time bin, infections by generation, outbreak
             size / detected / sum of infection times / offspring counts, an offspring histogram, a smooth
             observable and its ancestral score.
  jax_front  the baseline worth quoting: ONE global generation frontier pooled over all N outbreaks, with each
             child slot mapped back to its parent by a `searchsorted` on the cumulative offspring counts, so
             there is no (parent, offspring) rectangle and no compaction scatter.  A generation costs
             O(front log front) instead of O(front x max_offspring).  `front` must be chosen in advance and
             must cover the largest generation of the whole call; `--full` sizes it from a measured run and
             every row reports `jax_dropped`, which must be zero.
  jax_gen    what most people would write first: `jax.vmap` over outbreaks with a (live, kmax) children block
             per outbreak.  Sized from the measured maxima it asks for tens of gigabytes and cannot run;
             shrunk so that it does run, it silently drops children, and the row reports how many.

Gradients, theta = (log R0, Weibull shape, Weibull scale, logit p_det).  Only the two generation-interval
parameters have a non-zero pathwise derivative, and that is itself a measured result: the other two enter
through discrete draws, so their pathwise derivative is exactly zero in both arms and the right estimator for
them is the score tally, which is free.  The honest recommendation for four parameters is `jacfwd`, not `grad`:
see the README.

  python run.py --quick
  python run.py --full
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _demo as D  # noqa: E402

import tallymap as tm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LIBS = ("jax",)
NT, T_OBS = 32, 40.0                 # incidence bins, observation horizon (days)
GMAX = 10                            # generation cap
NU_MAX = 400.0                       # clip on the individual reproduction number (a numerical guard)
SIGMA = {"low": 0.0, "mid": 1.0, "high": 1.8}       # spread of nu_i; 0 = plain Poisson offspring
THETA0 = np.array([np.log(1.55), 2.5, 6.0, -1.2], np.float32)   # log R0, Weibull shape, scale, logit p_det
NOFF = 256                           # offspring-count histogram bins
MAXOFF = 512                         # max_steps: the cap on ONE individual's offspring loop
ARMS = ("tm", "jax_front", "jax_gen")
LABEL = {"tm": "tallymap spawn tree", "jax_front": "global generation frontier",
         "jax_gen": "per-outbreak padding"}


def draw_nu(key, logR0, sigma):
    return jnp.minimum(jnp.exp(logR0 + sigma * jax.random.normal(key) - 0.5 * sigma * sigma), NU_MAX)


def poisson_count(key, lam, cap):
    """K ~ Poisson(lam) by exponential interarrivals: K = max{k : sum_{i<k} E_i <= lam}.  f32-safe."""
    def body(c):
        n, s, i = c
        u = jax.random.uniform(jax.random.fold_in(key, i), minval=1e-20)
        return n + 1, s - jnp.log(u), i + 1
    n, _, _ = lax.while_loop(lambda c: (c[1] <= lam) & (c[0] < cap), body,
                             (jnp.int32(0), jnp.float32(0.0), jnp.int32(0)))
    return jnp.maximum(n - 1, 0)


def weibull(key, alpha, beta):
    """G ~ Weibull(alpha, beta) by inversion: beta (-log U)^(1/alpha).  Smooth in alpha and beta."""
    return beta * jnp.power(-jnp.log(jax.random.uniform(key, minval=1e-20)), 1.0 / alpha)


def smooth_w(t):
    """A smooth bounded weight on the infection time: differentiable in t, unlike a histogram bin."""
    return jnp.exp(-0.5 * ((t - 14.0) / 7.0) ** 2)


def tbin(t):
    return jnp.clip((t * (NT / T_OBS)).astype(jnp.int32), 0, NT - 1)


# ------------------------------------------------------------------------------------- tallymap arm (spawn tree)
def make_tm(sigma, cap, **o):
    def individual(t, p, theta):
        """p = dict(t=infection time, g=generation, key=PRNG key, sc=ancestral score of log R0)."""
        logR0, alpha, beta, lp = theta[0], theta[1], theta[2], theta[3]
        kd, kn, ko, kg = jax.random.split(p["key"], 4)
        detected = jax.random.uniform(kd) < jax.nn.sigmoid(lp)
        nu = draw_nu(kn, logR0, sigma)
        n = jnp.where(detected, 0, poisson_count(ko, nu, MAXOFF))
        sc_child = p["sc"] + jnp.where(detected, 0.0, n.astype(jnp.float32) - nu)
        sw = smooth_w(p["t"])
        t = t.add("inc", tbin(p["t"]), 1.0)
        t = t.add("gen", jnp.clip(p["g"], 0, GMAX), 1.0)
        t = t.add("cum", jnp.arange(4), jnp.stack([1.0, detected.astype(jnp.float32), p["t"],
                                                   n.astype(jnp.float32)]))
        t = t.add("smooth", 0, sw)
        t = t.add("score", 0, sw * p["sc"])
        t = t.add("off", jnp.clip(n, 0, NOFF - 1), 1.0)       # the last bin means "this many or more"
        # children DRAWN by generation.  This is exactly what the frontier arm has to be sized for: it must
        # hold every child drawn in one generation, including the ones that then die of the horizon or the
        # generation cap, because they still occupy a slot.
        t = t.add("offgen", jnp.clip(p["g"], 0, GMAX), n.astype(jnp.float32))

        def body(c):
            i, t = c
            ki = jax.random.fold_in(kg, i)
            tc = p["t"] + weibull(ki, alpha, beta)
            ok = (tc < T_OBS) & (p["g"] + 1 <= GMAX)
            t = t.spawn(dict(t=tc, g=p["g"] + 1, key=jax.random.fold_in(ki, 7), sc=sc_child), when=ok)
            return i + 1, t
        _, t = lax.while_loop(lambda c: c[0] < n, body, (jnp.int32(0), t))
        return t
    outs = {"inc": tm.Sum((NT,)), "gen": tm.Sum((GMAX + 1,)), "cum": tm.Sum((4,)),
            "smooth": tm.Sum((1,)), "score": tm.Sum((1,)), "off": tm.Sum((NOFF,)),
            "offgen": tm.Sum((GMAX + 1,))}
    return tm.vmap(individual, outputs=outs, in_axes=(0, None), spawn=tm.Queue(cap), max_steps=MAXOFF, **o)


# ------------------------------------------------------------------ plain-JAX arm 1: per-outbreak padding
def make_jax_gen(sigma, live, kmax):
    """`jax.vmap` over outbreaks, each a scan over generations holding a padded (live,) buffer, with children
    produced as a (live, kmax) block and compacted by cumsum + scatter.  It pays live x kmax work per
    generation per outbreak however small the outbreak really is, and anything past the caps is dropped."""
    def one_outbreak(root, theta):
        logR0, alpha, beta, lp = theta[0], theta[1], theta[2], theta[3]
        p_det = jax.nn.sigmoid(lp)
        off = jnp.arange(kmax)

        def gen_step(c, _):
            tt, gg, kk, sc, al, drop = c
            ks = jax.vmap(lambda x: jax.random.split(x, 4))(kk)
            det = jax.vmap(jax.random.uniform)(ks[:, 0]) < p_det
            nu = jax.vmap(lambda x: draw_nu(x, logR0, sigma))(ks[:, 1])
            n = jax.vmap(lambda x, l: poisson_count(x, l, MAXOFF))(ks[:, 2], nu)
            n = jnp.where(al & ~det, n, 0)
            sw = jnp.where(al, smooth_w(tt), 0.0)
            out = (jnp.zeros((NT,), jnp.float32).at[tbin(tt)].add(jnp.where(al, 1.0, 0.0)),
                   jnp.zeros((GMAX + 1,), jnp.float32).at[jnp.clip(gg, 0, GMAX)].add(jnp.where(al, 1.0, 0.0)),
                   jnp.stack([jnp.sum(jnp.where(al, 1.0, 0.0)), jnp.sum(jnp.where(al & det, 1.0, 0.0)),
                              jnp.sum(jnp.where(al, tt, 0.0)), jnp.sum(n.astype(jnp.float32))]),
                   jnp.sum(sw), jnp.sum(sw * sc))
            scc = sc + jnp.where(al & ~det, n.astype(jnp.float32) - nu, 0.0)
            kg = ks[:, 3]
            kid = jax.vmap(lambda x: jax.vmap(lambda j: jax.random.fold_in(x, j))(off))(kg)
            tc = tt[:, None] + jax.vmap(lambda s: jax.vmap(lambda k_: weibull(k_, alpha, beta))(s))(kid)
            ok = (off[None, :] < n[:, None]) & (tc < T_OBS) & ((gg[:, None] + 1) <= GMAX)
            ck = jax.vmap(lambda x: jax.vmap(lambda j: jax.random.fold_in(jax.random.fold_in(x, j), 7))(off))(kg)
            fo, ft = ok.ravel(), tc.ravel()
            fk = ck.reshape(-1, ck.shape[-1])
            fs = jnp.broadcast_to(scc[:, None], ok.shape).ravel()
            fg = jnp.broadcast_to((gg + 1)[:, None], ok.shape).ravel()
            slot = jnp.cumsum(fo.astype(jnp.int32)) - 1
            nkid = jnp.sum(fo.astype(jnp.int32))
            slot = jnp.where(fo & (slot < live), slot, live)                 # `live` is the trash slot
            z = jnp.zeros((live + 1,), jnp.float32)
            return ((z.at[slot].set(ft)[:live], jnp.zeros((live + 1,), jnp.int32).at[slot].set(fg)[:live],
                     jnp.zeros((live + 1,) + fk.shape[1:], fk.dtype).at[slot].set(fk)[:live],
                     z.at[slot].set(fs)[:live], jnp.zeros((live + 1,), bool).at[slot].set(fo)[:live],
                     drop + jnp.maximum(nkid - live, 0) + jnp.sum(jnp.maximum(n - kmax, 0))), out)

        pad = lambda v, dt_: jnp.concatenate([jnp.asarray(v)[None], jnp.zeros((live - 1,), dt_)])  # noqa: E731
        c0 = (pad(root["t"], jnp.float32), pad(root["g"], jnp.int32),
              jnp.concatenate([root["key"][None],
                               jnp.zeros((live - 1,) + root["key"].shape, root["key"].dtype)]),
              pad(root["sc"], jnp.float32),
              jnp.concatenate([jnp.ones((1,), bool), jnp.zeros((live - 1,), bool)]), jnp.int32(0))
        c, outs = lax.scan(gen_step, c0, None, length=GMAX + 1)
        return tuple(jnp.sum(o, axis=0) for o in outs) + (c[-1],)

    def call(roots, theta):
        o = jax.vmap(lambda r: one_outbreak(r, theta))(roots)
        return tuple(jnp.sum(x, axis=0) for x in o)
    return call


# -------------------------------------------------------- plain-JAX arm 2: ONE global generation frontier
def make_front(sigma, front):
    """`lax.scan` over generations holding ONE global frontier of `front` slots, pooled over all N outbreaks,
    with no per-parent padding at all.

    The trick that makes this the best plain-JAX program: draw the offspring counts of the whole frontier, take
    their cumulative sum, and map each CHILD slot back to its parent with one `searchsorted`.  The children are
    packed densely, so there is no (parent, offspring) rectangle and no compaction scatter.  Slots whose child
    died (t > T_OBS or g > GMAX) become holes, which the next generation's cumsum skips for free.

    What it still pays, and the spawn tree does not: `front` must be chosen in advance and must cover the
    LARGEST generation of the whole call, children past it are dropped and counted, every generation touches
    all `front` slots whether or not the epidemic is still that wide, and reverse mode stores the whole
    (GMAX+1) x front trajectory of the scan.
    """
    def call(roots, theta):
        logR0, alpha, beta, lp = theta[0], theta[1], theta[2], theta[3]
        p_det = jax.nn.sigmoid(lp)
        n0 = roots["t"].shape[0]
        slots = jnp.arange(front)
        pad = lambda a, v: jnp.concatenate([a, jnp.full((front - n0,) + a.shape[1:], v, a.dtype)])  # noqa: E731

        def gen_step(c, _):
            tt, gg, kk, sc, al, drop = c
            ks = jax.vmap(lambda x: jax.random.split(x, 4))(kk)
            det = jax.vmap(jax.random.uniform)(ks[:, 0]) < p_det
            nu = jax.vmap(lambda x: draw_nu(x, logR0, sigma))(ks[:, 1])
            n = jnp.where(al & ~det, jax.vmap(lambda x, l: poisson_count(x, l, MAXOFF))(ks[:, 2], nu), 0)
            sw = jnp.where(al, smooth_w(tt), 0.0)
            out = (jnp.zeros((NT,), jnp.float32).at[tbin(tt)].add(jnp.where(al, 1.0, 0.0)),
                   jnp.zeros((GMAX + 1,), jnp.float32).at[jnp.clip(gg, 0, GMAX)].add(jnp.where(al, 1.0, 0.0)),
                   jnp.stack([jnp.sum(jnp.where(al, 1.0, 0.0)), jnp.sum(jnp.where(al & det, 1.0, 0.0)),
                              jnp.sum(jnp.where(al, tt, 0.0)), jnp.sum(n.astype(jnp.float32))]),
                   jnp.sum(sw), jnp.sum(sw * sc))
            scc = sc + jnp.where(al & ~det, n.astype(jnp.float32) - nu, 0.0)
            cs = jnp.cumsum(n)
            total = cs[-1]
            par = jnp.minimum(jnp.searchsorted(cs, slots, side="right"), front - 1)   # child slot -> parent
            idx = slots - (cs - n)[par]                                               # offspring index
            kg = ks[:, 3][par]
            ki = jax.vmap(jax.random.fold_in)(kg, idx)
            tc = tt[par] + jax.vmap(lambda k_: weibull(k_, alpha, beta))(ki)
            gc = gg[par] + 1
            ok = (slots < total) & (tc < T_OBS) & (gc <= GMAX)
            kc = jax.vmap(lambda k_: jax.random.fold_in(k_, 7))(ki)
            return ((jnp.where(ok, tc, 0.0), jnp.where(ok, gc, 0), kc, scc[par], ok,
                     drop + jnp.maximum(total - front, 0)), out)

        c0 = (pad(roots["t"], 0.0), pad(roots["g"], 0), pad(roots["key"], 0), pad(roots["sc"], 0.0),
              jnp.concatenate([jnp.ones((n0,), bool), jnp.zeros((front - n0,), bool)]), jnp.int32(0))
        c, outs = lax.scan(gen_step, c0, None, length=GMAX + 1)
        return tuple(jnp.sum(o, axis=0) for o in outs) + (c[-1],)
    return call


def roots_of(n):
    rng = np.random.default_rng(0)
    return dict(t=jnp.asarray(rng.uniform(0.0, 4.0, n), jnp.float32), g=jnp.zeros((n,), jnp.int32),
                key=jax.random.split(jax.random.PRNGKey(11), n), sc=jnp.zeros((n,), jnp.float32))


# ------------------------------------------------------------------------------------------------ one measurement
def one(a):
    D.banner(f"branch arm={a.arm} mode={a.mode} n={a.n} div={a.div} cap={a.cap} front={a.front} "
             f"live={a.live} kmax={a.kmax}", LIBS)
    sigma, n = SIGMA[a.div], a.n
    roots = roots_of(n)
    theta = jnp.asarray(THETA0)
    front = a.front or int(3.2 * n)
    base = dict(demo="branching_epidemic", arm=a.arm, mode=a.mode, n=n, div=a.div, sigma=sigma,
                cap=a.cap, front=front, live=a.live, kmax=a.kmax, gmax=GMAX)
    ftm = make_tm(sigma, a.cap, rev_stats=bool(a.revstats))

    if a.mode == "sizes":           # measure what the padded baselines have to be sized for
        rt = jax.jit(lambda th: ftm(roots, th))(theta)
        off = np.asarray(rt.outputs["off"])
        D.row(**base, status="ok", tm_spawned=int(rt.stats["spawned"]),
              tm_dropped=int(rt.stats["dropped"]), tm_gen_max=int(np.max(np.asarray(rt.outputs["gen"]))),
              tm_front_needed=int(np.max(np.asarray(rt.outputs["offgen"]))),
              tm_max_offspring=int(np.max(np.nonzero(off)[0])),
              tm_off_overflow=float(off[NOFF - 1]),
              size_per_root=float(rt.outputs["cum"][0]) / n)
        return

    if a.arm == "tm":
        sim = lambda th: ftm(roots, th).outputs["inc"]                                   # noqa: E731
        loss = lambda th: ftm(roots, th).outputs["smooth"][0] / n                        # noqa: E731
    else:
        gj = make_front(sigma, front) if a.arm == "jax_front" else make_jax_gen(sigma, a.live, a.kmax)
        sim = lambda th: gj(roots, th)[0]                                                # noqa: E731
        loss = lambda th: gj(roots, th)[3] / n                                           # noqa: E731
    fn = {"fwd": sim, "grad": jax.grad(loss), "jacfwd": jax.jacfwd(loss),
          "jvp": lambda th: jax.jvp(loss, (th,), (jnp.ones_like(th),))[1]}[a.mode]

    def go():
        out, r = D.measure(fn, (theta,), reps=a.reps)
        r.update(base, status="ok", out_norm=float(jnp.linalg.norm(jnp.asarray(out).ravel())),
                 finite=bool(np.all(np.isfinite(np.asarray(out)))))
        if a.arm == "tm":
            res = jax.jit(lambda th: ftm(roots, th))(theta)
            r.update(tm_spawned=int(res.stats["spawned"]), tm_dropped=int(res.stats["dropped"]),
                     n_truncated=int(res.stats["n_truncated"]),
                     size_per_root=float(res.outputs["cum"][0]) / n)
        else:
            o = jax.jit(lambda th: gj(roots, th))(theta)
            r.update(jax_dropped=int(o[-1]), size_per_root=float(o[2][0]) / n)
        D.row(**r)
    D.guard(go, base)


# -------------------------------------------------------------------------------------------------------- --quick
def quick(a):
    n = min(a.n, 1024)
    sigma = SIGMA["mid"]
    D.banner(f"quick n={n} div=mid", LIBS)
    roots = roots_of(n)
    theta = jnp.asarray(THETA0)
    checks = []

    # size the frontier arm from what the spawn tree MEASURES, not from a guess
    probe = make_tm(sigma, 1 << 22)
    rp = jax.jit(lambda th: probe(roots, th))(theta)
    need = int(np.max(np.asarray(rp.outputs["offgen"])))
    front = max(n, int(1.05 * need) + 1)
    checks.append((f"the spawn tree measures what the frontier arm must be sized for (front={front:,})",
                   int(rp.stats["dropped"]) == 0,
                   f"{int(rp.stats['spawned']):,} children spawned, {need:,} drawn in the largest "
                   f"generation, {int(rp.stats['dropped'])} dropped"))

    ftm = make_tm(sigma, 1 << 22)
    gj = make_front(sigma, front)
    rt = jax.jit(lambda th: ftm(roots, th))(theta)
    o = jax.jit(lambda th: gj(roots, th))(theta)
    checks.append(("the frontier arm dropped no children at that size", int(o[-1]) == 0,
                   f"jax_dropped = {int(o[-1])}"))
    ri, rg = D.rel(rt.outputs["inc"], o[0]), D.rel(rt.outputs["gen"], o[1])
    checks.append(("incidence curve and infections by generation vs the frontier arm", max(ri, rg) < 1e-5,
                   f"incidence rel = {ri:.2g}, generation rel = {rg:.2g}"))
    st, sj = float(rt.outputs["cum"][0]) / n, float(o[2][0]) / n
    checks.append(("outbreak size per index case", abs(st - sj) / sj < 1e-4,
                   f"{st:.4f} vs {sj:.4f}"))

    loss_tm = lambda th: ftm(roots, th).outputs["smooth"][0] / n                          # noqa: E731
    loss_jx = lambda th: gj(roots, th)[3] / n                                             # noqa: E731
    g = np.asarray(jax.jit(jax.grad(loss_tm))(theta))
    J = np.asarray(jax.jit(jax.jacfwd(loss_tm))(theta))
    gx = np.asarray(jax.jit(jax.grad(loss_jx))(theta))
    checks.append(("the tree adjoint is finite and agrees with jacfwd", np.all(np.isfinite(g))
                   and D.rel(g, J) < 1e-5, f"grad = {np.array2string(g, precision=5)}, "
                   f"grad vs jacfwd rel = {D.rel(g, J):.2g}"))
    checks.append(("the gradient agrees with the frontier arm's", D.rel(J, gx) < 1e-5,
                   f"rel = {D.rel(J, gx):.2g}"))
    checks.append(("only the two generation-interval parameters have a pathwise derivative",
                   g[0] == 0.0 and g[3] == 0.0 and abs(g[1]) + abs(g[2]) > 0,
                   f"d/d log R0 = {g[0]:.3g}, d/d logit p_det = {g[3]:.3g} (both exactly zero, so use the "
                   f"score tally for those two)"))

    # the negative control: the tree adjoint needs a Queue that covers ALL children of the call.  Too small,
    # and the gradient is NaN and the diagnostics say why.  A gradient is never silently wrong.
    small = make_tm(sigma, 1 << 8, rev_stats=True)
    gs = np.asarray(jax.jit(jax.grad(lambda th: small(roots, th).outputs["smooth"][0] / n))(theta))
    tr = tm.diagnostics()["reverse"]["tree"]
    ov = bool(tr["overflow"]) if tr is not None else None
    checks.append(("an undersized spawn Queue gives NaN, not a wrong number",
                   bool(np.any(~np.isfinite(gs))) and ov is True,
                   f"grad = {np.array2string(gs, precision=3)}, diagnostics overflow = {ov}"))
    return D.verdict(checks)


# --------------------------------------------------------------------------------------------------------- --full
NS = (16384, 65536, 262144, 1048576)


def plots(rows, meta):
    """Redraw plots/ from `rows`.  Called by --full and by --plots."""
    P = os.path.join(HERE, "plots")
    fwd = [r for r in rows if r.get("mode") == "fwd" and r.get("div") == "mid"]
    D.panel(os.path.join(P, "fwd_time.png"), fwd, ARMS, LABEL,
            "Branching-process outbreaks, forward: time per call", "ms per call", xlabel="index cases N")
    D.panel(os.path.join(P, "fwd_mem.png"), fwd, ARMS, LABEL,
            "Branching-process outbreaks, forward: peak device memory", "peak device MB",
            xlabel="index cases N", ykey="peak_mb")
    for mode, t in (("grad", "reverse mode (grad)"), ("jacfwd", "forward mode (jacfwd over 4 parameters)")):
        rr = [r for r in rows if r.get("mode") == mode and r.get("div") == "mid"]
        D.panel(os.path.join(P, f"{mode}_time.png"), rr, ("tm", "jax_front"), LABEL,
                f"Derivative of the smooth observable, {t}: time", "ms per call", xlabel="index cases N")
        D.panel(os.path.join(P, f"{mode}_mem.png"), rr, ("tm", "jax_front"), LABEL,
                f"Derivative of the smooth observable, {t}: peak device memory", "peak device MB",
                xlabel="index cases N", ykey="peak_mb")

def full(a):
    script = os.path.join(HERE, "run.py")
    out = os.path.join(HERE, "results", "branching_epidemic.json")
    # stage 1: measure what the frontier arm must be sized for
    size_jobs = [dict(arm="tm", mode="sizes", n=n, div=d, cap=1 << 21)
                 for n in NS for d in ("low", "mid", "high")]
    rows = D.run_jobs(script, size_jobs)
    need = {}
    for r in rows:
        if r.get("status") == "ok":
            need[(r["n"], r["div"])] = max(r["n"], int(1.05 * r["tm_front_needed"]) + 1)
    front = lambda n, d: need.get((n, d), int(3.5 * n))                                  # noqa: E731
    # Forward and forward-mode need a Queue that holds the PENDING frontier; the tree adjoint needs one that
    # covers EVERY child of the call (README "Sizing the Queue"), which is about 15.5 per index case here.
    # Every row reports `tm_dropped`, which must be zero, and `--quick` shows what happens when it is not.
    FCAP = 1 << 21
    rcap = {n: 1 << int(np.ceil(np.log2(max(1 << 20, 1.3 * 15.5 * n)))) for n in NS}

    jobs = []
    for n in NS:
        for arm in ("tm", "jax_front"):
            jobs.append(dict(arm=arm, mode="fwd", n=n, div="mid", cap=FCAP, front=front(n, "mid")))
    # the obvious per-outbreak arm, sized so that it runs at all; it drops children and the row says how many
    for n in (16384, 65536):
        jobs.append(dict(arm="jax_gen", mode="fwd", n=n, div="mid", live=64, kmax=32))
    for n in (16384, 65536, 262144):
        for arm in ("tm", "jax_front"):
            jobs.append(dict(arm=arm, mode="grad", n=n, div="mid", cap=rcap[n], front=front(n, "mid")))
            jobs.append(dict(arm=arm, mode="jacfwd", n=n, div="mid", cap=FCAP, front=front(n, "mid")))
    for d in ("low", "high"):
        for arm in ("tm", "jax_front"):
            jobs.append(dict(arm=arm, mode="fwd", n=262144, div=d, cap=FCAP, front=front(262144, d)))
            jobs.append(dict(arm=arm, mode="jacfwd", n=65536, div=d, cap=FCAP, front=front(65536, d)))
    rows += D.run_jobs(script, jobs)
    rows, meta = D.write_results(
        out, rows, libs=LIBS,
        extra_meta=dict(ns=list(NS), fwd_cap=FCAP, grad_caps=rcap,
                        front_sizes={f"{k[0]}:{k[1]}": v for k, v in need.items()}))
    plots(rows, meta)

    print(f"\n{len(rows)} rows -> {out}")
    return 0


def main():
    p = D.parser(cap=1 << 24, live=64, kmax=32, front=0, revstats=0)
    a = p.parse_args()
    if a.one:
        one(a)
        return 0
    if a.plots:
        rows, meta = D.load(os.path.join(HERE, "results", "branching_epidemic.json"))
        plots(rows, meta)
        return 0
    return quick(a) if a.quick else full(a)


if __name__ == "__main__":
    sys.exit(main())
