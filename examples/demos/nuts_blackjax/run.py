"""BlackJAX samplers over many chains: the documented vmap-over-chains pattern against one chain per item.

The incumbent is blackjax 1.7.1, called unmodified.  Two of its kernels have a data-dependent inner loop:

  nuts      `blackjax.nuts`.  One transition is two nested `lax.while_loop`s, the doubling loop
            (`blackjax/mcmc/trajectory.py:719`) and the sub-trajectory integration inside it
            (`trajectory.py:368`), plus a third in the iterative U-turn check (`termination.py:95`).  The
            number of leapfrog steps per draw is between 1 and 2^max_num_doublings - 1 and is data dependent.
  ess       `blackjax.elliptical_slice`.  Its kernel shrinks an angle bracket in a `lax.while_loop` until the
            proposal is inside the slice (`elliptical_slice.py:211-237`).

BlackJAX's own multi-chain how-to states the consequence: under `jax.vmap` across chains, "each step is as long
as the slowest chain".  It recommends `jax.pmap`, which is limited to the device count, so on one GPU it is not
an alternative; `--prog nuts --arm bj_pmap` runs it anyway and reports what one device gives.

Arms for `nuts` and `ess`, both calling the SAME `alg.step`:
  bj_vmap   the documented pattern: `lax.scan` over draws of `jax.vmap(alg.step)` across chains.
  tm        one chain = one tallymap item; the item runs the `lax.scan` over draws of the same `alg.step`.

A third program, `--prog essgrad`, is about a capability rather than a speed-up: `jax.grad` through a short
elliptical-slice chain.  JAX cannot reverse-differentiate a `lax.while_loop` at all, so a BlackJAX user who
wants this gradient today has to rewrite the kernel as a padded, masked `lax.scan`.  Its four arms are `tm`,
`while` (the identical `jax.vmap` + `while_loop` program; `grad` raises, and the message is recorded),
`scan` (the rewrite, padded to the observed maximum shrinkage count) and `ckpt` (`scan` with a checkpointed
body).  The chain there is written out in this file rather than taken from blackjax, because the point is the
comparison between the three loop lowerings of one program; `--prog ess` is the one that runs blackjax's kernel.

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
import blackjax  # noqa: E402

import tallymap as tm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LIBS = ("blackjax", "jax")

# ----------------------------------------------------------------------------------------------------- nuts
NUTS_DIVS = {"low": 1.0, "mid": 3.0, "high": 5.0}       # decades of spread in the target's log standard deviations


def nuts_target(dim, div):
    """A correlated Gaussian plus a mild non-Gaussian tail term.

    The tail term (a log-sum-exp mixture in the first coordinate) is what makes the required tree depth vary
    strongly from chain to chain: chains in the narrow mode need many more leapfrog steps than chains in the
    broad one.
    """
    e = NUTS_DIVS[div]
    sd = jnp.float32(10.0) ** jnp.linspace(-e / 2, e / 2, dim).astype(jnp.float32)

    def logdensity(x):
        g = -0.5 * jnp.sum((x / sd) ** 2)
        m = jax.nn.logsumexp(jnp.array([-0.5 * ((x[0] - 2.0) / 0.3) ** 2,
                                        -0.5 * ((x[0] + 2.0) / 1.5) ** 2]))
        return g + 0.3 * m
    return logdensity, sd


def nuts_adapt(logdensity, dim, key, n_adapt):
    """ONE blackjax window_adaptation run.  Both arms then sample with its result, so both arms sample the
    same kernel.  Not timed."""
    wa = blackjax.window_adaptation(blackjax.nuts, logdensity)
    (state, params), _ = wa.run(key, jnp.zeros((dim,), jnp.float32), num_steps=n_adapt)
    return float(params["step_size"]), jnp.asarray(params["inverse_mass_matrix"], jnp.float32)


def nuts_build(a):
    logdensity, sd = nuts_target(a.dim, a.div)
    ss, imm = nuts_adapt(logdensity, a.dim, jax.random.PRNGKey(3), a.nadapt)
    alg = blackjax.nuts(logdensity, step_size=ss, inverse_mass_matrix=imm, max_num_doublings=a.doublings)
    x0 = 0.5 * jax.random.normal(jax.random.PRNGKey(11), (a.n, a.dim), jnp.float32) * sd
    return alg, x0, ss


def nuts_counter(info):
    return info.num_integration_steps


def ess_build(a):
    """blackjax.elliptical_slice on a Gaussian prior with a Gaussian likelihood of precision DIVS[div].

    A sharper likelihood means the first angle is almost always rejected, so the bracket shrinks many more
    times: `--div` is a direct knob on the per-chain trip count.
    """
    s = {"low": 1.0, "mid": 25.0, "high": 400.0}[a.div]
    prec = s * (jnp.arange(1.0, a.dim + 1.0) / a.dim)
    mu = jnp.full((a.dim,), 0.3, jnp.float32)
    alg = blackjax.elliptical_slice(lambda x: -0.5 * jnp.sum(prec * (x - mu) ** 2),
                                    mean=jnp.zeros(a.dim, jnp.float32), cov=jnp.ones(a.dim, jnp.float32))
    x0 = 0.1 * jax.random.normal(jax.random.PRNGKey(7), (a.n, a.dim), jnp.float32)
    return alg, x0, 0.0


def ess_counter(info):
    return info.subiter


COUNTER = {"nuts": nuts_counter, "ess": ess_counter}
CNAME = {"nuts": "leapfrogs", "ess": "shrinkage steps"}


def arm_bj_vmap(alg, prog, x0, keys, n_draws, mode):
    """The pattern blackjax's multi-chain how-to shows: vmap the kernel across chains, scan the draws."""
    nch, dim = x0.shape
    init = jax.vmap(alg.init)(x0)
    count = COUNTER[prog]

    def one_step(carry, k):
        st, a1, a2, nl = carry
        st, info = jax.vmap(alg.step)(jax.random.split(k, nch), st)
        n = count(info).sum().astype(jnp.int32)
        if mode == "store":
            return (st, a1, a2, nl + n), st.position
        return (st, a1 + st.position, a2 + st.position ** 2, nl + n), None

    z = jnp.zeros_like(x0)
    carry, out = lax.scan(one_step, (init, z, z, jnp.int32(0)), keys)
    if mode == "store":
        return out.swapaxes(0, 1), carry[3]
    return (carry[1], carry[2]), carry[3]


def arm_bj_pmap(alg, prog, x0, keys, n_draws, mode):
    """What blackjax's how-to recommends instead of vmap.  On one GPU `jax.local_device_count()` is 1, so this
    is the same lockstep loop with a device axis of length one; it is reported, not hidden."""
    nd = jax.local_device_count()
    nch = x0.shape[0]
    per = nch // nd
    xs = x0[:per * nd].reshape(nd, per, x0.shape[1])

    def chunk(xc):
        return arm_bj_vmap(alg, prog, xc, keys, n_draws, mode)
    out, nl = jax.pmap(chunk)(xs)
    if mode == "store":
        return out.reshape(-1, *out.shape[2:]), nl.sum()
    return (out[0].reshape(-1, out[0].shape[-1]), out[1].reshape(-1, out[1].shape[-1])), nl.sum()


def arm_tm(alg, prog, n_draws, mode, nch, dim, max_steps):
    """One chain = one item.  Same kernel, same number of draws, same outputs."""
    if mode == "store":
        outputs = {"draws": tm.Sum((nch, n_draws, dim)), "count": tm.Sum((1,), dtype=jnp.int32)}
    else:
        outputs = {"m1": tm.Sum((nch, dim)), "m2": tm.Sum((nch, dim)),
                   "count": tm.Sum((1,), dtype=jnp.int32)}
    cols = jnp.arange(dim, dtype=jnp.int32)
    count = COUNTER[prog]

    def item(t, x0i, key, cid):
        st = alg.init(x0i)

        def body(c, i):
            st, t = c
            st, info = alg.step(jax.random.fold_in(key, i), st)
            if mode == "store":
                t = t.add("draws", (cid, i, cols), st.position)
            else:
                t = t.add("m1", (cid, cols), st.position)
                t = t.add("m2", (cid, cols), st.position ** 2)
            t = t.add("count", 0, count(info).astype(jnp.int32))
            return (st, t), None
        (st, t), _ = lax.scan(body, (st, t), jnp.arange(n_draws, dtype=jnp.int32))
        return t, st.position

    f = tm.vmap(item, outputs=outputs, max_steps=max_steps)

    def call(x0, keys, cids):
        r = f(x0, keys, cids)
        if mode == "store":
            return r.outputs["draws"], r.outputs["count"][0]
        return (r.outputs["m1"], r.outputs["m2"]), r.outputs["count"][0]
    return call, f


# ----------------------------------------------------------------------- essgrad: the gradient through a sampler
GDIM = 6
GMAXSUB = 24           # the padded-scan bound: the observed maximum shrinkage count, not an inflated cap


def g_loglik(div, dim=GDIM):
    s = {"low": 1.0, "mid": 25.0, "high": 400.0}[div]
    prec = s * (jnp.arange(1.0, dim + 1.0) / dim)
    mu = jnp.full((dim,), 0.3, jnp.float32)
    return lambda x: -0.5 * jnp.sum(prec * (x - mu) ** 2)


def g_draw_while(loglik, theta, x, lx, key):
    """One elliptical-slice transition, written the way blackjax writes it: a `lax.while_loop`."""
    k_nu, k_u, k_th, k_sl = jax.random.split(key, 4)
    nu = theta * jax.random.normal(k_nu, (GDIM,))         # prior N(0, theta^2 I): theta enters the ellipse
    logy = lx + jnp.log(jax.random.uniform(k_u))
    th0 = 2 * jnp.pi * jax.random.uniform(k_th)

    def prop(t):
        p = x * jnp.cos(t) + nu * jnp.sin(t)
        return p, loglik(p)
    p, lp = prop(th0)

    def body(v):
        _, i, _, tmin, tmax, _ = v
        t = jax.random.uniform(jax.random.fold_in(k_sl, i), minval=tmin, maxval=tmax)
        p, lp = prop(t)
        return lp, i + 1, t, jnp.where(t < 0, t, tmin), jnp.where(t > 0, t, tmax), p
    lp, i, t, _, _, p = lax.while_loop(lambda v: v[0] <= logy, body,
                                       (lp, 1, th0, th0 - 2 * jnp.pi, th0, p))
    return p, lp, i


def g_draw_scan(loglik, theta, x, lx, key, maxsub):
    """The rewrite a user has to write today: the same transition as a masked `lax.scan`."""
    k_nu, k_u, k_th, k_sl = jax.random.split(key, 4)
    nu = theta * jax.random.normal(k_nu, (GDIM,))
    logy = lx + jnp.log(jax.random.uniform(k_u))
    th0 = 2 * jnp.pi * jax.random.uniform(k_th)

    def prop(t):
        p = x * jnp.cos(t) + nu * jnp.sin(t)
        return p, loglik(p)
    p0, lp0 = prop(th0)

    def body(c, i):
        lp, tmin, tmax, p, live, n = c
        t = jax.random.uniform(jax.random.fold_in(k_sl, i + 1), minval=tmin, maxval=tmax)
        pn, lpn = prop(t)
        upd = live & (lp <= logy)
        lp = jnp.where(upd, lpn, lp)
        p = jnp.where(upd, pn, p)
        tmin = jnp.where(upd & (t < 0), t, tmin)
        tmax = jnp.where(upd & (t > 0), t, tmax)
        return (lp, tmin, tmax, p, live, n + upd.astype(jnp.int32)), None
    c, _ = lax.scan(body, (lp0, th0 - 2 * jnp.pi, th0, p0, jnp.bool_(True), 1), jnp.arange(maxsub))
    return c[3], c[0], c[5]


def g_chain(loglik, n_draws, kind, maxsub, remat):
    def chain(theta, x0, key):
        lx = loglik(x0)

        def step(c, i):
            x, lx, acc, ns = c
            k = jax.random.fold_in(key, i)
            if kind == "while":
                x, lx, n = g_draw_while(loglik, theta, x, lx, k)
            else:
                x, lx, n = g_draw_scan(loglik, theta, x, lx, k, maxsub)
            return (x, lx, acc + x, ns + n), None
        body = jax.checkpoint(step) if remat else step
        c, _ = lax.scan(body, (x0, lx, jnp.zeros((GDIM,), jnp.float32), 0), jnp.arange(n_draws))
        return c[2], c[3]
    return chain


def g_arm(arm, loglik, n_draws, n, maxsub):
    """-> loss(theta, x0, keys) -> (scalar loss, total shrinkage steps)."""
    if arm == "tm":
        cols = jnp.arange(GDIM, dtype=jnp.int32)

        def item(t, x0, key, theta):
            lx = loglik(x0)

            def step(c, i):
                x, lx, ns, t = c
                x, lx, nn = g_draw_while(loglik, theta, x, lx, jax.random.fold_in(key, i))
                t = t.add("sum", cols, x)
                return (x, lx, ns + nn, t), None
            (x, lx, ns, t), _ = lax.scan(step, (x0, lx, 0, t), jnp.arange(n_draws))
            t = t.add("steps", 0, ns.astype(jnp.float32))
            return t
        f = tm.vmap(item, outputs={"sum": tm.Sum((GDIM,)), "steps": tm.Sum((1,))},
                    in_axes=(0, 0, None), max_steps=4096)

        def loss(theta, x0, keys):
            r = f(x0, keys, theta)
            return (jnp.sum(r.outputs["sum"] ** 2) / jnp.float32(float(n) * float(n_draws)) ** 2,
                    r.outputs["steps"][0])
        return loss
    kind = "while" if arm == "while" else "scan"
    chain = g_chain(loglik, n_draws, kind, maxsub, arm == "ckpt")

    def loss(theta, x0, keys):
        s, ns = jax.vmap(chain, in_axes=(None, 0, 0))(theta, x0, keys)
        return jnp.sum(s.sum(0) ** 2) / jnp.float32(float(n) * float(n_draws)) ** 2, ns.sum()
    return loss


# ------------------------------------------------------------------------------------------------ one measurement
def one(a):
    D.banner(f"{a.prog} arm={a.arm} mode={a.mode} n={a.n} div={a.div} draws={a.draws} dim={a.dim}", LIBS)
    base = dict(demo="nuts_blackjax", prog=a.prog, arm=a.arm, mode=a.mode, n=a.n, div=a.div,
                draws=a.draws, dim=a.dim)
    if a.prog == "essgrad":
        loglik = g_loglik(a.div)
        x0 = 0.2 * jax.random.normal(jax.random.PRNGKey(7), (a.n, GDIM), jnp.float32)
        keys = jax.random.split(jax.random.PRNGKey(0), a.n)
        theta = jnp.float32(1.0)
        loss = g_arm(a.arm, loglik, a.draws, a.n, a.maxsub)
        base.update(dim=GDIM, maxsub=a.maxsub)
        if a.mode == "fwd":
            fn = loss
        elif a.mode == "grad":
            fn = lambda th, x, k: jax.grad(lambda t: loss(t, x, k)[0])(th)               # noqa: E731
        else:
            raise SystemExit(a.mode)

        def go_g():
            out, r = D.measure(fn, (theta, x0, keys), reps=a.reps)
            D.row(**base, **r, status="ok",
                  value=float(np.asarray(out if a.mode == "grad" else out[0])),
                  count_total=(float(np.asarray(out[1])) if a.mode == "fwd" else -1.0))
        return D.guard(go_g, base)

    build = {"nuts": nuts_build, "ess": ess_build}[a.prog]
    alg, x0, ss = build(a)
    base["step_size"] = ss
    if a.prog == "nuts":
        base["doublings"] = a.doublings
    if a.arm == "tm":
        keys = jax.random.split(jax.random.PRNGKey(0), a.n)
        fn, _ = arm_tm(alg, a.prog, a.draws, a.mode, a.n, a.dim, a.maxsteps)
        args = (x0, keys, jnp.arange(a.n, dtype=jnp.int32))
    elif a.arm in ("bj_vmap", "bj_pmap"):
        keys = jax.random.split(jax.random.PRNGKey(0), a.draws)
        g = arm_bj_vmap if a.arm == "bj_vmap" else arm_bj_pmap
        fn = lambda x, k: g(alg, a.prog, x, k, a.draws, a.mode)                          # noqa: E731
        args = (x0, keys)
    else:
        raise SystemExit(f"unknown arm {a.arm}")

    def go():
        out, r = D.measure(fn, args, reps=a.reps)
        D.row(**base, **r, count_total=int(np.asarray(out[1])), status="ok",
              devices=jax.local_device_count())
    D.guard(go, base)


# -------------------------------------------------------------------------------------------------------- --quick
class _A:
    pass


def quick(a):
    checks = []
    n = min(a.n, 128)
    D.banner(f"quick chains={n}", LIBS)

    # (1) and (2): both arms run blackjax's kernel.  They consume different random streams (different key
    # plumbing), so the contract is statistical: pool each chain's draws into a chain mean and compare the
    # two ensembles.  The number of inner-loop steps must agree to Monte-Carlo noise.
    for prog, draws, dim in (("nuts", 8, 4), ("ess", 16, 6)):
        q = _A()
        q.n, q.div, q.dim, q.draws, q.doublings, q.nadapt = n, "mid", dim, draws, 6, 200
        alg, x0, _ = {"nuts": nuts_build, "ess": ess_build}[prog](q)
        kb = jax.random.split(jax.random.PRNGKey(0), draws)
        pb, cb = jax.jit(lambda x, k: arm_bj_vmap(alg, prog, x, k, draws, "store"))(x0, kb)
        kt = jax.random.split(jax.random.PRNGKey(0), n)
        call, _ = arm_tm(alg, prog, draws, "store", n, dim, 1 << 16)
        pt, ct = jax.jit(call)(x0, kt, jnp.arange(n, dtype=jnp.int32))
        P, Q = np.asarray(pb), np.asarray(pt)
        cmb, cmt = P.mean(1), Q.mean(1)
        se = np.sqrt(cmb.var(0, ddof=1) / n + cmt.var(0, ddof=1) / n)
        z = float(np.max(np.abs(cmb.mean(0) - cmt.mean(0)) / (se + 1e-12)))
        sd = float(np.max(np.abs(P.reshape(-1, dim).std(0) - Q.reshape(-1, dim).std(0))
                          / P.reshape(-1, dim).std(0)))
        ratio = float(int(ct) / max(int(cb), 1))
        checks.append((f"{prog}: posterior agrees with blackjax vmap (chain-pooled z)", z < 4.0,
                       f"z = {z:.2f}, marginal sd within {100 * sd:.1f}%"))
        checks.append((f"{prog}: both arms do the same work ({CNAME[prog]})", 0.8 < ratio < 1.25,
                       f"tallymap {int(ct):,} vs blackjax {int(cb):,} ({ratio:.4f})"))

    # (3) essgrad: tallymap's gradient must equal the padded rewrite's, and `grad` of the `while_loop` arm
    # must raise.  Both arms must take the same number of shrinkage steps, or they are not the same program.
    nn, draws = min(a.n, 256), 8
    loglik = g_loglik("mid")
    x0 = 0.2 * jax.random.normal(jax.random.PRNGKey(7), (nn, GDIM), jnp.float32)
    keys = jax.random.split(jax.random.PRNGKey(0), nn)
    th = jnp.float32(1.0)
    ls = {k: g_arm(k, loglik, draws, nn, GMAXSUB) for k in ("tm", "while", "scan")}
    v = {k: jax.jit(f)(th, x0, keys) for k, f in ls.items()}
    gt = float(jax.jit(jax.grad(lambda t: ls["tm"](t, x0, keys)[0]))(th))
    gs = float(jax.jit(jax.grad(lambda t: ls["scan"](t, x0, keys)[0]))(th))
    jt = float(jax.jit(lambda t: jax.jvp(lambda u: ls["tm"](u, x0, keys)[0], (t,), (1.0,))[1])(th))
    steps = {k: float(np.asarray(v[k][1])) for k in v}
    checks.append(("essgrad: the padded scan covers every shrinkage step",
                   steps["scan"] == steps["while"], f"scan {steps['scan']:.0f} vs while {steps['while']:.0f}"))
    checks.append(("essgrad: forward, tallymap vs the same while_loop program under vmap",
                   D.rel(v["tm"][0], v["while"][0]) < 1e-5,
                   f"rel = {D.rel(v['tm'][0], v['while'][0]):.2g}"))
    checks.append(("essgrad: grad, tallymap vs the padded-scan rewrite",
                   abs(gt - gs) / (abs(gs) + 1e-30) < 1e-4,
                   f"{gt:.8g} vs {gs:.8g}, rel = {abs(gt - gs) / (abs(gs) + 1e-30):.2g}"))
    checks.append(("essgrad: jvp agrees with grad inside tallymap",
                   abs(jt - gt) / (abs(gt) + 1e-30) < 1e-4,
                   f"rel = {abs(jt - gt) / (abs(gt) + 1e-30):.2g}"))
    try:
        jax.jit(jax.grad(lambda t: ls["while"](t, x0, keys)[0]))(th)
        msg, raised = "no error: JAX now reverse-differentiates while_loop", False
    except Exception as e:                                                               # noqa: BLE001
        msg, raised = f"{type(e).__name__}: {str(e)[:90]}", True
    checks.append(("essgrad: plain JAX cannot take this gradient", raised, msg))
    return D.verdict(checks)


# --------------------------------------------------------------------------------------------------------- --full
def plots(rows, meta):
    """Redraw plots/ from `rows`.  Called by --full and by --plots."""
    P = os.path.join(HERE, "plots")
    lab2 = {"tm": "tallymap", "bj_vmap": "blackjax + jax.vmap", "bj_pmap": "blackjax + jax.pmap"}
    D.panel(os.path.join(P, "nuts_fwd_time.png"),
            [r for r in rows if r.get("prog") == "nuts" and r.get("mode") == "store"
             and r.get("div") == "mid" and r.get("dim") == 8],
            ("tm", "bj_vmap"), lab2, "BlackJAX NUTS, 32 draws/chain, dim 8: time per call",
            "ms per call", xlabel="chains")
    D.panel(os.path.join(P, "nuts_dim.png"),
            [r for r in rows if r.get("prog") == "nuts" and r.get("mode") == "expect"],
            ("tm", "bj_vmap"), lab2,
            "BlackJAX NUTS at 4,096 chains: time vs model dimension (per-chain state size)",
            "ms per call", xlabel="dimension", xkey="dim",
            note="Each dimension is adapted on its own target, so read the ratio, not the absolute times.")
    D.panel(os.path.join(P, "ess_fwd_time.png"),
            [r for r in rows if r.get("prog") == "ess"], ("tm", "bj_vmap"), lab2,
            "BlackJAX elliptical slice, 64 draws/chain, dim 10: time per call",
            "ms per call", xlabel="chains")
    lab4 = {"tm": "tallymap", "while": "vmap + while_loop", "scan": "padded scan", "ckpt": "padded scan + remat"}
    for mode, title in (("fwd", "forward"), ("grad", "reverse-mode gradient")):
        D.panel(os.path.join(P, f"essgrad_{mode}_time.png"),
                [r for r in rows if r.get("prog") == "essgrad" and r.get("mode") == mode],
                ("tm", "while", "scan", "ckpt"), lab4,
                f"Gradient through an elliptical-slice chain, {title}: time",
                "ms per call", xlabel="chains",
                note=("`grad` of the while_loop arm raises in JAX, so it has no line here."
                      if mode == "grad" else None))
    D.panel(os.path.join(P, "essgrad_grad_mem.png"),
            [r for r in rows if r.get("prog") == "essgrad" and r.get("mode") == "grad"],
            ("tm", "scan", "ckpt"), lab4,
            "Gradient through an elliptical-slice chain: peak device memory",
            "peak device MB", xlabel="chains", ykey="peak_mb")

def full(a):
    script = os.path.join(HERE, "run.py")
    out = os.path.join(HERE, "results", "nuts_blackjax.json")
    jobs = []
    for n in (256, 1024, 4096, 16384):
        for arm in ("tm", "bj_vmap"):
            jobs.append(dict(prog="nuts", arm=arm, mode="store", n=n, div="mid", draws=32, dim=8,
                             doublings=8))
    jobs.append(dict(prog="nuts", arm="bj_pmap", mode="store", n=4096, div="mid", draws=32, dim=8,
                     doublings=8))
    for d in ("low", "high"):
        for arm in ("tm", "bj_vmap"):
            jobs.append(dict(prog="nuts", arm=arm, mode="store", n=4096, div=d, draws=32, dim=8,
                             doublings=8))
    for dim in (2, 8, 16, 32, 64):
        for arm in ("tm", "bj_vmap"):
            jobs.append(dict(prog="nuts", arm=arm, mode="expect", n=4096, div="mid", draws=16, dim=dim,
                             doublings=6))
    for n in (1024, 4096, 16384, 65536):
        for arm in ("tm", "bj_vmap"):
            jobs.append(dict(prog="ess", arm=arm, mode="expect", n=n, div="mid", draws=64, dim=10))
    for n in (4096, 16384, 65536):
        for arm in ("tm", "while", "scan", "ckpt"):
            for mode in ("fwd", "grad"):
                jobs.append(dict(prog="essgrad", arm=arm, mode=mode, n=n, div="mid", draws=16))
    rows, meta = D.sweep(script, jobs, out, libs=LIBS)

    plots(rows, meta)

    print(f"\n{len(rows)} rows -> {out}")
    return 0


def main():
    p = D.parser(draws=32, dim=8, doublings=8, maxsteps=200000, nadapt=600, maxsub=GMAXSUB)
    p.add_argument("--prog", default="nuts", choices=("nuts", "ess", "essgrad"))
    a = p.parse_args()
    if a.one:
        one(a)
        return 0
    if a.plots:
        rows, meta = D.load(os.path.join(HERE, "results", "nuts_blackjax.json"))
        plots(rows, meta)
        return 0
    return quick(a) if a.quick else full(a)


if __name__ == "__main__":
    sys.exit(main())
