"""A stiff ODE ensemble: diffrax as it ships against the same solver run as a tallymap item.

The problem is a forced Van der Pol oscillator with a per-trajectory stiffness `mu`, augmented with a path
functional so that the quantity being tallied is smooth and under the same error control in both arms:

    x' = v,   v' = mu (1 - x^2) v - w2 x + A cos(om t),   z' = x^2,   t in [0, 20]

`theta = (w2, A, om)` are the three differentiated parameters; `mu` is the divergence knob, drawn
log-uniformly per trajectory.  The solver is Kvaerno3, a 4-stage A-L-stable ESDIRK with an embedded 2nd-order
error estimate, with diffrax's `VeryChord` root finder and its default `PIDController(rtol=1e-4, atol=1e-6)`.
The item therefore contains three nested data-dependent loops -- the adaptive time loop, the Runge-Kutta stage
sequence and a chord solve per stage -- which is diffrax's own structure (`_integrate.py:685`,
`runge_kutta.py:1143-1160`, `runge_kutta.py:986-1008`).

Arms:
  diffrax     `diffrax.diffeqsolve(ODETerm(...), Kvaerno3(), PIDController(rtol=1e-4, atol=1e-6))` under
              `jax.vmap`, with `RecursiveCheckpointAdjoint()` (its default) for `grad` and `ForwardMode()` for
              `jvp`/`jacfwd`, `throw=False` (which is what you need under `vmap`, because a per-element solver
              error cannot be raised) and `max_steps` set to the observed maximum rounded up.
  jax_while   `jax.vmap` of the item function in `kv3.py`, with its `lax.while_loop`.  Forward only: JAX
              cannot reverse-differentiate a `while_loop`.  This is the fairness control that matters, because
              it runs the IDENTICAL Python function as `tm`.
  tm          the same `kv3.solve` as a tallymap item.

IMPORTANT, and the README says it again: `kv3.py` is a re-implementation of Kvaerno3, not diffrax's code.  It
agrees with diffrax on the answer but takes a different number of internal steps, so every forward table is
reported twice: raw wall time, and nanoseconds per item-solver-step, which divides the step-count difference
out.  The `tm` versus `jax_while` columns are unaffected, because those two arms run the same code.

  python run.py --quick
  python run.py --full
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _demo as D  # noqa: E402
import kv3  # noqa: E402

import tallymap as tm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LIBS = ("diffrax", "optimistix", "jax")
NB = 32
T_END = 20.0
Z_TARGET = 30.0        # the event: the path functional z reaches this value
DIV = {"low": (8.0, 12.0), "mid": (1.0, 30.0), "high": (1.0, 300.0)}       # the per-item mu range
THETA0 = np.asarray([1.0, 0.3, 0.9], np.float32)
ARMS = ("tm", "jax_while", "diffrax")
LABEL = {"tm": "tallymap", "jax_while": "vmap + while_loop", "diffrax": "diffrax (as it ships)"}


def inputs(n, div, seed=0):
    lo, hi = DIV[div]
    r = np.random.default_rng(seed)
    mu = np.exp(r.uniform(np.log(lo), np.log(hi), n)).astype(np.float32)
    y0 = np.stack([np.full(n, 2.0), np.zeros(n), np.zeros(n)], 1).astype(np.float32)
    gid = (np.arange(n) % NB).astype(np.int32)
    return jnp.asarray(y0), jnp.asarray(mu), jnp.asarray(gid)


# ------------------------------------------------------------------------------------------------- tallymap arm
def make_tm(ev, maxsteps, **opts):
    def item(t, y0, mu, gid, th):
        y, te, ns, na, ci = kv3.solve(y0, mu, th, jnp.float32(T_END), event=ev, z_target=Z_TARGET)
        t = t.add("obs", gid, y[2])
        t = t.add("cnt", jnp.asarray([0, 1]), jnp.stack([ns.astype(jnp.float32), na.astype(jnp.float32)]))
        return t, jnp.stack([y[0], y[1], y[2], te, ns.astype(jnp.float32)])
    return tm.vmap(item, outputs={"obs": tm.Sum((NB,)), "cnt": tm.Sum((2,))},
                   in_axes=(0, 0, 0, None), max_steps=maxsteps, **opts)


def _reduce(y, te, ns, na, gids):
    obs = jnp.zeros((NB,), jnp.float32).at[gids].add(y[:, 2])
    cnt = jnp.stack([ns.sum().astype(jnp.float32), na.sum().astype(jnp.float32)])
    return {"obs": obs, "cnt": cnt}, jnp.stack([y[:, 0], y[:, 1], y[:, 2], te, ns.astype(jnp.float32)], 1)


# ------------------------------------------------------------------------------------------- plain jax.vmap arm
def make_jax_while(ev):
    def one(y0, mu, th):
        y, te, ns, na, ci = kv3.solve(y0, mu, th, jnp.float32(T_END), event=ev, z_target=Z_TARGET)
        return y, te, ns, na

    def f(y0s, mus, gids, th):
        return _reduce(*jax.vmap(one, in_axes=(0, 0, None))(y0s, mus, th), gids)
    return f


# ------------------------------------------------------------------------------------------------- diffrax arm
def make_diffrax(ev, maxsteps, adj="ckpt"):
    import diffrax as dfx
    import optimistix as optx

    term = dfx.ODETerm(lambda t, y, args: kv3.vf(t, y, args[0], args[1]))
    solver = dfx.Kvaerno3()
    ctrl = dfx.PIDController(rtol=kv3.RTOL_DEF, atol=kv3.ATOL_DEF)
    adjoint = {"ckpt": dfx.RecursiveCheckpointAdjoint(), "direct": dfx.DirectAdjoint(),
               "forward": dfx.ForwardMode()}[adj]
    event = None
    if ev:
        event = dfx.Event(lambda t, y, args, **kw: y[2] - jnp.float32(Z_TARGET),
                          optx.Newton(rtol=1e-6, atol=1e-6, norm=optx.rms_norm))

    def one(y0, mu, th):
        s = dfx.diffeqsolve(term, solver, t0=0.0, t1=T_END, dt0=None, y0=y0, args=(mu, th),
                            stepsize_controller=ctrl, max_steps=maxsteps, adjoint=adjoint, event=event,
                            saveat=dfx.SaveAt(t1=True), throw=False)
        return s.ys[-1], s.ts[-1], s.stats["num_steps"], s.stats["num_accepted_steps"]

    def f(y0s, mus, gids, th):
        return _reduce(*jax.vmap(one, in_axes=(0, 0, None))(y0s, mus, th), gids)
    return f


def wrap(arm_fn):
    def g(y0s, mus, gids, th):
        r = arm_fn(y0s, mus, gids, th)
        return r.outputs, r.per_item
    return g


def build(arm, a, adj=None):
    if arm == "tm":
        opts = {}
        if a.revck != 16:
            opts["rev_checkpoints"] = a.revck
        return wrap(make_tm(a.ev, a.maxsteps, **opts))
    if arm == "jax_while":
        return make_jax_while(a.ev)
    if arm == "diffrax":
        return make_diffrax(a.ev, a.maxsteps, adj or a.adj)
    raise SystemExit(f"unknown arm {arm}")


def loss_of(f):
    def loss(th, y0s, mus, gids):
        o, _ = f(y0s, mus, gids, th)
        return jnp.sum(o["obs"] ** 2)
    return loss


# ------------------------------------------------------------------------------------------------ one measurement
def one(a):
    D.banner(f"stiff arm={a.arm} mode={a.mode} n={a.n} div={a.div} ev={int(a.ev)} "
             f"maxsteps={a.maxsteps} adj={a.adj}", LIBS)
    y0s, mus, gids = inputs(a.n, a.div)
    th = jnp.asarray(THETA0)
    base = dict(demo="stiff_ode", arm=a.arm, mode=a.mode, n=a.n, div=a.div, ev=int(a.ev),
                maxsteps=a.maxsteps, adj=a.adj, revck=a.revck)

    if a.mode == "calib":                               # the observed maximum step count, which sizes diffrax
        f = build("tm", a)
        _, pi = jax.jit(f)(y0s, mus, gids, th)
        ns = np.asarray(pi[:, 4])
        D.row(**base, status="ok", steps_mean=float(ns.mean()), steps_max=float(ns.max()),
              steps_min=float(ns.min()), t_end_min=float(np.asarray(pi[:, 3]).min()))
        return

    # diffrax's reverse-mode adjoint is RecursiveCheckpointAdjoint; its forward-mode-capable loop is
    # ForwardMode().  Pick the one the mode needs rather than making the library look bad with the other.
    adj = a.adj if a.adj != "auto" else ("forward" if a.mode in ("jvp", "jacfwd") else "ckpt")
    base["adj"] = adj
    f = build(a.arm, a, adj)

    def go():
        if a.mode == "fwd":
            o, r = D.measure(lambda th_: f(y0s, mus, gids, th_), (th,), reps=a.reps)
            od, pi = o
            ns = np.asarray(pi[:, 4])
            D.row(**base, status="ok", **r, obs0=float(od["obs"][0]), steps=float(od["cnt"][0]),
                  accepted=float(od["cnt"][1]), steps_mean=float(ns.mean()), steps_max=float(ns.max()),
                  ns_per_item_step=float(r["ms"] * 1e6 / max(float(od["cnt"][0]), 1.0)))
        elif a.mode == "grad":
            o, r = D.measure(jax.grad(loss_of(f)), (th, y0s, mus, gids), reps=a.reps)
            D.row(**base, status="ok", **r, grad=np.asarray(o))
        elif a.mode == "jvp":
            tg = jnp.asarray([1.0, 0.0, 0.0], jnp.float32)
            o, r = D.measure(lambda th_, tg_: jax.jvp(lambda q: loss_of(f)(q, y0s, mus, gids),
                                                      (th_,), (tg_,)), (th, tg), reps=a.reps)
            D.row(**base, status="ok", **r, val=float(o[0]), tang=float(o[1]))
        elif a.mode == "jacfwd":
            o, r = D.measure(jax.jacfwd(lambda th_: loss_of(f)(th_, y0s, mus, gids)), (th,), reps=a.reps)
            D.row(**base, status="ok", **r, jac=np.asarray(o))
        else:
            raise SystemExit(f"unknown mode {a.mode}")
    D.guard(go, base)


# -------------------------------------------------------------------------------------------------------- --quick
class _A:
    pass


def quick(a):
    q = _A()
    q.n, q.div, q.ev, q.maxsteps, q.adj, q.revck = min(a.n, 256), "mid", False, 4096, "ckpt", 16
    D.banner(f"quick n={q.n} div={q.div}", LIBS)
    y0s, mus, gids = inputs(q.n, q.div)
    th = jnp.asarray(THETA0)
    out = {}
    for arm in ARMS:
        f = build(arm, q)
        o, pi = jax.block_until_ready(jax.jit(f)(y0s, mus, gids, th))
        g = None
        if arm != "jax_while":
            g = np.asarray(jax.block_until_ready(jax.jit(jax.grad(loss_of(f)))(th, y0s, mus, gids)))
        out[arm] = (np.asarray(o["obs"]), np.asarray(o["cnt"]), np.asarray(pi), g)
    ob, cb, pb, gb = out["diffrax"]
    checks = []
    for arm in ("tm", "jax_while"):
        o_, c_, p_, g_ = out[arm]
        ro, rz = D.rel(o_, ob), D.rel(p_[:, 2], pb[:, 2])
        # f32 and a different step sequence, so the contract here is the solver tolerance, not bitwise
        checks.append((f"{LABEL[arm]} vs diffrax: tally and terminal z", max(ro, rz) < 2e-3,
                       f"tally rel = {ro:.2g}, per-trajectory z rel = {rz:.2g}"))
    ot, ct, pt, gt = out["tm"]
    ow, cw, pw, _ = out["jax_while"]
    checks.append(("tallymap vs the identical while_loop program: same answer and same step counts",
                   D.rel(ot, ow) < 1e-5 and float(np.mean(pt[:, 4] == pw[:, 4])) == 1.0,
                   f"tally rel = {D.rel(ot, ow):.2g}, step counts equal for "
                   f"{100 * float(np.mean(pt[:, 4] == pw[:, 4])):.0f}% of trajectories"))
    rg = D.rel(gt, gb)
    checks.append(("the gradient agrees with diffrax's", rg < 5e-2,
                   f"rel = {100 * rg:.2f}% on a gradient of norm {np.linalg.norm(gb):.3g}; the two arms "
                   f"take different step sequences, so this is not a bitwise contract"))
    checks.append(("the re-implementation's step count is within 2x of diffrax's",
                   0.5 < ct[0] / cb[0] < 2.0,
                   f"{int(ct[0]):,} attempted steps vs diffrax's {int(cb[0]):,} "
                   f"(ratio {ct[0] / cb[0]:.2f}); forward tables are also reported per item-solver-step"))
    ns = pt[:, 4]
    checks.append(("the step count varies across trajectories", ns.max() > 1.2 * ns.mean(),
                   f"mean {ns.mean():.0f}, max {ns.max():.0f}"))

    # the event configuration: both arms must detect the SAME event
    q.ev = True
    oe = {}
    for arm in ("tm", "diffrax"):
        f = build(arm, q)
        o, pi = jax.block_until_ready(jax.jit(f)(y0s, mus, gids, th))
        oe[arm] = np.asarray(pi)
    rt = D.rel(oe["tm"][:, 3], oe["diffrax"][:, 3])
    checks.append(("with event detection, the stopping time agrees with diffrax's", rt < 5e-3,
                   f"event-time rel = {rt:.2g}"))
    return D.verdict(checks)


# --------------------------------------------------------------------------------------------------------- --full
NS = (16384, 65536, 262144)


def plots(rows, meta):
    """Redraw plots/ from `rows`.  Called by --full and by --plots."""
    P = os.path.join(HERE, "plots")
    fwd = [r for r in rows if r.get("mode") == "fwd" and r.get("div") == "mid" and not r.get("ev")]
    D.panel(os.path.join(P, "fwd_time.png"), fwd, ARMS, LABEL,
            "Stiff Van der Pol ensemble, Kvaerno3, forward: time per call", "ms per call")
    D.panel(os.path.join(P, "fwd_per_step.png"), fwd, ARMS, LABEL,
            "Forward cost per item-solver-step (this removes the step-count difference)",
            "ns per item-solver-step", ykey="ns_per_item_step")
    gr = [r for r in rows if r.get("mode") == "grad" and r.get("div") == "mid"]
    D.panel(os.path.join(P, "grad_time.png"), gr, ("tm", "diffrax"), LABEL,
            "Reverse-mode gradient w.r.t. the three forcing parameters: time", "ms per call",
            note="`grad` of the while_loop arm raises in JAX, so it has no line here.")
    D.panel(os.path.join(P, "grad_mem.png"), gr, ("tm", "diffrax"), LABEL,
            "Reverse-mode gradient: peak device memory", "peak device MB", ykey="peak_mb")

def full(a):
    script = os.path.join(HERE, "run.py")
    out = os.path.join(HERE, "results", "stiff_ode.json")
    MS = 896                      # the observed maximum of 875 at div=mid, rounded up to a multiple of 64
    jobs = [dict(arm="tm", mode="calib", n=4096, div=d, maxsteps=4096) for d in DIV]
    jobs += [dict(arm=arm, mode="fwd", n=n, div="mid", maxsteps=MS) for n in NS for arm in ARMS]
    jobs += [dict(arm=arm, mode="grad", n=n, div="mid", maxsteps=MS) for n in NS
             for arm in ("tm", "diffrax")]
    jobs += [dict(arm=arm, mode=m, n=65536, div="mid", maxsteps=MS, adj="auto")
             for m in ("jvp", "jacfwd") for arm in ("tm", "diffrax")]
    jobs += [dict(arm=arm, mode="fwd", n=65536, div=d, maxsteps=MS) for d in ("low", "high") for arm in ARMS]
    jobs += [dict(arm=arm, mode="fwd", n=65536, div="mid", maxsteps=MS, ev=True) for arm in ARMS]
    rows, meta = D.sweep(script, jobs, out, libs=LIBS, extra_meta=dict(ns=list(NS), maxsteps=MS))

    plots(rows, meta)

    print(f"\n{len(rows)} rows -> {out}")
    return 0


def main():
    p = D.parser(maxsteps=896, revck=16)
    p.add_argument("--ev", action="store_true", help="stop each trajectory when the path functional hits 30")
    p.add_argument("--adj", default="ckpt", choices=("ckpt", "direct", "forward", "auto"))
    a = p.parse_args()
    if a.one:
        one(a)
        return 0
    if a.plots:
        rows, meta = D.load(os.path.join(HERE, "results", "stiff_ode.json"))
        plots(rows, meta)
        return 0
    return quick(a) if a.quick else full(a)


if __name__ == "__main__":
    sys.exit(main())
