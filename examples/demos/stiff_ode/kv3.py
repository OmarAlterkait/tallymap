"""Kvaerno3 (ESDIRK 3/2) + VeryChord + diffrax's default PI controller, written as ONE item function.

This file is NOT diffrax.  It is a faithful re-implementation of what
`diffrax.diffeqsolve(..., Kvaerno3(), PIDController())` does for a single trajectory, so that the same
algorithm can be run as a tallymap item, under `jax.vmap` + `lax.while_loop`, and checked against diffrax
itself.  Why a re-implementation is needed, and what it costs in fairness, is in this demo's README.

Sources for every constant and formula (line numbers are diffrax 0.7.2, the version the measurements ran
against; the same code is at different lines in 0.6.2 and nothing below changed between them):

  tableau                 diffrax/_solver/kvaerno3.py:15-42
  k_i is dt * f           diffrax/_solver/runge_kutta.py:250-287 (`_implicit_relation_f` / `_implicit_relation_k`)
  chord iteration + stop  diffrax/_root_finder/_verychord.py:114-166 (kappa=1e-2 at :73, >= 2 iterations at
                          :157); at most 10 chord steps from diffrax/_solver/kvaerno3.py:74-75
  error + step control    diffrax/_step_size_controller/pid.py:394-530 (scaled_error form (1) at :492,
                          `safety` outside the exponent, factormin 0.2, factormax 10, safety 0.9)
  error_order = 4         diffrax/_solver/base.py:294-300 (for an ODE: order + 1; Kvaerno3.order() = 3)
  initial step            diffrax/_step_size_controller/pid.py:33-88 (Hairer's algorithm)

With the default `PIDController(pcoeff=0, icoeff=1, dcoeff=0)` the P and D factors are exactly 1
(pid.py:512-520), so the controller state is just `dt`.

The state is 3-D: y = (x, v, z) with z' = x^2, so the quantity the demo tallies and differentiates is a smooth
path functional that BOTH arms compute inside the solver and under the same error control.
"""
import jax.numpy as jnp
from jax import lax

G = 0.43586652150
A21 = G
A31 = (-4 * G ** 2 + 6 * G - 1) / (4 * G)
A32 = (-2 * G + 1) / (4 * G)
A41 = (6 * G - 1) / (12 * G)
A42 = -1 / ((24 * G - 12) * G)
A43 = (-6 * G ** 2 + 6 * G - 1) / (6 * G - 3)
TH = 1 / (2 * G)
AL21, AL31, AL32 = 1.0, 1.0 - TH, TH
AL41, AL42, AL43 = A31, A32, G
C2, C3, C4 = 2 * G, 1.0, 1.0
B1, B2, B3, B4 = A41, A42, A43, G
E1, E2, E3, E4 = A41 - A31, A42 - A32, A43 - G, G
ERROR_ORDER = 4.0
SAFETY, FACTORMIN, FACTORMAX = 0.9, 0.2, 10.0
KAPPA = 1e-2
CHORD_MAX = 10
RTOL_DEF, ATOL_DEF = 1e-4, 1e-6        # f32-appropriate; the same numbers go to diffrax


def rms(x):
    return jnp.sqrt(jnp.mean(x * x))


def maxnorm(x):
    return jnp.max(jnp.abs(x))


# ---------------------------------------------------------------------------------- vector field (3-D)
def vf(t, y, mu, th):
    w2, amp, om = th[0], th[1], th[2]
    return jnp.stack([y[1],
                      mu * (1.0 - y[0] * y[0]) * y[1] - w2 * y[0] + amp * jnp.cos(om * t),
                      y[0] * y[0]])


def _jac(y, mu, th):
    """d vf / d y (3x3, analytic)."""
    z = jnp.zeros_like(y[0])
    o = jnp.ones_like(y[0])
    return jnp.stack([jnp.stack([z, o, z]),
                      jnp.stack([-2.0 * mu * y[0] * y[1] - th[0], mu * (1.0 - y[0] * y[0]), z]),
                      jnp.stack([2.0 * y[0], z, z])])


def _solve3(m, r):
    """3x3 solve m x = r by Cramer's rule."""
    c0 = m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]
    c1 = m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2]
    c2 = m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]
    det = m[0, 0] * c0 + m[0, 1] * c1 + m[0, 2] * c2
    a0 = r[0] * c0 + r[1] * (m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2]) + r[2] * (m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1])
    a1 = r[0] * c1 + r[1] * (m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]) + r[2] * (m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2])
    a2 = r[0] * c2 + r[1] * (m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1]) + r[2] * (m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])
    return jnp.stack([a0 / det, a1 / det, a2 / det])


def init_dt(t0, y0, f0, mu, th, rtol, atol):
    """diffrax `_select_initial_step` (pid.py:33-88)."""
    scale = atol + jnp.abs(y0) * rtol
    d0, d1 = rms(y0 / scale), rms(f0 / scale)
    c = (d0 < 1e-5) | (d1 < 1e-5)
    d1s = jnp.where(c, 1.0, d1)
    h0 = jnp.where(c, 1e-6, 0.01 * (d0 / d1s))
    f1 = vf(t0 + h0, y0 + h0 * f0, mu, th)
    d2 = rms((f1 - f0) / scale) / h0
    md = jnp.maximum(d1, d2)
    h1 = jnp.where(md <= 1e-15, jnp.maximum(1e-6, h0 * 1e-3), (0.01 / md) ** (1.0 / ERROR_ORDER))
    return jnp.minimum(100.0 * h0, h1)


def _chord(ti, ypart, fpred, m, h, mu, th, rtol, atol):
    """VeryChord on F(f) = f - vf(ti, ypart + G h f) = 0 with the frozen matrix m = I - G h J.

    The root variable is the DERIVATIVE f, not the increment h f: diffrax solves `_implicit_relation_f`
    (runge_kutta.py:250-267, taken whenever `eval_fs`), so VeryChord's convergence scale is
    `atol + rtol |f|` and not `atol + rtol |h f|` (_verychord.py:134-136).  (MEASURED: matching this made no
    difference to the step counts - the chord converges in the minimum two iterations either way - but it is the
    faithful form, so it is what this does.)

    -> (f, iterations, failed); a failed solve makes the whole STEP fail (optimistix/_iterate.py:251-256,
    runge_kutta.py:1194-1197)."""
    res = 10.0 ** (2 - jnp.finfo(jnp.float32).precision)

    def body(s):
        i, fi, ds, _ = s
        F = fi - vf(ti, ypart + (G * h) * fi, mu, th)
        d = _solve3(m, F)
        fn = fi - d
        scale = atol + rtol * jnp.abs(fn)
        return i + 1, fn, maxnorm(d / scale), ds

    def cond(s):
        i, _, ds, dsp = s
        rate = ds / dsp
        factor = ds * rate / (1.0 - rate)
        small = ds < res
        diverged = jnp.invert(jnp.isfinite(rate)) | (rate > 2.0)
        converged = (factor > 0) & (factor < KAPPA)
        return jnp.invert((i >= 2) & (small | diverged | converged)) & (i < CHORD_MAX)

    i, fi, ds, dsp = lax.while_loop(cond, body, (jnp.int32(0), fpred, jnp.float32(jnp.inf), jnp.float32(1.0)))
    rate = ds / dsp
    factor = ds * rate / (1.0 - rate)
    small = ds < res
    diverged = jnp.invert(jnp.isfinite(rate)) | (rate > 2.0)
    converged = (factor > 0) & (factor < KAPPA)
    term = (i >= 2) & (small | diverged | converged)
    fail = jnp.invert(term) | (jnp.invert(small) & (diverged | jnp.invert(converged)))
    return fi, i, fail


def kv3_step(t0, h, y0, f0, mu, th, rtol, atol):
    """One Kvaerno3 step from (t0, y0), f0 = vf(t0, y0) (FSAL). -> (y1, yerr, f1, chord_iters).

    The Jacobian is evaluated ONCE per step at the second-stage predictor, which is what
    `CalculateJacobian.second_stage` means for an ESDIRK (runge_kutta.py:960-978, `jac_stage_index = 1` at :968),
    and it is stop-gradient-ed exactly as diffrax does (`_filter_stop_gradient`, runge_kutta.py:973)."""
    k1 = h * f0
    ylin = y0 + (A21 + G) * k1                                     # stage-2 partial + G h * predictor(= f0)
    m = lax.stop_gradient(jnp.eye(3, dtype=y0.dtype) - (G * h) * _jac(ylin, mu, th))
    f2, i2, b2 = _chord(t0 + C2 * h, y0 + A21 * k1, AL21 * f0, m, h, mu, th, rtol, atol)
    k2 = h * f2
    f3, i3, b3 = _chord(t0 + C3 * h, y0 + A31 * k1 + A32 * k2, AL31 * f0 + AL32 * f2, m, h, mu, th, rtol, atol)
    k3 = h * f3
    f4, i4, b4 = _chord(t0 + C4 * h, y0 + A41 * k1 + A42 * k2 + A43 * k3,
                        AL41 * f0 + AL42 * f2 + AL43 * f3, m, h, mu, th, rtol, atol)
    k4 = h * f4
    y1 = y0 + B1 * k1 + B2 * k2 + B3 * k3 + B4 * k4
    yerr = E1 * k1 + E2 * k2 + E3 * k3 + E4 * k4
    yerr = jnp.where(b2 | b3 | b4, jnp.float32(jnp.inf), yerr)
    return y1, yerr, f4, i2 + i3 + i4                              # FSAL: f4 = vf(t0 + h, y1)


def control(h, y0, y1, yerr, rtol, atol):
    """PIDController(pcoeff=0, icoeff=1, dcoeff=0).adapt_step_size -> (keep, h_next).  pid.py:394-530."""
    scale = atol + jnp.maximum(jnp.abs(y0), jnp.abs(y1)) * rtol
    err = rms(yerr / scale)
    keep = err < 1.0
    fmin = jnp.where(keep, 1.0, FACTORMIN)
    fac = jnp.clip(SAFETY * (1.0 / err) ** (1.0 / ERROR_ORDER), fmin, FACTORMAX)
    fac = lax.stop_gradient(fac)                 # diffrax stop-gradients the factor (pid.py:522-530)
    return keep, h * fac


# ---------------------------------------------------------------------------------- the solve, as a pure function
def hermite(t0, h, y0, y1, f0, f1, s, j):
    """Cubic Hermite interpolation of component j at t0 + s h (diffrax uses 3rd-order Hermite for Kvaerno3,
    kvaerno3.py:70-72)."""
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    h11 = s ** 3 - s ** 2
    return h00 * y0[j] + h10 * h * f0[j] + h01 * y1[j] + h11 * h * f1[j]


def solve(y0, mu, th, t_end, rtol=RTOL_DEF, atol=ATOL_DEF, event=False, z_target=30.0, nbisect=40):
    """Integrate ONE trajectory.  -> (y, t, nsteps, naccept, chord_iters).

    `event`: stop when the path functional z reaches `z_target`, and locate the crossing time by bisection on the
    cubic Hermite interpolant of z.  This is exactly what `diffrax.Event(lambda t, y, args, **kw: y[2] - Z,
    root_finder)` locates: z is monotone in t, so there is one crossing and the two arms detect the same event
    (a sign-change detector on an oscillating component would not be the same event in the two arms)."""
    f0 = vf(jnp.float32(0.0), y0, mu, th)
    h0 = init_dt(jnp.float32(0.0), y0, f0, mu, th, rtol, atol)

    def body(c):
        t, h, y, f, ns, na, ci, done = c
        hc = jnp.minimum(h, t_end - t)
        y1, yerr, f1, k = kv3_step(t, hc, y, f, mu, th, rtol, atol)
        keep, hn = control(hc, y, y1, yerr, rtol, atol)
        if event:
            cross = keep & (y1[2] >= z_target)

            def refine(_):
                lo, hi = jnp.float32(0.0), jnp.float32(1.0)

                def bb(s):
                    j, lo, hi = s
                    mid = 0.5 * (lo + hi)
                    zm = hermite(t, hc, y, y1, f, f1, mid, 2) - z_target
                    return j + 1, jnp.where(zm < 0, mid, lo), jnp.where(zm < 0, hi, mid)
                j, lo, hi = lax.while_loop(lambda s: s[0] < nbisect, bb, (0, lo, hi))
                s = 0.5 * (lo + hi)
                ye = jnp.stack([hermite(t, hc, y, y1, f, f1, s, j) for j in range(3)])
                return t + s * hc, ye
            te, ye = lax.cond(cross, refine, lambda _: (t + hc, y1), None)
            tn = jnp.where(cross, te, jnp.where(keep, t + hc, t))
            yn = jnp.where(cross, ye, jnp.where(keep, y1, y))
            return tn, hn, yn, jnp.where(keep, f1, f), ns + 1, na + keep, ci + k, done | cross
        tn = jnp.where(keep, t + hc, t)
        return tn, hn, jnp.where(keep, y1, y), jnp.where(keep, f1, f), ns + 1, na + keep, ci + k, done

    init = (jnp.float32(0.0), h0, y0, f0, jnp.int32(0), jnp.int32(0), jnp.int32(0), jnp.bool_(False))
    t, h, y, f, ns, na, ci, done = lax.while_loop(lambda c: (c[0] < t_end) & jnp.invert(c[7]), body, init)
    return y, t, ns, na, ci
