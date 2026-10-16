"""A gradient that is exactly computed and still not the derivative you want.

    python examples/gradient_pitfall.py          # CPU, under a minute

tallymap's job is to return the exact derivative of the program you wrote.  That is not the same thing as the
derivative of the quantity you are estimating.  This example shows the gap twice, on a random walk with absorbing
boundaries, and shows the check that detects it.

A. theta scales the step size.  `grad` and `jacfwd` agree to a few parts in 1e7, so the mechanics are right, and
   the answer is still about an order of magnitude away from the true slope.  The reason: the NUMBER of steps a
   walker takes depends on theta, and a pathwise derivative differentiates the steps a walker took, not the
   decision to take one more.  The missing piece is a boundary term.

B. p is the probability of stepping left, so theta enters only through a discrete comparison.  The derivative of
   a comparison is zero almost everywhere, so `grad` returns exactly 0 while the true slope is large.  Here you
   need a score-function (REINFORCE) estimator, which tallymap cannot invent for you.

The detector in both cases is a finite difference on the SAME per-item keys (common random numbers), used only to
TEST the gradient, never to compute it.  Common random numbers are what make the difference cheap enough to be
usable: without them the Monte Carlo noise swamps the signal.

See README.md, "Gradients: exactly computed is not the same as what you want", and docs/validation.md.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm

L = 10.0                              # the walk is absorbed outside [0, L]
N = 8192                              # items (walkers)
MAX_STEPS = 20_000                    # a safety net only: the observed maximum is a few hundred

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def make_walk(scaled):
    """Build the item function.  scaled=True: theta scales the step.  scaled=False: p is a left-step probability."""
    def item(t, x0, key, theta):
        def body(c):
            x, k, t = c
            u = jax.random.uniform(jax.random.fold_in(key, k), (2,))
            if scaled:
                step = theta * u[0] * jnp.where(u[1] < 0.5, -1.0, 1.0)
            else:
                step = jnp.where(u[1] < theta, -1.0, 1.0)
            return x + step, k + 1, t
        x, k, t = lax.while_loop(lambda c: (c[0] > 0.0) & (c[0] < L), body, (x0, jnp.int32(0), t))
        t = t.add("exit", jnp.where(x <= 0.0, 0, 1), 1.0)                  # which boundary absorbed it
        t = t.add("overshoot", 0, jnp.where(x <= 0.0, -x, x - L))          # how far past it, theta-dependent
        return t, k
    return item


OPTS = dict(outputs={"exit": tm.Sum((2,)), "overshoot": tm.Sum((1,))}, in_axes=(0, 0, None),
            max_steps=MAX_STEPS)
scaled_walk = tm.vmap(make_walk(True), **OPTS)
discrete_walk = tm.vmap(make_walk(False), **OPTS)

xs = jnp.full((N,), 0.5 * L, jnp.float32)
keys = jax.random.split(jax.random.PRNGKey(1), N)


def crn_fd(f, x, h):
    """A central finite difference at the SAME keys.  Only ever used to test a gradient."""
    return (float(jax.jit(f)(jnp.float32(x + h))) - float(jax.jit(f)(jnp.float32(x - h)))) / (2.0 * h)


# ---- A: a continuous parameter, a theta-dependent stopping time
theta = jnp.float32(1.0)
res = jax.jit(lambda th: scaled_walk(xs, keys, th))(theta)
steps = np.asarray(res.per_item)
print(f"A  walkers {N}  mean steps {steps.mean():.1f}  max steps {steps.max()}  "
      f"truncated {int(res.stats['n_truncated'])}")
check("A no walker was truncated (a truncated item would make the tallies NaN)",
      int(res.stats["n_truncated"]) == 0)

loss_a = lambda th: scaled_walk(xs, keys, th).outputs["overshoot"][0]
g_a = float(jax.jit(jax.grad(loss_a))(theta))
j_a = float(jax.jit(jax.jacfwd(loss_a))(theta))
fd_a = crn_fd(loss_a, 1.0, 1e-3)
print(f"A  d(total overshoot)/d(theta):  grad {g_a:.4f}   jacfwd {j_a:.4f}   CRN finite difference {fd_a:.4f}")

rel = abs(g_a - j_a) / max(abs(j_a), 1e-30)
check("A the mechanics are right: grad == jacfwd", rel <= 1e-4, f"rel {rel:.1e}")
check("A and the answer is still wrong: grad is far from the CRN finite difference",
      abs(g_a - fd_a) > 2.0 * abs(fd_a), f"grad/fd {g_a / fd_a:.2f}x")

# ---- B: the same walk, with theta entering only a discrete decision
p = jnp.float32(0.45)
loss_b = lambda pp: discrete_walk(xs, keys, pp).outputs["exit"][0]
g_b = float(jax.jit(jax.grad(loss_b))(p))
fd_b = crn_fd(loss_b, 0.45, 0.05)
print(f"B  d(left exits)/dp:  grad {g_b}   CRN finite difference {fd_b:.1f}")
check("B the gradient of a discrete decision is exactly zero", g_b == 0.0)
check("B while the true slope is large", abs(fd_b) > 100.0, f"fd {fd_b:.1f}")

print("""
What to do about it:
  A  the stopping time depends on theta, so a boundary term is missing.  Either reformulate so the loop
     condition does not depend on the parameter (fixed step count, fixed horizon, a parameter that enters only
     the emitted weight), or add the missing term, or accept a biased gradient knowingly.
  B  theta enters a comparison, so the pathwise derivative is zero by construction.  Use a score-function
     estimator: emit log p(decision; theta) * payoff as well, and differentiate that.
  Both  keep a CRN finite-difference check in your own test suite, at a point away from the optimum.  It is the
     only cheap way to notice that an exactly computed gradient is answering a different question.
""")

print(f"TOTAL {len(PASS) + len(FAIL)} PASS {len(PASS)}")
print("GRADIENT_PITFALL " + ("OK" if not FAIL else "FAIL " + ",".join(FAIL)))
raise SystemExit(0 if not FAIL else 1)
