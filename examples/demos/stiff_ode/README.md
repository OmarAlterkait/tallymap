# A stiff ODE ensemble against diffrax

A forced Van der Pol oscillator with a per-trajectory stiffness `mu`, augmented with a path functional so that
the quantity being tallied is smooth and under the same error control in both arms:

```
x' = v,   v' = mu (1 - x^2) v - w2 x + A cos(om t),   z' = x^2,   t in [0, 20]
```

`theta = (w2, A, om)` are the three differentiated parameters; `mu` is drawn log-uniformly per trajectory, which
is what makes the number of adaptive steps vary across the ensemble.  The item contains **three nested
data-dependent loops**: the adaptive time loop, the Runge-Kutta stage sequence, and a chord solve per stage.

## The incumbent

[diffrax](https://github.com/patrick-kidger/diffrax) **0.7.2** with
[optimistix](https://github.com/patrick-kidger/optimistix) **0.1.0**, pinned in `requirements.txt`.  The demo
calls `diffrax.diffeqsolve` with `Kvaerno3()`, `PIDController(rtol=1e-4, atol=1e-6)` (its default controller),
`throw=False` (which is what you need under `jax.vmap`, because a per-element solver error cannot be raised),
`RecursiveCheckpointAdjoint()` (its default) for `grad` and `ForwardMode()` for `jvp` and `jacfwd`, and
`max_steps` set to the observed maximum rounded up to a multiple of 64.

## The item is a re-implementation of Kvaerno3, not diffrax's code

**This is the one place in the demos where an algorithm is re-implemented rather than called, so read this
before reading the tables.**  `kv3.py` is a faithful re-implementation of what
`diffrax.diffeqsolve(..., Kvaerno3(), PIDController())` does for a single trajectory: the Kvaerno3 tableau,
diffrax's `VeryChord` root finder with its convergence test, the Jacobian evaluated once per step at the
second-stage predictor with diffrax's `stop_gradient`s, and diffrax's PI controller formula.  Every constant and
formula cites the diffrax 0.7.2 line it came from, in that file's header.

Why: diffrax's `diffeqsolve` is built out of `equinox` modules and `lax.while_loop`s whose carry holds a
`SaveAt` buffer, and getting that to run inside a tallymap item is a separate piece of work.  That work is
under way: the literal `diffrax.diffeqsolve` call will run inside an item, **this demo will switch to it once
that lands**, and the hand-written item will go away.

What it costs in fairness, measured rather than hoped:

- The two implementations **agree on the answer** to the solver tolerance (the tally to a few parts in 1e6, the
  per-trajectory terminal state to a few parts in 1e5, the gradient to under a percent of a gradient of
  magnitude ~1e6).  `--quick` checks all of that.
- They **do not take the same number of internal steps**.  The re-implementation takes about 0.7 of diffrax's
  attempted steps on this problem.  The tableau, the chord formulation and its convergence test, the Jacobian's
  linearisation point, the `stop_gradient`s and the controller formula were all checked line by line against
  diffrax's source, and the remaining difference is unaccounted for.
- So **every forward table below is reported twice**: raw wall time, and nanoseconds per item-solver-step,
  which divides the step-count difference out entirely.  One "item-solver-step" is one attempted step of one
  trajectory, accepted or rejected, and both arms report their own count.  **The per-step table is the one to
  quote.**
- The `tm` and `jax_while` columns are unaffected by any of this, because those two arms run the identical
  Python function.  They agree to one f32 ulp and have identical step counts, which `--quick` checks.

## The three arms

| arm | what it is | which modes |
|---|---|---|
| `diffrax` | `diffrax.diffeqsolve` under `jax.vmap`, as described above | all |
| `jax_while` | `jax.vmap` of `kv3.solve`, with its `lax.while_loop` | forward only: JAX cannot reverse-differentiate a `while_loop` |
| `tm` | the same `kv3.solve` as a tallymap item | all |

## Measured

Measured on one **NVIDIA L40S (gpu)**, JAX **0.11.2**, Python 3.12.14, diffrax 0.7.2, jax 0.11.2, optimistix 0.1.0, tallymap **0.4.0**, an unreleased development build (commit `2061bb4b8695` of the private pre-release history).

Median `jax.jit` wall time per call after warm-up, one measurement per process so the peak-memory counter is clean.  Every number below is a row of `results/` in this folder.

### Forward, raw wall time

| trajectories N | tallymap (ms) | vmap + while_loop (ms) | diffrax (as it ships) (ms) | x vs best baseline | attempted steps, item arms | attempted steps, diffrax |
|---|---|---|---|---|---|---|
| 16,384 | 1.24 | 171 | 462 | **138x** | 7.7M | 10.6M |
| 65,536 | 2.44 | 235 | 575 | **96.2x** | 31.0M | 42.3M |
| 262,144 | 7.92 | 337 | 1,043 | **42.5x** | 124.1M | 169.1M |

### Forward, per item-solver-step

This removes the step-count difference between the re-implementation and diffrax entirely.

| trajectories N | tallymap (ns/step) | vmap + while_loop (ns/step) | diffrax (as it ships) (ns/step) | x vs best baseline | x vs diffrax |
|---|---|---|---|---|---|
| 16,384 | 0.16 | 22.1 | 43.8 | **138x** | **274x** |
| 65,536 | 0.0788 | 7.58 | 13.6 | **96.2x** | **173x** |
| 262,144 | 0.0638 | 2.71 | 6.16 | **42.5x** | **96.6x** |



### Reverse-mode gradient

diffrax uses its default `RecursiveCheckpointAdjoint`, which is genuinely good: it stores only about
`sqrt(2 max_steps)` states per trajectory rather than the whole trajectory, which is why the memory margin here
is far smaller than it is against a padded scan in the other demos.

| trajectories N | tallymap (ms) | diffrax (as it ships) (ms) | x vs best baseline | tallymap peak MB | diffrax peak MB | x less memory |
|---|---|---|---|---|---|---|
| 16,384 | 17.4 | 2,197 | **126x** | 64.3 | 128 | **2x** |
| 65,536 | 166 | 3,916 | **23.6x** | 65.8 | 301 | **4.58x** |
| 262,144 | 654 | 12,776 | **19.5x** | 70 | 1,203 | **17.2x** |



### Forward-mode derivatives

diffrax with `adjoint=ForwardMode()`, which is its forward-mode-capable loop.

| mode (at N = 65,536) | tallymap (ms) | tallymap, x its own forward | diffrax ForwardMode() (ms) | diffrax, x its own forward | x vs diffrax |
|---|---|---|---|---|---|
| `jvp`, one tangent | 3.56 | 1.5x | 800 | 1.4x | **225x** |
| `jacfwd` over all three parameters | 5.03 | 2.1x | 878 | 1.5x | **175x** |

### The divergence sweep

| div (at N = 65,536) | steps mean / max | total item-solver-steps | tm / while / diffrax (ms) | x vs best baseline |
|---|---|---|---|---|
| low | 400 / 466 | 26.2M | 1.66 / 136 / 461 | **82.2x** |
| mid | 473 / 690 | 31.0M | 2.44 / 235 / 575 | **96.2x** |
| high | 300 / 690 | 19.7M | 2.4 / 238 / 578 | **99.4x** |

### With event detection

The event is "stop when the accumulated path functional `z` reaches 30", located by diffrax's
`Event(cond_fn, root_finder=optx.Newton(1e-6, 1e-6))` in one arm and by 40 bisections on the cubic Hermite
interpolant of `z` in the other.  `z` is monotone in `t`, so there is exactly one crossing and the two arms
detect the **same** event; `--quick` checks that the stopping times agree.

| trajectories N | tallymap (ms) | vmap + while_loop (ms) | diffrax (ms) | x vs best baseline |
|---|---|---|---|---|
| 65,536 | 1.56 | 259 | 366 | **166x** |

Gradients are **not** reported for the event configuration, deliberately.  The tally there is `z` at the event
time, which is 30 for every trajectory that fired, so the true derivative is about zero, and comparing two
near-zero gradients relatively is meaningless.  Differentiating an event *time* needs the implicit derivative
`dT = -(dc/dtheta)/(dc/dt)`, which is the user's job in both arms.

## Where tallymap loses, or the margin shrinks

- **The item is a re-implementation of Kvaerno3, and it takes a different number of steps.**  At `--div mid` it
  attempts 124.1M item-solver-steps where diffrax attempts 169.1M, a ratio of 0.73. That is why the per-step table is
  the one to quote, and why the `tm` versus `jax_while` columns (the same code, so identical step counts) are the
  clean comparison. Separate work is under way to make the literal `diffrax.diffeqsolve` call run inside an item; this demo
  switches to it when that lands.
- **tallymap's gradient costs far more than its own forward pass.**  654 ms against 7.9 ms at 262,144 trajectories,
  about 83x, where diffrax's costs 12x its own. The replay adjoint of an item with three nested loops and a 3x3 linear
  solve per chord iteration re-executes the whole trajectory and then applies the VJP of one time step at a time, and
  a time step here is itself two nested loops. So the gradient margin (19.5x at this N) is an order of magnitude
  smaller than the forward margin.
- **The memory margin is small, because diffrax's adjoint is genuinely good.**  17.2x at 262,144 trajectories, against
  the hundreds-fold margins the padded-scan demos show. `RecursiveCheckpointAdjoint` stores about `sqrt(2 max_steps)`
  states per trajectory rather than the whole trajectory. Do not expect a padded-scan memory story here.
- **Compile time is a real cost, though on this item it is not larger than diffrax's.**  tallymap's gradient lowers in
  2.95 s (trace plus the NVRTC build of a large item) plus 0.16 s of XLA, 3.11 s in total, against 5.00 s for diffrax;
  forward, 0.65 s against 2.35 s. diffrax pays for building an equinox program and tallymap pays for emitting and
  compiling a CUDA kernel, and here those happen to be comparable. The kernel build grows with the item, and this is a
  large item: a cold cubin cache adds seconds on top.
- **No gradient for the event configuration.**  The tally there is the path functional at the event time, which is its
  target value for every trajectory that fired, so the true derivative is about zero and comparing two near-zero
  gradients relatively is meaningless. Differentiating an event time needs the implicit derivative, which is the
  user's job in both arms.

## How to rerun

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install tallymap

python run.py --quick     # CPU, about half a minute: all three arms, the agreement checks, PASS/FAIL
python run.py --full      # one GPU, regenerates results/stiff_ode.json and plots/
```
