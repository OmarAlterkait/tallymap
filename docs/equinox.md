# Equinox-ecosystem solvers inside an item

Diffrax, Optimistix, Lineax and Quadax all build on `equinox.internal.while_loop`.  tallymap runs their solvers
inside an item: you write the `diffrax.diffeqsolve(...)` call you would write anyway, and tallymap maps it over
items with one kernel.

```python
import os
os.environ.setdefault("EQX_ON_ERROR", "off")       # BEFORE importing equinox / diffrax; see below

import diffrax as dfx
import jax.numpy as jnp
import tallymap as tm

def item(t, mu):                                   # ONE item, ONE trajectory
    sol = dfx.diffeqsolve(
        dfx.ODETerm(lambda s, y, p: jnp.stack([y[1], p * ((1 - y[0] ** 2) * y[1] - y[0])])),
        dfx.Kvaerno5(), t0=0.0, t1=2.0, dt0=0.01, y0=jnp.array([2.0, 0.0]), args=mu,
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-7),
        adjoint=dfx.ForwardMode(),                 # <- see "The adjoint" below
        max_steps=16384)
    y = sol.ys[0, 0]
    return t.add("mean_y", 0, y), y

f = tm.vmap(item, outputs={"mean_y": tm.Sum((1,))}, max_steps=16384)
res = f(mus)                                       # res.outputs["mean_y"], res.per_item
```

`examples/diffrax_ensemble.py` is this program, checked against `jax.vmap(diffeqsolve)` and timed.

## The two settings an item needs

| setting | why |
|---|---|
| `adjoint=diffrax.ForwardMode()` | its loop is a plain `lax.while_loop`, which tallymap differentiates itself |
| `EQX_ON_ERROR=off` or `nan` in the environment | Equinox's default raises from a host callback; a kernel cannot call the host |

### EQX_ON_ERROR

Equinox's runtime checks (`equinox.error_if`, and every check the libraries build on it) raise from a host
callback under the default `EQX_ON_ERROR=raise`.  An item runs inside one kernel and cannot call the host, so an
item has to pick another mode.  The variable is read when `equinox` is imported, so set it before the import (or
in the shell).  If you forget, the error message says what to set.

| mode | in an item |
|---|---|
| `nan` | a failed check writes NaN into the value it guards. The right choice for a FORWARD-only item: the NaN says which item failed. |
| `off` | no check at all. Needed to differentiate an implicit solver (below). |
| `raise`, `breakpoint` | a host callback. The cuda backend raises `UnsupportedError` with this advice; the lockstep backend can call the host, so there it works. |

MEASURED, and the reason this page recommends `off` for any item you differentiate: with `EQX_ON_ERROR=nan`,
`jax.grad` of a `Kvaerno3` solve inside an item is NaN for every item, while `jax.jacfwd` of the same program is
right.  A check inside the implicit solver's COTANGENT solve reports a failure, and Lineax passes `throw=True` to
that solve unconditionally ("there is nowhere we can pipe their error to", `lineax/_solve.py:306`), so the only
place the failure can go is the value.  Two ways out, both MEASURED to give the gradient `jacfwd` gives:

* `EQX_ON_ERROR=off`; or
* `root_finder=optimistix.Newton(rtol=..., atol=...)` on the diffrax solver.  It is the frozen Jacobian of the
  default `VeryChord` that the cotangent solve trips over; a Newton root finder refactorises and does not.

An explicit solver (`Tsit5`), `optimistix.root_find`, `lineax.linear_solve` and `quadax` all differentiate fine
under `nan`.  Either way you still have tallymap's own divergence signal, which does not go through Equinox:
`Result.truncated` and `stats['n_truncated']` for an item that hit `max_steps`.

## What works

MEASURED by `tests/equinox_compat.py` on the lockstep backend (CPU) and the cuda backend (L40S), with
equinox 0.13.8, diffrax 0.7.2, optimistix 0.1.0, lineax 0.1.1, quadax 0.3.0 and jax 0.11.2.  Each row ran inside
`tm.vmap` and was compared with `jax.vmap` of the same function.

| solver | forward | jvp / jacfwd | grad |
|---|---|---|---|
| diffrax `Tsit5`, constant step | yes | yes | yes |
| diffrax `Tsit5` + `PIDController` | yes | yes | yes |
| diffrax `Kvaerno3` / `Kvaerno5` + `PIDController` (`VeryChord`) | yes | yes | yes |
| diffrax `Kvaerno3` with `optimistix.Newton` | yes | yes | yes |
| diffrax SDE: `Euler` + `VirtualBrownianTree` + `MultiTerm` | yes | yes | yes |
| diffrax event-terminated solve (`diffrax.Event`) | yes | yes | yes |
| diffrax `RecursiveCheckpointAdjoint` (the default) | yes | no (clear error) | yes, by a different route |
| diffrax `DirectAdjoint` | yes | yes | cuda: no (clear error); lockstep: yes |
| optimistix `root_find`: `Newton`, `Chord`, `Bisection` | yes | yes | yes |
| optimistix `least_squares`: `LevenbergMarquardt` | yes | yes | yes |
| optimistix `minimise`: `BFGS` | yes | yes | yes |
| lineax `linear_solve`: `LU`, `QR`, `Cholesky`, `CG` | yes | yes | yes |
| quadax `quadgk`, `quadts` | yes | yes | yes |

"grad" means `jax.grad` of a function containing the `tm.vmap` call.  For every diffrax row with `ForwardMode`
that gradient does not exist in plain JAX at all: `jax.grad(jax.vmap(diffeqsolve(..., adjoint=ForwardMode())))`
raises `Reverse-mode differentiation does not work for lax.while_loop`.  It exists here because tallymap replaces
the solver's loop with a checkpointed replay loop it can reverse-differentiate (`docs/design.md`,
`tallymap/_reverse.py`).

### How closely it agrees

An adaptive controller decides its step sequence by comparing floating-point numbers, and tallymap's arithmetic is
not XLA's bit for bit (`docs/float_semantics.md`).  Two runs of the same adaptive solve can therefore take
DIFFERENT steps and then agree only to about the solver's own tolerance, not to f32 rounding.  MEASURED relative
difference from `jax.vmap` of the same function, over every row of the table, at `rtol=1e-5` where the solver is
adaptive:

| | lockstep backend | cuda backend |
|---|---|---|
| lineax, optimistix, quadax (no adaptive step control) | **exactly 0** on every row | up to 2.6e-6 (`lineax.QR`: this library's Householder QR against cuSOLVER's) |
| diffrax, constant step or `Tsit5` + PID | 0 to 2e-7 | 0 to 2e-7 |
| diffrax, `Kvaerno3` / `Kvaerno5` + PID | up to 7.8e-6 | up to 3.8e-6 |

Gradients, where plain JAX can take one to compare with, agree with `jax.grad(jax.vmap(f))` to between 0 and
2.3e-6.  For a statistical statement about a particular program, use the `compare` protocol (`docs/validation.md`).

## The adjoint

tallymap differentiates the item's loops itself: `_rewrite.retrace(rev=...)` replaces every `lax.while_loop` in
the item by a checkpointed replay loop with a `custom_vjp`.  So the adjoint you want from diffrax is the one that
leaves the loop alone.

| adjoint | what diffrax builds | inside an item |
|---|---|---|
| `ForwardMode()` | `eqxi.while_loop(kind="lax")`, i.e. a plain `lax.while_loop` | **use this**: forward, jvp and grad all work |
| `RecursiveCheckpointAdjoint()` (default) | a `custom_vjp` around a treeverse loop | forward and grad work, jvp raises; carries the loop's buffers for nothing |
| `DirectAdjoint()` | a nest of `lax.scan` under `jax.checkpoint` | forward and jvp work; grad raises on cuda (it runs on lockstep) |

`RecursiveCheckpointAdjoint`'s gradient works, but not through diffrax's treeverse loop: because that loop sits
inside a `custom_vjp` whose primal contains a `while_loop`, tallymap's reverse retrace inlines the primal and
replays it with its own checkpointed loop, and diffrax's backward rule is never used.  MEASURED: it agrees with
`jax.grad(jax.vmap(diffeqsolve))` to float32.  You still pay for the checkpointed loop's buffers and its step
counters in the item, for a backward pass that is then discarded, so `ForwardMode` is the better choice.  Its
`jvp` does NOT work: JAX cannot forward-differentiate a `custom_vjp` at all.

`DirectAdjoint`'s backward pass introduces an `eval_jaxpr` primitive that no kernel can run, so its `grad`
raises on the cuda backend (the lockstep backend runs it, and MEASURED agrees with `jacfwd` to 4e-7).  Both
messages name `ForwardMode`; neither adjoint returns a wrong number.

Two notes on `ForwardMode` inside an item:

* it is not slower here.  Outside tallymap, `ForwardMode` means "differentiate the solver forward", whose cost
  grows with the number of parameters.  Inside an item it only selects the loop; the gradient is tallymap's
  reverse-mode replay, whose cost is independent of the parameter count.
* the gradient is the derivative of the discrete solver (differentiate-the-discretisation), not the continuous
  adjoint.  That is the same thing `RecursiveCheckpointAdjoint` computes, and it is not what
  `BacksolveAdjoint` computes.

## What is not supported inside an item, and why

| feature | why |
|---|---|
| `diffrax.RecursiveCheckpointAdjoint` under jvp or jacfwd | a `custom_vjp`; JAX cannot forward-differentiate one. Its `grad` does work (above) |
| `diffrax.DirectAdjoint` under grad, on the cuda backend | its backward pass emits `eval_jaxpr`, which a kernel cannot run. It runs on the lockstep backend |
| `diffrax.BacksolveAdjoint` | a continuous adjoint ODE; it is also a `custom_vjp` |
| `equinox.error_if` with `EQX_ON_ERROR=raise` or `breakpoint` | a host callback |
| `equinox.debug.announce_transform` | prints from inside the program |
| `equinox.internal.noinline` | keeps a function out of the enclosing program, which one fused kernel cannot do |
| `quadax`'s differentiable integration limits | its adjoint has its own primitive (`quadax_leibniz_tangent`) |
| a dense factorisation above 16x16, a batch of more than 64 of them, or a complex matrix | see "How it works" |
| anything that needs more per-item memory than a thread has | the usual item limit, not an Equinox one |

One performance limitation rather than a functional one: an item that calls a Lineax or Optimistix solver does
not use tallymap's dual tangent lanes under `jacfwd`.  Batching the item turns the foreign primitive into
Equinox's batched `_vprim_p`, and MEASURED on an L40S a `lineax.CG` solve then gives NaN in every lane but the
first on the cuda backend (the same iteration written by hand is right, and so is a `lineax.LU` solve, which has
no loop inside the primitive).  tallymap therefore runs such a program one lane at a time, which is measured to
be right; `tallymap/_foreign.lanes_unsafe` is the rule and says so.  The NaN no longer reproduces on the test
suite's `lineax.CG` case (MEASURED 2026-11-04, L40S and 2080 Ti, JAX 0.5.3 and 0.11.2), so the rule may go once a
larger case is measured.

The lockstep backend (`backend='lockstep'`, and the default off a GPU) runs every JAX primitive, so the rows above
that are only a kernel restriction - the callbacks, `noinline`, the dense limit - do work there.  The autodiff rows
do not: they are JAX restrictions, not tallymap's.

## How it works

Two mechanisms, both in the library:

**1. The Equinox primitives are expanded into their own implementation** (`tallymap/_foreign.py`).
`equinox.internal.while_loop` wraps the predicate of every loop in `unvmap_any` and masks every carry update with
`select_if_vmap`, so that ONE loop can serve a whole `jax.vmap` batch.  An item is un-batched by construction,
which is exactly the case each of those primitives documents a meaning for - `unvmap_any` is `jnp.any`,
`select_if_vmap(pred, x, y)` is `x`, `nonbatchable` is the identity - and that meaning is each primitive's own
`impl`, which is also its registered XLA lowering.  tallymap traces that `impl` on the equation's avals.  Nine
names: `unvmap_all`, `unvmap_any`, `unvmap_max`, `select_if_vmap`, `maybe_set`, `nonbatchable`, `nontraceable`,
`nondifferentiable_backward` and (at lowering only) Lineax's `linear_solve`.

MEASURED, before and after, on the `jax.make_jaxpr` of these items: 1710 occurrences of 11 unsupported primitives
became 82 occurrences of 7, and the remainder is dense linear algebra.  The whole masking apparatus disappears:
797 `select_if_vmap` equations expand to nothing at all, because inside an item every one of them selects its
first argument.

`linear_solve` is expanded LATER than the rest - by the cuda emitter, not at the item trace - because its autodiff
rules are not the autodiff rules of its implementation.  Its implementation reads the factorisation out of the
solver state, while its jvp rule asserts the state has no tangent and rebuilds the derivative from the operator
through the implicit function theorem.  Expanding it before autodiff silently drops the whole dependence on the
matrix (MEASURED: a gradient of 0 instead of -0.056).

**2. Dense linear algebra is lowered to plain JAX after autodiff.**  XLA lowers `lu`, `triangular_solve`,
`cholesky`, `geqrf` and `ormqr` to cuSOLVER / cuBLAS custom calls, which a fused kernel cannot make, and every
implicit solver needs them: a Kvaerno stage is a Newton solve, and a Newton step is an `lu` plus two
`triangular_solve`s.  The cuda emitter substitutes plain-JAX algorithms for them on the program it is about to
emit, after every autodiff rule has been applied, so the derivative is JAX's own:

* `lu`, `lu_pivots_to_permutation` and `custom_linear_solve` (behind `jnp.linalg.solve` / `inv`) become JAX's own
  pure-JAX implementations, the ones JAX uses on a backend without a solver library; `triangular_solve` becomes a
  substitution loop (`tallymap/_foreign.py`).  Measured against straight-line versions on a 2080 Ti
  (`jnp.linalg.solve` of n = 3 / 6 / 12 inside 1M items): faster in 8 of 9 cells (forward, grad, jacfwd) and 9%
  slower in one.  `lu` has the same size limit as the factorisations below: a matrix dimension above 16 raises.
* `qr` (behind `jnp.linalg.qr`, and `lineax.QR` on Lineax 0.0.x) becomes JAX's own lowering onto `geqrf` and
  `householder_product` (Q's first columns: `ormqr` applied to the identity); without this it raised on cuda.
* `cholesky`, `geqrf` and `ormqr`, which JAX has no pure-JAX form of, are textbook unblocked algorithms with
  LAPACK's conventions, unrolled in Python (`tallymap/backends/cuda/dense.py`).  MEASURED against LAPACK through
  `jax.lax.linalg` itself, in float32 and float64, for 1x1 to 16x16 and three rectangular shapes: the factors and
  `taus` agree to 1e-11 (f64) and 2e-4 (f32).  Straight-line code pays only while the matrix is small, so a
  dimension above 16 raises with a message saying so; one leading batch axis of up to 64 is mapped over, which is
  what a vmapped factorisation needs (`jacfwd`'s tangent lanes, a batch folded into the item axis).

## Sizing and the step budget

`max_steps` on `tm.vmap` caps every top-level `while_loop` in the item, which for a diffrax solve is the solver's
own loop.  An item that hits the cap is flagged in `Result.truncated` and poisons every float tally of the call
with NaN (`docs/float_semantics.md`).  Set it to at least diffrax's own `max_steps`, so that diffrax's budget is
the one that bites and you get its NaN per item instead of a poisoned tally:

```python
f = tm.vmap(item, outputs=..., max_steps=16384)     # >= the diffeqsolve max_steps
```

`rev_checkpoints` (default 16) trades gradient recomputation against per-item memory.  It is not monotone for a
heavy item: measure it rather than deriving it from the step count.

## The measured numbers

MEASURED on one L40S (`ada`), jax 0.11.2, float32, van der Pol with `mu` spread across the ensemble, `t1 = 2`,
`PIDController(rtol=1e-5, atol=1e-7)`.  Arms:

* **tallymap**: `tm.vmap` of an item calling `diffeqsolve(..., adjoint=ForwardMode())`;
* **jax.vmap**: `jax.vmap(diffeqsolve)`, with `ForwardMode` for the forward pass and diffrax's default
  `RecursiveCheckpointAdjoint` for the gradient, because JAX cannot reverse-differentiate `ForwardMode`.  That is
  what a diffrax user writes today, so it is the arm to beat.

N is the number of items; `fwd` is one forward call and `grad` is `jax.grad` of a sum over items.  Peak device
memory is the device allocator's own counter, one measurement per process.

**Tsit5 + PID (explicit), stiffness spread 9 to 11** - time ms / peak MB

| mode | N | tallymap | jax.vmap | speedup | memory |
|---|---|---|---|---|---|
| fwd | 4,096 | 0.2 / 64 | 9.6 / 144 | **59x** | 2.3x |
| fwd | 65,536 | 0.5 / 64 | 13.7 / 145 | **25x** | 2.3x |
| fwd | 1,048,576 | 6.6 / 68 | 53.4 / 260 | **8.0x** | 3.8x |
| grad | 4,096 | 2.5 / 64 | 40.4 / 144 | **16x** | 2.3x |
| grad | 65,536 | 42.1 / 64 | 136.6 / 1,198 | **3.2x** | 18.6x |
| grad | 1,048,576 | 728 / 68 | 2,275 / 19,168 | **3.1x** | **282x** |

With the wider spread (2 to 40) the forward margin is 11x to 73x and the gradient margin 4.1x to 17x, because the
`jax.vmap` arm pays the batch maximum of the step count and tallymap pays each item's own.

**Kvaerno5 + PID (stiff, implicit), stiffness spread 9 to 11** - time ms / peak MB

| mode | N | tallymap | jax.vmap | speedup | memory |
|---|---|---|---|---|---|
| fwd | 4,096 | 5.5 / 64 | 581 / 192 | **105x** | 3.0x |
| fwd | 65,536 | 19.7 / 64 | 702 / 193 | **36x** | 3.0x |
| fwd | 1,048,576 | 270 / 68 | 3,726 / 301 | **14x** | 4.4x |

With the wider spread (2 to 40) the forward margin reaches **136x** at a million items: 647 ms against 87.7 s.

The gradient needs `rev_checkpoints` chosen for the item, so it gets its own section.

### Choose rev_checkpoints for the item: the default is not it

`rev_checkpoints` (C) is the number of loop states the reverse pass keeps per checkpoint buffer.  The README
already says it is not monotone in the step count; a diffrax item is the clearest case of that, because a diffrax
solver's state is large (the solution, the controller, the stage buffers, several counters) so `2C` copies of it
leave registers, while a small C pays quadratic recomputation once the loop is longer than C^2 steps.

MEASURED, `Kvaerno5` gradient in ms, the best C in bold:

| spread | N | C = 4 | C = 8 | C = 16 (default) | C = 32 | C = 64 |
|---|---|---|---|---|---|---|
| narrow | 65,536 | **714** | 1,164 | 4,936 | 7,136 | 20,280 |
| narrow | 262,144 | **2,755** | 4,735 | 21,506 | 30,184 | - |

and, across N, the best of C = 4 and the default C = 16 against `jax.vmap(diffeqsolve)` with diffrax's
`RecursiveCheckpointAdjoint`:

| spread | N | C = 16 | C = 4 | jax.vmap | best speedup |
|---|---|---|---|---|---|
| narrow | 4,096 | **115** | 158 | 2,528 | 22x |
| narrow | 16,384 | 388 | **200** | 2,699 | 13x |
| narrow | 65,536 | 4,937 | **714** | 4,313 | 6.0x |
| narrow | 262,144 | 21,526 | **2,755** | 11,206 | 4.1x |
| narrow | 1,048,576 | 86,752 | **11,008** | 49,194 | 4.5x |
| wide | 4,096 | **476** | 1,965 | 10,875 | 23x |
| wide | 16,384 | **757** | 1,919 | 11,462 | 15x |
| wide | 65,536 | **10,939** | 67,342 | 18,556 | 1.7x |
| wide | 262,144 | **34,326** | 73,320 | over 1,200,000 | over 35x |
| wide | 1,048,576 | 130,592 | **101,654** | over 1,200,000 | over 9x |

The last two `jax.vmap` entries are a bound, not a time: one gradient call of that ensemble did not finish inside
the 20-minute per-configuration limit of the measurement, where the item version takes 34 s and 102 s.

Read that as: the gradient is **1.7x to 23x** faster than `jax.vmap` at **19x to 282x** less memory, and **which C
you want depends on the ensemble, not just on the item**.  The default is 6.9x off at the narrow spread and
N = 65,536; C = 4 is 6.2x off at the wide spread and the same N, where the stiffest items run a long loop and
quadratic recomputation dominates.  Peak memory does not move with C at these sizes (64-68 MB throughout), so the
measurement is purely of time.  Recomputing each step in checkpointed segments inside its VJP did not help
either way (4,968 ms at S = 4 against 4,936 at S = 0); that option is gone (docs/design.md section 6).

Two or three values of C on your own ensemble is the whole tuning procedure.

The table above was measured BEFORE the checkpoint stride was fixed to C (`docs/design.md` section 3). That fix
removes most of what made C so sharply non-monotone -- the sweep used to allocate a C-slot segment buffer per
segment and fill one slot of it -- so the diffrax numbers here are an upper bound on today's spread and have not
been re-measured. On the six `benchmarks/vs_jax` problems, which were, C = 8, 16, 32 and 64 now lie within a
factor of 1.6 of each other, where before the spread reached 20.

### What running diffrax inside the item costs, against writing the stepper yourself

MEASURED like-for-like: the same problem, the same `Kvaerno3` tableau, the same `VeryChord` root finder, the same
PI controller and the same tolerances, as THREE arms - the solver hand-written as an item, `diffeqsolve` called
inside an item, and `jax.vmap(diffeqsolve)`.

| N | mode | hand-written item | `diffeqsolve` in an item | `jax.vmap(diffeqsolve)` |
|---|---|---|---|---|
| 65,536 | fwd | 2.4 ms | 17.4 ms | 580 ms |
| 1,048,576 | fwd | 26.7 ms | 255 ms | 4,053 ms |
| 65,536 | grad | 164 ms / 66 MB | 5,443 ms / 66 MB | 4,231 ms / 1,026 MB |
| 1,048,576 | grad | 2,664 ms / 72 MB | 94,874 ms / 72 MB | 57,948 ms / 14,614 MB |

So calling `diffeqsolve` inside the item keeps most of the forward margin (16x against `jax.vmap`, where the
hand-written item gets 152x) and costs **9.5x** the hand-written item's forward time.  In reverse mode the same
gap is **33x to 36x**, which is why the gradient needs `rev_checkpoints` tuned to stay ahead at all.  The reason is the
same in both: diffrax's step is three nested data-dependent loops (the time loop, the Runge-Kutta stage loop, and
a root find per stage) with buffers and step counters threaded through them, where the hand-written item's step is
straight-line code.  tallymap's reverse pass replays that whole step, so a heavier step costs more in the
gradient than in the forward pass.

Choose accordingly: `diffeqsolve` inside an item for diffrax's full solver set with one line changed, a
hand-written item when the gradient is the inner loop of a fit.

## Pointers

* `examples/diffrax_ensemble.py` - the example above, with the check and the timing.
* `tests/equinox_compat.py` - every row of "What works", as a test.
* `tallymap/_foreign.py` - the expansion, and the argument for each primitive in it.
* `tallymap/backends/cuda/dense.py` - the Cholesky and QR factorisations and their conventions.
* `docs/float_semantics.md` - what "agrees with `jax.vmap`" means in general.
* `docs/design.md` - the checkpointed replay loop that makes the gradient exist.
