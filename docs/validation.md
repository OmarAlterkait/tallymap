# Validation

Two questions, kept apart, and the one frozen statistical contract between implementations.

## A (mechanics) and B (estimator)

- **A. Mechanics.** Is the derivative the exact derivative of the program as written?
  - Compare against the same transformation of the dense reference (`tm.reference`, which is `jax.vmap(item)`
    with real per-item tally arrays, summed) at small N.
  - Use the same keys, use f64 or `Sum(exact=True)` wherever bitwise agreement is expected, and compare reverse
    mode against forward mode (`jacfwd`) on the same keys.
  - tallymap is responsible for A. `tests/` checks it: `tests/edge.py`, `tests/transforms.py`,
    `tests/reverse.py`, `tests/reverse_spawn.py`.
- **B. Estimator.** Is the derivative of the program an unbiased estimate of d E[loss] / d theta?
  - Boundary terms, score-function terms (the REINFORCE or DiCE correction for a sampling distribution that
    depends on the parameter) and hard bin edges all matter here.
  - B belongs to the program, not to the library. Check it with finite differences on the expected-loss curve
    using common random numbers, away from the truth, over many keys, and compare per-output VECTORS rather than
    scalar ratios.

A derivative can pass A perfectly and still be the wrong answer to B. `examples/gradient_pitfall.py` is a
runnable case: `grad` and `jacfwd` agree to 1.5e-7 and the gradient is still an order of magnitude away from the
true slope, because the number of loop steps depends on the parameter and the pathwise derivative misses that
boundary term. The README section "Gradients: exactly computed is not the same as what you want" explains it.

Report which question each number tests. Most arguments about a gradient being "wrong" are an A number being
used to answer a B question.

## Rules for any A/B comparison

- **Pass the evaluation point as a `jit` ARGUMENT, never by closure.** XLA constant-folds closed-over arithmetic
  on the host. The inputs then differ by one unit in the last place from the jitted ones, and chaotic Monte Carlo
  paths diverge from there to O(1) differences. This has produced more false alarms than any other mistake on
  this list.
- Expect bitwise agreement only from deterministic sums: CPU, or `Sum(exact=True)` (int64 fixed point). f32
  tallies on a GPU are accumulated with atomics on the CUDA backend and with a GPU scatter-add on the lockstep
  backend, so their summation order, and their last bits, change from run to run. Rerun the same keys twice as
  the control before blaming a difference on anything else.
- Set `jax_default_matmul_precision='highest'` for any GPU parity test. JAX's default on recent NVIDIA GPUs is a
  TF32 matmul, which biases results at the 1e-4 level.
- Before computing anything, assert that the two arms really differ end to end: both a program hash and an
  output. `assert_arms_differ` in `tests/ab_rule.py` does the first half.
- Never validate a gradient at the truth point, where its expectation is zero. Use common-random-number finite
  differences with a large step.

## The `compare` criterion

`compare(f, *args, other=..., floor_fn=...)` in `tests/compare_protocol.py` compares two implementations of the
same item program, for example `backend='cuda'` against `backend='lockstep'`, on the same per-item keys. It is a
test protocol, not library API; `tests/edge.py` runs it on the CUDA backend. It is a statistical test, not an
equality check, because the two backends do not produce bitwise identical f32 sums by design
(`docs/float_semantics.md`).

Its three legs, all required, are in `tests/compare_protocol.py::criteria()` and are frozen:

| leg | bound |
|---|---|
| flip rate: the fraction of items whose per-item result or summed tally differs by more than `rtol` | at most 3 x the control's flip rate + 3/N |
| `rms_z` | at most 3 x the control's `rms_z` + 0.05 |
| `max_z` | at most `max(5, 2 x` the control's `max_z)` |

where `z` per bin is `|f - other| / sqrt(sum w^2 + median_over_hit_bins(sum w^2))`, the floor coming from
`floor_fn`, the same item program with every emitted value squared. The **control** is `other` run on float inputs
perturbed by one unit in the last place: it measures how much the same program moves under a perturbation that
should not matter, and every bound is relative to it. Bins hit by `f` but never by the floor run are reported
(`empty_bin_mismatch`) and not judged; that hole was stated in advance and is still there.

"Frozen" means the criterion was pre-registered: written down, with its predicted outcomes, before the runs that
tested it, and not tuned afterwards. Two things about that registration are worth keeping:

- One leg was regularised before freezing, because the first run failed on a single sparse bin. That change is
  disclosed, and the criterion as frozen is the regularised one.
- The power was then measured with negative controls, on a workload that is not public. The criterion fails on a
  defect that removes about a fifth of the signal and on a 1% change to a physical parameter of the simulation;
  it does not fail on a defect that removes about a thousandth. The `max_z` leg never fired once the floor was
  regularised, so as frozen that leg is lax and the power comes from the `rms_z` leg. This is recorded rather
  than fixed, because a criterion that is changed after seeing its results is no longer pre-registered.
- One predicted failure did not happen, and the cause was the instrument, not the criterion: the injected defect
  turned out not to change the observable at all. Hence the rule below.

**Before every negative control, measure that the injected defect changes the observable.** A negative control
that passes because it changed nothing has tested nothing. The instrument checks in `tests/` and
`benchmarks/perf_negative_control.py` exist for this reason.

The full registration record, the protocol drivers and the replication runs are kept privately with the engines
they were run on. Nothing in this repository depends on them.

## The repository's own evidence

| what | where |
|---|---|
| every transformation against the same transformation of the dense reference | `tests/transforms.py` |
| reverse mode, including sparse table cotangents, the single-loop adjoint and the replay gate | `tests/reverse.py` |
| the spawn tree adjoint against `jax.grad` of a dense plain-JAX reference, with four negative controls | `tests/reverse_spawn.py` |
| per-operation parity of the CUDA emitter against `jax.jit(jax.vmap(op))` and the host-exact result, with a bound per class and a per-GPU baseline | `tests/ops.py`, `tests/ops_baseline.json` |
| edge cases and every documented failure mode | `tests/edge.py` |
| a long chain of 64-bit integer multiplies compiles in seconds and matches lockstep bitwise | `tests/mulchain.py` |
| speed, against a recorded baseline, as a regression net | `benchmarks/perf.py`, `benchmarks/perf_baseline.json` |
| speed and memory against plain JAX | `docs/performance.md` |

## When to rerun the `compare` protocol

Rerun it whenever `backends/cuda` math or the operation table changes in a way that `tests/ops.py` reports as a
changed number. Re-record `tests/ops_baseline.json` only after that, and in a commit of its own.
