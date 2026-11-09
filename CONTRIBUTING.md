# Contributing

Bug reports and small patches are welcome. The two things that make a patch easy to accept are a test cell and,
for anything that touches speed or numbers, a measurement of both arms.

## Running the tests

```
python tests/run_all.py                      # CPU: the xla backend
python tests/run_all.py --gpu                # + the CUDA backend, the spawn tree adjoint, per-operation parity
python tests/run_all.py --only edge,reverse  # one or more suites
python tests/run_all.py --x64                # with jax_enable_x64
python tests/run_all.py --unfused            # the two-node JVP design (the fallback if JAX drops a registry)
python tests/run_all.py --canary             # DeprecationWarning and FutureWarning become errors
python benchmarks/perf.py                    # jit wall times against benchmarks/perf_baseline.json
```

`python tests/run_all.py` is the entry point. **`pytest` is not the runner.** Each suite needs fresh JAX state,
so each runs as its own process, and the runner reports a suite as passing only if its summary line appeared and
every cell is a PASS or an expected raise. Exit status 0 means every suite passed. `TEST_LOG=<prefix>` writes the
full output of a failing suite to `<prefix>.<suite>.log`.

The suites are `edge` (edge cases and every documented failure mode), `transforms` (the
transformation-composability matrix), `reverse`, `reverse_spawn` (the tree adjoint), `ops` (per-operation CUDA
parity, `--gpu` only) and `examples` (every file in `examples/` run as a script).

The tests are the specification. A test cell compares against the same transformation of the dense reference
(`tm.reference`, which is `jax.vmap(item)` with real per-item tally arrays, summed), because that is what a
tallymap call *means*. If a change makes a cell fail and the cell looks wrong, say so in the pull request rather
than editing the cell quietly: the cells have negative controls precisely so that a cell which cannot fail gets
noticed.

The CPU suite takes about ten minutes and runs in GitHub Actions on every pull request, at both ends of the
supported JAX range. The GPU cells need hardware Actions does not have, so they run before each release
(`docs/maintenance.md`).

## How the operation table works

The CUDA backend compiles the item's jaxpr to C by walking its equations and emitting one piece of code per JAX
primitive. The table of those rules, `tallymap/backends/cuda/op_table.py`, is the main maintenance surface: a new
or renamed JAX primitive in a user's program shows up as
`tm.UnsupportedError: primitive 'foo' is not supported`.

**Adding a primitive is usually one rule plus one `tests/ops.py` cell.**

1. Write the rule in `op_table.py`:

   ```python
   @rule("my_primitive")
   def _my_primitive(em, eqn, ins, **params):
       ...
       return value          # or a list of values, one per output
   ```

   The emitter helpers (`Scalar`, `Local`, `Table`, `View`, `elem`, `lit`, `ctype`, ...) come from `codegen.py`.
   IEEE semantics only: round-to-nearest add, subtract, multiply, divide and square root with no
   fused-multiply-add contraction, and transcendentals from the CUDA math library. Do not try to reproduce XLA's
   own algebraic rewrites; the backends agree statistically, not bitwise (`docs/float_semantics.md`).

2. If the primitive is sugar for primitives that already have rules, put it in
   `tallymap/backends/cuda/canon.py` instead, which rewrites it before emission. That is where `square`,
   `integer_pow`, `clamp`, `squeeze`, `split` and the total-order comparisons live. A canonicalisation is
   preferable to a rule: it cannot drift from the primitives it decomposes into.

3. Add a cell to the `OPS` table in `tests/ops.py`:

   ```python
   "my_op": (lambda a, b: ..., ("wide", "wide"), "exact", None),
   ```

   The tuple is `(function, input kinds, class, (bound for ieee, bound for approx))`. The class decides what the
   cell is judged against: `exact` means bitwise equal to `jax.jit(jax.vmap(op))`, `cr` means correctly-rounded
   arithmetic judged in units in the last place against a float64 host evaluation, `lib` means a CUDA math
   library function judged against the maximum error documented in the CUDA C++ Programming Guide, and `info`
   means reported but not judged. **State the bound before the first run.** Changing a bound after seeing a
   result is a commit of its own, with the reason in the message, and `tests/ops.py` has two such disclosures in
   it already.

4. `tm.supported_primitives()` and `docs/primitives.md` are generated; regenerate them.

5. The edge cell "op table: names are JAX primitives" checks every table name against
   `jax.extend.core.primitives`, so a typo in a rule name fails a test rather than lying dormant.

The xla backend needs no work for a new primitive: it is plain JAX, so it already runs everything JAX can
run. That is also why `backend='xla'` is the workaround for any `UnsupportedError`.

## The A/B rule for performance and numerics

Any change that is meant to make something faster, or to change a number, needs **both arms measured in the same
job**, and an assertion that the two arms are genuinely different programs:

```python
from ab_rule import assert_arms_differ      # tests/ab_rule.py (tests/ on sys.path)
assert_arms_differ(f_a, f_b, *args)         # cuda: the cubin hashes; xla: the StableHLO
```

Comparing two identical programs measures noise, and it is an easy mistake to make, because an option can be
silently ignored. Then:

- Measure `jax.jit` **wall** time per call, after warm-up, as the median or minimum of repeats. Wall time is the
  point: only wall time shows a lost `jax.jit` fast path, which is worth an order of magnitude and which a kernel
  profiler will not show you at all.
- Report compile time separately from run time, and peak memory per process.
- Pass the evaluation point as a `jit` **argument**, never by closure: XLA constant-folds closed-over arithmetic
  on the host, the inputs then differ by one unit in the last place, and chaotic Monte Carlo paths diverge from
  there (`docs/validation.md`).
- For a numerics change, compare per-output **vectors**, not scalar ratios, and rerun the same keys twice first
  as the control. f32 tally sums on a GPU are not reproducible run to run by design; `Sum(fixed_point=True)` is.
- **Before a negative control, measure that the injected defect changes the observable at all.** A negative
  control that passes because it changed nothing has tested nothing. This has happened here.

If a change is declared "no behaviour change", prove it: identical StableHLO, identical generated CUDA source and
identical cubins, with bitwise-equal outputs wherever the base's own reruns are bitwise equal.

Re-record `benchmarks/perf_baseline.json` or `tests/ops_baseline.json` only in a commit of its own, and only when
the change in the numbers is intended.

## Code style

```
ruff check .
```

Settings are in `pyproject.toml`: rule sets E, F, W and I, a line length of 120, and a target of Python 3.10.
`E731` (assigning a lambda) and `E741` (the names `l`, `I` and `O`) are deliberately allowed, because they read
better in index arithmetic and in small closures. Ruff runs in CI on every pull request and must be clean.

Beyond ruff: keep the public surface small (a new public name is a decision, not a convenience), keep every
JAX-version difference inside `_compat.py` or `_jaxpr.py` so that a JAX upgrade stays a bounded job, and make new
options private (a leading underscore) unless a user has a reason to reach for them.

Prose, in comments and in docs: plain and direct, one claim per sentence, with the measurement next to the claim
that rests on it.

## Pull requests

One logical change per commit, with the test that covers it in the same commit. The commit message should say
what changed and, if anything was measured, the numbers and the hardware. If you had to choose between two
designs, a sentence about the one you did not pick is worth more than a paragraph about the one you did.

## Reporting a bug

Open an issue with the output of `tm.diagnostics()`, which carries the versions, the GPU, the compute capability,
the driver, the CUDA compiler, the relevant counters and the last generated kernels. There is a template that
asks for it. A smaller reproducer is always better, and the xla backend (`backend='xla'`) is usually
the fastest way to find out whether a problem is in the CUDA emitter or in the library's semantics.
