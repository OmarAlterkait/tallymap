# A branching-process outbreak as a spawn tree

The individual-level branching process used for early-outbreak risk assessment (Lloyd-Smith et al. 2005 on
superspreading; Hellewell et al. 2020) is simulated from N index cases: each infected individual draws an
individual reproduction number, then a Poisson number of offspring, each offspring an infection time from a
Weibull generation interval, and is detected (and so isolated) with probability `p_det`.  The outputs are the
epidemic curve, infections by generation, outbreak size and a smooth observable with a derivative.

An overdispersed offspring distribution means most lineages die out at once while a few explode, so the number
of live individuals swings over orders of magnitude from one outbreak to the next and from one generation to
the next.  In tallymap this is a `tally.spawn`: the item is one individual, and its offspring loop spawns each
child as a new item which runs the same function.

## The incumbent

**There is no incumbent JAX library for individual-level branching processes, and that is the honest framing.**
The established tools (`epichains`, `ringbp`, `bpmodels`) are R packages that walk one lineage at a time on the
CPU, so a head-to-head against them would be measuring R against a GPU and would tell you nothing about
execution models.  The baselines here are plain-JAX programs for the same model: the obvious one, and the best
one available.

## The three arms

| arm | what it is |
|---|---|
| `tm` | item = one individual; its offspring loop `spawn`s each child, and the child runs the same function |
| `jax_front` | the baseline worth quoting: ONE global generation frontier pooled over all N outbreaks, with each child slot mapped back to its parent by a `searchsorted` on the cumulative offspring counts.  No `(parent, offspring)` rectangle and no compaction scatter, so a generation costs `O(front log front)` instead of `O(front x max_offspring)` |
| `jax_gen` | what most people write first: `jax.vmap` over outbreaks with a `(live, kmax)` children block per outbreak, compacted by cumsum and scatter |

## Why the comparison is fair

- All three arms draw the same distributions by the same explicit transforms of uniforms (Weibull by inversion,
  Poisson by exponential interarrivals), not library rejection samplers, so there is no sampler difference
  hiding in the comparison.  The two arms turn out to be **path-identical**, because both derive a child's PRNG
  key the same way from its parent's.
- `jax_front` must be told its frontier width in advance, and it must cover every child **drawn** in the
  largest generation, including the ones that then die of the horizon or the generation cap, because those
  still occupy a slot.  That number is **measured**: the tallymap item tallies children drawn per generation,
  `--mode sizes` reports it, and `--full` sizes `jax_front` at 1.05 times the measurement.  Every row reports
  `jax_dropped`, which is zero everywhere in the tables below.  `--quick` does the same sizing and checks it.
- `jax_gen` is the counter-example rather than a serious baseline.  Sized from the measured maxima
  (117 offspring for one individual, hundreds of live individuals in one generation of one outbreak) it asks
  for tens of gigabytes and cannot run at all.  Shrunk to `live=64, kmax=32` so that it does run, it silently
  drops children, and its row reports exactly how many.  It is in the table because "just pad it" is the first
  thing a reader will suggest.
- Agreement: the incidence tally and infections by generation are bitwise equal to `jax_front`'s, outbreak size
  per index case agrees to six digits, and `grad` agrees with `jacfwd` and with the frontier arm's `grad`.

## The gradient, and which transform to use

theta is `(log R0, Weibull shape, Weibull scale, logit p_det)`.  Only the two generation-interval parameters
have a non-zero pathwise derivative, and that is itself a measured result: `log R0` and `logit p_det` enter only
through discrete draws, so their pathwise derivative is **exactly zero** in both arms.  `--quick` measures that.
The right estimator for those two is the ancestral score, which the item already tallies in the same forward
pass for free.

For the two parameters that do have one, **`jacfwd` is the transform to use, not `grad`**: see the two tables
below.

## Sizing the spawn Queue

Forward mode and `jacfwd` need a `tm.Queue` that holds the pending frontier.  The tree adjoint needs one that
covers **every** child of the call, which is about 15.5 per index case here.  An undersized Queue does not give
a wrong answer: the gradient is NaN and `tm.diagnostics()['reverse']['tree']` reports `generation_overflow=1`.  `--quick`
runs that negative control deliberately and checks both.  The capacities used in the tables are in
`results/branching_epidemic.json` under `meta.fwd_cap` and `meta.grad_caps`.

## Measured

Measured on one **NVIDIA L40S (gpu)**, JAX **0.11.2**, Python 3.12.14, jax 0.11.2, tallymap **0.4.0**, an unreleased development build (commit `2de020619934` of the private pre-release history).

Median `jax.jit` wall time per call after warm-up, one measurement per process so the peak-memory counter is clean.  Every number below is a row of `results/` in this folder.

### Forward

| index cases N | tallymap spawn tree (ms) | global generation frontier (ms) | per-outbreak padding (ms) | x vs best baseline | children in the tree | frontier width | children dropped |
|---|---|---|---|---|---|---|---|
| 16,384 | 4.32 | 16.2 | 53 | **3.75x** | 234,977 | 50,310 | 0 / 0 / 2,603 |
| 65,536 | 4.78 | 25.8 | 376 | **5.4x** | 946,687 | 202,031 | 0 / 0 / 9,089 |
| 262,144 | 8.23 | 56.8 | - | **6.9x** | 3,759,997 | 794,415 | 0 / 0 |
| 1,048,576 | 16.7 | 225 | - | **13.5x** | 15,175,202 | 3,208,900 | 0 / 0 |

The "children dropped" column is tallymap / frontier / per-outbreak padding. The per-outbreak arm is sized at `live=64, kmax=32` so that it runs at all; sized from the measured maxima it asks for tens of gigabytes. Its drops are silent in the program and only visible because the arm counts them.

Peak device memory on the same runs:

| index cases N | tallymap spawn tree (MB) | global generation frontier (MB) | per-outbreak padding (MB) |
|---|---|---|---|
| 16,384 | 64.3 | 96.5 | 718 |
| 65,536 | 65.3 | 65.3 | 3,337 |
| 262,144 | 70 | 134 | - |
| 1,048,576 | 84 | 308 | - |



### What the tree looks like, measured

These are the `--mode sizes` rows, and they are what the frontier arm is sized from.

| index cases N | children in the tree | outbreak size per index case | children drawn in the largest generation | as a multiple of N | largest offspring count for one individual |
|---|---|---|---|---|---|
| 16,384 | 234,977 | 15.342 | 47,914 | 2.92 N | 117 |
| 65,536 | 946,687 | 15.445 | 192,410 | 2.94 N | 117 |
| 262,144 | 3,759,997 | 15.343 | 756,585 | 2.89 N | 195 |
| 1,048,576 | 15,175,202 | 15.472 | 3,056,095 | 2.91 N | 195 |

### Derivatives of the smooth observable

Reverse mode (`grad`):

| index cases N | tallymap spawn tree (ms) | global generation frontier (ms) | x vs best baseline | tallymap peak MB | frontier peak MB | spawn Queue capacity |
|---|---|---|---|---|---|---|
| 16,384 | 9.59 | 16.6 | **1.73x** | 256 | 96.5 | 1,048,576 |
| 65,536 | 15.4 | 26.1 | **1.7x** | 277 | 129 | 2,097,152 |
| 262,144 | 47.3 | 56.4 | **1.19x** | 1,107 | 279 | 8,388,608 |

Forward mode (`jacfwd` over all four parameters):

| index cases N | tallymap spawn tree (ms) | global generation frontier (ms) | x vs best baseline | jacfwd, x faster than tallymap's own grad | tallymap peak MB | frontier peak MB |
|---|---|---|---|---|---|---|
| 16,384 | 3.38 | 16.1 | **4.76x** | 2.84x | 128 | 96.5 |
| 65,536 | 3.64 | 24.4 | **6.72x** | 4.22x | 129 | 65.3 |
| 262,144 | 6.54 | 49.3 | **7.55x** | 7.23x | 134 | 134 |




### The divergence sweep

`--div` sets the lognormal spread `sigma` of the individual reproduction number: 0 is plain Poisson offspring,
1.8 is strong superspreading.

| sigma | fwd tm / frontier (ms), N = 262,144 | x | jacfwd tm / frontier (ms), N = 65,536 | x |
|---|---|---|---|---|
| 0 (plain Poisson) | 4.47 / 16.8 | **3.76x** | 1.78 / 5.45 | **3.06x** |
| 1.0 | 8.23 / 56.8 | **6.9x** | 3.64 / 24.4 | **6.72x** |
| 1.8 (superspreading) | 25.5 / 166 | **6.5x** | 15.2 / 113 | **7.39x** |

## Where tallymap loses, or the margin shrinks

- **Reverse mode is roughly a tie, and at low overdispersion it is a loss.**  1.73x at 16,384 index cases falling to
  1.19x at 262,144. The tree adjoint also uses more memory than the frontier scan (1,107 MB against 279 MB at
  262,144), because its memory is proportional to a spawn Queue capacity that has to cover the whole tree. For a four-
  parameter branching model the honest recommendation is `jacfwd`, not `grad`, and the two tables above say so.
- **Two of the four parameters have no pathwise derivative at all.**  `log R0` and `logit p_det` enter only through
  discrete draws, so their pathwise derivative is exactly zero in both arms. The estimator for those is the ancestral
  score, which the item tallies for free in the same forward pass, and which this demo reports but does not benchmark
  against anything.
- **The frontier arm has to be told its width, and the demo tells it the right one.**  The frontier must cover every
  child drawn in the largest generation, which the spawn tree measures at about 2.9N here. If you give it that number
  it drops nothing and is a good program. The point of the comparison is that you have to know the number in advance;
  it is not that the arm is badly written.
- **The forward margin at zero overdispersion is the smallest in the sweep.**  3.76x at sigma = 0 against 6.5x at
  sigma = 1.8. Without superspreading there is much less for the execution model to exploit.
- **No incumbent library, so these numbers are against plain JAX.**  The established branching-process tools are R
  packages that walk one lineage at a time on the CPU. What is measured here is tallymap against the best plain-JAX
  program for the same model, not against `epichains`.
- **Compile time.**  The gradient lowers in 2.44 s (trace plus the NVRTC build of the item) plus 2.70 s of XLA, 5.14 s
  in total, against 0.18 s plus 0.99 s, 1.17 s, for the frontier scan. That is 4.4x the baseline's, paid once per
  program shape.

## How to rerun

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install tallymap

python run.py --quick     # CPU, about a minute: all arms, the agreement checks, the NaN control, PASS/FAIL
python run.py --full      # one GPU, regenerates results/branching_epidemic.json and plots/
```

`--full` runs in two stages: first `--mode sizes` at every N and divergence level to measure what the frontier
arm has to be sized for, then the measurement sweep using those sizes.
