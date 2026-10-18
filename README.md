# tallymap

Dr.Jit-style per-thread execution of divergent Monte Carlo loops, as a JAX primitive that composes with `jit`,
`vmap` and `grad`.

You write one ordinary JAX function of ONE item, with its own `lax.while_loop`, its own trip count, its own
gathers and its own `jax.random` draws: one walker, one photon, one trajectory, one chain. `tm.vmap` maps it over
N independent items whose results go into commutative sums, called tallies. On an NVIDIA GPU the item compiles to
a single CUDA kernel in which a lane picks up a new item as soon as its current one finishes, so a batch costs
the *average* item rather than the longest. Everywhere else it runs the lockstep backend, a masked plain-JAX
interpreter that doubles as the reference implementation. Because it is all one JAX primitive, `jit`, `vmap`,
`grad`, `jvp`, `jacfwd`, `jacrev`, `hessian`, `checkpoint` and `shard_map` work on it, and the result means
exactly `jax.vmap(item)` followed by a sum. tallymap was built for in-house photon-transport and particle-shower
engines and has nothing in it about either.

```python
import tallymap as tm
```

## Will it help you?

<!-- perf:begin regimes: from benchmarks/vs_jax/readme_table.md (make_tables.py) -->
Measured on one L40S, JAX 0.5.3, at the largest size of each problem, against the FASTEST plain-JAX arm that does
the same job. Every padded arm is padded to the trip count measured in the same sweep, not to a guess.
`docs/performance.md` has the method, every size, the 2080 Ti and JAX 0.11.2 rows, the agreement checks and the
compile times.

Measured against the fastest plain-JAX arm of each problem (`docs/performance.md` has the method, the padding and the losses):

| problem | kind | size | forward: tallymap vs best plain JAX | gradient: tallymap vs best plain JAX | gradient peak memory |
|---|---|---|---|---|---|
| random walk | natural | N = 4,194,304 | 3.1 vs 127 ms (padded scan (oracle)): **40.86x** faster | 8.84 vs 149 ms (padded scan (oracle)): **16.86x** faster | 112 MB vs 4,840 MB (43x less) |
| transport | natural | N = 4,194,304 | 1.77 vs 72.8 ms (padded scan (oracle)): **41.11x** faster | 16.3 vs 160 ms (padded scan (oracle)): **9.81x** faster | 96 MB vs 11.8 GB (126x less) |
| walk on spheres | natural | N = 4,194,304 | 1.27 vs 46.5 ms (padded scan (oracle)): **36.52x** faster | 8.68 vs 82.6 ms (padded scan (oracle)): **9.53x** faster | 128 MB vs 4,940 MB (39x less) |
| ode ensemble | natural | N = 1,048,576 | 5.32 vs 15.6 ms (padded scan (oracle)): **2.92x** faster | 25.9 vs 187 ms (padded scan (oracle)): **7.21x** faster | 28 MB vs 25.5 GB (931x less) |
| gillespie | natural | N = 1,048,576 | 30 vs 247 ms (padded scan (oracle)): **8.22x** faster | - | - |
| shower | natural | N = 262,144 | 1.66 vs 5.95 ms (generation vmap + compaction (oracle)): **3.58x** faster | 14.9 vs 10.3 ms (generation vmap + compaction (oracle)): **1.45x** **slower** | 673 MB vs 263 MB (**2.6x MORE**) |
| barrier | natural | N = 4,194,304 | 8.16 vs 10.8 ms (padded scan (oracle)): **1.32x** faster | 38.3 vs 123 ms (padded scan (oracle)): **3.21x** faster | 96 MB vs 12.9 GB (137x less) |
| interacting md | interacting | N = 262,144 | 1.44 vs 2.44 ms (padded neighbour list (oracle)): **1.70x** faster | 63.1 vs 244 ms (padded neighbour list (oracle)): **3.86x** faster | 132 MB vs 2,414 MB (18x less) |
| particle filter | interacting | N = 262,144 | 4.86 vs 13.5 ms (padded scan (oracle)): **2.77x** faster | 19.4 vs 222 ms (padded scan + remat (oracle)): **11.41x** faster | 65 MB vs 260 MB (4x less) |
| uniform | bad-fit | N = 4,194,304 | 1.6 vs 4.7 ms (jax.vmap + while_loop): **2.93x** faster | 5.59 vs 9.83 ms (padded scan (oracle)): **1.76x** faster | 96 MB vs 1,120 MB (12x less) |
| dense state | bad-fit | D = 64 | 0.251 vs 0.142 ms (jax.vmap + while_loop): **1.77x** **slower** | 52.3 vs 0.621 ms (padded scan (oracle)): **84.29x** **slower** | 8 MB vs 132 MB (16x less) |
| trajectory | bad-fit | L = 256 | 4.99 vs 1.7 ms (padded scan (oracle)): **2.93x** **slower** | - | - |

**Good fit:**
- many independent items whose loop lengths depend on the data;
- results that are sums;
- gradients through those loops;
- reverse-mode memory that must not grow with the trip count.

The win is not only divergence: a per-thread kernel keeps the loop state in registers instead of writing it to
memory every step.

**Interacting systems work when each time step is one call.** Write an outer `lax.scan` over time steps. Inside
each step, one `tm.vmap` call lets every item read the step's shared state (positions, a cell list, a field) and
write its own result. The interaction happens between calls, in ordinary JAX.

| problem | forward | `grad` | gradient memory |
|---|---|---|---|
| short-range molecular dynamics, cell list rebuilt each step, against the JAX-MD-style padded neighbour list | 1.7x (L40S), 2.7x (2080 Ti) | 3.9x, 1.4x | 18x less |
| particle filter: variable-length propagation in the call, resampling between calls | 2.8x, 2.2x | 11x, 5.4x | 4x to 32x less |

What cannot work is items reading each other, or a tally, INSIDE one call: pairwise forces within a step, sorting
across items, resampling inside the loop. Those raise `TallyError` at trace time.

**Bad fit**, measured so that the answer is a number:
- per-item dense state: a D = 64 matrix per item is 1.8x slower forward and 84x slower in `grad`;
- per-step trajectories as the output: 3x slower at 256 stored samples;
- a single item;
- the shower's reverse mode, which is 1.45x slower than a compacted plain-JAX frontier and uses 2.6x the memory
  (use `jacfwd` there).

Uniform loops still gain 1.8 to 2.9x here, but nothing like the divergent cases.
<!-- perf:end -->

## Why not `jax.vmap`, `lax.map`, Dr.Jit, Warp or Pallas?

- **`jax.vmap` / `lax.map(batch_size=)`.** `jax.vmap` runs every item in lockstep, so a batch costs the *longest*
  item, and every item needs its own copy of the output array. `lax.map` just chunks that. **The decisive point
  is not speed: plain JAX cannot reverse-differentiate a `lax.while_loop` at all** (`ValueError: Reverse-mode
  differentiation does not work for lax.while_loop`). To get a gradient in plain JAX you must rewrite the loop as
  a masked `lax.scan` padded to a step cap, which costs the cap for every item and holds memory proportional to
  it. If your items are short and uniform, or your output is small and N is small, use `jax.vmap`: in that regime
  the measured gap is around 12x, and what the library buys you is the gradient, not the speed.
- **Dr.Jit / Mitsuba 3.** The closest real overlap, and the mature version of this idea: a just-in-time compiler
  that fuses divergent Monte Carlo loops into megakernels, with reverse-mode differentiation by trajectory
  replay. The difference is the host language. Dr.Jit programs are Dr.Jit programs; tallymap programs are JAX
  programs, so they compose with the rest of a JAX codebase: an optimiser, neural-network parameters, a
  likelihood written in `jax.numpy`. **If you are not already in JAX, use Dr.Jit.**
- **NVIDIA Warp, Taichi.** Python-authored GPU kernels with per-thread control flow and their own automatic
  differentiation. The right tools if you want to write kernels. tallymap's claim is the opposite: you write one
  ordinary JAX function of one item, you never see a kernel, and the derivative is JAX's derivative of that
  function, checked against `jax.vmap` of the same code. Neither composes with JAX transformations.
- **Numba-CUDA, CuPy raw kernels, hand-written CUDA.** Without tallymap, the natural hand-written version of the
  quickstart (a masked `lax.while_loop` with one shared `.at[].add` tally) measured 32x slower than tallymap's
  generated kernel on an L40S, and it has no gradient at all. The honest
  version: if you are willing to hand-write AND hand-differentiate one kernel for one program, you can beat
  tallymap. On the one workload where that was measured, a hand-tuned kernel was slightly ahead. The claim is
  that you get close to that without writing CUDA, and you get `jvp`, `vmap`, `grad` and `hessian` for free.
- **`jax.experimental.pallas`.** The one that looks like it should be the answer. Pallas on GPU lowers through
  Triton, which is built around uniform block-level control flow; a per-lane data-dependent `while_loop` with a
  work queue is not that programming model, and Pallas is explicitly experimental.

<!-- perf:begin libraries -->
**Against the libraries people use.** `examples/demos/` holds eight head-to-head demos. Each runs a real library's
own code or problem as the library ships it, and the same computation through tallymap. Measured on one L40S,
JAX 0.11.2; each demo's README states where the margin shrinks or tallymap loses.

| demo | incumbent | measured |
|---|---|---|
| `nuts_blackjax` | BlackJAX 1.7.1 NUTS / elliptical slice | 14.5-33x / 23-60x against `jax.vmap` over chains; `grad` through a sampler chain becomes possible (1.05x by dimension 64) |
| `gymnax_rollouts` | gymnax 1.0.0 `RolloutWrapper` | 18-26x, bitwise-identical returns |
| `gillespie_crn` | crn-jax 0.3.1, its own `simulate_interval` unmodified | 25-56x, bitwise per replicate, 44-128x less memory |
| `radiative_transfer` | Ray-trax, its own problem | 10-24x forward, 10-20x gradient, 243x less memory at 64^3 |
| `stiff_ode` | diffrax 0.7.2 Kvaerno3 | 97-274x per solver step, 20-126x gradient |
| `sdf_inverse_rendering` | a padded plain-JAX sphere tracer | 108-180x gradient; forward only 5-11x |
| `adaptive_quadrature` | quadax 0.3.0 `quadgk` | 3-18x forward, 12x gradient, 98-380x less memory |
| `branching_epidemic` | a plain-JAX generation-frontier program | 4-14x forward, 5-8x `jacfwd`; reverse is a tie |

diffrax itself now runs inside an item (`docs/equinox.md`, `examples/diffrax_ensemble.py`).

**On CPU** there is only the lockstep backend, which is plain JAX. Measured at N = 65,536, its forward pass is
within 0.7x to 1.3x of the best plain-JAX program on the natural problems. Its gradient is 3x to 7x SLOWER than a
padded `lax.scan` given the oracle step cap (`docs/performance.md`, "cpu"). What it buys you there is a gradient
through a `lax.while_loop` without choosing a cap, and memory proportional to the outputs. It does not buy speed.
<!-- perf:end -->

## Install

```
pip install tallymap                 # CPU: the lockstep backend, plain JAX
pip install 'tallymap[cuda]'         # + JAX's CUDA 12 plugin (quote it: zsh treats [] as a glob)
```

A **C++17 compiler is needed once**, the first time you use the GPU backend, to build the `jax.ffi` launch shim.
It is about 200 lines and the result is cached per toolchain and jaxlib version. If `$CXX` is set it is used.

**GPU memory.** The CUDA driver reserves a kernel's per-thread local memory (large per-item arrays live there)
outside JAX's memory pool, and JAX preallocates 75% of the GPU by default. If a call fails with `out of memory` at
kernel launch, the error says how much the kernel needs; set `XLA_PYTHON_CLIENT_PREALLOCATE=false`, or a lower
`XLA_PYTHON_CLIENT_MEM_FRACTION`, before JAX starts.

Tested on Linux with NVIDIA GPUs of compute capability 7.0 and above (the spawn queue needs 7.0; the library is
tested on 7.5 and 8.9). The CPU path is plain JAX and needs nothing extra. The wheel holds only the package: the
tests, examples, benchmarks and docs are in the sdist and in the git repository.

## Quickstart

```python
import jax, jax.numpy as jnp
import numpy as np
from jax import lax
import tallymap as tm

NB = 16                                                    # histogram bins

def walker(t, x, key, theta):
    """ONE item.  t is its Tally: thread it through the loop like a PRNG key."""
    def body(c):
        k, x, t = c
        u = jax.random.uniform(jax.random.fold_in(key, k))
        t = t.add("hist", (x * NB).astype(jnp.int32), theta * (1.0 - x))
        return k + 1, x * (0.5 + 0.5 * u), t
    k, x, t = lax.while_loop(lambda c: c[1] > 0.05, body, (0, x, t))
    return t, k                                            # the Tally, plus an optional per-item result

f = tm.vmap(walker, outputs={"hist": tm.Sum((NB,))}, in_axes=(0, 0, None), max_steps=100)

N = 10_000
xs = jnp.linspace(0.1, 0.99, N)
keys = jax.random.split(jax.random.PRNGKey(0), N)
theta = jnp.float32(2.0)

res = jax.jit(f)(xs, keys, theta)
print(res.outputs["hist"], res.per_item.mean(), res.stats)
g = jax.jit(jax.grad(lambda th: f(xs, keys, th).outputs["hist"].sum()))(theta)
```

`examples/quickstart.py` is this program with its checks. The other examples are listed at the end.

**The rules of a Tally.** `t = t.add(name, index, value)` returns a NEW Tally. Tallies are write-only and used
linearly: no reads, no forks, no dropped results. That is checked when the function is traced, and a violation
raises `tm.TallyError`, because it would silently disagree with the dense meaning. `index` is an int array (a flat
index for a 1-D output) or a tuple of int arrays, one per output dimension.

A tally indexed by a TUPLE is the right way to get per-item vector outputs: `Sum((N, K))` indexed by
`(item_id, slot)` gives each item its own K-element result without a per-item array in the kernel.

## The mental model

Three sentences, and everything else follows from them.

1. **One item is one GPU thread running its own loop.** Not one lane of a wide batch: a real thread with its own
   trip count, whose state lives in registers. When it finishes, the lane takes the next item. This is why the
   batch costs the average item and not the longest one, and why a per-item state of tens of scalars is fast
   while a per-item D x D matrix is not.
2. **Tallies are atomic sums.** There is no per-item output array. Every `t.add` goes into one accumulator, in
   shared memory when the output is small enough and in global memory otherwise. Sums are commutative, so the
   order is not fixed, which is where the float semantics below come from.
3. **Reverse mode replays each item from its inputs.** The forward pass stores nothing per step. The adjoint
   re-runs each item with two-level checkpointing and applies the vector-Jacobian product of one step at a time,
   so **memory is O(1) per item in the trip count** and there is no step cap anywhere. That is the property
   plain JAX cannot give you, and it is the reason to pay for a CUDA emitter in your dependency tree.

## Nested loops: solvers inside an item

An item may contain loops inside loops, each with its own data-dependent trip count: a `while_loop` inside a
`while_loop` inside a `cond`. The cuda backend compiles every level into an ordinary loop of the item's own
thread. Every transformation handles the nesting: `jvp`, `jacfwd` and `grad`. Reverse mode checkpoints each
loop level recursively, so memory stays O(1) per item.

This matters most for numerical methods that are themselves iterative inside each step. An implicit (stiff) ODE
solver has three nested loops: adaptive time steps, the Runge-Kutta stages of each step, and a Newton or chord
solve inside each stage. Under `jax.vmap` every one of those loops runs to the batch maximum, and each finished
lane executes masked iterations. In a tallymap item each loop runs exactly as often as that item needs.

Measured on one L40S. The workload is an ensemble of stiff 3-state ODEs integrated with a Kvaerno3 / chord /
PID-controller algorithm, written as a tallymap item, against `jax.vmap(diffrax.diffeqsolve(...))` with the same
solver settings:

| | forward | `grad` |
|---|---|---|
| tallymap item | 99 to 280x faster (step-count matched) | 19 to 125x faster, 2 to 67x less memory |
| at 1M trajectories | | 2.8 s / 72 MB against 53 s / 4.8 GB |

The user's literal `diffrax.diffeqsolve(...)` now runs inside an item too (`docs/equinox.md`). Measured on a
stiff implicit ensemble against `jax.vmap(diffeqsolve)`: forward 14 to 136x faster, gradient 1.7 to 23x faster
at 19 to 282x less memory. The margin is smaller than for the hand-written item because of diffrax's per-step
overheads inside the item.

Two costs to know:

- Replaying three nested loops is expensive. On that solver tallymap's `grad` costs about 84x its own forward
  pass, where diffrax's checkpointed adjoint costs about 12x. The gradient margin is therefore one order of
  magnitude where the forward margin is two.
- `rev_checkpoints` trades recomputation against per-item state, and it is not monotone. On that solver,
  raising it from the default 16 to 32 made `grad` 3x slower. Measure it per item rather than deriving it from
  the trip count. (Those numbers predate the checkpoint-stride fix, which made C far less sensitive everywhere it
  was re-measured: on the `benchmarks/vs_jax` problems the spread between C = 8, 16, 32 and 64 is now within a
  factor of 1.6, against up to 20 before. The advice stands, but try fewer values.)

## A plain-JAX pitfall tallymap avoids: `cond` around a loop under `vmap`

Under `jax.vmap`, a `lax.cond` whose predicate differs per item is turned into a select. The batching rule
broadcasts **every operand** of the `cond` to the batch size, including arrays the branch closes over. When the
branch is straight-line code, XLA fuses that broadcast away. When the branch contains a `while_loop` that reads the
array, the broadcast array becomes loop state and is materialised: **one full copy per item**.

Measured (XLA temp memory, CPU, JAX 0.5.3 and 0.11.2 identical). The setup is a 1.2 MB table read inside the loop:

| item body | N = 1,024 | N = 8,192 |
|---|---|---|
| `cond(p, while-loop reading the table, 0)` under `jax.vmap` | 1,152 MiB | 9,216 MiB (1.00 copy per item) |
| the same work with `p` moved into the trip count, no `cond` | 0.0 MiB | 0.1 MiB |
| `cond` around straight-line reads | 0.0 MiB | 0.0 MiB |
| tallymap, lockstep backend (the cuda backend runs a real branch) | 0.0 MiB | 0.3 MiB |

A ray tracer that tests a BVH leaf's triangles in an inner loop hits exactly this. With 32,768 triangles,
`jax.vmap` asked for 104 GiB at 65,536 rays, and 6.5 TiB at 4.2M.

**In plain JAX:** do not wrap a loop in a per-item `cond`. Fold the predicate into the loop's trip count
(`n = jnp.where(p, n, 0)`), or unroll the inner loop if its maximum is small and static. **In tallymap** none of
this is needed: a `cond` is a real branch of the item's thread.

## Correctness rules you need to know

### `max_steps` and the budget idiom

A tallymap call means exactly `jax.vmap(item)` followed by a sum. It deviates only where a tallymap **resource
limit** gets in the way, and then it says so: every float tally that includes an affected item becomes NaN.
There are two such limits, `max_steps` and the spawn `Queue` capacity, and they behave identically.

- **`max_steps=None` is the default.** No limit, no deviation, and therefore also no protection: an item whose
  loop does not terminate does not terminate, and on a GPU that means a hung kernel. **Set `max_steps` as a
  safety net**, generously, at a value no correct run should reach.
- **When an item hits `max_steps`**, `stats['n_truncated']` counts it, `Result.truncated` flags it per item, and
  every float tally of that call is NaN. Per-item outputs are not poisoned; they carry their own flag in the same
  `Result`. Integer tallies cannot be NaN, so for those the counter is the only signal: check it.
- **There is no opt-out, because a deliberate budget is not a tallymap limit.** Write it into the loop condition:

  ```python
  def item(t, ...):
      def cond(c):
          alive, step = c[0], c[1]
          return alive & (step < K)          # a budget you chose: exact jax.vmap semantics, nothing is poisoned
      ...
  ```

  That program means what it says, and `max_steps` stays a safety net above it.

### Out-of-range emit indices

An emit whose index is negative or at least the output size is **dropped**, exactly as JAX's own scatter drops an
out-of-bounds update, so this is not a deviation from the dense meaning: `tm.reference` does the same. It is
counted in `stats['dropped_emits']`. Nothing is poisoned, because dropping is the defined behaviour, but a
non-zero count almost always means an off-by-one in your binning: `(x * NB).astype(int32)` when `x` can reach
1.0, a forgotten `clip`, a negative index out of a `floor`. Check it.

### f32 tally accumulation

`tm.Sum(shape, dtype=float32, exact=False)`. f32 sums over millions of emits lose low-order bits, and forward and
reverse mode accumulate in different orders, so their derivatives agree only to that accumulation error, which
can be 1e-4 relative on a large tree and varies from run to run on a GPU. With `exact=True` the sum is
accumulated in int64 fixed point: bitwise order-independent, identical across backends and schedules (the
lockstep backend needs `jax_enable_x64` for it). Unrepresentable values are dropped and counted in
`stats['exact_overflow']`. Use `exact=True` when derivatives of very large sums must agree tightly, or when you
need run-to-run reproducibility on a GPU.

### Sizing the spawn Queue

`t.spawn(child, when=...)` needs `tm.vmap(..., spawn=tm.Queue(capacity))`. The child runs the same function and
its per-item result is discarded. The Queue holds the **pending** children, not all of them. A child that does
not fit is dropped, counted in `stats['dropped']`, and makes the call's float tallies NaN.

How many children are pending at once depends on your program and on the scheduling, so **measure it once per
configuration**: run with a large Queue, read `stats['spawned']` (the tree size), then size the Queue and confirm
`stats['dropped'] == 0`. As a rule of thumb the forward pass needs a power of two at least 0.65 times the
children per call. **Reverse mode needs capacity >= ALL children of the call**, because the tree adjoint records
the whole tree, and a tree larger than the capacity gives a NaN gradient rather than a wrong one. The tree
adjoint's memory is proportional to capacity times the per-item state size, which is what bounds how much of a
workload fits in one call: split bigger workloads into several calls. The CUDA spawn queue needs compute
capability 7.0 or newer.

### One key per item, and the 32-bit limit

tallymap invents no randomness: pass per-item keys (`jax.random.split`), or an item id and `fold_in` it. Results
never depend on scheduling. But `fold_in` takes a 32-bit message, so a key built from ONE counter such as
`fold_in(key, id * stride + step)` repeats after `2**32 / stride` items: with a stride of 64, items `i` and
`i + 67_108_864` draw the same random numbers. Fold the item id and the step separately,
`fold_in(fold_in(key, id), step)`, or chunk the work and fold the chunk index first,
`fold_in(fold_in(key, chunk), local_id * stride + step)`. See `tests/edge.py`, the cell "per-item keys".

## Gradients: exactly computed is not the same as what you want

tallymap returns the exact derivative of the program you wrote. Whether that is the derivative of the quantity
you are estimating is a different question, and it is the most likely way to get a wrong published number out of
this library.

Take a random walk with absorbing boundaries, where `theta` scales the step size, and differentiate the total
overshoot past the boundary (`examples/gradient_pitfall.py`, measured):

```
A  walkers 8192  mean steps 85.5  max steps 631  truncated 0
A  d(total overshoot)/d(theta):  grad 43387.6562   jacfwd 43387.6523   CRN finite difference 5808.4717
```

`grad` and `jacfwd` agree to 9e-8, so the mechanics are right: two independent modes of differentiation computed
the same derivative of the same program. The number is still **7.5x the true slope** (it was 9.3x in a larger configuration on a GPU; the factor
depends on the problem, the point is that it is not 1). The reason: the *number of steps* a walker takes depends
on theta, and a pathwise derivative differentiates the steps that were taken, not the decision to take one more.
The missing piece is a boundary term.

The same walk with theta entering only a discrete decision (`u < theta`, so theta is the probability of stepping
left) is worse:

```
B  d(left exits)/dp:  grad 0.0   CRN finite difference 31520.0
```

Exactly zero, where the true slope is 31,520. The derivative of a comparison is zero almost everywhere. Here you
need a score-function (REINFORCE) estimator, which no automatic-differentiation system can invent for you.

**How to detect it.** A central finite difference on the SAME per-item keys, used only to TEST the gradient and
never to compute it:

```python
loss = lambda th: f(xs, keys, th).outputs["overshoot"][0]
g  = jax.grad(loss)(theta)
h  = 1e-3
fd = (loss(theta + h) - loss(theta - h)) / (2 * h)      # same keys: common random numbers
assert abs(g - fd) < 0.2 * abs(fd)                      # fails for the walk above, as it should
```

Common random numbers are what make this cheap enough to keep in a test suite: without them the Monte Carlo noise
swamps the signal. Run the check at a point **away from the optimum**, where the true slope is not zero, and
compare per-output vectors rather than scalar ratios. `docs/validation.md` calls this question B, as against
question A, which is whether the derivative is the exact derivative of the program; tallymap owns A and your
program owns B.

### An adaptive step-size controller: stop-gradient its factor

Differentiating THROUGH a step-size controller gives an exactly computed derivative whose value is decided by
rounding. Measured on the ensemble of `examples/ode_ensemble.py` (4096 forced Van der Pol trajectories,
Bogacki-Shampine 3(2), T = 20) in plain JAX, no tallymap involved:

| | controller differentiated | factor stop-gradient-ed |
|---|---|---|
| f32 gradient, first component | -216,307 | -508.3 |
| f64 gradient, first component | +116,690 | -503.8 |
| f32 after theta * (1 + 1e-7) | -1,199,912 | -508.6 |
| largest per-trajectory derivative | 2.0e5 | 364 |

The controller's feedback is chaotic for the stiff trajectories, so through it the gradient has no stable value.
The symptom to look for is `grad` and `jacfwd` of the same program agreeing on one platform and not on another.
They agree only when the two compiled programs round identically, and XLA:CPU from JAX 0.8 does not make them do so.
Write `fac = lax.stop_gradient(fac)`, as diffrax does. The gradient is then that of the solution on the steps the
solver chose.

## When a gradient is NaN, check in this order

tallymap uses NaN as its own failure signal, which makes NaN ambiguous. Resolve it in this order, because the
common cause is first:

1. **Is `jvp` or `jacfwd` NaN too?** Then no reverse-mode machinery was involved: it is your own arithmetic. Look
   for overflow or a division by zero in the tangent of your own program, which is easy to produce in a diverging
   iteration.
2. **Otherwise look at `stats`.** A truncated item (`stats['n_truncated']`) or a dropped spawn child
   (`stats['dropped']`) poisons float tallies on purpose, and the forward pass is NaN too.
3. **Only then suspect a reverse-mode check.** Reverse mode replays every item and verifies the replay with a
   per-item fingerprint; the tree adjoint also checks that the recorded tree fits the Queue and that its two
   recording passes agree. Any failed check makes that call's float cotangents NaN, so a gradient is never
   silently wrong. Rerun with `rev_stats=True` and read the counters:

   ```python
   tm.diagnostics()["reverse"]    # {'replay': {'calls', 'mismatch', 'last'}, 'tree': {...} or None}
   ```

`tm.diagnostics()` is the one place to look when something is wrong, and what to paste into a bug report:
versions, GPU, compute capability, driver, the CUDA compiler and cache directory, these counters, and the last
item jaxprs and generated kernels. `tm.diagnostics(f, *args)` reports what `jax.jit(f).lower(*args)` builds.

## Options

| option | default | meaning |
|---|---|---|
| `backend` | `'auto'` | `'auto'`: cuda on a GPU, where a program the cuda emitter cannot compile raises `tm.UnsupportedError`; lockstep elsewhere, with a one-time warning. `'cuda'`: cuda or raise. `'lockstep'`: always the plain-JAX interpreter, slow by design, the reference. |
| `allow_fallback` | `False` | with `backend='auto'` on a GPU, run lockstep with a warning instead of raising. It is about 100x slower, which is why it is not the default. |
| `cuda_math` | `'approx'` | cuda f32 division, sqrt and rsqrt. `'approx'`: the PTX XLA:GPU itself emits, so it is the arithmetic a plain JAX program on the same GPU gets. `'ieee'`: correctly rounded, the portable reference, about 1.5x slower. |
| `cuda_min_blocks` | `0` | cuda `__launch_bounds__` minimum blocks per multiprocessor, which caps registers. Performance only; the best value is per kernel. |
| `lockstep_block` | `2**18` | lockstep backend: lanes per block, which bounds its memory. |
| `rev_checkpoints` | `16` | reverse mode: checkpoint buffer size C. O(K) work for loops of K <= C^2 steps; 2C states of memory per item. |
| `rev_step_segments` | `0` | reverse mode: recompute each loop step in S checkpointed segments inside its VJP. Helps when a step's residuals spill out of the register file; costs recomputation otherwise. |
| `rev_stats` | `False` | reverse mode: report replay mismatches and tree-adjoint counts to the host. Off by default because a host callback takes `jax.jit` off its C++ fast path, which can cost an order of magnitude. |

Any other option name raises. Options starting with `_` are private measurement and negative-control switches
(`tallymap/_api.py::_OPTS`). The only environment variable is `TALLYMAP_CUDA_CACHE`, the cubin cache directory,
which defaults to `~/.cache/tallymap`.

### Which derivative to use

**Forward mode** (`jvp`, `jacfwd`, `linearize`): one kernel computes the primal and the tangent together.
`jacfwd` with P parameters runs one kernel with P dual lanes sharing the primal, the control flow, the gathers
and the random numbers. **Reverse mode** (`grad`, `vjp`, `jacrev`): the forward kernel stores nothing per step
and the adjoint replays each item. Large read-only tables receive their cotangents by atomic scatter into one
accumulator, never a dense per-item copy. Programs that spawn use the tree adjoint, one launch per generation,
deepest first.

| situation | use |
|---|---|
| few parameters, or you need the Jacobian itself (Gauss-Newton) | `jacfwd` / `jvp` |
| many parameters and a scalar loss (per-bin tables, network weights): the cost does not depend on P | `grad` |
| a few globals plus big tables | `grad` for the loss; `jacfwd` over the globals only if you need a metric |

Reverse mode costs roughly 3.5 to 10 times the forward pass independent of P, while forward dual lanes cost about
forward x (1 + alpha P). So the break-even in P is the ratio of those, typically in the tens, and it is worth
measuring once for your own program. Do not run forward and reverse on one call: `jax.value_and_grad` computes
the value once.

Measured `jax.jit` wall time per call on the toy walk of `benchmarks/` (1M items, a 256-bin table, L40S, JAX
0.5.3, `benchmarks/perf_baseline.json`): forward 0.65 ms, `jvp` 0.72 ms, `jacfwd(8)` 4.92 ms, `grad` 2.25 ms.

Second order (`hessian`, Hessian-vector products, `jacrev(jacfwd)`, grad-of-grad) composes from the same rules,
except through the spawn tree adjoint, which raises.

## Support boundary

What is explicitly out of scope, so that you do not have to find out by trying:

- **No TPU and no ROCm.** Any non-CUDA platform runs the lockstep backend, which is correct and about 100x
  slower than the CUDA backend.
- **`jax.export` of a CUDA program is unsupported.** Its custom call is not on JAX's export allow-list and its
  kernels are registered per process, so a deserialised CUDA program is not portable. Lockstep programs export
  normally.
- **No second-order derivatives through spawn.** The tree adjoint raises for them.
- **A100 and H200 are untested.** Nothing suggests they will not work; no tuning constant was measured on them.
- **Coupling between items inside the loop raises**, and so do reading, forking or dropping a tally, `n_active`
  together with spawn, effects inside the item (`jax.debug.print`, callbacks), `grad` through a `lax.while_loop`
  AROUND a call (that is JAX's own limitation, not tallymap's), 2^31 items or more in one call, and unknown
  option names.
- **An `UnsupportedError` on a primitive the CUDA emitter does not know is a normal bug report**, not a dead end.
  Add `backend='lockstep'` and your program will run; attach `tm.diagnostics()` to the issue. Adding a primitive
  is usually one rule plus one test row (`CONTRIBUTING.md`).

## Tests, examples, citation, licence

```
python tests/run_all.py [--gpu] [--x64] [--unfused] [--canary] [--only edge,reverse]
python benchmarks/perf.py
```

`python tests/run_all.py` is the entry point. **`pytest` is not the runner**: each suite needs fresh JAX state and
runs as its own process. The suites are edge cases, the transformation-composability matrix, reverse mode, the
spawn tree adjoint, per-operation CUDA parity (`--gpu`) and the examples.

| example | what it shows |
|---|---|
| `examples/quickstart.py` | the program above, with its checks |
| `examples/photon_toy.py` | transport into sensor tallies, `jacfwd` and `grad` |
| `examples/shower_toy.py` | `spawn`, and the tree-adjoint gradient |
| `examples/ensemble_toy.py` | per-item results and `n_active` |
| `examples/gradient_pitfall.py` | the gradient section above, runnable |

Design decisions and the measurements behind them: `docs/design.md`. Validation and the A/B rules:
`docs/validation.md`. Float semantics: `docs/float_semantics.md`. Terms: `docs/glossary.md`. Supported JAX
versions, the release process and a "when X breaks, do Y" table: `docs/maintenance.md`. Covered primitives:
`docs/primitives.md`. Speed against plain JAX: `docs/performance.md`.

If you use tallymap in published work, please cite it: see `CITATION.cff`.

Licensed under the Apache License 2.0; see `LICENSE` and `NOTICE`.
