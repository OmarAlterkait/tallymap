# tallymap

`jax.vmap` for per-item loops of different lengths whose results are sums.

You write an ordinary JAX function of ONE item (one photon, one walker, one trajectory, one chain) with its own
`lax.while_loop`, its own trip count and its own `jax.random` draws. `tm.vmap` maps it over N independent items
whose results go into commutative sums, called tallies. On an NVIDIA GPU the item compiles to a single CUDA kernel
in which each thread runs one item for exactly as many steps as that item needs, then claims the next. Everywhere
else it runs the xla backend, a plain-JAX interpreter. It is one JAX primitive, so `jit`, `vmap`, `grad`, `jvp`,
`jacfwd`, `jacrev`, `hessian`, `checkpoint` and `pmap` work on it, and the result means exactly `jax.vmap(item)`
followed by a sum.

What it gives you that plain JAX does not:

- **Divergent loops at the cost of the average item, not the longest.** 1.3 to 41x faster forward on the natural
  problems below (one L40S).
- **Reverse mode through `lax.while_loop`.** Plain JAX cannot reverse-differentiate a `while_loop` at all. tallymap
  replays each item from its inputs, so gradient memory does not grow with the trip count and there is no step cap.

**Status:** 0.7.0, unreleased. JAX >= 0.6.2. The kernel backend runs on Linux, on an NVIDIA GPU of compute
capability 7.0 or newer or on the CPU (with a C++17 compiler); the xla backend runs anywhere JAX does. Not on PyPI: install from GitHub (below).
tallymap was built for in-house photon-transport and particle-shower engines and has nothing in it about either.

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

`examples/quickstart.py` is this program with its checks; the other examples are listed at the end. Install it first (next section).

**The rules of a Tally.** `t = t.add(name, index, value)` returns a NEW Tally. Tallies are write-only and used
linearly: no reads, no forks, no dropped results. That is checked when the function is traced, and a violation
raises `tm.TallyError`, because it would silently disagree with the dense meaning. `index` is an int array (a flat
index for a 1-D output) or a tuple of int arrays, one per output dimension.

A tally indexed by a TUPLE is the right way to get per-item vector outputs: `Sum((N, K))` indexed by
`(item_id, slot)` gives each item its own K-element result without a per-item array in the kernel.

## Install

```
pip install "tallymap @ git+https://github.com/OmarAlterkait/tallymap@v0.6.0rc1"         # CPU
pip install "tallymap[cuda] @ git+https://github.com/OmarAlterkait/tallymap@v0.6.0rc1"   # + JAX's CUDA 12 plugin and NVRTC
```

For the tests, examples and benchmarks, clone the repository and install it editable:

```
git clone https://github.com/OmarAlterkait/tallymap && cd tallymap && pip install -e ".[cuda]"
```

In a project that depends on tallymap, pin a tag (`@v0.6.0rc1`) as above. Until 1.0 a minor release may change the
API; `CHANGELOG.md` lists every change.

A **C++17 compiler** (GCC or Clang; `$CXX` if set) builds the `jax.ffi` launch shim once per toolchain and jaxlib
version, and on the CPU it also compiles each program's kernel (about 0.5-2 s, cached on disk).

**GPU memory.** The CUDA driver reserves a kernel's per-thread local memory (large per-item arrays live there)
outside JAX's memory pool, and JAX preallocates 75% of the GPU by default. If a call fails with `out of memory` at
kernel launch, the error says how much the kernel needs; set `XLA_PYTHON_CLIENT_PREALLOCATE=false`, or a lower
`XLA_PYTHON_CLIENT_MEM_FRACTION`, before JAX starts.

Tested on Linux with NVIDIA GPUs of compute capability 7.5 (2080 Ti) and 8.9 (L40S); the spawn queue needs 7.0.

## Will it help you?

<!-- perf:begin regimes: from benchmarks/vs_jax/readme_table.md (make_tables.py) -->
Measured on one L40S, JAX 0.5.3, at the largest size of each problem, against the FASTEST plain-JAX arm that does
the same job. Every padded arm is padded to the trip count measured in the same sweep, not to a guess.
`docs/performance.md` is the summary; `docs/performance_detail.md` has the method, every size, the 2080 Ti and JAX
0.11.2 rows and the agreement checks.


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

## The mental model

Three sentences, and everything else follows from them.

1. **One item is one GPU thread running its own loop.** Not one element of a wide batch: a real thread with its
   own trip count, whose state lives in registers. When it finishes, the thread takes the next item (with
   `schedule='step'`, already at the end of each loop step). This is why the batch does not cost its longest item
   times the batch size, and why a per-item state of tens of scalars is fast
   while a per-item D x D matrix is not.
2. **Tallies are atomic sums.** There is no per-item output array. Every `t.add` goes into one accumulator, in
   shared memory when the output is small enough and in global memory otherwise. Sums are commutative, so the
   order is not fixed, which is where the float semantics below come from.
3. **Reverse mode replays each item from its inputs.** The forward pass stores only the first C states of each
   loop (an item whose loop is that short is never replayed). The adjoint
   re-runs each item with two-level checkpointing and applies the vector-Jacobian product of one step at a time,
   so **memory is O(1) per item in the trip count** and there is no step cap anywhere. That is the property
   plain JAX cannot give you, and it is the reason to pay for a CUDA emitter in your dependency tree.

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
  Triton, which is built around uniform block-level control flow; a per-thread data-dependent `while_loop` with a
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

**On CPU** the same kernel runs compiled for the host: on one 32-core socket, forward 2.8-5.1x faster than the
best plain-JAX program on the divergent problems and at parity on long uniform loops, gradients 8.6-54x faster
(the table is in "Backends").
<!-- perf:end -->

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

`tm.Sum(shape, dtype=float32, fixed_point=False)`. f32 sums over millions of emits lose low-order bits, and forward and
reverse mode accumulate in different orders, so their derivatives agree only to that accumulation error, which
can be 1e-4 relative on a large tree and varies from run to run on a GPU. With `fixed_point=True` the sum is
accumulated in int64 fixed point (each value rounded to a multiple of `1/scale`): bitwise order-independent,
identical across backends and schedules (the xla backend needs `jax_enable_x64` for it). Unrepresentable values
are dropped and counted in `stats['fixed_point_overflow']`, and the bin they were emitted to is NaN. Each kernel
flags its own bins, so a tangent overflow makes that tangent bin NaN on every jvp / jacfwd / linearize path and
leaves the primal finite. Use `fixed_point=True` when derivatives of very large sums must agree tightly, or when you
need run-to-run reproducibility on a GPU.

### Sizing the spawn Queue

`t.spawn(child, when=...)` needs `tm.vmap(..., spawn=tm.Queue(capacity))`. The child runs the same function and
its per-item result is discarded. The Queue holds the **pending** children, not all of them. A child that does
not fit is dropped, counted in `stats['children_dropped']`, and makes the call's float tallies NaN.

How many children are pending at once depends on your program and on the scheduling, so **measure it once per
configuration**: run with a large Queue, read `stats['spawned']` (the tree size), then size the Queue and confirm
`stats['children_dropped'] == 0`. As a rule of thumb the forward pass needs a power of two at least 0.65 times the
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

## Derivatives

### Which derivative to use

**Forward mode** (`jvp`, `jacfwd`, `linearize`): one kernel computes the primal and the tangent together.
`jacfwd` with P parameters runs one kernel with P dual lanes sharing the primal, the control flow, the gathers
and the random numbers. **Reverse mode** (`grad`, `vjp`, `jacrev`): the forward kernel stores the first C states
of each loop and the adjoint replays each item from there. Large read-only tables receive their cotangents by atomic scatter into one
accumulator, never a dense per-item copy. Programs that spawn use the tree adjoint, one launch per generation,
deepest first.

| situation | use |
|---|---|
| few parameters, or you need the Jacobian itself (Gauss-Newton) | `jacfwd` / `jvp` |
| many parameters and a scalar loss (per-bin tables, network weights): the cost does not depend on P | `grad` |
| a few globals plus big tables | `grad` for the loss; `jacfwd` over the globals only if you need a metric |

Reverse mode costs roughly 3.5 to 10 times the forward pass independent of P, while forward dual lanes cost about
forward x (1 + alpha P). So the break-even in P is the ratio of those, typically in the tens, and it is worth
measuring once for your own program. Through spawn (the tree adjoint, measured on four spawning programs, L40S)
the rule depends on whether `jacfwd` gets dual lanes: with lanes it wins up to about 64 parameters; when lanes are
refused (a large per-lane loop state; `tm.diagnostics()['kernels']` then shows no `dual_lanes` tag) `jacfwd` is
one launch per parameter and the tree adjoint wins from about 4 parameters (4-24x at 16-64). Do not run forward and reverse on one call: `jax.value_and_grad` computes
the value once.

Measured `jax.jit` wall time per call on the toy walk of `benchmarks/` (1M items, a 256-bin table, L40S, JAX
0.5.3, the `benchmarks/perf.py` baseline): forward 0.65 ms, `jvp` 0.72 ms, `jacfwd(8)` 4.92 ms, `grad` 2.25 ms.

Second order (`hessian`, Hessian-vector products, `jacrev(jacfwd)`, grad-of-grad) composes from the same rules,
except through the spawn tree adjoint, which raises.

### Exactly computed is not the same as what you want

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

### When a gradient is NaN, check in this order

tallymap uses NaN as its own failure signal, which makes NaN ambiguous. Resolve it in this order, because the
common cause is first:

1. **Is `jvp` or `jacfwd` NaN too?** Then no reverse-mode machinery was involved: it is your own arithmetic. Look
   for overflow or a division by zero in the tangent of your own program, which is easy to produce in a diverging
   iteration.
2. **Otherwise look at `stats`.** A truncated item (`stats['n_truncated']`) or a dropped spawn child
   (`stats['children_dropped']`) poisons float tallies on purpose, and the forward pass is NaN too.
3. **Only then suspect a reverse-mode check.** Reverse mode through spawn (the tree adjoint) checks that the
   recorded tree fits the Queue and that its two recording passes agree; a failed check makes that call's float
   cotangents NaN, so a gradient is never silently wrong. Rerun with `rev_stats=True` and read the counters:

   ```python
   tm.diagnostics()["reverse"]    # {'single': {'used', 'why'}, 'tree': {...} or None}
   ```

`tm.diagnostics()` is the one place to look when something is wrong, and what to paste into a bug report:
versions, GPU, compute capability, driver, the CUDA compiler and cache directory, these counters, and the last
item jaxprs and generated kernels. `tm.diagnostics(f, *args)` reports what `jax.jit(f).lower(*args)` builds.

## Backends

There is one compiled backend, the **kernel** backend, and JAX's device placement decides where it runs: tallymap
never picks a device. A computation placed on a CUDA GPU runs the CUDA kernel; one placed on the CPU
(`jax.default_device`, `jax.device_put`, or simply no GPU) runs the same generated C compiled for the host, one host
thread per GPU thread, as many threads as `OMP_NUM_THREADS` or else the CPUs the process may use (its cpuset or
Slurm allocation). Every feature works on both: forward, step schedule, jvp and dual lanes, both reverse modes,
spawn and the tree adjoint, fixed-point sums, truncation, `n_active`. `backend='xla'` is the plain-JAX reference
interpreter, for checking and for the platforms the kernel backend does not compile for.

**Several devices.** Shard the item axis the usual JAX way and every device runs its own items; tallies and
stats are summed across devices and per-item results stay sharded like the items. There is no tallymap API for it:

```python
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
mesh = Mesh(jax.devices(), ("d",))                       # the quickstart's f, xs, keys, theta
items = NamedSharding(mesh, P("d"))
res = jax.jit(f)(jax.device_put(xs, items), jax.device_put(keys, items), theta)
# res.outputs["hist"]: summed over the devices; res.per_item: sharded like xs
```

Derivatives and spawn work the same way. As with any JAX sharding, the item count must divide evenly among the
devices. A spawn `Queue` holds its capacity per device. On the CPU the devices of
one process share its CPUs, so the threads are divided among them.

On the CPU every operation is IEEE (subnormals kept, no FMA contraction) and the math library is libm, so
transcendentals differ from XLA:CPU by a few ulps (libm is the more accurate of the two:
`docs/maintenance.md`). `cuda_math` has no effect there.

CPU, one AMD EPYC 7713 socket (32 cores, exclusive node, every arm pinned to the same cores), JAX 0.6.2. "vs best"
is the fastest plain-JAX arm, often `pmap` over 32 host devices, which needs
`XLA_FLAGS=--xla_force_host_platform_device_count=32` and code that shards the items; "vs jit" is plain `jax.jit` on
one device, what a JAX program gets without that setup:

| problem | N | forward ms | vs best | vs jit | gradient ms | vs best (= jit) |
|---|---|---|---|---|---|---|
| random_walk | 65,536 | 3.8 | 5.1x | 82x | 7.4 | 44x |
| transport | 524,288 | 26.8 | 4.0x | 123x | 85.8 | 48x |
| walk_on_spheres | 65,536 | 4.3 | 4.6x | 129x | 9.9 | 54x |
| ode_ensemble | 65,536 | 36.1 | 1.0x | 32x | 202 | 18x |
| barrier | 65,536 | 30.1 | 1.1x | 27x | 111 | 8.6x |

Forward passes of long, fairly uniform loops (ode_ensemble, barrier) are at parity with a sharded plain-JAX program;
the gain there is the gradient. All rows and the method: `docs/performance.md`.

## Options

| option | default | meaning |
|---|---|---|
| `backend` | `'auto'` | `'auto'`: the kernel backend on the platform JAX places the computation on (CUDA GPU or CPU); a program it cannot compile raises `tm.UnsupportedError` (pass `backend='xla'` to run it; it never falls back on its own, because the xla backend is far slower); other platforms run the xla backend with a one-time warning. `'xla'`: always the plain-JAX interpreter, slow by design. (`'cuda'` was removed in 0.7: use `'auto'`.) |
| `cuda_math` | `'approx'` | GPU only (the CPU is always IEEE). f32 division, sqrt and rsqrt. `'approx'`: the PTX XLA:GPU itself emits, so it is the arithmetic a plain JAX program on the same GPU gets. `'ieee'`: correctly rounded and portable, about 1.5x slower. |
| `schedule` | `'item'` | kernel backend: `'item'`, a thread runs one whole item at a time. `'step'`, the step schedule: a thread is refilled with a new item at loop-step granularity, so it does not idle while the longest item of its warp finishes. `'step'` needs an item with exactly one top-level `while_loop` and raises `tm.UnsupportedError` (saying why) otherwise. Neither wins everywhere (3.0x either way on real workloads), so measure yours; `tm.diagnostics()['kernels']` names the schedule each kernel ran. |
| `xla_batch_size` | `2**18` | xla backend: items per batch, which bounds its memory. |
| `rev_checkpoints` | `'auto'` | reverse mode: checkpoint buffer size C. `'auto'`: 16 for a loop step of fewer than 1000 equations, else the largest of 4 ... 64 whose 3C+1 states (single-loop adjoint: 2C+1) fit 64 KiB. A loop of K <= C steps is not replayed; O(K) work for K <= C^2. |
| `rev_stats` | `False` | reverse through spawn: report the tree adjoint's counts to the host. Off by default because a host callback takes `jax.jit` off its C++ fast path, which can cost an order of magnitude. |

Any other option name raises, and there are no private options: the measurement switches and negative controls
are test-only module constants. tallymap's one environment variable is `TALLYMAP_CACHE`, the kernel cache
directory (default `~/.cache/tallymap`). On the CPU the standard OpenMP and compiler variables apply:
`OMP_NUM_THREADS` (default: the CPUs the process may use), `OMP_STACKSIZE` (the worker threads' stack, where per-item
arrays live; tallymap sets it to 16M if it is unset, and a kernel needing more than half of it raises before it runs)
and `CXX`.

## Nested loops: solvers inside an item

An item may contain loops inside loops, each with its own data-dependent trip count: a `while_loop` inside a
`while_loop` inside a `cond`. The kernel backend compiles every level into an ordinary loop of the item's own
thread. Every transformation handles the nesting: `jvp`, `jacfwd` and `grad`. Reverse mode checkpoints each
loop level recursively, so memory stays O(1) per item.

This matters most for numerical methods that are themselves iterative inside each step. An implicit (stiff) ODE
solver has three nested loops: adaptive time steps, the Runge-Kutta stages of each step, and a Newton or chord
solve inside each stage. Under `jax.vmap` every one of those loops runs to the batch maximum, and each finished
item executes masked iterations. In a tallymap item each loop runs exactly as often as that item needs.

Measured on one L40S. The workload is an ensemble of stiff 3-state ODEs integrated with a Kvaerno3 / chord /
PID-controller algorithm, written as a tallymap item, against `jax.vmap(diffrax.diffeqsolve(...))` with the same
solver settings:

| | forward | `grad` |
|---|---|---|
| tallymap item | 99 to 280x faster (step-count matched) | 19 to 125x faster, 2 to 67x less memory |
| at 1M trajectories | | 2.8 s / 72 MB against 53 s / 4.8 GB |

Two costs to know:

- Replaying three nested loops is expensive. On that solver tallymap's `grad` costs about 84x its own forward
  pass, where diffrax's checkpointed adjoint costs about 12x. The gradient margin is therefore one order of
  magnitude where the forward margin is two.
- `rev_checkpoints` trades recomputation against per-item state, and it is not monotone. On that solver,
  raising it from 16 to 32 made `grad` 3x slower. Measure it per item rather than deriving it from
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
| tallymap, xla backend (the kernel backend runs a real branch) | 0.0 MiB | 0.3 MiB |

A ray tracer that tests a BVH leaf's triangles in an inner loop hits exactly this. With 32,768 triangles,
`jax.vmap` asked for 104 GiB at 65,536 rays, and 6.5 TiB at 4.2M.

**In plain JAX:** do not wrap a loop in a per-item `cond`. Fold the predicate into the loop's trip count
(`n = jnp.where(p, n, 0)`), or unroll the inner loop if its maximum is small and static. **In tallymap** none of
this is needed: a `cond` is a real branch of the item's thread.

## Support boundary

What is explicitly out of scope, so that you do not have to find out by trying:

- **No TPU and no ROCm.** Those platforms run the xla backend, which is correct and far slower.
- **The CPU kernel needs Linux and a C++17 compiler (GCC or Clang).** Without one, or on macOS and Windows,
  `backend='auto'` on the CPU raises `tm.UnsupportedError` naming the fix; `backend='xla'` runs there.
- **CPU results are not bitwise equal to XLA:CPU** on transcendentals (libm vs XLA's own approximations, 1-7 ulps)
  or on subnormals (XLA:CPU flushes them to zero; the kernel is IEEE). On a GPU the bitwise checks against XLA stand.
- **`shard_map` around a call is unsupported**: shard the items with `NamedSharding` and `jax.jit` instead
  (see "Backends"). `jax.pmap` works.
- **`jax.export` of a kernel program is unsupported.** Its custom call is not on JAX's export allow-list and its
  kernels are registered per process, so a deserialised program is not portable. Programs on the xla backend export
  normally.
- **No second-order derivatives through spawn.** The tree adjoint raises for them.
- **A100 and H200 are untested.** Nothing suggests they will not work; no tuning constant was measured on them.
- **Coupling between items inside the loop raises**, and so do reading, forking or dropping a tally, `n_active`
  together with spawn, effects inside the item (`jax.debug.print`, callbacks), `grad` through a `lax.while_loop`
  AROUND a call (that is JAX's own limitation, not tallymap's), 2^31 items or more in one call, and unknown
  option names.
- **An `UnsupportedError` on a primitive the kernel emitter does not know is a normal bug report**, not a dead end.
  Add `backend='xla'` and your program will run; attach `tm.diagnostics()` to the issue. Adding a primitive
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
| `examples/gradient_pitfall.py` | the gradient pitfalls in "Derivatives", runnable |

Why each part exists: `docs/design.md` (the full record of measurements: `docs/design_notes.md`). Validation and the A/B rules:
`docs/validation.md`. Float semantics: `docs/float_semantics.md`. Terms: `docs/glossary.md`. Supported JAX
versions, the release process and a "when X breaks, do Y" table: `docs/maintenance.md`. Covered primitives:
`docs/primitives.md`. Speed against plain JAX: `docs/performance.md` (every table and plot: `docs/performance_detail.md`).

If you use tallymap in published work, please cite it: see `CITATION.cff`.

Licensed under the Apache License 2.0; see `LICENSE` and `NOTICE`.