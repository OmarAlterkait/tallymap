# Design

How tallymap is built and why each part exists, one paragraph and one measurement per part. The full record,
with every measurement and the alternatives that were built, measured and removed, is `docs/design_notes.md`
(section numbers below point there). Terms: `docs/glossary.md`.

A part stays only if removing it costs at least 1.5x somewhere or it is needed for correctness. Timings are an
NVIDIA L40S or RTX 2080 Ti and JAX 0.5.3 unless stated otherwise.

## The shape

```
tm.vmap(item)                     _api.py        user contract: Tally, Sum, Queue, Result, max_steps, n_active
  -> ItemProgram (item jaxpr)     _rewrite.py    tally checks, max_steps counters, re-tracing
  -> tallymap_call(node)          _primitive.py  ONE primitive: jvp, transpose, batching, lowering, sharding rules
       derived item programs      _api.jvp_prog, _reverse.py, _reverse_spawn.py, dual lanes
  -> lowering, per platform
       'cuda' | 'cpu'             backends/kernel   item jaxpr -> C (codegen, op_table) -> CUDA (NVRTC) | host (C++)
       anything else / 'xla'      backends/xla.py   masked plain-JAX interpreter: the reference
```

## The user contract (notes section 1)

- **The Tally is threaded, not an effect.** `t = t.add(name, idx, val)` costs one loop carry and gives a data
  dependency nothing can eliminate, plus a dense meaning: `jax.vmap(item)` followed by a sum IS the definition
  (`tm.reference`), so JAX's own rules define every transformation's answer. Write-only, linear use is checked on
  the jaxpr (`TallyError`).
- **No init / step / done.** The emitter finds the item's loops itself. Two schedules: `'item'` (default: a thread
  runs a whole item) and `'step'` (a thread is refilled at loop-step granularity). Neither wins everywhere: the
  step schedule is up to 1.7x faster on divergent items on a 2080 Ti and 3.0x slower on LUCiD's forward, so it is
  an option and the default never splits the item.
- **Resource limits poison, never truncate silently.** An item that hits `max_steps` or a child the `Queue`
  drops makes every float tally NaN, with the count in `Result.stats`. Out-of-range emits are dropped, as JAX's
  scatter drops them, and counted.

## One primitive (notes section 2)

Every transformation builds a new item program with `jax.jvp` / `jax.vjp` / `jax.vmap` applied to ONE item and
binds the primitive again, so rules recurse to any order (`hessian`, grad-of-jvp, ...). The earlier
`custom_jvp` / `custom_vjp` integration could not compose at all.

- **Fused jvp** (a second primitive, `tallymap_jvp`): primal and tangent in one kernel, 0.68 ms against 1.22 ms
  in two launches. Split back wherever JAX partially evaluates it; off by feature detection if JAX drops the two
  legacy registries it uses, and the suite passes either way (`--unfused`).
- **Dual lanes**: `jacfwd`, `jacrev` and vmap-of-jvp run T tangent lanes in one kernel, in chunks of at most 16:
  `jacrev` of 256 outputs took 176 ms in chunks of 16 against 4.3 s with all lanes at once; a guard splits lanes
  with large loop state (`jax.hessian` 1629 -> 13.5 ms). `jax.vmap` of a call with a batched non-linear operand
  is `lax.map`.
- **Sharding** (notes section 12): a `custom_partitioning` rule, so a sharded `jax.jit` runs each device on its
  own items and sums tallies with `psum`; without it XLA replicated the whole call (4 L40S 0.93 ms against 0.22 ms).

## Reverse mode (notes sections 3, 4, 6)

- **Checkpointed replay.** Each while loop stores at most 3C+1 states per item (C = `rev_checkpoints`, chosen
  from the step size) and replays segments; an item of K <= C steps is not replayed. Replaying from the start
  instead was 13x slower at K = 100 and over 100x at K = 1000. Stride C rather than ceil(K/C): transport's
  gradient 15.6 -> 7.4 ms.
- **Sparse table cotangents.** A large table read only by gathers gets one atomic scatter-add accumulator instead
  of a dense per-item cotangent, which otherwise runs out of memory on a GPU.
- **Uninitialised checkpoint buffers.** Zero-filling them made cheap-step gradients up to 5x slower; a test fills
  them with 0x00 and 0xFF and requires bitwise-equal results.
- **The tree adjoint** for reverse through spawn: a spawn is an emit of the child's initial state, so the
  ordinary replay adjoint carries the child's cotangent to the parent; one adjoint launch per generation, a flat
  2.4-4.1x the forward. It beats `jacfwd` from about 4 parameters when lanes are refused, and loses to lanes below
  about 64 parameters otherwise. Any overflow or replay mismatch makes the gradient NaN.

## The kernel backend (notes sections 1d, 7, 9-11)

- **One emitter, two targets.** The C is IEEE without FMA contraction on both; the CUDA target uses
  `cuda_math='approx'` by default (XLA:GPU's own f32 div / sqrt PTX: `ieee` costs 1.5x forward, 1.65x jvp), the
  CPU target is always IEEE with libm. A program the emitter cannot compile raises `UnsupportedError` rather than
  falling back to the xla backend, which is about 100x slower.
- **Run-or-raise on resources.** A CUDA kernel over the 512 KiB per-thread local limit, or a CPU kernel whose
  stack bound is over half of `OMP_STACKSIZE`, is refused before it runs.
- **Code size is compile time.** NVRTC is superlinear in function size; the levers kept are a cap on lazily
  inlined expressions (1543 -> 834 s on a 4 MB kernel), one non-inlined wrapper per f32 math function (100-stage
  benchmark 195 -> 5.3 s), cutting long 64-bit multiply chains (60 -> 1.7 s), a structural digest instead of the
  printed jaxpr as cache key (95 -> 4.2 s per build), and in-place updates and buffer assignment at loop / cond
  boundaries (smaller frames, up to 1.59x).
- **CPU target** (notes section 11): an OpenMP team kept between launches (3.1 ms against 6.4 with a thread per
  launch), per-thread IEEE environment, per-item arrays on worker stacks.

## The xla backend

A masked plain-JAX interpreter of the item jaxpr over batches of items (`xla_batch_size`). It is the reference
the kernel is tested against and the fallback on platforms with no kernel target (TPU, ROCm, Metal); it runs every
JAX primitive without side effects and is slow by design, at about plain-JAX speed or below.

## Fewer knobs (notes section 5)

Six public options; unknown names raise; measurement switches and negative controls are test-only module
constants. Removed with their measurements (notes section 5-6): 20 environment switches, event scheduling, the
batch fold, step segmentation, `cuda_min_blocks`, block-size and shared-memory options, constant folding, the
replay fingerprint, and the forward-stores-checkpoints and one-kernel-per-phase adjoints.
