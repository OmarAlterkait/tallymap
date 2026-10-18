# Glossary

Terms used in the code, the docs and the error messages.

## User-facing

| term | meaning |
|---|---|
| **item** | one mapped element: one walker, path, chain, particle.  The user writes the function of ONE item. |
| **Tally** | the item's write-only, commutative outputs.  Threaded linearly through the item like a pseudo-random-number-generator key: `t = t.add(name, index, value)`.  Outside tallymap it is a dense array; inside, the backends write straight into one shared accumulator. |
| **tally output** / `Sum` | one named output: the sum over items of everything emitted into it.  `Sum(exact=True)` sums in int64 fixed point, which is bitwise order-independent. |
| **emit** | one `t.add(...)`: `tally[index] += value`.  An out-of-range or negative index is dropped, exactly as JAX's own scatter drops an out-of-bounds update, and counted in `stats['dropped_emits']`. |
| **spawn** / `Queue` | `t.spawn(child, when)` appends a new item with the same input structure; `Queue(capacity)` bounds the PENDING children.  A child that does not fit is dropped, counted in `stats['dropped']`, and makes the call's float tallies NaN. |
| **truncation** | an item that hits `max_steps`.  Counted in `stats['n_truncated']`, flagged per item in `Result.truncated`, and it makes the call's float tallies NaN, exactly like a dropped spawn child. |
| **resource limit** | the two limits that can make a tallymap result differ from `jax.vmap` of the same item: `max_steps` and the spawn `Queue` capacity.  Both report themselves.  With `max_steps=None`, the default, and no spawn, there is no limit and no deviation. |
| **per-item result** | the optional second return value of the item: `Result.per_item[N, ...]`. |
| **dense reference** | `tm.reference`: `jax.vmap(item)` with real per-item tally arrays, summed.  The definition every backend and every transformation must reproduce. |
| **compare** | `tm.compare`: the frozen statistical comparison of two implementations of one item program (`docs/validation.md`). |
| **question A / question B** | A (mechanics): is the derivative the exact derivative of the program as written?  tallymap's job.  B (estimator): is it an unbiased estimate of d E[loss] / d theta?  The program's job. |

## Backends

| term | meaning |
|---|---|
| **cuda** | the item jaxpr compiled to one CUDA kernel with NVRTC, NVIDIA's runtime compiler, launched through one `jax.ffi` handler (`tallymap_cuda_launch`). |
| **lockstep** | the plain-JAX masked lockstep interpreter: the reference implementation and the fallback, on every platform.  Slow by design, about 100 times the cuda backend. |
| **step refill** | cuda scheduling: a lane takes a new item when its current one finishes, at loop-STEP granularity.  Needs exactly one top-level while loop; not used with spawn. |
| **privatised tally** | a small sum output kept in shared memory per block and flushed once, instead of going to global atomics. |
| **op table** | `backends/cuda/op_table.py`: one emission rule per JAX primitive.  The main maintenance surface. |
| **cuda_math** | `approx`, the default, uses the f32 division, sqrt and rsqrt PTX that XLA:GPU itself emits; `ieee` is correctly rounded.  `docs/float_semantics.md`. |

## Transformations (internal)

| term | meaning |
|---|---|
| **ItemProgram** | the item jaxpr plus its operand and output layout: tally specs, and which operands are per-item and which are broadcast. |
| **node** | an ItemProgram plus how it was derived; the parameter of the primitive. |
| node kind **primal** | the user's program. |
| node kind **tangent** | the linear tangent map: the tangent part of `jax.jvp(item)`, taking the primal operands as inputs. |
| node kind **adjoint** | the replay vector-Jacobian product of the item (`vjp_prog`). |
| node kind **dual_lanes** | T tangent or cotangent lanes of a linear node in ONE kernel (`jacfwd`, `jacrev`, vmap of a jvp). |
| node kind **folded_batch** | a `jax.vmap` batch of B members folded into the item axis: B*N items in one launch. |
| node kind **fused_jvp** | primal and tangent in one kernel (`jax.jvp` of a call); split back into primal plus tangent wherever JAX partially evaluates it.  A separate primitive, `tallymap_jvp`. |
| **activity** | which loop carries can depend on perturbed inputs.  Only active float carries are differentiated in a step VJP. |

## Reverse mode (internal)

| term | meaning |
|---|---|
| **checkpointed loop** | the reverse-differentiable replacement of every `while_loop` in an adjoint: the forward stores nothing per step; the VJP replays from the initial carry with two-level checkpointing. |
| **checkpoint buffer** | C states per item (`rev_checkpoints`, default 16): O(K) work for loops of K <= C^2 steps, 2C states of memory. The checkpoint stride is C, so every buffer the sweep allocates is full. |
| **step segments** | `rev_step_segments=S`: recompute each loop step in S checkpointed segments inside its VJP, so fewer residuals spill. |
| **sparse table cotangent** | the cotangent of a large read-only table (32 elements or more, read only by gathers): an atomic scatter of the read cotangents into ONE accumulator through a threaded cotangent token, never a dense per-item array. |
| **replay fingerprint** | a per-item hash of the final carries of the top-level loops, computed by the forward kernel and again by the adjoint's replay.  A mismatch makes the call's cotangents NaN. |
| **tree adjoint** | reverse mode through spawn: the run is recorded generation by generation, then one adjoint launch per generation, deepest first. |
| **drop risk** | the recorded spawn tree may exceed the Queue capacity, which means the forward may have dropped children: the tree adjoint's gradient is NaN. |
| **rev_stats** | opt-in host reporting of replay mismatches and tree-adjoint counts (`tm.diagnostics()['reverse']`).  It costs `jax.jit`'s C++ fast path, so it is off by default. |
