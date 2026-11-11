## CPU (the kernel backend's CPU target)

One AMD EPYC 7713 socket (milano): an exclusive node, every process pinned to the same 32 cores of socket 0, OpenMP
threads placed close on cores; JAX 0.6.2, `div=mid`, tallymap 8c3ed7c; job 44213951 (medians; the minimum is within
2% of the median in every row, and the CPU kernel keeps 30-32 cores busy). host = `backend='auto'` on the CPU with 32
threads; best plain = the fastest of `jax_while` / `jax_scan` on one device and the same sharded over 32 forced host
devices with `pmap` (which needs `XLA_FLAGS=--xla_force_host_platform_device_count=32` and code that shards the items:
a plain `jax.jit` program gets the "vs jit" column).

| problem | N | fwd host / best plain (ms) | x | vs jit | grad host / best plain (ms) | x |
|---|---|---|---|---|---|---|
| random_walk | 65,536 | 3.8 / 19.4 (pmap) | 5.1 | 82 | 7.4 / 329 (1 dev) | 44 |
| random_walk | 524,288 | 21.5 / 59.6 (pmap) | 2.8 | 77 | 41.5 / 1,958 (1 dev) | 47 |
| transport | 65,536 | 4.6 / 22.1 (pmap) | 4.8 | 117 | 13.1 / 574 (1 dev) | 44 |
| transport | 524,288 | 26.8 / 106 (pmap) | 4.0 | 123 | 85.8 / 4,079 (1 dev) | 48 |
| walk_on_spheres | 65,536 | 4.3 / 19.9 (pmap) | 4.6 | 129 | 9.9 / 539 (1 dev) | 54 |
| ode_ensemble | 65,536 | 36.1 / 37.5 (pmap) | 1.0 | 32 | 202 / 3,603 (1 dev) | 17.8 |
| barrier | 65,536 | 30.1 / 32.4 (pmap) | 1.1 | 27 | 111 / 959 (1 dev) | 8.6 |

The gradient's best plain arm is one device in every row: a padded scan's reverse pass is memory-bound and `pmap` does
not speed it up. The forward of long, fairly uniform loops (ode_ensemble, barrier) is at parity with the sharded plain
program; the gain there is the gradient. On shared nodes the same cells varied up to 4x from node to node (threads
sleeping and waking around other jobs); `OMP_WAIT_POLICY=active` changed nothing on the exclusive node (random_walk
forward 3.85 vs 3.83 ms), so the default stays.

Why the CPU target is built as it is (OpenMP team, IEEE environment, `erf_inv` in double): `docs/design_notes.md`
section 11.
