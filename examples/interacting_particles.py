"""Interacting particles: the items interact at every step, and the interaction happens BETWEEN the calls.

    python examples/interacting_particles.py

A Lennard-Jones fluid in 2-D with periodic boundaries.  tallymap forbids one item from reading another item's
state inside a call, which looks like it rules this out.  It does not, because that is not the structure of the
problem:

    lax.scan over time steps                              (plain JAX)
      |-- build the cell list from the current positions  (plain JAX: argsort + searchsorted)
      |-- ONE tm.vmap call.  ITEM = one particle.  It reads the SHARED state of this step -- every position and
      |   the cell list, as broadcast arguments -- walks the members of its 3x3 neighbouring cells, and returns
      |   its own force.  TALLY = Sum((NSLAB,)), the potential energy per x slab.
      |-- velocity-Verlet update                          (plain JAX)

The walk length is the particle's own neighbour count, which is where the divergence comes from: in a clustered
configuration it varies by an order of magnitude between particles.

Measured: the comparison against the JAX-MD-style padded neighbour list, which has to size every particle's
candidate list from the FULLEST cell, is in docs/performance_detail.md.  The answer is not uniform: tallymap wins when
the neighbour counts are spread out and loses when they are not.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm

NSLAB, RC, DT, NSTEP = 16, 1.0, 1.0e-3, 5
RMIN = 0.5                   # soft core: see lj() below


def lj(xi, xj, theta, side):
    """Force on i from j and the pair energy: Lennard-Jones, truncated at RC, minimum image.

    The pair separation is floored at RMIN, which makes the core SOFT.  Without it a randomly placed start has
    overlapping pairs, r^-12 overflows f32 and the whole configuration is NaN after one step.  Real MD starts
    from a lattice instead; a soft core is the shorter way to a program whose cost is the thing being measured.
    """
    d = xi - xj
    d = d - side * jnp.round(d / side)
    r2 = jnp.sum(d * d)
    s6 = (theta[1] * theta[1] / jnp.maximum(r2, RMIN * RMIN)) ** 3
    live = (r2 < RC * RC) & (r2 > 1e-10)
    f = jnp.where(live, 24.0 * theta[0] * (2.0 * s6 * s6 - s6) / jnp.maximum(r2, RMIN * RMIN), 0.0)
    return f * d, jnp.where(live, 4.0 * theta[0] * (s6 * s6 - s6), 0.0)


def cell_list(pos, nc, side):
    """Plain JAX, rebuilt every step: the CSR form of a cell list.

    order[start[c] : start[c+1]] are the particles of cell c."""
    h = side / nc
    cid = (jnp.clip((pos[:, 1] / h).astype(jnp.int32), 0, nc - 1) * nc
           + jnp.clip((pos[:, 0] / h).astype(jnp.int32), 0, nc - 1))
    order = jnp.argsort(cid)
    return order, jnp.searchsorted(cid[order], jnp.arange(nc * nc + 1, dtype=cid.dtype))


def particle(t, xi, order, start, pos, theta, side, nc):
    """ONE item: walk the members of the 3x3 cells around xi, tally the energy, return the force.

    The carry is (which of the 9 cells, the slot in `order`, force, energy, trips); one trip is one neighbour."""
    h = side / nc

    def body(c):
        g, p, fs, u, k = c
        ix = jnp.clip((xi[0] / h).astype(jnp.int32), 0, nc - 1)
        iy = jnp.clip((xi[1] / h).astype(jnp.int32), 0, nc - 1)
        cid = ((iy + g // 3 - 1) % nc) * nc + ((ix + g % 3 - 1) % nc)
        lo, hi = start[cid], start[cid + 1]
        p = jnp.maximum(p, lo)
        inside = p < hi
        fj, uj = lj(xi, pos[order[jnp.clip(p, 0, pos.shape[0] - 1)]], theta, side)
        nxt = p + 1
        done = ~(nxt < hi)
        return (g + done.astype(jnp.int32), jnp.where(done, 0, nxt),
                fs + jnp.where(inside, 1.0, 0.0) * fj, u + jnp.where(inside, uj, 0.0), k + 1)
    g, p, fs, u, k = lax.while_loop(lambda c: c[0] < 9, body,
                                    (jnp.int32(0), jnp.int32(0), jnp.zeros((2,), jnp.float32),
                                     jnp.float32(0.0), jnp.int32(0)))
    slab = jnp.clip((xi[0] / side * NSLAB).astype(jnp.int32), 0, NSLAB - 1)
    return t.add("u", slab, 0.5 * u), (fs, k)


N, DENSITY, NBLOB = 2048, 2.0, 8
SIDE = float(np.sqrt(N / DENSITY))                               # the mean density is 2 particles per unit area
NC = max(3, int(SIDE / RC))
rng = np.random.default_rng(0)
# A clustered start: 8 dense blobs holding 80% of the particles at 4x the mean density, the rest spread out.
# That is where the divergence comes from: a particle in a blob has many more neighbours than one outside.
blob_r = float(np.sqrt(0.8 * N / 8.0 / (NBLOB * np.pi)))
ctr = rng.uniform(0, SIDE, (NBLOB, 2))
pos0 = np.mod(ctr[rng.integers(0, NBLOB, N)] + rng.normal(0, blob_r / 2.0, (N, 2)), SIDE)
bg = rng.random(N) >= 0.8                                        # the sparse background
pos0[bg] = rng.uniform(0, SIDE, (int(bg.sum()), 2))
pos0 = jnp.asarray(pos0, jnp.float32)
vel0 = jnp.asarray(rng.normal(0, 0.2, (N, 2)), jnp.float32)
theta = jnp.array([1.0, 0.4], jnp.float32)

f = tm.vmap(lambda t, xi, o, s, p, th: particle(t, xi, o, s, p, th, SIDE, NC),
            outputs={"u": tm.Sum((NSLAB,))}, in_axes=(0, None, None, None, None), max_steps=1 << 14)


def one_step(pos, theta):
    """The forces and the energy tally at these positions: cell list in plain JAX, then ONE tm.vmap call."""
    order, start = cell_list(pos, NC, SIDE)
    res = f(pos, order, start, pos, theta)
    (force, trips) = res.per_item
    return force, res.outputs["u"], trips


force, energy, trips = jax.jit(one_step)(pos0, theta)
print("neighbour-walk trips per particle: mean", round(float(trips.mean()), 1), " max", int(trips.max()),
      "  (9 of them are the cell boundaries)")
print("potential energy per slab[:4]", np.round(np.asarray(energy[:4]), 2))

# Check the cell-list walk against the direct O(N^2) sum on a smaller set: they must be the same forces.
n2 = 384
sub = pos0[:n2]
f_sub = tm.vmap(lambda t, xi, o, s, p, th: particle(t, xi, o, s, p, th, SIDE, NC),
                outputs={"u": tm.Sum((NSLAB,))}, in_axes=(0, None, None, None, None), max_steps=1 << 14)


def direct(pos, theta):
    fs, us = jax.vmap(lambda xi: jax.vmap(lambda xj: lj(xi, xj, theta, SIDE))(pos))(pos)
    return jnp.sum(fs, 1), jnp.sum(us, 1)


def cells(pos, theta):
    order, start = cell_list(pos, NC, SIDE)
    r = f_sub(pos, order, start, pos, theta)
    return r.per_item[0], r.outputs["u"]


fd, ud = jax.jit(direct)(sub, theta)
fc, uc = jax.jit(cells)(sub, theta)
e_force = float(jnp.max(jnp.abs(fc - fd)) / jnp.max(jnp.abs(fd)))
print(f"cell-list forces vs the direct O(N^2) sum at N={n2}: rel {e_force:.1e}")

# The gradient of an observable after NSTEP velocity-Verlet steps, with respect to (epsilon, sigma).  The outer
# scan is plain JAX with jax.checkpoint on its body; reverse mode goes through every step.
wslab = jnp.asarray(rng.uniform(0.5, 1.5, NSLAB), jnp.float32)


def observable(theta):
    def body(c, _):
        p, v = c
        fs, u, _k = one_step(p, theta)
        v = v + DT * fs
        return (jnp.mod(p + DT * v, SIDE), v), u
    (p, v), us = lax.scan(jax.checkpoint(body), (pos0, vel0), None, length=NSTEP)
    return jnp.sum(wslab * us[-1])


g = jax.jit(jax.grad(observable))(theta)
J = jax.jit(jax.jacfwd(observable))(theta)
e_grad = float(jnp.max(jnp.abs(g - J)) / jnp.max(jnp.abs(J)))
print("d observable / d (epsilon, sigma) after", NSTEP, "steps: grad", np.round(np.asarray(g), 4))
print(f"grad vs jacfwd: rel {e_grad:.1e}")

bad = [n for n, e in (("forces", e_force), ("grad-vs-jacfwd", e_grad)) if not (e < 1e-3)]
print("INTERACTING_PARTICLES " + ("OK" if not bad else "FAIL " + ",".join(bad)))
assert not bad, bad
