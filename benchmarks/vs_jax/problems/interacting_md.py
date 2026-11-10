"""INTERACTING SYSTEM 1: short-range interacting particles (a Lennard-Jones fluid in 2-D).

This is the problem that answers "does tallymap help when the items interact at every step?".  The interaction
does not happen INSIDE one tm.vmap call; it happens BETWEEN calls:

    lax.scan over time steps                                 (plain JAX)
      |-- build the cell list from the current positions     (plain JAX: argsort + searchsorted)
      |-- ONE tm.vmap call: item = one particle.  It reads the SHARED state of this step (every position, the
      |   cell list) as closed-over broadcast arrays, walks the members of its 9 neighbouring cells -- a
      |   VARIABLE count -- and writes its own force (a per-item output) and its potential energy into a
      |   Sum((32,)) deposit grid indexed by its x slab.
      |-- velocity-Verlet update                             (plain JAX)

ITEM      one particle.  Its loop runs over the members of its 3x3 neighbouring cells, so the trip count is the
          particle's own neighbour count plus 9 (one trip per cell boundary).
TALLY     Sum((32,)): potential energy per x slab.  Per-item output: the 2-component force.
DIVERGE   the spatial distribution.  `uniform` gives every particle about the same neighbour count; `clustered`
          puts most particles in a few dense blobs, so neighbour counts vary by an order of magnitude.
THETA     (epsilon, sigma): the two Lennard-Jones parameters.

Arms, beyond the usual ones: `jax_pad` is the JAX-MD-style padded neighbour list.  Its cell list has a fixed
capacity per cell (the observed maximum occupancy), each particle's candidate list is the 9 cells' slots
concatenated, and `jax.vmap` evaluates all 9 * capacity candidates with a distance mask.  That is what a JAX
user writes today, and its cost is set by the FULLEST cell, not by the average one.  `jax_pad2x` doubles the
capacity, which is what a program that must not drop a neighbour has to do.

Two notes on the padding of this problem.  The capacity and the walk's padded trip count are both taken from the
INITIAL configuration, and in the gradient mode the particles then move; with DT = 2e-3 over 10 steps they move
about 0.004 of a cell, so the occupancy does not change, and the cross-arm agreement cell confirms it.  The
`jax_pad` capacity is therefore still an oracle: a real program does not know the fullest cell in advance.

Modes: `fwd` is ONE step (cell list, forces, tally).  `grad` is the gradient of a scalar observable after
KSTEPS velocity-Verlet steps with respect to (epsilon, sigma), taken in reverse mode through the outer scan with
`jax.checkpoint` on the step body.
"""
import arms as A
import harness as H
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import tallymap as tm

NAME = "interacting_md"
TITLE = "Short-range interacting particles (Lennard-Jones, 2-D)"
GROUP = "interacting"
AXIS = "n"
SWEEP = {"full": (1 << 12, 1 << 14, 1 << 16, 1 << 18), "quick": (1 << 9, 1 << 10)}
DIVS = ("clustered", "uniform")
ARMS = ("tm_cuda", "tm_xla", "jax_while", "jax_scan", "jax_scan2x", "jax_ckpt", "jax_ckpt2x",
        "jax_pad", "jax_pad2x")
MODES = ("fwd", "grad")
CHECK_VAL = 1 << 9
DIV_VAL = 1 << 16      # the other divergence levels are measured at this one size

NG = 32                    # deposit-grid slabs
RC = 1.0                   # cutoff, and the cell size
RMIN = 0.5                 # soft core: see pair()
DENSITY = 2.0              # mean particles per unit area, the same for both distributions
BLOB_DENSITY = 8.0         # inside a blob of the `clustered` distribution
BLOB_FRACTION = 0.8        # fraction of the particles that sit in blobs
NBLOB = 16
DT = 1.0e-3
KSTEPS = 10                # velocity-Verlet steps for the gradient mode
TH0 = (1.0, 0.4)
NBR = 9                    # 3x3 neighbouring cells in 2-D


def box_of(n):
    """(side length, cells per side).  The box is sized so the MEAN density is DENSITY whatever n is, which is
    what keeps the per-item work independent of n: otherwise the neighbour count grows with n and the scaling
    panel measures two things at once."""
    side = float(np.sqrt(n / DENSITY))
    nc = max(3, int(np.floor(side / RC)))
    return side, nc


def cell_list(pos, nc, side):
    """Plain JAX, rebuilt every step: sort the particles by cell and record where each cell starts.

    -> (order, start).  `order` lists the particle indices sorted by cell id; `start[c]` is the first slot of
    cell c in `order`, so cell c owns order[start[c] : start[c+1]].  This is the CSR form of a cell list."""
    h = side / nc
    cid = cell_id(pos, nc, h)
    order = jnp.argsort(cid)
    start = jnp.searchsorted(cid[order], jnp.arange(nc * nc + 1, dtype=cid.dtype))
    return order, start


def cell_id(pos, nc, h):
    ix = jnp.clip((pos[..., 0] / h).astype(jnp.int32), 0, nc - 1)
    iy = jnp.clip((pos[..., 1] / h).astype(jnp.int32), 0, nc - 1)
    return iy * nc + ix


def nbr_cell(xi, g, nc, h):
    """The flat id of the g-th of the 3x3 cells around the cell containing xi (periodic)."""
    ix = jnp.clip((xi[0] / h).astype(jnp.int32), 0, nc - 1)
    iy = jnp.clip((xi[1] / h).astype(jnp.int32), 0, nc - 1)
    return ((iy + g // 3 - 1) % nc) * nc + ((ix + g % 3 - 1) % nc)


def pair(xi, xj, th, side):
    """Lennard-Jones force on i from j and the pair energy, truncated at RC, minimum image.  Shared by every
    arm.  -> (force[2], energy).

    The separation is floored at RMIN, which makes the core SOFT.  A randomly placed start has overlapping
    pairs, and without the floor r^-12 overflows f32 and the configuration is NaN after one step.  The
    arithmetic per pair is the same either way, which is what this benchmark measures."""
    d = xi - xj
    d = d - side * jnp.round(d / side)
    r2 = jnp.sum(d * d)
    r2c = jnp.maximum(r2, RMIN * RMIN)
    inv = (th[1] * th[1]) / r2c
    s6 = inv * inv * inv
    f = 24.0 * th[0] * (2.0 * s6 * s6 - s6) / r2c
    u = 4.0 * th[0] * (s6 * s6 - s6)
    live = (r2 < RC * RC) & (r2 > 1e-10)
    return jnp.where(live, f, 0.0) * d, jnp.where(live, u, 0.0)


def walk_step(c, xi, order, start, pos, th, nc, h, side):
    """ONE trip of the neighbour walk of ONE particle.  Shared by the tallymap, while and scan arms.

    carry = (g, p, force, energy, k): g is which of the 9 cells, p the running slot in `order`."""
    g, p, fs, u, k = c
    cid = nbr_cell(xi, g, nc, h)
    lo, hi = start[cid], start[cid + 1]
    p = jnp.maximum(p, lo)
    inside = p < hi
    j = order[jnp.clip(p, 0, order.shape[0] - 1)]
    fj, uj = pair(xi, pos[j], th, side)
    fs = fs + jnp.where(inside, 1.0, 0.0) * fj
    u = u + jnp.where(inside, uj, 0.0)
    nxt = p + 1
    done = ~(nxt < hi)
    return (g + done.astype(jnp.int32), jnp.where(done, 0, nxt), fs, u, k + 1)


def walk_cond(c):
    return c[0] < NBR


def walk_c0():
    return (jnp.int32(0), jnp.int32(0), jnp.zeros((2,), jnp.float32), jnp.float32(0.0), jnp.int32(0))


def slab(xi, side):
    return jnp.clip((xi[0] / side * NG).astype(jnp.int32), 0, NG - 1)


def setup(n, div):
    side, nc = box_of(n)
    rng = np.random.default_rng(0)
    if div == "uniform":
        pos = rng.uniform(0, side, (n, 2))
    else:
        # NBLOB dense blobs holding BLOB_FRACTION of the particles at BLOB_DENSITY, the rest spread over the
        # remaining area.  The blob radius is set from n so the two densities, and therefore the two neighbour
        # counts, do not depend on n: about 25 neighbours inside a blob and 2 outside, at every size.
        r = float(np.sqrt(BLOB_FRACTION * n / BLOB_DENSITY / (NBLOB * np.pi)))
        ctr = rng.uniform(0, side, (NBLOB, 2))
        pos = ctr[rng.integers(0, NBLOB, n)] + rng.normal(0, r / 2.0, (n, 2))
        bg = rng.random(n) >= BLOB_FRACTION
        pos[bg] = rng.uniform(0, side, (int(bg.sum()), 2))
        pos = np.mod(pos, side)
    return dict(n=n, side=side, nc=nc, h=side / nc,
                pos=jnp.asarray(pos, jnp.float32),
                vel=jnp.asarray(rng.normal(0, 0.2, (n, 2)), jnp.float32),
                th=jnp.asarray(TH0, jnp.float32),
                wg=jnp.asarray(rng.uniform(0.5, 1.5, NG), jnp.float32))


def telemetry(ctx):
    """Trip counts of the neighbour walk, and the cell occupancy the padded arms have to provision for."""
    pos, nc, h, side = ctx["pos"], ctx["nc"], ctx["h"], ctx["side"]
    order, start = jax.jit(cell_list, static_argnums=(1, 2))(pos, nc, side)

    def one(xi):
        c = lax.while_loop(walk_cond, lambda c: walk_step(c, xi, order, start, pos, ctx["th"], nc, h, side),
                           walk_c0())
        return c[4]
    k = jax.jit(jax.vmap(one))(pos)
    occ = np.diff(np.asarray(start))
    return dict(**H.trip_stats(k), cell_occ_max=int(occ.max()), cell_occ_mean=float(occ.mean()),
                ncells=int(nc * nc))


def cap_of(ctx, arm):
    """The per-cell capacity the padded-neighbour-list arms provision: the observed maximum occupancy, doubled
    for the `2x` arm."""
    pos, nc, side = ctx["pos"], ctx["nc"], ctx["side"]
    cid = np.asarray(cell_id(pos, nc, side / nc))
    occ = np.bincount(cid, minlength=nc * nc)
    return int(occ.max()) * (2 if arm.endswith("2x") else 1)


def forces_of(ctx, arm, maxs):
    """-> step_forces(pos, th) -> (force[N,2], tally[NG]) for one arm, at the current positions."""
    nc, h, side, n = ctx["nc"], ctx["h"], ctx["side"], ctx["n"]
    if arm.startswith("tm_"):
        f = tm.vmap(lambda t, xi, order, start, pos, th: _tm_item(t, xi, order, start, pos, th, nc, h, side),
                    outputs={"u": tm.Sum((NG,))}, in_axes=(0, None, None, None, None),
                    backend="auto" if arm == "tm_cuda" else "xla")

        def call(pos, th):
            order, start = cell_list(pos, nc, side)
            r = f(pos, order, start, pos, th)
            return r.per_item, r.outputs["u"]
        return call
    if arm.startswith("jax_pad"):
        cap = cap_of(ctx, arm)

        def call(pos, th):
            order, start = cell_list(pos, nc, side)
            slots = jnp.arange(cap, dtype=jnp.int32)

            def one(xi):
                cid = jax.vmap(lambda g: nbr_cell(xi, g, nc, h))(jnp.arange(NBR, dtype=jnp.int32))
                lo, hi = start[cid], start[cid + 1]
                p = lo[:, None] + slots[None, :]                       # (9, cap) candidate slots
                ok = p < hi[:, None]
                j = order[jnp.clip(p, 0, n - 1)]
                fj, uj = jax.vmap(jax.vmap(lambda jj: pair(xi, pos[jj], th, side)))(j)
                m = ok.astype(jnp.float32)
                return jnp.sum(fj * m[..., None], (0, 1)), jnp.sum(uj * m)
            fs, u = jax.vmap(one)(pos)
            return fs, A.scatter_add(NG, jax.vmap(lambda x: slab(x, side))(pos), 0.5 * u)
        return call

    def call(pos, th):
        order, start = cell_list(pos, nc, side)

        def one(xi):
            body = lambda c: walk_step(c, xi, order, start, pos, th, nc, h, side)
            if arm == "jax_while":
                c = lax.while_loop(walk_cond, body, walk_c0())
            else:
                c, _ = A.masked_scan(body, walk_cond, walk_c0(), maxs, remat=A.remat_of(arm))
            return c[2], c[3]
        fs, u = jax.vmap(one)(pos)
        return fs, A.scatter_add(NG, jax.vmap(lambda x: slab(x, side))(pos), 0.5 * u)
    return call


def _tm_item(t, xi, order, start, pos, th, nc, h, side):
    """ONE particle: walk its neighbouring cells, tally its energy, return its force."""
    c = lax.while_loop(walk_cond, lambda c: walk_step(c, xi, order, start, pos, th, nc, h, side), walk_c0())
    return t.add("u", slab(xi, side), 0.5 * c[3]), c[2]


def sim_of(ctx, arm, maxs):
    """`fwd`: one step at the initial positions -> the deposit grid (the per-item forces come with it)."""
    step_forces = forces_of(ctx, arm, maxs)
    pos = ctx["pos"]
    return lambda th: step_forces(pos, th)[1]


def loss_of(ctx, arm, maxs):
    """A scalar observable after KSTEPS velocity-Verlet steps: the weighted deposit grid of the last step.

    The outer loop is a plain `lax.scan` with `jax.checkpoint` on its body, so the reverse pass recomputes each
    step instead of storing it.  Only the inner per-particle loop differs between arms."""
    step_forces = forces_of(ctx, arm, maxs)
    pos0, vel0, wg, side = ctx["pos"], ctx["vel"], ctx["wg"], ctx["side"]

    def loss(th):
        def body(c, _):
            p, v = c
            f, u = step_forces(p, th)
            v = v + DT * f
            p = jnp.mod(p + DT * v, side)
            return (p, v), u
        (p, v), us = lax.scan(jax.checkpoint(body), (pos0, vel0), None, length=KSTEPS)
        return jnp.sum(wg * us[-1])
    return loss


def measured(ctx, arm, mode, maxs):
    extra = dict(nslabs=NG, side=ctx["side"], ncells=ctx["nc"] ** 2, ksteps=KSTEPS,
                 padded_steps=A.padded_count(arm, maxs), jacfwd_k=A.jacfwd_k(ctx["th"]))
    if arm.startswith("jax_pad"):
        extra["padded_steps"] = NBR * cap_of(ctx, arm)
        extra["cell_capacity"] = cap_of(ctx, arm)
    fn = sim_of(ctx, arm, maxs) if mode == "fwd" else jax.grad(loss_of(ctx, arm, maxs))
    return fn, (ctx["th"],), extra


def check(ctx, maxs):
    return A.cross_arm_check(lambda a: sim_of(ctx, a, maxs), lambda a: loss_of(ctx, a, maxs), ctx["th"],
                             arms=("tm_cuda", "jax_while", "jax_scan", "jax_pad"))
