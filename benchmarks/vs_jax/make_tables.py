"""Turn the measured rows in `results/` into the documentation.

    python benchmarks/vs_jax/make_tables.py            # needs numpy and matplotlib; no JAX

Writes
    docs/performance.md                  the whole measured comparison, per problem and per platform
    docs/img/<problem>_<mode>.png        one panel per MODE, never two modes in one panel
    benchmarks/vs_jax/readme_table.md    a 12-row headline table for the README

Plot rules kept deliberately tight, because these figures go in the public documentation:
  * one panel per mode.  Forward time, gradient time and gradient memory are three figures, never three axes of
    one figure, and there is no second y axis anywhere.
  * at most three series per panel, so the categorical palette clears the all-pairs colour-vision checks.
  * identity is carried by colour AND marker AND a label at the end of the line, so the figure survives being
    printed in grey.
  * every number in a figure is also in a table.
"""
import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# arm -> (colour, marker, label).  Fixed per entity, never assigned by rank.
ARM = {
    "tm_cuda":     ("#2a78d6", "o", "tallymap (cuda)"),
    "tm_xla": ("#7c4dbd", "v", "tallymap (xla)"),
    "jax_while":   ("#eb6834", "s", "jax.vmap + while_loop"),
    "jax_scan":    ("#1baf7a", "^", "padded scan (oracle)"),
    "jax_scan2x":  ("#1baf7a", "^", "padded scan (2x cap)"),
    "jax_ckpt":    ("#eda100", "D", "padded scan + remat (oracle)"),
    "jax_ckpt2x":  ("#eda100", "D", "padded scan + remat (2x cap)"),
    "jax_pad":     ("#c2185b", "P", "padded neighbour list (oracle)"),
    "jax_pad2x":   ("#c2185b", "P", "padded neighbour list (2x cap)"),
    "jax_gen":     ("#00897b", "X", "generation vmap + compaction (oracle)"),
    "jax_gen2x":   ("#00897b", "X", "generation vmap + compaction (2x cap)"),
    "diffrax":     ("#5d4037", "*", "diffrax"),
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
MODE_LABEL = {"fwd": "Forward", "grad": "Reverse-mode gradient", "jvp": "Forward-mode jvp",
              "jacfwd": "jacfwd over all parameters", "score": "Forward with the score carry"}
AXIS_LABEL = {"n": "items N", "d": "state dimension D", "L": "trajectory length L"}

# The three series each panel shows, per problem.  Keep it at three.
PANELS = {
    "default": {"fwd": ("tm_cuda", "jax_while", "jax_scan2x"),
                "grad": ("tm_cuda", "jax_scan2x", "jax_ckpt2x"),
                "jvp": ("tm_cuda", "jax_scan2x", "jax_ckpt2x"),
                "jacfwd": ("tm_cuda", "jax_scan2x", "jax_ckpt2x")},
    "ode_ensemble": {"fwd": ("tm_cuda", "jax_while", "diffrax"),
                     "grad": ("tm_cuda", "jax_ckpt2x", "diffrax"),
                     "jvp": ("tm_cuda", "jax_scan2x", "diffrax"),
                     "jacfwd": ("tm_cuda", "jax_scan2x", "diffrax")},
    "interacting_md": {"fwd": ("tm_cuda", "jax_while", "jax_pad2x"),
                       "grad": ("tm_cuda", "jax_ckpt2x", "jax_pad2x")},
    "shower": {"fwd": ("tm_cuda", "jax_gen", "jax_gen2x"),
               "grad": ("tm_cuda", "jax_gen", "jax_gen2x")},
    "trajectory": {"fwd": ("tm_cuda", "jax_scan",)},
    "dense_state": {"fwd": ("tm_cuda", "jax_while", "jax_scan"),
                    "grad": ("tm_cuda", "jax_scan")},
    "gillespie": {"fwd": ("tm_cuda", "jax_while", "jax_scan2x"),
                  "score": ("tm_cuda", "jax_while", "jax_scan2x")},
}
# Figures are drawn for these modes only.  jvp, jacfwd and score stay in the tables: a panel per mode for every
# mode of every problem would be fifty figures, and the two that carry the argument are forward and gradient.
FIG_MODES = ("fwd", "grad")
ORDER = ["random_walk", "transport", "walk_on_spheres", "ode_ensemble", "gillespie", "shower", "barrier",
         "interacting_md", "particle_filter", "uniform", "dense_state", "trajectory", "small_n"]


def panels_for(problem, mode):
    return PANELS.get(problem, {}).get(mode) or PANELS["default"].get(mode) or ()


def load(results):
    rows = []
    for f in sorted(os.listdir(results)):
        if f.endswith(".json"):
            rows += json.load(open(os.path.join(results, f)))["rows"]
    return rows


def sel(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r.get(k) == v]
    return out


def ok(rows, **kw):
    return sorted(sel(rows, status="ok", **kw), key=lambda r: r["val"])


def platforms(rows):
    """(platform, jax) pairs present, primary first.

    The primary platform is the GPU with the most measured rows, which is the one that got the full sweep; the
    others are reported in their own section."""
    n = {}
    for r in rows:
        n[(r["platform"], r["jax"])] = n.get((r["platform"], r["jax"]), 0) + 1
    return sorted(n, key=lambda p: (p[0] == "cpu", -n[p], p))


def style(ax, xlabel, ylabel, title, logx=True):
    if logx:
        ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
    ax.grid(True, which="minor", color=GRID, lw=0.3, alpha=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def panel(rows, problem, mode, field, ylabel, out, plat, div):
    arms = panels_for(problem, mode)
    series = [(a, ok(rows, problem=problem, mode=mode, arm=a, div=div, platform=plat[0], jax=plat[1]))
              for a in arms]
    series = [(a, rs) for a, rs in series if any(r.get(field) is not None for r in rs)]
    if not series:
        return None
    axis = series[0][1][0]["axis"]
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    for a, rs in series:
        c, m, lab = ARM[a]
        x = np.array([r["val"] for r in rs], float)
        y = np.array([np.nan if r.get(field) is None else r[field] for r in rs], float)
        good = np.isfinite(y)
        if not good.any():
            continue
        ax.plot(x[good], y[good], color=c, marker=m, ms=6, lw=2.0, zorder=3, label=lab,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.annotate(lab, (x[good][-1], y[good][-1]), textcoords="offset points", xytext=(6, 0),
                    color=INK2, fontsize=7.5, va="center")
    title = f"{MODE_LABEL.get(mode, mode)}: {problem.replace('_', ' ')}"
    style(ax, AXIS_LABEL.get(axis, axis), ylabel, title)
    ax.margins(x=0.34)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=min(3, len(series)))
    fig.text(0.01, 0.005, f"{plat[0]}, JAX {plat[1]}, divergence '{div}'", color=INK2, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    name = f"{problem}_{mode}{'_mem' if field == 'peak_mb' else ''}.png"
    path = os.path.join(out, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return name


def best_baseline(rows, problem, mode, val, div, plat, exclude=("tm_cuda", "tm_xla")):
    """The FASTEST plain-JAX arm at this point: the number tallymap actually has to beat."""
    rs = [r for r in sel(rows, status="ok", problem=problem, mode=mode, val=val, div=div,
                         platform=plat[0], jax=plat[1]) if r["arm"] not in exclude and r.get("ms")]
    return min(rs, key=lambda r: r["ms"]) if rs else None


def mb(x):
    """Megabytes, readable: 96 MB, 1,120 MB, 12.1 GB."""
    if x is None:
        return "-"
    return f"{x / 1024:.3g} GB" if x >= 10240 else f"{x:,.0f} MB"


def fmt(x, n=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:.{n}g}"


def table(rows, problem, mode, div, plat, arms, field="ms", unit="ms"):
    rs = sel(rows, problem=problem, mode=mode, div=div, platform=plat[0], jax=plat[1])
    if not rs:
        return ""
    vals = sorted({r["val"] for r in rs})
    # drop an arm that produced no usable row here (diffrax on a platform where it is not installed, say):
    # a column of SKIPs is noise
    arms = [a for a in arms if any(r.get("status") == "ok" for r in sel(rs, arm=a))]
    if not arms:
        return ""
    axis = rs[0]["axis"]
    head = (f"\n{MODE_LABEL.get(mode, mode)}, divergence `{div}` ({unit}):\n\n| {axis.upper()} | "
            + " | ".join(ARM[a][2] for a in arms) + " |\n|" + "---|" * (1 + len(arms)) + "\n")
    body = []
    for v in vals:
        cells = []
        for a in arms:
            r = sel(rs, arm=a, val=v)
            if not r:
                cells.append("-")
            elif r[0].get("status") != "ok":
                cells.append(r[0]["status"])
            else:
                cells.append(fmt(r[0].get(field)))
        body.append(f"| {v:,} | " + " | ".join(cells) + " |")
    return head + "\n".join(body) + "\n"


def pad_table(rows, problem, div, plat):
    """The measured trip counts (from the probe cells) next to the padding each arm was given."""
    probes = ok(rows, problem=problem, mode="probe", div=div, platform=plat[0], jax=plat[1])
    fwd = ok(rows, problem=problem, mode="fwd", div=div, platform=plat[0], jax=plat[1])
    if not probes:
        return ""
    ORACLE, TWOX = ("jax_scan", "jax_ckpt", "jax_pad", "jax_gen"), ("jax_scan2x", "jax_ckpt2x", "jax_pad2x",
                                                                   "jax_gen2x")
    out = [f"\nMeasured trip counts, divergence `{div}`, and the padding each arm was given:\n",
           "| " + probes[0]["axis"].upper()
           + " | steps mean | steps p99 | steps max | padded: oracle | padded: 2x cap |",
           "|---|---|---|---|---|---|"]
    for p in probes:
        v = p["val"]
        po = next((r.get("padded_steps") for r in fwd if r["val"] == v and r["arm"] in ORACLE), None)
        p2 = next((r.get("padded_steps") for r in fwd if r["val"] == v and r["arm"] in TWOX), None)
        out.append(f"| {v:,} | {fmt(p.get('steps_mean'))} | {fmt(p.get('steps_p99'))} | "
                   f"{fmt(p.get('steps_max'))} | {fmt(po)} | {fmt(p2)} |")
    return "\n".join(out) + "\n"


def speedups(rows, problem, div, plat):
    """-> list of (val, mode, tm ms, best baseline arm, best baseline ms, ratio).  The tallymap arm is the cuda
    backend, or the xla backend on a platform without one (the CPU rows)."""
    out = []
    for mode in ("fwd", "grad"):
        tm = [ok(rows, problem=problem, mode=mode, arm=a, div=div, platform=plat[0], jax=plat[1])
              for a in ("tm_cuda", "tm_xla")]
        for r in tm[0] or tm[1]:
            b = best_baseline(rows, problem, mode, r["val"], div, plat)
            if b:
                out.append((r["val"], mode, r["ms"], b["arm"], b["ms"], b["ms"] / r["ms"],
                            r.get("peak_mb"), b.get("peak_mb")))
    return out


# The headline table is capped at 12 rows: the seven natural problems, the two interacting systems and three of
# the four bad fits.  `small_n` is in the per-problem section only; its message (N = 1 gains nothing) needs the
# whole curve, not one row.
HEADLINE = ["random_walk", "transport", "walk_on_spheres", "ode_ensemble", "gillespie", "shower", "barrier",
            "interacting_md", "particle_filter", "uniform", "dense_state", "trajectory"]


def headline(rows, plat):
    """<= 12 rows, one per problem at its largest measured size: forward and gradient speedup, and memory."""
    lines = ["| problem | kind | size | forward: tallymap vs best plain JAX | gradient: tallymap vs best "
             "plain JAX | gradient peak memory |",
             "|---|---|---|---|---|---|"]
    for p in HEADLINE:
        rs = sel(rows, problem=p, platform=plat[0], jax=plat[1])
        if not rs:
            continue
        divs = sorted({r["div"] for r in rs if r.get("div")}, key=lambda d: (d != "mid", d))
        sp = speedups(rows, p, divs[0], plat)
        if not sp:
            continue
        v = max(x[0] for x in sp)
        axis = rs[0]["axis"].upper()
        cells = []
        mem = "-"
        for mode in ("fwd", "grad"):
            hit = [x for x in sp if x[0] == v and x[1] == mode]
            if not hit:
                cells.append("-")
                continue
            _, _, tms, barm, bms, ratio, tmb, bmb = hit[0]
            word = "faster" if ratio >= 1 else "**slower**"
            cells.append(f"{fmt(tms)} vs {fmt(bms)} ms ({ARM[barm][2]}): "
                         + (f"**{ratio:.2f}x** {word}" if ratio >= 1 else f"**{1 / ratio:.2f}x** {word}"))
            if mode == "grad" and tmb and bmb and tmb > 0:
                r_ = bmb / tmb
                mem = (f"{mb(tmb)} vs {mb(bmb)} "
                       + (f"({r_:.0f}x less)" if r_ >= 1 else f"(**{1 / r_:.1f}x MORE**)"))
        lines.append(f"| {p.replace('_', ' ')} | {rs[0].get('group', '')} | {axis} = {v:,} | "
                     + " | ".join(cells) + f" | {mem} |")
    return "\n".join(lines) + "\n"


def agreement(rows, plat):
    rs = [r for r in rows if r.get("mode") == "check" and r["platform"] == plat[0] and r["jax"] == plat[1]]
    if not rs:
        return ""
    keys = sorted({k for r in rs for k in r if k.startswith("rel_")})
    out = ["\n| problem | divergence | " + " | ".join(k[4:] for k in keys) + " |",
           "|" + "---|" * (2 + len(keys))]
    for r in sorted(rs, key=lambda r: (ORDER.index(r["problem"]) if r["problem"] in ORDER else 99, r["div"])):
        out.append(f"| {r['problem']} | {r['div']} | "
                   + " | ".join(("-" if r.get(k) is None else
                                 (r[k] if isinstance(r[k], str) else f"{r[k]:.1e}")) for k in keys) + " |")
    return "\n".join(out) + "\n"


PREAMBLE = """# Measured performance: tallymap against plain JAX

Generated by `benchmarks/vs_jax/make_tables.py` from the committed rows in `benchmarks/vs_jax/results/`.
Reproduce any row with

    python benchmarks/vs_jax/run.py --problem <name>          # or --quick for the small sweep

## What is measured

One measurement is one subprocess, because the device peak-memory counter is cumulative for the life of a
process. Each cell reports the MEDIAN `jax.jit` wall time per call over at least 7 calls after a warm-up, with
`jax.block_until_ready` on the result. Compile time is reported separately (`lower_s` and `xla_compile_s` in the
raw rows) and is never folded into the per-call time. Peak device memory comes from the device allocator's own
`peak_bytes_in_use`; on CPU there is no such counter and the memory columns are empty.

Every row records the device, the JAX version and the git commit it was measured at. Rows measured before the 0.5.0
public release carry commit hashes from the private pre-release history, which do not resolve in this
repository.

## The arms

| arm | what it is | modes |
|---|---|---|
| `tallymap (cuda)` | `tm.vmap`, one persistent-thread CUDA kernel | all |
| `tallymap (xla)` | `tm.vmap`, the masked plain-JAX interpreter (the only backend on CPU) | all |
| `jax.vmap + while_loop` | `jax.vmap` of the SAME item with its `lax.while_loop`, then one `.at[].add` | forward only |
| `padded scan` | `jax.vmap` of the same item rewritten as a masked `lax.scan` | all |
| `padded scan + remat` | the same with a nested `jax.checkpoint`-ed inner scan | all |
| `padded neighbour list` | the JAX-MD-style fixed-capacity cell list (the interacting-particles problem) | all |
| `generation vmap + compaction` | fixed-capacity generations, children compacted between them (the shower) | all |
| `diffrax` | diffrax's own `diffeqsolve` with `Bosh3`, a `PIDController` and `RecursiveCheckpointAdjoint` | all |

`jax.vmap + while_loop` has no gradient row anywhere in this document. JAX cannot reverse-differentiate a
`lax.while_loop` whose trip count depends on the data; `jax.grad` of one raises. Every gradient-capable baseline
here had to be rewritten as a padded, masked `lax.scan` first, and that rewrite is the cost tallymap removes.

## How the padding was chosen

A padded scan has to be told a trip count, and the number chosen decides the answer. Nothing here is padded to a
number picked by hand. For every (problem, size, divergence level) the runner first measures the trip count of
the item loop with an UNBOUNDED loop construct, and then:

* the arms labelled **oracle** are padded to EXACTLY that observed maximum. No real program can do this: it would
  have to know the longest item before running it. The oracle arms are the best a padded scan could possibly be.
* the arms labelled **2x cap** are padded to twice the observed maximum, which is what a careful user picks when
  they do not know the maximum in advance.

Both numbers appear in the trip-count table of every problem, next to the maximum the items actually needed, so
the waste is visible rather than asserted.

The tallymap arms set no `max_steps` at all here (`max_steps=None`, the default, which is exactly `jax.vmap`
semantics). In production code `max_steps` should be set as a safety net; it costs one extra per-item counter.

## The agreement tolerance

These are f32 programs. Two arms sum the same per-item values in a different order, so the contract is
reassociation-level agreement, not bitwise equality. The tolerances the agreement cell is read against:

* **a tally, arm against arm: 1e-6 relative.** One f32 ulp is about 1.2e-7, and a sum over millions of emits
  loses a few more low-order bits. `tm.Sum(fixed_point=True)` is the fix when bit-reproducibility matters.
* **a tally whose index comes from a comparison: 1e-4 relative.** An escape sensor or a knock-out decision can
  flip on a 1-ulp difference. The items that flip are the ones sitting on the boundary and they contribute
  almost nothing to the tally, but the relative figure is larger than an ulp.
* **reverse against forward mode of the SAME program: 1e-6 relative, up to 1e-5 for an adaptive solver.** The
  two modes sum the per-item contributions in a different order; the adaptive ODE at its high divergence level
  measured 8.5e-6. That figure needs a well-conditioned derivative: both adaptive-step problems stop-gradient the
  step-size factor, as diffrax does. Differentiated THROUGH the controller, the ODE gradient was decided by
  rounding (its f32 and f64 values had opposite signs; README, "An adaptive step-size controller").
* **a CHAOTIC item: no useful tolerance on the tally at all; compare per item instead.** A transport packet
  takes up to a few hundred scattering flights, and a 1-ulp difference in an early flight sends it to a
  different sensor entirely. The `transport` tally differs between tallymap and plain JAX by 4e-2 at its high
  divergence level for that reason, while the two tallymap backends agree to 4e-7 with each other. The useful
  question there is what FRACTION of packets land in the same sensor, which the agreement table reports as
  `same_sensor_<arm>`; a tally tolerance would be measuring the chaos, not the mechanics.
* **an integer tally: exact.** Integers cannot lose bits, so a difference means the items took different paths.

The agreement table at the end of this document gives the measured figure for every problem, so a reader can
check these claims rather than take them.
"""

NOTES = {
    "random_walk": "The tally is written every step, so the plain-JAX arms cannot put the loop inside the "
                   "`jax.vmap`: a shared tally has to be a loop carry, and the loop runs outside, one batched "
                   "`.at[].add` per iteration. The `while_loop` arm's condition is `jnp.any(alive)`, so it runs "
                   "the batch maximum.",
    "uniform": "THE FLOOR of what tallymap buys, in two shapes. Every item runs exactly 64 steps and the step "
               "is a handful of flops, so there is no divergence to exploit and the oracle padded scan is "
               "padded to nothing. The two levels are not divergence levels: they are where the tally is "
               "written. At `end_only` the item emits once and the plain-JAX arm puts the whole loop inside "
               "the `jax.vmap`, finishing with one scatter-add: about the cheapest program XLA can be given. "
               "At `per_step` the item emits every step, a shared tally has to be a loop carry, and the "
               "plain-JAX arm writes N values to HBM 64 times while tallymap's emit stays in shared memory. "
               "This problem was written expecting a loss, because a uniform loop is what `lax.scan` is for. "
               "It is not one. A uniform trip count on its own is NOT a reason to avoid tallymap on this "
               "hardware; the losses below (`dense_state`, `trajectory`) have uniform loops too, and they lose "
               "because the per-item STATE leaves the registers, not because the loop is uniform.",
    "dense_state": "A BAD FIT, measured on purpose. The sweep axis is the state dimension D, not N. A "
                   "persistent-thread kernel runs a D x D matvec as scalar code; `jax.vmap` turns it into one "
                   "batched GEMM. The crossover in D is the number to quote, and the gradient crosses over "
                   "much earlier than the forward pass does. Note also the agreement: this is the one problem "
                   "where the cuda backend and the xla backend differ by more than an ulp (about 4e-3). "
                   "A D x D matvec is a `dot_general`, the cuda backend lowers it as scalar code, XLA sums it "
                   "in a different order, and eight `tanh` steps amplify the difference. It is a reassociation "
                   "difference, not a defect, and it is another reason this shape belongs to `jax.vmap`.",
    "trajectory": "A BAD FIT, measured on purpose. The item must return all L of its samples, not a sum. "
                  "tallymap has no per-step output, so the history goes into a per-item buffer in local memory, "
                  "while `lax.scan` stacks its per-step outputs for free. The break-even in L is the number to "
                  "quote.",
    "small_n": "A BAD FIT, measured on purpose. At N = 1 there is one GPU thread and no parallelism at all, "
               "and the absolute time is fixed overhead in both arms rather than work. Read the ratio here as "
               "a statement about overheads, not about throughput.",
    "interacting_md": "An INTERACTING system. The interaction is between calls, not inside one: an outer "
                      "`lax.scan` over time steps, the cell list rebuilt in plain JAX each step, and one "
                      "`tm.vmap` call in which every particle reads the shared positions and cell list and walks "
                      "the members of its 3x3 neighbouring cells. `fwd` is one step; `grad` is the gradient of a "
                      "scalar observable after 10 velocity-Verlet steps, in reverse mode through the outer scan "
                      "with `jax.checkpoint` on the step.",
    "particle_filter": "An INTERACTING system. Resampling is an all-to-all over the particles and stays in plain "
                       "JAX between the calls; the variable-length adaptive propagation between two observations "
                       "is the `tm.vmap` call. The gradient is the standard stop-gradient-through-resampling "
                       "estimator, which is biased; that is the usual estimator in practice, not a property of "
                       "tallymap.",
    "gillespie": "A jump process has no pathwise derivative of its histogram with respect to the rate constants, "
                 "so there is no gradient row. The `score` rows carry the path score along the item and tally "
                 "score * X(T), a likelihood-ratio (REINFORCE) gradient estimate obtained in ONE forward pass; "
                 "compare them with the `fwd` rows to see what that estimator costs.",
    "barrier": "A knock-out is discontinuous in theta, so the pathwise Greeks measured here are BIASED estimators "
               "of the true Greeks. tallymap returns the exact derivative of the program as written, and it "
               "agrees with the plain-JAX arm's derivative of the same program.",
    "shower": "`jax.vmap` cannot grow its batch, so the plain-JAX arm has to run fixed-capacity generations with "
              "the children compacted between them. The oracle arm is given each generation's measured "
              "population; a real program does not know the population of generation 5 before it runs. Read "
              "the gradient rows carefully: this is the one natural problem where tallymap LOSES at the "
              "largest size. The tree adjoint records every child of the call, so its memory grows with the "
              "tree, which is not the O(1)-per-item replay the flat problems get, and the compaction baseline "
              "is cheap once the generations are large enough to fill the machine. tallymap still wins the "
              "forward pass at every size, and wins the gradient below about 10^5 roots.",
}


def problem_section(rows, p, plat, out):
    rs = sel(rows, problem=p, platform=plat[0], jax=plat[1])
    if not rs:
        return ""
    divs = sorted({r["div"] for r in rs if r.get("div")}, key=lambda d: (d != "mid", d))
    group = rs[0].get("group", "")
    s = [f"\n### {p} ({group})\n", f"\n{NOTES[p]}\n" if p in NOTES else ""]
    figs = []
    for mode in ("fwd", "grad", "jvp", "jacfwd", "score"):
        if not sel(rs, mode=mode):
            continue
        arms = [a for a in ARM if sel(rs, mode=mode, arm=a)]
        for d in divs:
            s.append(table(rows, p, mode, d, plat, arms))
            if mode == "grad":
                s.append(table(rows, p, mode, d, plat, arms, field="peak_mb", unit="peak MB"))
        if mode in FIG_MODES:
            f = panel(rows, p, mode, "ms", f"{MODE_LABEL.get(mode, mode)} time per call (ms)", out, plat,
                      divs[0])
            if f:
                figs.append(f)
        if mode == "grad":
            f = panel(rows, p, mode, "peak_mb", "peak device memory (MB)", out, plat, divs[0])
            if f:
                figs.append(f)
    for d in divs:
        s.append(pad_table(rows, p, d, plat))
    for f in figs:
        s.append(f"\n![{p} {f}](img/{f})\n")
    return "".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--repo", default=ROOT, help="where docs/performance.md and docs/img/ are written")
    a = ap.parse_args()
    rows = load(a.results)
    if not rows:
        print("no rows in " + a.results)
        return 1
    img = os.path.join(a.repo, "docs", "img")
    os.makedirs(img, exist_ok=True)
    plats = platforms(rows)
    prim = plats[0]
    print(f"{len(rows)} rows, platforms {plats}, primary {prim}")

    doc = [PREAMBLE]
    doc.append(f"\n## Headline\n\nLargest measured size of each problem on {prim[0]}, JAX {prim[1]}. "
               f"`speedup` is against the FASTEST plain-JAX arm at that point, not against the slowest.\n\n")
    doc.append(headline(rows, prim))
    doc.append("\nThe same table, as a file: `benchmarks/vs_jax/readme_table.md`.\n")
    doc.append("\n## Per problem\n\nAll rows below are " + f"{prim[0]}, JAX {prim[1]}.\n")
    for p in ORDER:
        doc.append(problem_section(rows, p, prim, img))
    dfx = next((pl for pl in plats if sel(rows, platform=pl[0], jax=pl[1], arm="diffrax", status="ok")), None)
    if dfx:
        doc.append(f"\n## Against diffrax\n\nMeasured on {dfx[0]}, JAX {dfx[1]}, where diffrax installs. "
                   "diffrax is the library a JAX user actually reaches for when they have an ensemble of "
                   "adaptive solves, so it is the baseline that matters for the ODE problem.\n\n"
                   "One finding before the numbers: diffrax's default adjoint, `RecursiveCheckpointAdjoint`, "
                   "is a `custom_vjp`, so forward-mode autodiff through it raises outright ('can't apply "
                   "forward-mode autodiff (jvp) to a custom_vjp function'). The `jvp` and `jacfwd` rows below "
                   "use `diffrax.ForwardMode()` instead, which is a different program. `max_steps` is a hard "
                   "bound diffrax raises on, and its step-size controller is not bit-identical to the one the "
                   "other arms share, so diffrax is given the realistic 2x cap rather than the oracle one.\n")
        for mode in ("fwd", "jvp", "jacfwd", "grad"):
            doc.append(table(rows, "ode_ensemble", mode, "mid", dfx,
                             ["tm_cuda", "tm_xla", "jax_scan2x", "jax_ckpt2x", "diffrax"]))
        doc.append(table(rows, "ode_ensemble", "grad", "mid", dfx,
                         ["tm_cuda", "jax_ckpt2x", "diffrax"], field="peak_mb", unit="peak MB"))
    doc.append("\n## Other platforms\n\n"
               "The same `fwd` and `grad` rows on every machine measured. The CPU rows have only the xla "
               "backend: the cuda backend needs a GPU, and `tm.vmap` says so.\n\n"
               "A CPU row is a shared machine: 16 cores of a 64-core node, with other jobs on the rest of it. "
               "Every arm of one problem is measured in the same job, one cell after another, so a RATIO within "
               "a problem is sound, but the same code measured in two jobs has come out up to 2.6x apart, so a "
               "CPU ratio should not be compared against the same ratio from an earlier run of this table.\n")
    for pl in plats[1:]:
        doc.append(f"\n### {pl[0]}, JAX {pl[1]}\n")
        doc.append(headline(rows, pl))
    doc.append("\n## Cross-arm agreement\n\n"
               "`rel_<arm>` is the largest relative difference between that arm's tally and the xla "
               "backend's, at the problem's check size with every arm in one process. `rel_grad_vs_jacfwd` is "
               "reverse against forward mode of the same tallymap program. One f32 ulp is about 1e-7.\n")
    doc.append(agreement(rows, prim))
    doc.append("\n## Compile time\n\n"
               "Compile time is not in the tables above. It is in the raw rows as `lower_s` (trace and lower, "
               "including the CUDA kernel build) and `xla_compile_s`. tallymap builds and caches a kernel per "
               "program; see `benchmarks/vs_jax/results/` for the per-cell figures.\n")
    perf = os.path.join(a.repo, "docs", "performance.md")
    open(perf, "w").write("".join(doc))
    print("wrote " + perf)

    rt = os.path.join(a.repo, "benchmarks", "vs_jax", "readme_table.md")
    open(rt, "w").write(f"<!-- generated by benchmarks/vs_jax/make_tables.py; {prim[0]}, JAX {prim[1]} -->\n"
                        "Measured against the fastest plain-JAX arm of each problem "
                        "(`docs/performance.md` has the method, the padding and the losses):\n\n"
                        + headline(rows, prim))
    print("wrote " + rt)
    big = [(f, os.path.getsize(os.path.join(img, f))) for f in sorted(os.listdir(img))
           if f.endswith(".png") and os.path.getsize(os.path.join(img, f)) > 200 * 1024]
    for f, n in big:
        print(f"PNG TOO BIG {f} {n / 1024:.0f} KB")
    print(f"MAKE_TABLES {'FAIL' if big else 'PASS'} "
          f"{len([f for f in os.listdir(img) if f.endswith('.png')])} figures")
    return 1 if big else 0


if __name__ == "__main__":
    raise SystemExit(main())
