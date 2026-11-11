# A branching-process outbreak as a spawn tree

The individual-level branching process used for early-outbreak risk assessment (Lloyd-Smith et al. 2005;
Hellewell et al. 2020), simulated from N index cases.  An overdispersed offspring distribution makes most
lineages die out at once while a few explode.  In tallymap the item is one infected individual and its offspring
loop `spawn`s each child as a new item.

**There is no incumbent JAX library** for this: the established tools (`epichains`, `ringbp`, `bpmodels`) are R
packages that walk one lineage at a time on the CPU.  The baseline is the best plain-JAX program, one global
generation frontier pooled over all outbreaks, with its width sized from what the spawn tree measures.

`run.py` runs both, checks that the incidence curve, infections by generation and outbreak size agree and that the
tree adjoint matches `jacfwd` and the frontier's gradient, and times the forward pass at one size.

## Measured

One NVIDIA L40S, JAX 0.11.2, N = 16,384 to 1,048,576 index cases: forward **3.8-13.5x** and `jacfwd` over the
four parameters **4.8-7.6x** against the frontier, with the incidence tally bitwise equal.

Where it loses: **reverse mode is a tie** (1.2-1.7x) and uses 2-4x *more* memory than the frontier scan, because
the tree adjoint needs a spawn Queue covering the whole tree; use `jacfwd` here.  Two of the four parameters
(`log R0`, `logit p_det`) have no pathwise derivative at all.  At zero overdispersion the forward margin is 3.8x.

## Run

```
python run.py              # CPU, N = 1024: checks and a timing, prints PASS/FAIL
python run.py --n 262144   # on a GPU
```
