# Convergence

Each `optimization_params` stage exits when one of two things happens:

1. The L2 norm of the path-integrated loss gradient,
   $\big\| \int \nabla_\theta \mathcal{L} \, \mathrm{d}t \big\|_2$,
   stays below `threshold` for `patience` consecutive iterations.
2. `num_optimizer_iterations` is reached.

Whichever fires first.

Picking `threshold` is the only stage-2 number you really have to
think about, and there's a closed-form recipe that converts your
physical $F_2$-target (the L2 norm of the force vector at the
predicted transition state) into `threshold` plus the integrator
tolerances. **You don't have to pilot-and-divide.**

## The derivation chain

Three knobs are determined by `F2_target`:

```
F2_target  →  δ            (loss δ, only for pseudo-Huber)
           →  threshold     (the |g|_2 convergence trigger)
           →  atol, rtol    (integrator tolerances)
```

### 1. δ (Pseudo-Huber loss shape parameter)

`pvre_pseudo_huber` has a δ knob that interpolates between L1-like
(small δ) and L2-like (large δ) behavior around the saddle. The
gradient magnitude near the saddle scales as δ, so δ should be set
relative to the force you want there:

$$\delta = F_2^{\text{target}} \cdot \|\Delta x_{R \to P}\|_{\text{lb}}$$

where $\|\Delta x_{R \to P}\|_{\text{lb}}$ is a lower bound on the
reactant→product displacement norm. The **safe a priori bound is 1**
(works for any chemistry system; round up if your endpoints are
unusually close in configuration space).

### 2. `threshold` (the |g|_2 convergence trigger)

The IBP-derived relationship between the integrated loss gradient and
the residual force at the TS (full derivation in the appendix below) is

$$\text{threshold} \;=\; \delta \cdot 2 \, \sigma_{\min}(J_\text{path}) \cdot F_2^{\text{target}}$$

where $\sigma_{\min}(J_\text{path})$ is the smallest singular value of
the path Jacobian $\partial x(t)/\partial \theta$ at the saddle. For
the shipped path representation (`width=128` `MLPpath` with REPAR
input), $\sigma_{\min} \approx 1$ is **system-independent** —
measured 0.90 ± 0.005 across $3N$ from 2 (Müller–Brown) to 213
(rost50). See `paths.md` for the calibration.

So in practice:

$$\text{threshold} \;\approx\; \delta \cdot 2 \, F_2^{\text{target}}$$

### 3. `atol`, `rtol` (integrator noise budget)

The adaptive Gauss–Kronrod integrator needs a noise budget small
enough not to mask the trigger, large enough not to over-refine. The
"half-EXTREME" pair sits at:

$$\text{atol} = \text{threshold} / 2, \qquad \text{rtol} = 0.5, \qquad \text{tol\_mode} = \text{`l2'}$$

`tol_mode='l2'` makes the integrator use a scalar `atol + rtol·|g|_2`
denominator that matches the trigger metric. With this pair the
total integrator noise at the trigger is
`atol + rtol·|g|_2 ≈ threshold/2 + 0.5·threshold = threshold` —
comparable to the trigger itself, which is loose enough that GK
doesn't over-subdivide but tight enough that the trigger fires
cleanly.

## Worked example

At the universal `F2_target = 0.05` (eV/Å for chemistry, dimensionless
for toy potentials), with $\sigma_{\min} \approx 1$:

| knob | value | how |
| --- | --- | --- |
| `F2_target` | 0.05 | physical input |
| δ | 0.05 | $F_2 \cdot \|\Delta x_{R\to P}\|_{\text{lb}} = 0.05 \cdot 1$ |
| `threshold` | 5e-3 | $\delta \cdot 2 \cdot 1 \cdot 0.05$ |
| `atol` | 2.5e-3 | $\text{threshold} / 2$ |
| `rtol` | 0.5 | half-EXTREME |

## Shipped recipe

Every system (chemistry and 2D toys) ships at `F2_target = 0.05`,
which collapses to a single stage-2 recipe:

```yaml
integrator_params:
  path_integrand_names: pvre_pseudo_huber
  path_integrand_kwargs:
    pvre_pseudo_huber: {delta: 0.05}
  rtol: 0.5
  atol: 2.5e-3
  tol_mode: l2
  method: gk7
optimizer_params:
  optimizer: {name: adam, lr: 5.0e-3}
  threshold: 5.0e-3
  patience: 1
```

All systems use the same `width=128` `MLPpath`, so the
$\sigma_{\min} \approx 1$ calibration holds without re-derivation.

## Why L2 (not L∞)?

Earlier popcornn used the L∞ norm of the integrated gradient. L∞ has
more seed-to-seed variance because a single component max can spike
on adaptive-quadrature noise. L2 averages over all $D$ components,
which damps the noise. The trade-off is that L2 nominally scales with
$\sqrt{D}$ — but the shipped path-MLP is `width=128`-fixed (so $D$ is
the same for every system at a given output dimension), and the
$\sigma_{\min}(J_\text{path})$ calibration that the threshold derivation
relies on is also system-independent at that width (see
[paths](paths.md)). Net result: L2 is the practical convergence
metric.

## How to pick `patience`

Default is `1`. With the half-EXTREME tolerances the trigger is
designed to fire near-monotonically, so dipping below threshold once
is a reliable convergence signal. Override only if you have a
specific reason — typically you don't.

- **Higher (3+)** for unusually noisy MLIPs or seeds where the
  gradient norm wobbles a lot near convergence.
- Stay at 1 for the shipped recipe.

## Disabling the trigger

`threshold: null` (the default if you omit it) skips the convergence
check entirely and always runs the full `num_optimizer_iterations`.
Use this for:

- The initial pilot pass when you genuinely don't know `F2_target`
  yet.
- Fixed-iteration-budget runs (benchmarks, schedule-driven sweeps).

## Monitoring loss and TS

Convergence is driven by the gradient norm. To also see the loss
integral $\int \mathcal{L}\,\mathrm{d}t$ and the transition state
energy / force per iteration, set on `Popcornn(...)`:

```python
Popcornn(..., track_loss=True, track_ts=True)
```

`track_loss=True` runs a detached scalar-loss integral each iter at
the same `(rtol, atol)`. `track_ts=True` triggers `path.ts_search`
each iter, populating `path.ts_time`, `path.ts_energy`,
`path.ts_force`, and `path.barrier`. Both cost extra forward passes
— enable for diagnostic runs, leave off for production sweeps.

The per-iter stdout columns and the JSONL metrics log
(`metrics_log_path`) both pick these up automatically; see
`Popcornn._StageLogger`.

## Where the formula comes from

The `threshold = δ · 2·σ_min · F2_target` relationship is derived
step-by-step in [derivation](derivation.md) from the pseudo-Huber
gradient, the IBP cancellation at pinned endpoints, and the
σ_min(J) bound between param-space and config-space norms. Read it
when you want to re-derive a constant after changing the path
representation, loss family, or target quantity.
