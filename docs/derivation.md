# Derivation of the recipe chain

This page derives the closed-form recipe used by the shipped popcornn
configurations:

$$\text{threshold} \;=\; \delta \cdot 2\,\sigma_{\min}(J_\text{path}) \cdot F_2^{\text{target}}, \qquad
\text{atol} = \text{threshold}/2, \qquad \text{rtol} = 0.5.$$

If you just want to *use* the recipe, see [convergence](convergence.md).
This page exists so you can re-derive the constants when you change
the path representation, swap loss families, or move to a target
quantity other than $F_2$.

The presentation builds up in two steps:

1. First, the simpler [pVRE](loss-functions.md#pvre-pvre) loss.
   The derivation has the same structure as the shipped pseudo-Huber
   case but only one factor to track.
2. Then the [pseudo-Huber](loss-functions.md#pvre_pseudo_huber-shipped-default)
   form that the shipped configs actually use. The only change is one
   extra $\delta$ factor.

## Setup

Notation:

- $x(t; \theta) \in \mathbb{R}^{3N}$ — path configuration at parametric
  time $t \in [0, 1]$, parameterized by neural-network weights $\theta$
  ($D$ scalars).
- $v(t; \theta) = \partial_t x(t; \theta)$ — path velocity.
- $E(x), F(x) = -\nabla_x E(x)$ — potential energy and force from the
  attached potential.
- $J(t; \theta) = \partial x(t; \theta) / \partial \theta \in \mathbb{R}^{3N \times D}$
  — path Jacobian w.r.t. parameters at time $t$.
- $F_2^{\text{target}}$ — desired $\|F\|_2$ at the predicted transition
  state. The recipe's only physical input.
- $s(t;\theta) = v(t;\theta) \cdot F(x(t;\theta))$ — the "alignment"
  scalar both pVRE and pseudo-Huber act on. Vanishes when
  $v \perp F$ (the saddle-point condition).

## Common machinery (applies to both losses)

These two steps are shared and only need to be done once.

### $\nabla_\theta s$ at the saddle

Integrating by parts in $t$ on the $v \cdot F$ component cancels the
endpoint terms (paths are pinned: $v(0) = v(1) = 0$ in the
boundary-enforcing ansatz $x(t) = (1-t)R + tP + t(1-t)\,\text{MLP}(2t-1)$;
see [paths](paths.md)). The leading contribution to $\nabla_\theta s$
near a clean saddle at $t = t^\ast$ comes from $\nabla_\theta x$
projected onto $F$:

$$\nabla_\theta s \;\approx\; J(t^\ast)^\top F^\ast \quad \text{near } t^\ast,$$

where $F^\ast = F(x(t^\ast))$ is the residual force at the predicted TS.

### $\sigma_{\min}$ bridges param-space and config-space norms

Using the singular-value bound
$\|J^\top u\|_2 \geq \sigma_{\min}(J) \cdot \|u\|_2$ for $u$ in the
column space of $J$:

$$\|\nabla_\theta s\|_2 \;\gtrsim\; \sigma_{\min}\big(J(t^\ast)\big) \cdot \|F^\ast\|_2
\;=\; \sigma_{\min}\big(J(t^\ast)\big) \cdot F_2^{\text{target}},$$

where $\|F^\ast\|_2 = F_2^{\text{target}}$ identifies the L2 norm of
the residual force with the quantity the recipe targets.

For the shipped `width=128` `MLPpath` with REPAR input,
$\sigma_{\min}(J) \approx 1$ at the midpath — see
[paths](paths.md) for the calibration measurement.

## Part A — pVRE

`pvre` uses the L1 loss on $s$:

$$\ell_{\text{pVRE}}(s) \;=\; |s|, \qquad
\mathcal{L}_{\text{pVRE}}(\theta) = \int_0^1 |s(t;\theta)| \, \mathrm{d}t.$$

### A1 — Loss gradient

$\partial |s| / \partial s = \text{sign}(s)$. Differentiating in
$\theta$:

$$\nabla_\theta \mathcal{L}_{\text{pVRE}}
\;=\; \int_0^1 \text{sign}(s) \, \nabla_\theta s \, \mathrm{d}t.$$

Near the saddle $t^\ast$, $\text{sign}(s)$ flips from $-1$ to $+1$ (or
vice versa). The bracket around the sign change contributes a kink to
the integrand, but the *L2 norm* of the contribution is set by
$\|\nabla_\theta s\|_2$ on **both sides** of $t^\ast$, picking up a
combined factor of 2:

$$\big\|\nabla_\theta \mathcal{L}_{\text{pVRE}}\big\|_2
\;\gtrsim\; 2 \cdot \|\nabla_\theta s\|_2
\;=\; 2 \cdot \sigma_{\min}(J) \cdot F_2^{\text{target}}.$$

### A2 — Inverting for `threshold`

We want the trigger to fire when the residual $F_2$ at the predicted
TS drops below `F2_target`. So:

$$\boxed{\text{threshold}_{\text{pVRE}} \;=\; 2\,\sigma_{\min}(J) \cdot F_2^{\text{target}}.}$$

For the shipped `width=128` `MLPpath`, $\sigma_{\min} \approx 1$,
giving $\text{threshold} \approx 2 F_2^{\text{target}}$. At
$F_2^{\text{target}} = 0.05$ this is $\text{threshold} \approx 0.1$.

That's the whole pVRE derivation. The shipped recipe extends it by
one $\delta$ factor for pseudo-Huber.

## Part B — pseudo-Huber

`pvre_pseudo_huber` replaces $|s|$ with the smooth Huber-style form

$$\ell_\delta(s) \;=\; \delta^2 \left(\sqrt{1 + (s/\delta)^2} - 1\right),
\qquad \mathcal{L}_\delta(\theta) = \int_0^1 \ell_\delta(s(t;\theta)) \, \mathrm{d}t.$$

The motivation is integrator cost: pVRE's `sign(s)` produces a kink at
$s = 0$ — exactly the point the optimizer is trying to reach — so
adaptive Gauss–Kronrod has to refine indefinitely near the saddle.
The pseudo-Huber form is $C^\infty$-smooth at $s = 0$ and quadrature
converges cleanly.

### B1 — Loss gradient

$\ell_\delta'(s) = s / \sqrt{1 + (s/\delta)^2}$. Two limits:

- For $|s| \ll \delta$: $\ell_\delta'(s) \approx s$ (quadratic regime;
  loss looks like $\tfrac{1}{2}s^2$).
- For $|s| \gg \delta$: $\ell_\delta'(s) \approx \delta \cdot \text{sign}(s)$
  (linear regime; loss looks like $\delta \cdot |s|$ — pVRE rescaled
  by $\delta$).

The optimizer spends most of its time in the linear regime — that's
the whole point of using pseudo-Huber: same sign-driven gradient as
pVRE far from $s = 0$, but smooth at the saddle. In that regime:

$$\nabla_\theta \mathcal{L}_\delta
\;\approx\; \int_0^1 \delta \cdot \text{sign}(s) \, \nabla_\theta s \, \mathrm{d}t
\;=\; \delta \cdot \nabla_\theta \mathcal{L}_{\text{pVRE}}.$$

So the entire pVRE bound from A1 just gets multiplied by $\delta$:

$$\big\|\nabla_\theta \mathcal{L}_\delta\big\|_2
\;\gtrsim\; \delta \cdot 2 \, \sigma_{\min}(J) \cdot F_2^{\text{target}}.$$

### B2 — Inverting for `threshold`

$$\boxed{\text{threshold}_{\delta} \;=\; \delta \cdot 2\,\sigma_{\min}(J) \cdot F_2^{\text{target}}.}$$

For $\delta = 0.05$, $\sigma_{\min} \approx 1$, and
$F_2^{\text{target}} = 0.05$: $\text{threshold} \approx 5\!\times\!10^{-3}$.
That's where the worked example in [convergence](convergence.md)
came from.

### B3 — δ from $F_2^{\text{target}}$

The gradient bound is linear in $\delta$, so picking $\delta$ smaller
than necessary just shrinks the trigger value. But you don't want
$\delta$ to be so small that the optimizer never enters the linear
regime that B1 assumed. The "$\delta$ matches the typical $|s|$ at
convergence" choice is

$$\delta = F_2^{\text{target}} \cdot \|\Delta x_{R \to P}\|_{\text{lb}},$$

since $s = v \cdot F$ scales as $\|v\| \cdot \|F\|$, and $\|v\|$ is
$O(\|\Delta x_{R \to P}\|)$ along the path. The safe a priori bound
$\|\Delta x_{R \to P}\|_{\text{lb}} = 1$ gives
$\delta = F_2^{\text{target}}$ for chemistry endpoints (which are
typically well-separated).

## Integrator noise budget

The integrator's tolerance pair `(atol, rtol)` (with `tol_mode='l2'`)
produces a per-step relative noise of
$\text{noise}/|g|_2 = \text{atol}/|g|_2 + \text{rtol}$. At the trigger
$|g|_2 = \text{threshold}$, so noise / threshold = $(\text{atol} / \text{threshold}) + \text{rtol}$.
The "half-EXTREME" choice $\text{atol} = \text{threshold}/2$,
$\text{rtol} = 1/2$ gives noise / threshold $= 1$ — comparable to the
trigger, loose enough that GK doesn't over-subdivide near convergence
but tight enough that the trigger fires cleanly.

## What's left empirical

Three constants in the recipe are pinned down by validation rather
than derivation:

- $\sigma_{\min}(J) \approx 0.90$ rather than exactly 1 — see
  [paths](paths.md) for the per-system measurement (it really is
  system-independent at the `width=128` default).
- The 0.5 in `rtol` — smaller is more conservative but slower, larger
  lets quadrature noise leak into the trigger. 0.5 is what 3-seed
  validation found wall-optimal across the shipped systems.
- `method=gk7` over higher-order GK variants — a wall vs panel-count
  trade documented in [loss-functions](loss-functions.md).

Everything else in the chain is closed-form.
