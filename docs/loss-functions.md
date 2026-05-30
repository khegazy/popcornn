# Loss functions

Popcornn optimizes by integrating a per-point quantity along the path
and minimizing the result:

$$\text{loss}(\theta) = \int_0^1 \ell\big(x(t; \theta)\big) \, \mathrm{d}t$$

The per-point quantity $\ell$ is whatever you put in
`integrator_params.path_integrand_names`. You can pass a single name
(the common case) or a list of names with `path_integrand_scales`
weights (for weighted combinations).

```yaml
integrator_params:
  path_integrand_names: pvre_pseudo_huber
  path_integrand_kwargs:
    pvre_pseudo_huber: {delta: 0.05}
  rtol: 0.5
  atol: 2.5e-3
  tol_mode: l2
  method: gk7
```

```yaml
integrator_params:
  path_integrand_names: ['pvre', 'vre']
  path_integrand_scales: [1.0, 0.1]
```

The shipped chemistry recipe uses `pvre_pseudo_huber` with
$\delta = 0.05$ — see [convergence](convergence.md) for why this
$\delta$ comes from the physical $F_2$-target you set.

## Available terms

All terms below take the path-evaluated quantities — positions $x$,
velocities $\mathbf{v} = \dot{x}$, energy $E$, forces $\mathbf{F}$ —
as needed. Popcornn fetches whichever fields are required and reuses
cached evaluations when it can.

### `pvre_pseudo_huber` (shipped default)

$$\ell_\delta = \delta^2 \left(\sqrt{1 + (s/\delta)^2} - 1\right),
\qquad s = \mathbf{v}\cdot\mathbf{F}$$

A smooth ($C^\infty$) Huber-style loss on $s = \mathbf{v} \cdot
\mathbf{F}$. Quadratic near $s = 0$ (so adaptive Gauss–Kronrod
quadrature converges without refining around the saddle), linear in
$|s|$ far from zero (so the path keeps moving while warm).
**This is the default for the shipped configs.**

$\delta$ is set by

$$\delta = F_2^{\text{target}} \cdot \|\Delta x_{R \to P}\|_{\text{lb}},$$

with the safe a priori bound $\|\Delta x_{R \to P}\|_{\text{lb}} = 1$
(chemistry endpoints are typically well-separated). For
$F_2^{\text{target}} = 0.05$, that pins $\delta = 0.05$ across all
shipped chemistry recipes. Derivation: [convergence](convergence.md)
and [derivation](derivation.md).

### `pvre` (pVRE)

$$\ell_{\text{pVRE}} = \big| \mathbf{v}(t) \cdot \mathbf{F}(t) \big|$$

The non-smooth $L_1$ form. Drives configurations where the force is
**perpendicular** to the path direction — the saddle-point
condition. The plain `pvre` integrand has a kink wherever
$\mathbf{v}\cdot\mathbf{F}$ crosses zero — i.e. exactly the points
the loss is trying to reach — so adaptive GK has to refine
indefinitely around each crossing. `pvre_pseudo_huber` was added
specifically to fix that: it has the same saddle-zero condition but
is smooth at the kink. **Prefer `pvre_pseudo_huber` for production.**

### `pvre_squared` (pVRE²)

$$\ell = \big( \mathbf{v}(t) \cdot \mathbf{F}(t) \big)^2$$

Same saddle-point physics as `pvre` (zero iff $\mathbf{v} \perp
\mathbf{F}$), $C^\infty$-smooth in $t$. Its gradient $\propto
2(\mathbf{v}\!\cdot\!\mathbf{F})$ vanishes near the saddle ridge, so
it drives the path most of the way to the MEP cheaply but plateaus
before pinpointing the TS. `pvre_pseudo_huber` interpolates between
this regime (small $|s|$, quadratic) and `pvre` (large $|s|$, linear)
with one $\delta$ knob, so a two-stage `pvre_squared → pvre`
schedule is rarely needed in current popcornn.

### `pvre_huber`

$$
\ell = \begin{cases}
\tfrac{1}{2} (\mathbf{v} \cdot \mathbf{F})^2
  & \text{if } |\mathbf{v} \cdot \mathbf{F}| \le \delta \\
\delta \, \big(|\mathbf{v} \cdot \mathbf{F}| - \tfrac{1}{2}\delta\big)
  & \text{otherwise}
\end{cases}
$$

Piecewise Huber loss on $s = \mathbf{v} \cdot \mathbf{F}$. Same
limits as `pvre_pseudo_huber` ($\delta \to \infty \Rightarrow$
$\tfrac{1}{2}\cdot$`pvre_squared`; $\delta \to 0 \Rightarrow
\delta\cdot$`pvre`) but only $C^1$-continuous at $|s| = \delta$. The
shipped recipes use `pvre_pseudo_huber` instead because the smooth
version's gradient is also smooth — GK quadrature doesn't have to
refine across the regime boundary.

### `pvre_mag`

$$\ell = \big\| \mathbf{v}(t) \odot \mathbf{F}(t) \big\|_2$$

Per-component product, then norm. A geometry-aware variant of pVRE.

### `vre` (VRE)

$$\ell_{\text{VRE}} = \|\mathbf{F}\|_2 \cdot \|\mathbf{v}\|_2$$

A magnitude-only product. Used in combination with pVRE so that the
difference $\ell_{\text{VRE}} - \ell_{\text{pVRE}}$ is a soft penalty
on the angle between force and velocity (zero when they're parallel).

### `vre_error`

$$\ell = \ell_{\text{VRE}} - \ell_{\text{pVRE}}$$

Force-velocity angular mismatch. Approaches zero on a true MEP (where
forces are tangent to the path).

### `geodesic`

$$\ell_{\text{geo}} = \big\| \mathbf{F}_{\text{decomposed}} \cdot \mathbf{v} \big\|_2$$

Path-length in a force-decomposed metric. With
`potential_params.name: repel`, this is the geodesic interpolation of
[Zhu et al. 2019](https://pubs.aip.org/aip/jcp/article/150/16/164103/198363/Geodesic-interpolation-for-reaction-pathways).
Use it as a pre-step to fix atom clashes. The shipped two-stage
configs (`gg3.yaml`, `gg3_uma.yaml`) use it for stage 1.

### `F_mag`

$$\ell = \|\mathbf{F}\|_2$$

Force magnitude. Useful as a transition-state region loss.

### `E`

$$\ell = E(x(t))$$

Raw energy. Integrating energy along the path gives a (poor) measure
of "how steep" the path is.

### `E_mean`

$$\ell = \overline{E}$$

Mean energy across whatever batched dimension the integrator hands
back. Sometimes useful as a TS-time loss.

## Picking a loss

For a typical reaction:

| What you want | Loss |
| --- | --- |
| Resolve atom clashes (pre-step) | `geodesic` with `potential_params.name: repel` |
| Find the minimum-energy path | **`pvre_pseudo_huber`** (shipped default in all production configs) |
| Same as above with a hand-tuned cutoff | `pvre_huber` with a per-system `delta` |
| Find the path *and* keep it short | combine `pvre` + `vre` with scales (see `examples/configs/loss_example.yaml`) |
| Maximize the TS energy | apply `E_mean` as a TS-region loss (see [Advanced](advanced.md)) |
| Minimize the TS force magnitude | apply `F_mag` as a TS-time loss |

If you're not sure, start with `pvre_pseudo_huber` at the shipped
$\delta = 0.05$ — that's the production default and the one whose
threshold derivation is closed-form.

## Combining terms with schedulers

The `loss_example.yaml` config ramps a geodesic-style term down and
a pVRE term up over the first 100 iterations using cosine schedulers,
so the path first untangles itself and then targets the TS. See
[Advanced](advanced.md) for the syntax.
