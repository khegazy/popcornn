# Paths

A *path* in popcornn is a smooth, differentiable mapping from a scalar
time $t \in [0, 1]$ to a configuration $x(t)$, with $x(0) = $ reactant
and $x(1) = $ product.

You select a path representation with `path_params.name`:

```yaml
initialization_params:
  path_params:
    name: mlp       # or 'linear'
    width: 128
    depth: 2
    activation: gelu
```

## Available representations

### `linear`

The simplest possible path:

$$x(t) = x_0 + t \, (x_1 - x_0)$$

A straight line in configuration space between reactant ($x_0$) and
product ($x_1$). Useful only as a starting reference — it doesn't
have any trainable parameters, so there's nothing to optimize.

### `mlp` (default)

A small multi-layer perceptron $\Delta(t'; \theta)$ added on top of
the linear path with input rescaled to $t' = 2t - 1 \in [-1, 1]$:

$$x(t) = \underbrace{x_0 + t\,(x_1 - x_0)}_{\text{linear base}}
       + \underbrace{(1 - t)\, t \cdot \Delta(2t - 1;\, \theta)}_{\text{MLP correction}}$$

Two design choices keep the path well-behaved:

- The $(1 - t)\, t$ envelope vanishes at $t = 0$ and $t = 1$, so the
  path is **pinned** to the reactant and product at the endpoints
  regardless of what the MLP outputs.
- The **REPAR** input rescaling $t' = 2t - 1$ centers the MLP input
  domain on zero, which makes the pre-activation at the midpath
  ($t = 0.5$, $t' = 0$) reduce to bias-only ($z = b$). That gives a
  symmetric $\sigma_{\min}(t)$ profile and a system-independent
  fresh-init $\sigma_{\min}$ calibration (see below).

| Key | Default | Effect |
| --- | --- | --- |
| `width` | `128` | Hidden layer width. Calibrated so $\sigma_{\min}(J_\text{path}) \approx 1$ at midpath (see below). |
| `depth` | `2` | Number of `Linear` layers. `depth=2` is `Linear(1, width) → GELU → Linear(width, 3N)`. |
| `activation` | `"gelu"` | Any nonlinearity in `torch.nn` (case-insensitive). |

You usually shouldn't touch `width`. It's set so that
$\sigma_{\min}(J_\text{path}) \approx 1$, which makes the [convergence
threshold derivation](derivation.md) system-independent.

## $\sigma_{\min}$ calibration (why `width=128`?)

The shipped recipe in [convergence](convergence.md) assumes
$\sigma_{\min}(J_\text{path}) \approx 1$ at the midpath. This is a
property of the path-MLP architecture and the initialization — not of
the chemistry. With `width=128` + REPAR input and PyTorch's default
`kaiming_uniform` Linear init, $\sigma_{\min}$ is empirically
system-independent:

| System | $3N$ | $D$ (params) | $\sigma_{\min}(\text{midpath})$ at fresh init |
| --- | ---:| ---:| ---:|
| Müller–Brown | 2 | 514 | 0.905 |
| LJ-13 | 39 | 5,287 | 0.902 |
| gg3 | 39 | 5,287 | 0.902 |
| gg9711 | 57 | 7,609 | 0.901 |
| rost50 | 213 | 27,733 | 0.901 |

All within 0.5% of each other across two orders of magnitude in $3N$.
The closed-form expression (derived in
[derivation](derivation.md)) is

$$\sigma_{\min}(J_\text{path}, t = 0.5) \;=\; \tfrac{1}{4} \cdot
\|h(t'=0)\|_2 \;\approx\; \tfrac{1}{4}\sqrt{W \cdot \mathbb{E}[\text{GELU}^2(b)]}$$

where $W$ is the hidden width and $b \sim U[-1, 1]$ is the bias-only
pre-activation at midpath under REPAR. The $\tfrac{1}{4}$ is the
$t(1-t)$ envelope evaluated at $t = \tfrac{1}{2}$.

For `width=128`, this gives $\sigma_{\min} \approx 1$ — which is why
that's the default. Changing `width` shifts the calibration by
$\sqrt{W / 128}$ and you'll need to re-derive `threshold`.

## Adding your own path representation

Subclass `BasePath` and implement `get_positions(time)`:

```python
from popcornn.paths.base_path import BasePath

class MyPath(BasePath):
    def __init__(self, my_param=1.0, **kwargs):
        super().__init__(**kwargs)
        # ... build whatever trainable params you need
        # use self.initial_position and self.final_position from BasePath

    def get_positions(self, time):
        # time has shape [N, 1] in [0, 1].
        # Return positions of shape [N, D] where D = len(initial_position).
        # Make sure x(0) = self.initial_position and x(1) = self.final_position.
        ...
```

Then register in `popcornn/paths/__init__.py`:

```python
path_dict = {
    "mlp" : MLPpath,
    "linear" : LinearPath,
    "mine" : MyPath,         # <--
}
```

`BasePath` handles the rest: velocity computation via autograd,
output reshaping, periodic-boundary wrapping, fixed-atom masking, and
the `forward` interface popcornn calls. It also exposes
`ts_search(samples)` which downstream consumers (the logger, the
convergence trigger) use to populate `ts_time`, `ts_energy`,
`ts_force`, `ts_force_mag`, and `barrier` on the path object.

## Why parameterize the path?

The neural-network path is what makes popcornn end-to-end
differentiable. Once you have a function $x(t; \theta)$, you can:

- Evaluate $x$ at any $t$ — no fixed mesh, no interpolation between
  images.
- Compute $\dot{x}(t)$ analytically via autograd.
- Define a loss that's an integral over $t$ and backpropagate to
  $\theta$ in one shot.

Methods like NEB or string method instead carry around a finite chain
of replicas and have to project forces along/perpendicular to the
chain. They work, but they're discrete and stiff. Popcornn's
continuous representation just trains.

See [Concepts](concepts.md) for more on why this matters.
