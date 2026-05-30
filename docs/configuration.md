# Configuration reference

Popcornn reads a YAML config split into two top-level sections:

```yaml
initialization_params:
  # ... how to set up the path
optimization_params:
  - # ... first optimization stage
  - # ... second optimization stage
  - # ...
```

The `examples/run.py` driver passes `initialization_params` to
`Popcornn(...)` and unpacks `optimization_params` (a list) into
`Popcornn.optimize_path(*params)`. You can mirror this from your own
Python code; see [Getting Started](getting-started.md).

## `initialization_params`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `images` | `str` or list | required | Path to an `xyz`/`traj`/`json`/`npy`/`pt` file, or a list of ASE `Atoms` objects, or a 2D coordinate list. First frame is the reactant, last is the product. Intermediate frames are non-fixed guesses the path is fitted through. |
| `unwrap_positions` | `bool` | `True` | For periodic systems, unwrap the product against the reactant using the minimum-image convention. Disable if any atom moves more than half a cell. |
| `path_params` | `dict` | `{}` | See below. |
| `track_loss` | `bool` | `False` | Run a detached scalar-loss integral each iteration (`∫L dt`). Required for the stage-end log line to report `loss=`. Costs one extra forward pass per iter. |
| `track_ts` | `bool` | `False` | Trigger `path.ts_search` each iteration. Populates `path.ts_time`, `path.ts_energy`, `path.ts_force`, and `path.barrier`. Required for per-iter `barrier`/`force` columns and the stage-end `barrier=`/`ts_force_norm=` fields. Costs one extra forward pass per iter. |
| `num_record_points` | `int` | `101` | Number of frames sampled along the optimized path by `get_discrete_path()`. |
| `device` | `str` or `None` | auto | `cuda` or `cpu`. `None` picks `cuda` if available. |
| `dtype` | `"float32"` or `"float64"` | `"float32"` | PyTorch dtype. |
| `seed` | `int` or `None` | `0` | RNG seed for reproducibility. `None` to skip seeding. |

### `path_params`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `"mlp"` or `"linear"` | required | Path representation. See [Paths](paths.md). |
| `width` | `int` | `128` | (`mlp` only) Hidden layer width. The default is calibrated so `σ_min(J_path) ≈ 1`, which makes the [threshold derivation](derivation.md) system-independent. Don't change unless you know why. |
| `depth` | `int` | `2` | (`mlp` only) Number of `Linear` layers. `depth=2` is `Linear(1, width) → GELU → Linear(width, 3N)`. |
| `activation` | `str` | `"gelu"` | Any activation in `torch.nn` (case-insensitive). |

## `optimization_params`

`optimization_params` is a **list of dicts**. Each dict is one stage
of optimization. Common two-stage pattern:

1. A cheap repulsive stage to resolve atom clashes (geodesic
   interpolation).
2. The expensive MLIP stage that finds the actual transition state.

Each stage has four sub-blocks:

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `potential_params` | `dict` | `{}` | Energy model. See below. |
| `integrator_params` | `dict` | `{}` | Path-integral and loss settings. See below. |
| `optimizer_params` | `dict` | `{}` | Adam / scheduler / convergence settings. See below. |
| `num_optimizer_iterations` | `int` | `1000` | Hard iteration cap. Actual run usually stops earlier when the convergence trigger fires. |

### `potential_params`

| Key | Description |
| --- | --- |
| `name` | One of: `wolfe_schlegel`, `muller_brown`, `schwefel`, `constant`, `sphere`, `lennard_jones`, `repel`, `morse`, `harmonic`, `mace`, `newtonnet`, `ani`, `leftnet`, `orb`, `escaip`, `uma`, `chgnet`. See [Potentials](potentials.md). |

Each potential takes its own extra keys (e.g. UMA needs `model_name`
and `task_name`; NewtonNet needs a `model_path`).

### `integrator_params`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `path_integrand_names` | `str` or list | `None` | Per-point integrand (or list of them). Options: `geodesic`, `pvre`, `pvre_squared`, `pvre_huber`, `pvre_pseudo_huber`, `pvre_mag`, `vre`, `vre_error`, `E`, `E_mean`, `F_mag`. See [Loss functions](loss-functions.md). |
| `path_integrand_scales` | float or list | `1.0` | Per-term weighting when `path_integrand_names` is a list. |
| `path_integrand_kwargs` | `dict` | `{}` | Per-term constructor kwargs, e.g. `{pvre_pseudo_huber: {delta: 0.05}}`. |
| `method` | `str` | `"gk7"` | Adaptive Gauss–Kronrod rule. `gk7` is 7-point, the popcornn default. |
| `rtol` | `float` | `0.5` | Relative tolerance. |
| `atol` | `float` | `2.5e-3` | Absolute tolerance. |
| `tol_mode` | `"l2"` or `"per_d"` | `"l2"` | Tolerance-aggregation rule. `"l2"` uses a scalar `atol + rtol·|g|_2` denominator (matches the `|g|_2` convergence trigger). `"per_d"` is the legacy per-component denominator. |
| `max_batch` | `int` or `None` | `None` | Max batch size at any quadrature step. `None` lets torchpathint auto-size and remember the value across calls. See [Memory & OOM](memory-and-oom.md). |

### `optimizer_params`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `optimizer` | `dict` | required | `{"name": "<torch.optim class>", ...kwargs}`. Names are case-insensitive (`adam`, `sgd`, `lbfgs`, …). All extra keys forward to the optimizer constructor (`lr`, `weight_decay`, …). |
| `lr_scheduler` | `dict` or `None` | `None` | `{"name": "<torch.optim.lr_scheduler class>", ...kwargs}`. Same convention. |
| `threshold` | `float` or `None` | `None` | `|g|_2` convergence trigger. `None` disables the trigger and runs the full `num_optimizer_iterations`. See [Convergence](convergence.md) for the derivation. |
| `patience` | `int` | `1` | Number of consecutive iterations the trigger must hold before the stage exits. |
| `path_integrand_schedulers` | `dict` | `None` | Schedulers on `path_integrand_scales`. Useful for ramping a geodesic term down as a pVRE term ramps up. See [Advanced](advanced.md). |
| `ts_time_loss_names`, `ts_time_loss_scales`, `ts_time_loss_schedulers` | various | `None` | Optional losses applied at the predicted transition-state time. See [Advanced](advanced.md). |

#### Scheduler dict

A scheduler entry has the form:

```yaml
path_integrand_schedulers:
  pvre:
    value: 1.0
    name: cosine          # or 'linear'
    start_value: 1.0
    end_value: 0.0
    last_step: 99
```

The `name` selects the scheduler class (`linear` or `cosine`); the
remaining keys are passed to its constructor. Schedulers step once
per optimization iteration.

## Shipped example: `gg3.yaml`

The current production config for the 13-atom organic `gg3` reaction
(UMA potential):

```yaml
initialization_params:
  images: configs/gg3.xyz
  path_params:
    name: mlp
    width: 128
    depth: 2
    activation: gelu
  track_loss: true
  track_ts: true
  num_record_points: 101
  device: cuda
  dtype: float32
  seed: 0
optimization_params:
  - potential_params:
      name: repel
    integrator_params:
      path_integrand_names: geodesic
      rtol: 0.5
      atol: 0.05
    optimizer_params:
      optimizer:
        name: adam
        lr: 5.0e-3
      threshold: 0.1
    num_optimizer_iterations: 1000
  - potential_params:
      name: uma
      model_name: uma-s-1p1
      task_name: omol
    integrator_params:
      path_integrand_names: pvre_pseudo_huber
      path_integrand_kwargs:
        pvre_pseudo_huber:
          delta: 0.05
      rtol: 0.5
      atol: 2.5e-3
    optimizer_params:
      optimizer:
        name: adam
        lr: 5.0e-3
      threshold: 5.0e-3
    num_optimizer_iterations: 1000
```

Two stages:

1. **Stage 1 (clash resolution).** `repel` + `geodesic`, `threshold =
   0.1`, `atol = threshold/2 = 0.05`. Triggers when `|g|_2` drops
   below 0.1 — a geometric check, not a chemistry one.
2. **Stage 2 (saddle search).** UMA-driven, `pvre_pseudo_huber` with
   $\delta = 0.05$, `threshold = 5e-3`. All numbers come from the
   $F_2^{\text{target}} = 0.05$ derivation chain in
   [Convergence](convergence.md): $\delta = F_2 \cdot 1 = 0.05$,
   `threshold` $= \delta \cdot 2 \cdot \sigma_{\min} \cdot F_2 \approx
   0.05 \cdot 2 \cdot 1 \cdot 0.05 = 5\!\times\!10^{-3}$.

Defaults fill in the rest: `method=gk7`, `tol_mode='l2'`, `patience=1`,
no LR scheduler, `width=128` for the path-MLP.
