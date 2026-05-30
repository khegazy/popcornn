# Advanced

Topics beyond the first-time user path.

## Multi-stage optimization

`Popcornn.optimize_path` accepts an arbitrary number of stage dicts.
Each stage picks up the path from where the previous one left off, then
swaps in its own potential, loss, and optimizer settings. The
canonical pattern is **clash-resolution then MLIP-driven saddle search**:

```python
mep.optimize_path(
    {
        # stage 1: cheap repulsive geodesic interpolation
        "potential_params": {"name": "repel"},
        "integrator_params": {
            "path_integrand_names": "geodesic",
            "rtol": 0.5, "atol": 0.05,
        },
        "optimizer_params": {
            "optimizer": {"name": "adam", "lr": 5.0e-3},
            "threshold": 0.1,
        },
        "num_optimizer_iterations": 1000,
    },
    {
        # stage 2: MLIP-driven TS search
        "potential_params": {
            "name": "uma",
            "model_name": "uma-s-1p1",
            "task_name": "omol",
        },
        "integrator_params": {
            "path_integrand_names": "pvre_pseudo_huber",
            "path_integrand_kwargs": {
                "pvre_pseudo_huber": {"delta": 0.05},
            },
        },
        "optimizer_params": {
            "optimizer": {"name": "adam", "lr": 5.0e-3},
            "threshold": 5.0e-3,
        },
        "num_optimizer_iterations": 1000,
    },
)
```

You can chain more stages — switch loss functions mid-optimization,
step the learning rate down across stages, swap potentials. The path's
network parameters are persistent state on the `Popcornn` instance.

### Choice of stage-2 loss

The shipped recipe uses `pvre_pseudo_huber` at the
$\delta = 0.05$ derived from $F_2^{\text{target}}$
(see [Convergence](convergence.md)). The pseudo-Huber form is smooth
at the saddle so adaptive Gauss–Kronrod doesn't have to refine across
the kink that plain `pvre` has there. For most reactions, no further
multi-stage loss-schedule is needed.

When you might want a separate warm-up:

- **`pvre_squared`** has a $C^\infty$-smooth integrand and a
  gradient $\propto 2(v\!\cdot\!F)$ that drives the path quickly toward
  the MEP but plateaus near the saddle. Sometimes useful as a stage-1
  warm-up before `pvre_pseudo_huber` — but in practice the
  pseudo-Huber's small-$|s|$ quadratic regime already covers what
  `pvre_squared` was for, so this combination is rarely shipped.
- **`pvre`** (the sharp $L_1$ form) keeps pushing once warm-started
  but pays for adaptive-quadrature refinement around its sign-kink.
  Prefer `pvre_pseudo_huber` unless you specifically want the
  sign-driven behavior.

## Schedulers

Three independent scheduler families are available per stage. Each is
a dict whose keys name what's being scheduled and whose values
configure the scheduler.

### `lr_scheduler` — learning rate

Any class from `torch.optim.lr_scheduler` (case-insensitive). Steps
once per optimization iteration.

```yaml
optimizer_params:
  optimizer:
    name: adam
    lr: 5.0e-3
  lr_scheduler:
    name: cosineannealinglr
    T_max: 1000
    eta_min: 1.0e-5
```

### `path_integrand_schedulers` — per-loss-term weights

Multiplies entries of `path_integrand_scales` (in `integrator_params`)
by a schedule. Useful for ramping one term down while another ramps up.

```yaml
integrator_params:
  path_integrand_names: ['pvre', 'vre']
  path_integrand_scales: [1.0, 0.1]

optimizer_params:
  path_integrand_schedulers:
    pvre:
      value: 1.0
      name: cosine
      start_value: 1.0
      end_value: 0.0
      last_step: 99
    vre:
      value: 1.0
      name: cosine
      start_value: 0.0
      end_value: 1.0
      last_step: 99
```

This config ramps the pVRE term from full weight to zero, and VRE
from zero to full weight, over the first 100 iterations.

Available scheduler types:

| `name` | Behavior |
| --- | --- |
| `linear` | Linear interpolation from `start_value` to `end_value` over `last_step`. |
| `cosine` | Cosine-anneal from `start_value` to `end_value` over `last_step`. |

## Transition-state losses

`ts_time_loss_names` / `ts_time_loss_scales` apply an extra loss at
the predicted TS time, useful e.g. for minimizing the force magnitude
at the TS (`F_mag` as a TS-time loss). The scales can be scheduled
with `ts_time_loss_schedulers`.

The TS itself is picked by `BasePath.ts_search`: linearly interpolate
$t$ at **every** interior `dE/dt = 0` sign change on the
per-quadrature-point sample cache the integrator already collects,
append the two endpoints ($t = 0$ and $t = 1$) as candidates, evaluate
the path at every candidate in one batched forward, and pick the
candidate with the highest model-truth energy. Two consequences:

- On wiggly paths (high-capacity MLP, adversarial quadrature,
  multi-basin landscapes) where there are multiple interior `dE/dt = 0`
  brackets, the model-truth ranking picks the rate-limiting saddle.
- For a **barrierless** reaction (no interior maximum), the
  argmax-over-candidates correctly picks the higher-energy endpoint,
  and the reported barrier is $\max(E_R, E_P) - E_R$ — either zero
  (exoergic) or $E_P - E_R$ (endoergic).

The same call also populates `path.ts_time`, `path.ts_energy`,
`path.ts_force`, `path.ts_force_mag`, and `path.barrier`.

## Logging per-iteration state

Every run prints a sparse per-iter table to stdout (one header at
stage start, one row per iter, plus the convergence + stage-end
lines). Columns: `iter / step_s / loss / grad / barrier / force`.
This is on by default — no setup needed.

For a programmatic record of the same scalar metrics, pass
`metrics_log_path=<file>` to `optimize_path`. The logger writes one
JSONL row per iteration to that exact path with fields:

```
iter, loss, grad_norm, lr, step_s, wall_s, converged, barrier, ts_force_norm
```

Rows are flushed each iteration so a killed run still leaves a valid
file. The `barrier` and `ts_force_norm` fields are non-null only when
`track_ts=True` was passed to the `Popcornn` constructor; the `loss`
field is non-null only when `track_loss=True` was passed.

After the optimizer-loop trigger fires, popcornn runs **one diagnostic
integration with `track_loss + track_ts` on** and reports `loss`,
`barrier`, and `|F|` on the stage-end line:

```
stage 1 done  iters=143  time=339.2s  time/iter=2.37e+00s  barrier=3.45  ts_force_norm=0.07
```

This is the most reliable single number for "how converged did this
stage actually get".

## Custom path integrands

To add a new per-point loss term, edit `popcornn/tools/integrand.py`,
write a subclass of `PathIntegrand`, and register it in
`PATH_INTEGRANDS`:

```python
class MyLoss(PathIntegrand):
    requires = ('forces', 'velocities')   # cache keys this term consumes

    def evaluate(self, variables) -> torch.Tensor:
        # return shape [N, 1]
        return ...

PATH_INTEGRANDS['my_loss'] = MyLoss
```

Then reference it from any config as `path_integrand_names: my_loss`.

## Custom path / potential classes

See [Paths](paths.md) and [Potentials](potentials.md).
