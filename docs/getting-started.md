# Getting started

This page walks from a clean machine to a custom reaction.

## 1. Install

```bash
conda create --name popcornn python=3.12
conda activate popcornn

git clone https://github.com/khegazy/popcornn.git
pip install -e ./popcornn
```

`pip install -e` is the editable install — when you pull updates from
the repo, you don't need to reinstall.

If you want to run a machine-learned interatomic potential (MLIP), you
also need to install that potential's package — see
[Potentials](potentials.md).

## 2. Run a built-in example

Two analytic examples ship with the repo, both in 2D so they run on
CPU in seconds:

```bash
cd examples
python run.py --config configs/wolfe.yaml
python run.py --config configs/muller_brown.yaml
```

A real-system example using the UMA MLIP also ships, but takes longer
and needs a GPU plus the UMA install:

```bash
python run.py --config configs/gg3.yaml
```

When `run.py` finishes on an **atomistic** input (anything that came
from an ASE `Atoms` list — including the `gg3` example) it writes:

- `popcornn_path.xyz` — the optimized path as a sequence of frames.
- `popcornn_ts.xyz` — the single frame at the predicted transition
  state.

You can open both in any standard molecular viewer (ASE GUI, VMD,
Avogadro, OVITO).

The 2D toy examples (`wolfe.yaml`, `muller_brown.yaml`) don't write
xyz — `run.py` only emits xyz when the input is `Atoms`. They still
return the optimized path through the Python API, so they're useful
for sanity-checking the optimization machinery.

## 3. Run on your own reaction

The shortest possible custom config (just sets the physical
$F_2$-target and lets the shipped defaults fill in everything else):

```yaml
initialization_params:
  images: my_reaction.xyz       # at least the reactant and product

optimization_params:
  - potential_params:
      name: uma                 # or mace, orb, leftnet, newtonnet, ...
      model_name: uma-s-1p1
      task_name: omol
    integrator_params:
      path_integrand_names: pvre_pseudo_huber
      path_integrand_kwargs:
        pvre_pseudo_huber:
          delta: 0.05           # = F_2_target × ‖Δx‖_lb
    optimizer_params:
      optimizer:
        name: adam
        lr: 5.0e-3
      threshold: 5.0e-3         # = δ × 2 × σ_min × F_2_target
    num_optimizer_iterations: 1000
```

`width=128, depth=2`, `method=gk7`, `tol_mode='l2'`, `rtol=0.5`,
`atol=2.5e-3`, `patience=1` all come from the shipped defaults.

Point `images:` at an `xyz` or `traj` file. The first frame is treated
as the reactant, the last frame as the product. Any frames in between
are intermediate guesses — popcornn will fit the path through them but
won't pin them.

To target a different $F_2^{\text{target}}$, see [Convergence](convergence.md) for the
formulas that convert it into `delta`, `threshold`, and `atol`.

Before you hand atoms to popcornn:

- The reactant and product must use the **same atom ordering**. Atom 1
  in the reactant is the same atom as atom 1 in the product.
- The two structures must be **rotationally and translationally
  aligned**. The path doesn't include rigid-body motion — it's
  configuration-space only.
- For periodic systems, the product should be **unwrapped** with
  respect to the reactant. By default popcornn does this for you using
  the minimum-image convention, but if any atom moves more than half a
  cell during the reaction, unwrap manually and pass
  `unwrap_positions: false` in `initialization_params`.

For chemistry with covalent-bond rearrangements, add a clash-resolution
warm-up stage first:

```yaml
optimization_params:
  - potential_params: {name: repel}
    integrator_params:
      path_integrand_names: geodesic
      rtol: 0.5
      atol: 0.05
    optimizer_params:
      optimizer: {name: adam, lr: 5.0e-3}
      threshold: 0.1
    num_optimizer_iterations: 1000
  - # ... MLIP saddle-search stage as above
```

Stage 1's `(threshold=0.1, atol=0.05)` are the geodesic-warm-up
counterpart to the chemistry stage's `(threshold=5e-3, atol=2.5e-3)`
— same `atol = threshold/2` rule, just at a looser geometric target.

## 4. Use the Python API directly

The `run.py` script is a thin wrapper. If you want full control,
import the `Popcornn` class:

```python
from ase.io import read, write
from popcornn import Popcornn

images = read("my_reaction.xyz", index=":")
for image in images:
    image.info = {"charge": 0, "spin": 1}  # if your MLIP needs it

mep = Popcornn(
    images=images,
    path_params={"name": "mlp", "width": 128, "depth": 2},
    track_loss=True,    # so stage-end log shows loss
    track_ts=True,      # so stage-end log shows barrier and |F|
)

mep.optimize_path(
    {
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

final_images = mep.get_discrete_path()
ts_image = mep.get_ts()
write("popcornn_path.xyz", final_images)
write("popcornn_ts.xyz", ts_image)
```

This is the same recipe the YAML driver uses. Each dict you pass to
`optimize_path` is one **stage** of optimization — you can chain as
many as you like. The example above runs a cheap repulsive pre-step
(geodesic interpolation, fixes atom clashes) before the expensive
MLIP-driven step.

## 5. What happens during optimization

Each stage:

1. Picks up the path from where the previous stage left it.
2. Picks up the potential and the loss you specified.
3. Repeatedly evaluates a path integral of the loss along the path,
   gets a gradient with respect to the path's neural-network
   parameters, and steps Adam.
4. Stops when the integrated gradient L2 norm `|g|_2` stays below
   `threshold` for `patience` consecutive iterations, or when
   `num_optimizer_iterations` is reached.
5. Runs one diagnostic integration with `track_loss + track_ts` on
   so the stage-end log line carries the loss integral, barrier, and
   `|F|` at the predicted TS.

For more on the convergence criterion and how to tune it, see
[Convergence](convergence.md).

## 6. Where to go next

- [Concepts](concepts.md) for the physics background.
- [Configuration reference](configuration.md) for every key in the
  YAML config.
- [Potentials](potentials.md) to install a particular MLIP.
- [Memory & OOM](memory-and-oom.md) if your run crashes with CUDA OOM.
