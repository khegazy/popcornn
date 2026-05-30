# Potentials

A *potential* in popcornn is an energy model: given atomic positions
it returns an energy (and ideally a force). Popcornn ships with a mix
of analytic toy potentials and wrappers around external machine-learned
interatomic potentials (MLIPs).

You select one with `potential_params.name` in your YAML config:

```yaml
optimization_params:
  - potential_params:
      name: uma            # <--
      model_name: uma-s-1p1
      task_name: omol
    ...
```

## Built-in analytic potentials

These work out of the box, no extra installs needed. Useful for
sanity-checking the optimization machinery on cheap 2D / few-atom
systems.

| `name` | What it is |
| --- | --- |
| `wolfe_schlegel` | 2D Wolfe–Quapp surface. Two minima, two saddles. |
| `muller_brown` | 2D Müller–Brown. Three minima, two saddles. Classic NEB benchmark. |
| `schwefel` | 2D Schwefel surface. |
| `sphere` | 2D quadratic bowl. |
| `constant` | Returns zero energy. Useful for testing. |
| `harmonic` | N-dimensional harmonic oscillator. |
| `lennard_jones` | All-atom Lennard-Jones. |
| `morse` | All-atom Morse pair potential. |
| `repel` | All-atom soft-repulsive potential. **Use this with `path_integrand_names: geodesic` as a pre-step to fix atom clashes.** |

## Machine-learned interatomic potentials

Each MLIP wraps an external package. The popcornn package itself
**does not install them** — you have to install the relevant package
separately into your `popcornn` conda environment.

| `name` | Package | Install pointer |
| --- | --- | --- |
| `uma` | `fairchem-core` ≥ 2.4.0 | Already pinned in `pyproject.toml`. For the UMA model weights you need a HuggingFace token; see [Rowan's setup guide](https://rowansci.com/blog/how-to-run-open-molecules-2025). |
| `mace` | [MACE](https://github.com/ACEsuit/mace) | `pip install mace-torch`. Pass a path to a `.pt` checkpoint as `model_path`. |
| `orb` | [Orb](https://github.com/orbital-materials/orb-models) | `pip install orb-models`. |
| `leftnet` | [LEFTNet](https://github.com/yuanqidu/LEFTNet) | Install per upstream README. |
| `newtonnet` | [NewtonNet](https://github.com/THGLab/NewtonNet) | `pip install newtonnet`. |
| `chgnet` | [CHGNet](https://github.com/CederGroupHub/chgnet) | `pip install chgnet`. |
| `escaip` | [EScAIP](https://github.com/atomistic-machine-learning/escaip) | Install per upstream README. |
| `ani` | [TorchANI](https://github.com/aiqm/torchani) | `pip install torchani`. |

If a potential isn't on this list, you'd need to add it — see "Adding
your own" below.

## Per-potential keys

`potential_params` always takes `name` as required. The rest are
potential-specific:

### `uma`

| Key | Description |
| --- | --- |
| `model_name` | UMA checkpoint name, e.g. `uma-s-1p1`. |
| `task_name` | UMA task, e.g. `omol`. |

### `mace`

| Key | Description |
| --- | --- |
| `model_path` | Path to a `.pt` checkpoint. |

### Toy potentials

The 2D analytic potentials (`wolfe_schlegel`, `muller_brown`,
`schwefel`, `sphere`, `constant`) take no extra keys.

### Force-field-style potentials

`repel`, `morse`, `harmonic`, `lennard_jones` accept their respective
parameters; see the source under `popcornn/potentials/` or the
auto-generated [Code reference](reference/popcornn.md).

## Adding your own potential

A potential is a `torch.nn.Module` subclass of
`popcornn.potentials.base_potential.BasePotential`. Two methods to
implement:

```python
from popcornn.potentials.base_potential import BasePotential, PotentialOutput

class MyPotential(BasePotential):
    def __init__(self, my_param=1.0, **kwargs):
        super().__init__(**kwargs)
        self.my_param = my_param

    def forward(self, positions):
        # positions: [N, 3 * n_atoms] (or [N, n_dim] for toy systems)
        # return PotentialOutput(energies=..., forces=...)
        energies = ...                                    # shape [N, 1]
        forces = self.calculate_conservative_forces(      # autograd
            energies, positions
        )
        return PotentialOutput(energies=energies, forces=forces)
```

Then register it in `popcornn/potentials/__init__.py` by adding a new
branch to `get_potential`:

```python
elif name == "my_potential":
    from .my_potential import MyPotential
    return MyPotential(**kwargs)
```

The base class already exposes `calculate_conservative_forces` (and
`calculate_conservative_forces_decomposed` for losses that need it),
so for energy-only models you only need to compute the energy and the
forces come out of autograd.

## Why a separate `repel` pre-step?

MLIPs are trained on physically reasonable geometries. If your
reactant/product alignment puts two atoms 0.3 Å apart, the MLIP will
return garbage energies and the optimization will diverge. The
soft-repulsive `repel` potential, run with
`path_integrand_names: geodesic`, performs **geodesic interpolation**:
it warps the path so atoms don't pass through each other, without
trying to find a transition state. Use it as a first stage, then swap
in your real MLIP for the second stage. The shipped `gg3.yaml` config
follows exactly this pattern.
