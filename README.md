# Popcornn

**Pop**cornn is a tool for finding how a chemical reaction gets from a
starting structure (the **reactant**) to an ending structure (the
**product**). Given those two structures, popcornn fits a smooth path
between them and then bends that path until it crosses the lowest-energy
barrier the chosen energy model can find.

The energy model can be a simple analytic toy (Wolfe, Müller–Brown), a
classical force field, or a modern machine-learned interatomic potential
(MACE, Orb, UMA, …). The path itself is represented by a small neural
network, so the whole thing is differentiable and trains by gradient
descent.

If "lowest-energy barrier" sounds unfamiliar, that's fine — see
[`docs/concepts.md`](docs/concepts.md) for a one-page background.

## Installation

```bash
conda create --name popcornn python=3.12
conda activate popcornn

git clone https://github.com/khegazy/popcornn.git
pip install -e ./popcornn
```

That gets you everything you need for the analytic potentials. Each
machine-learned potential (MACE, Orb, UMA, …) has its own install steps —
see [`docs/potentials.md`](docs/potentials.md).

## Examples

The shipped configs in `examples/configs/` go from analytic 2D toys
to a published organic-chemistry benchmark. Run any of them from the
`examples/` directory with

```bash
python run.py --config configs/<name>.yaml
```

| Config | What it is | Needs | Runtime* |
| --- | --- | --- | --- |
| `wolfe.yaml` | 2D Wolfe–Schlegel analytic surface | nothing beyond the base install | 17 s (opt. 2.2 s, 77 iterations) |
| `muller_brown.yaml` | 2D Müller–Brown analytic surface | nothing beyond the base install | 25 s (opt. 10.5 s, 211 iterations) |
| `loss_example.yaml` | Wolfe–Schlegel with the `pvre` and `vre` integrands combined | nothing beyond the base install | 25 s (opt. 10.1 s, 186 iterations) |
| `lj13.yaml` | 13-atom Lennard–Jones cluster rearrangement | nothing beyond the base install | 30 s (opt. 14.3 s, 174 iterations) |
| `gg3.yaml` | a Grambow–Green organic reaction (3rd entry of the GG benchmark set) | GPU + the UMA model installed (see [`docs/potentials.md`](docs/potentials.md)) | 11 min† (opt. 4.2 s repel stage, 49 iterations; 10.5 min UMA stage, 108 iterations) |
| `rost50.yaml` | a 71-atom reaction | GPU + the UMA model installed | several hours† (opt. 13 s repel stage, 59 iterations; UMA stage about 36 s per iteration over its first 60, not run to the end) |

\*Wall time of `python run.py --config configs/<name>.yaml` on one
NVIDIA A100-SXM4-80GB (NERSC Perlmutter GPU node), popcornn 0.4.0 with
padaquad 2.0.0, measured 2026-11-18. About 15 s of each run is spent
outside the optimization stages (start-up such as imports and CUDA
initialization, and writing the output). "opt."
is the optimization time of the stages and the number of iterations
until the convergence threshold was met. Every run except `rost50`
stopped at its threshold, so the iteration counts, and the times,
depend on the seed and the machine. †Timed without imports or writing
the output (the `Popcornn` constructor and `optimize_path` only); an
earlier `rost50` run had not converged after 648 UMA iterations (4 h).

Each run writes the optimized path and predicted transition state to
`popcornn_path.xyz` / `popcornn_ts.xyz`. To swap in your own reaction,
point `images:` in the config at an `xyz` or `traj` file with at least
the reactant and product structures. The
[Getting Started guide](docs/getting-started.md) walks through this
from scratch.

## Where to go next

- [Getting Started](docs/getting-started.md) — install through first
  custom run.
- [Concepts](docs/concepts.md) — what a reaction path is, what the
  transition state is, why a neural network represents the path.
- [Configuration reference](docs/configuration.md) — every key in the
  YAML config.
- [Potentials](docs/potentials.md) — what's built in and how to install
  the MLIPs.
- [Convergence](docs/convergence.md) — picking `threshold` and
  `patience` for your system.
- [Memory & OOM](docs/memory-and-oom.md) — what to do if you run out of
  GPU memory.
- [Advanced](docs/advanced.md) — multi-stage optimization, schedulers,
  transition-state losses.

## Citing

If you use popcornn in your research, please cite this repository. A
citable release is on the way; until then, link to the GitHub repo.

## Support and contributing

Popcornn is under active development. Open a
[GitHub issue](https://github.com/khegazy/popcornn/issues) for bugs or
questions, and pull requests are welcome.

Before your first commit, install the development tools and the git hook
that lints and formats your changes with ruff on every commit:

```bash
pip install -e "./popcornn[dev]"
cd popcornn && pre-commit install
git config blame.ignoreRevsFile .git-blame-ignore-revs  # git blame skips formatting-only commits
```

To check the whole repository by hand, run `pre-commit run --all-files`.
