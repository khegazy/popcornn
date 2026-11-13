# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Popcornn.optimize_path()` returns one convergence flag per stage (`True` if the `threshold` check fired, `False` if the stage reached `num_optimizer_iterations` or had no `threshold`) and keeps the list as `Popcornn.converged`.

### Fixed

- `MacePotential`, `LeftNetPotential` and `OrbPotential` could not be built: `process_images` stored atomic numbers as int32, which `one_hot` rejects. Atomic numbers are now int64.
- `MacePotential`, `LeftNetPotential` and `OrbPotential` run in the configured `dtype` instead of always float64, and the MACE and Orb models are moved to `device` after loading.

## [0.2.0] - 2026-11-10

Path integration now runs on [padaquad](https://github.com/khegazy/padaquad), and the optimizer integrates the gradient of the loss along the path, stops on the norm of that integrated gradient, and finds the transition state from the integrator's own samples. Pull requests #79, #80, #82 and #83.

### Added

- `PathIntegrator` (`popcornn/tools/integrator.py`), built on padaquad >= 1.3.0, replaces `ODEintegrator`. Settings in `integrator_params`: `method` (default `gk7`), `integrate_gradient` (integrate the per-point gradient, the default, or backpropagate through the loss integral), `rtol`/`atol`, `norm` (`'2'` or `'max'`), `max_batch`, `max_iter` (per-panel refinement cap), `track_loss`, `track_ts`. The mesh and batch size carry over between optimizer steps; a batch size reduced after an out-of-memory error is kept for the rest of the stage.
- Convergence check: a stage stops when the norm of the integrated gradient stays below `optimizer_params.threshold` for `patience` iterations (default 1).
- Transition-state finding from the energies at the integrator's quadrature points (`track_ts`), with no extra path evaluations.
- `Popcornn.get_discrete_path()` and `Popcornn.get_ts()` return the optimized path and the transition state.
- Path integrands, selected by name in `path_integrand_names` with optional `path_integrand_scales` and `path_integrand_kwargs`: `geodesic`, `pvre`, `pvre_mag`, `pvre_squared`, `pvre_huber`, `pvre_pseudo_huber`, `vre`, `vre_error`, `E`, `E_mean`, `F_mag`.
- Documentation pages for the mkdocs site (`docs/`): getting started, concepts, configuration, convergence, loss functions, memory and out-of-memory errors.
- Examples `muller_brown.yaml`, `lj13.yaml`, `gg3.yaml` (UMA) and `rost50.yaml` (UMA), with a sample `gg3` path and transition state.
- `popcornn.tools.traceplot` for plotting per-iteration traces.

### Changed

- **Breaking:** `Popcornn.optimize_path()` no longer returns the path and transition state, and no longer takes `output_ase_atoms`; call `get_discrete_path()` and `get_ts()` instead.
- **Breaking:** the MLP path takes `width` (hidden-layer width, default 128) instead of `n_embed`.
- **Breaking:** `repel` and `lennard_jones` default to no cutoff (`cutoff=None`, was 3.0), and their radius graph uses the minimum-image convention for periodic cells.
- `UMAPotential` runs on the GPU given by `device`, including an index such as `cuda:1`.
- Dependencies: requires `padaquad >= 1.3.0`; `torchpathdiffeq` and `tqdm` are no longer required.

### Removed

- `ODEintegrator`, `Metrics` (`popcornn/tools/metrics.py`) and the torchpathdiffeq solver options (`computation`, `sample_type`).
- The `energy_weight` and `growing_string` loss types; the loss is always the path integral.
- The `rxn0003` example.

## [0.1.0] - 2025-09-05

- First release on PyPI. No changelog entry was written for this release.

## [0.0.2]

### Added

- Added `.pre-commit-config.yaml`

### Changed

- Use `ruff` for formatting

## [0.0.1]

### Added

- The initial release!
