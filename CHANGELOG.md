# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-11-15

Code is linted and formatted with ruff, on every commit through a pre-commit hook and on every push in CI.

### Added

- A `lint` GitHub Actions workflow that runs the pre-commit hooks (`ruff check` and `ruff format`) on every push, on every branch.
- `pre-commit` in the `dev` extra. The hook runs `ruff check --fix` (safe fixes only) and `ruff format` on each commit; install it once per clone with `pre-commit install` (see the README).
- `.git-blame-ignore-revs`, so `git blame` skips the repository-wide formatting commit.

### Changed

- The whole repository, including the Python examples in `docs/`, is formatted with `ruff format`. Formatting only: no code changes.
- Ruff replaces black and isort: their `pyproject.toml` sections are removed, and the VS Code formatter setting points to the Ruff extension. Ruff's default keeps a bracketed list that ends in a trailing comma one item per line (black was configured to join it).
- The ruff version is set in one place, `.pre-commit-config.yaml` (0.17.4).
- `tests_ongoing/` is excluded from linting.
- `os.path` calls are replaced with `pathlib.Path`.
- `Popcornn` and `MinimaUpdate` default their dictionary and list arguments to `None` and `()` instead of mutable `{}` and `[]`.
- **Breaking:** `BasePotential.__init__` no longer accepts and ignores unknown keyword arguments; an unexpected key in a potential's configuration raises `TypeError`.
- Removed unused arguments: `log_frequency` from `MinimaUpdate.find_minimum`, and `max_neighbors` from the private `_radius_graph_mic`.

### Fixed

- `radius_graph(..., method="fairchem")` raised `NameError`: the lazy `fairchem` imports were in `radius_graph` instead of `_radius_graph_fairchem`, where the names are used.

### Known issues

- `_plot_added_rot_dof` in `popcornn/tools/visualize.py` raises `TypeError` because `_plot_path` no longer accepts `return_contour_vals` (#102).

## [0.2.2] - 2026-11-15

Progress output goes through Python's `logging` module, with new `log_file` and `quiet` options; popcornn no longer calls `print`.

### Added

- `Popcornn` constructor options `log_file` (also write the progress table to a file) and `quiet` (stop printing it to the screen).

### Changed

- Progress output (the per-iteration table and the path-initialization messages) goes through Python's `logging` module on loggers named `popcornn.*`. `Popcornn` still prints the table to stdout by default, once, even when another library has attached a handler to the root logger. The path-model summary is logged once, by `get_path`, at DEBUG, and the path-fit progress every 1000 steps is at DEBUG. A failure to write a table line (for example an encoding error or a closed pipe) no longer stops the run; `logging` reports it on stderr. Calling `initialize_path` or `randomly_initialize_path` directly no longer prints anything unless logging is configured.

### Removed

- The unused `popcornn.tools.logging` class, and leftover debug prints in `initialize_path`.

### Fixed

- The docs for the per-iteration metrics files named a nonexistent `metrics_log_path` argument; the argument is the constructor's `log_path` directory.

## [0.2.1] - 2026-11-15

Moves to padaquad 1.4.0, which fixes out-of-memory kills when padaquad picks the batch size, and therefore to Python 3.11. Also adds per-stage convergence flags and fixes the MACE, LEFTNet and Orb potentials. Pull requests #85, #86 and #87.

### Added

- `Popcornn.optimize_path()` returns one convergence flag per stage (`True` if the `threshold` check fired, `False` if the stage reached `num_optimizer_iterations` or had no `threshold`) and keeps the list as `Popcornn.converged`.

### Changed

- **Breaking:** requires Python 3.11 or newer (was 3.10). padaquad has required Python 3.11 since padaquad#61, so on Python 3.10 an installer can only resolve padaquad 1.3.0, which lacks the memory fix below. CI now tests Python 3.11 and 3.12.
- Dependencies: requires `padaquad >= 1.4.0` (was 1.3.0). padaquad 1.3.0 sized batches by assuming that the memory of one integrand call grows linearly with the number of points. With `integrate_gradient: true` (the default), popcornn computes each point's gradient in one batched backward pass, whose memory grows with the square of the number of points. A `PathIntegrator` built without `max_batch` could therefore try, or choose, a batch far larger than the available memory, and the operating system killed the process without a catchable error; this is what killed the GitHub CI runners. padaquad 1.4.0 fits the measured memory as a quadratic in the number of points.
- `examples/run.py` plots runs on 2D toy potentials as the optimized path and transition state on the potential's contours, colored by energy relative to the transition state, in `popcornn_path.pdf`. It no longer writes `popcornn_path.json` and `popcornn_ts.json`. Atomistic runs still write `popcornn_path.xyz` and `popcornn_ts.xyz`.

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
