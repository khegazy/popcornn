# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `initial_mesh` in `integrator_params`: the starting mesh of each integral's first call (increasing times, for example `[[0.0], [0.3], [0.7], [1.0]]`, cut or extended to the integration bounds). Later calls start from the previous call's refined mesh as before. The default, `null`, keeps padaquad's random starting mesh, so a run then depends on the random seed and on how padaquad draws that mesh; a fixed mesh removes both. The test suite uses one.

### Changed

- `initialize_path`'s "Finished path initialization after N iterations" message counts the Adam steps taken; before, it showed one less than the steps taken whenever the fit took at least one step. Its DEBUG progress line no longer shows the relative change of the loss. A negative `max_steps` raises `ValueError`. The fit threshold is the module constant `popcornn.optimization.initialize_path.INIT_LOSS_THRESHOLD` (1e-8, unchanged).
- `initialize_path` raises `ValueError` when its loss is NaN or Inf (a NaN or Inf target or path output); before, it ran all `max_steps` steps and returned NaN parameters with only a warning. Targets whose shape differs from the path's predictions also raise `ValueError` instead of `AssertionError`.
- `process_images` raises `ValueError` when the frames differ in shape, atomic numbers, pbc, cell, constraints, tags, charge or spin. Its checks could never fail before (`np.all` of a generator is always true), so such input was accepted and the first frame's values were used.

### Fixed

- `MorsePotential` and `HarmonicPotential` return energies in the configured `dtype`; in float32 runs they returned float64 energies (#94).
- `initialize_path`, which `Popcornn` runs when given more than two images, no longer takes one more Adam step after its loss falls below the threshold, so it returns the path that met it. Before, that step could leave the path above the threshold (in a 60-seed sweep at the default width, 3 runs ended above 1e-8, the worst at 1.75e-8). Runs started from intermediate images therefore give slightly different paths.
- `initialize_path` logs a WARNING when it reaches `max_steps` before its loss falls below the threshold; it returned silently (#91).
- Frames returned by `get_discrete_path()` and `get_ts()` for an input with fixed atoms carry the input's `FixAtoms` indices. They were built from the per-coordinate mask, so atom i became indices 3i to 3i+2, and writing such a frame (for example to extxyz, as `examples/run.py` does) raised `IndexError`. The optimization itself was not affected.
- A stage with `num_optimizer_iterations: 0` no longer raises `ZeroDivisionError` in its stage-end log line; it runs only the stage-end diagnostic and reports `time/iter=nan`.

## [0.4.0] - 2026-11-26

Moves to padaquad 2.0.0, which no longer keeps a warm-start mesh and batch size on its solver and no longer runs a memory benchmark. popcornn now passes each solver's state from one integration to the next itself.

### Added

- padaquad's four memory options in `integrator_params`, passed to both of `PathIntegrator`'s solvers: `trade_speed_for_memory` (default `true`; padaquad's default is `false`), `evaluation_buffer` (default 5 GiB; padaquad's default is `null`), `result_device` (default `cpu`) and `record_evaluations` (default `true`). They decide where and when integrand values are held during an integral and never change the evaluated points, the mesh or the integral. With `trade_speed_for_memory: true` the GPU holds at most the buffer plus one call's values instead of a refinement pass's, which matters when the gradient is very large, as in catalysis with many atoms kept in the path. `trade_speed_for_memory` and `evaluation_buffer` apply only with `integrate_gradient: true` and are ignored otherwise. The values are always recorded while `track_ts` is on, because the transition-state samples are read from them. See [Memory & OOM](docs/memory-and-oom.md#memory-options).
- The README's examples table gives each example's runtime, and the README gives the time `pip install` takes (about 2.5 min).
- padaquad's `runtime_checks` in `integrator_params`, passed to both solvers, default `false` (padaquad's default is `true`). `true` raises naming the points where the integrand is NaN or Inf and checks padaquad's own bookkeeping once per call; `false` skips these checks, which saves time and temporary arrays as large as a call's output.

### Changed

- **Breaking:** requires `padaquad >= 2.0.0, < 3` (was `>= 1.4.0`). padaquad 2.0.0 removed `same_f` and `solver.previous_max_batch`, which `PathIntegrator` used, and `integrate` now returns `(result, state)`. popcornn 0.4.0 does not run on padaquad 1.x, and popcornn 0.2.0 to 0.3.0 do not run on padaquad 2.x. Their requirements (`padaquad >= 1.3.0` or `>= 1.4.0`) do not exclude 2.x, so install them with `padaquad < 2`.
- `PathIntegrator` passes the `IntegrationState` each solver returned (its refined mesh and batch size) to that solver's next call, which keeps the warm start of earlier releases. A size reduced after an out-of-memory error is still kept for the rest of the stage. The returned `IntegrationResult` no longer has `mesh_optimal` (padaquad moved it to the state).
- With `max_batch: null`, padaquad no longer measures memory. It calls the integrand on 10, 100 and 1,000 copies of one point, starts from the largest call that ran, then grows the batch by factors of 10 until a call runs out of memory and retries that call at 0.75 times its size until it fits. Expect a few "Caught OOM" warnings in the first integration of a stage. On a CPU, pin `max_batch`: an allocation that does not fit can get the process killed instead of raising an error.
- The `track_loss` integral finds its own batch size instead of starting from the gradient integral's, which needs far more memory per point (and, once reduced after an out-of-memory error, would never grow again).
- With `integrate_gradient: false`, an out-of-memory error in the integrand or in a backward pass is retried with smaller batches instead of raised. popcornn passes the path's parameters to padaquad (`parameters=`), so the retry also works when a potential uses a custom autograd function.
- **A NaN or Inf integrand value no longer raises by default.** padaquad 1.x raised a `ValueError` naming the points; with popcornn's default `runtime_checks: false` the value is accepted and reaches the gradient. Set `runtime_checks: true` to get the error.
- The Gauss–Kronrod methods evaluate only their interior nodes (padaquad 2.0.0 no longer adds a zero-weight node at each panel end): a `gk7` panel (the default) costs 7 integrand evaluations instead of 9, and `gk21` 21 instead of 23. The integrals are unchanged. The transition-state samples no longer include `t = 0`, `t = 1` or the panel boundaries.
- The per-panel arrays of padaquad's result (nodes, tracked variables) are stored on the CPU by default (`result_device: cpu`) instead of the integration device; the transition-state samples are moved back to the integration device as before.
- Seeded runs do not reproduce earlier releases' iteration counts: padaquad 2.0.0 evaluates other points and sums in another order, and with the examples' tolerances the iteration at which a run stops is very sensitive to such changes (see Known issues). The paths and barriers are the same.
- `integrate_path` bounds that differ from the previous mesh's ends cut that mesh (a bound inside it) or extend it (a bound outside it), as before, except that padaquad 2.0.0 also drops the mesh points within a few rounding steps of a new bound; no caller in popcornn changes the bounds.

### Removed

- The `rost50.yaml` example (a 71-atom reaction with UMA) and its `rost50.xyz`: a run takes several hours (about 36 s per UMA iteration on an A100, and an earlier run had not converged after 648 iterations). `gg3.yaml` remains the UMA example.

### Fixed

- `examples/configs/loss_example.yaml` failed at start-up: it used integrand schedulers and transition-state losses, which `PathOptimizer` currently disables. It now combines `pvre` and `vre` with fixed scales.
- popcornn imported `einops` without listing it as a dependency; it came with padaquad 1.x, which padaquad 2.0.0 no longer depends on, so `import popcornn` failed in a fresh environment. The two uses in `BasePath` are replaced by `torch.flatten` and `torch.unflatten`, and popcornn no longer needs `einops`.
- The `tests` CI workflow installed `tests/requirements.txt`, which no longer exists.
- `PathOptimizer`'s docstring gave `patience` a default of 5; the default is 1.

### Known issues

- With the examples' integrator tolerances (`rtol: 1.0`, `atol: 0`), the measured gradient norm that the convergence test compares with `threshold` can be off by more than the threshold, so how many iterations a run takes is largely chance. On an already converged Müller–Brown path whose accurate gradient norm is 0.070, the examples' settings measured 0.06 to 15.7 depending on the random starting mesh, below the threshold of 0.1 in half the cases; over seeds 0–9 on a CPU, Müller–Brown took 48 to 504 iterations to the same barrier. Compare iteration counts and runtimes over several seeds, not single runs. The examples also do not use the tolerances `docs/convergence.md` recommends (`atol` = threshold/2, `rtol: 0.5`), which narrow the spread but do not remove it.

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
