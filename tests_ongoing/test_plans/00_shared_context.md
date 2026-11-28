# 00 — Shared context for the test plans

This folder holds one plan per recommended test (files `01`–`06`). This file collects the facts that apply to all of them, so the individual plans don't repeat each other. Everything here was checked against the code at commit `287709a` (branch `ci_linting`) on 2026-11-14 unless it is marked otherwise.

## Plans in this folder

| File | Test | Priority |
|---|---|---|
| `01_integral_and_gradient_correctness.md` | The path integral and its parameter gradient match exact values | 1 |
| `02_morse_harmonic_potentials.md` | Morse and harmonic energies and forces match independent references | 2 |
| `03_initialize_path.md` | Fitting the path to intermediate images | 3 |
| `04_atomic_end_to_end.md` | Two-stage `Popcornn` run on a molecule, returning ASE `Atoms` | 4 |
| `05_term_schedulers.md` | Per-term scale schedulers, once reinstated | 5 |
| `06_ts_time_loss.md` | Transition-state (TS) time loss, once reinstated | 6 |

Plans 05 and 06 cover features that are currently commented out of `PathOptimizer`. The decision (2026-11-14) is that both features will come back, so their tests are planned now and get written when the features return.

## Related issues

- #89: `PathIntegrator`'s per-point gradient uses O(N²) memory. This is relevant to test cost (see "Cost" below).
- #90: `PathOptimizer` silently ignores unknown `optimizer_params` keys. It blocks plans 05 and 06.
- #91: `initialize_path` returns silently when the fit does not converge. It is plan 03's error-path test.
- #92: `popcornn/tools/tests.py` is shipped and broken.
- #93: Size of `tests_ongoing/`.
- #94: Morse and harmonic return float64 energies in float32 runs. Test 02 exposes it.
- #95: `get_schedulers` modifies its config dict, and `last_step=1` divides by zero. Relevant to test 05.
- #96: `get_discrete_path()` returns `Atoms` without energies. Decided 2026-11-27: intended, and now documented; test 04 asserts nothing about those energies.
- #97: padaquad reports convergence while the `pvre` gradient is still off by about 2%. This is why test 01 needs tight tolerances.

## How to run

- **Environment.** Use the conda env at `~/.conda/envs/popcornn` (`~/.conda/envs/popcornn/bin/python -m pytest ...`). The system `python3` does not have popcornn installed.
- **Paths.** pytest works from any directory: `pyproject.toml` puts the repository root on `pythonpath`, and tests open data as `str(IMAGES_DIR / "<file>")` with `from tests import IMAGES_DIR` (never a working-directory-relative `"tests/images/..."`). Test ids name data files by file name.
- **Pytest settings.** These come from `pyproject.toml`: `testpaths = ["tests"]`, `--import-mode=importlib`, `--strict-markers` and `xfail_strict = true`.
- **Shared machinery** (`tests/conftest.py`, `tests/__init__.py`, tested in `tests/test_conftest.py`):
  - **Per-test seed.** An autouse fixture calls `torch.manual_seed(TEST_SEED)` (0) before every test, so no test depends on which tests ran before it. Don't start a test with `torch.manual_seed(0)`; reseed only to choose a different state.
  - **Pinned `max_batch`.** A session-wide guard raises if a `PathIntegrator` is built with `max_batch=None`. Only a test of padaquad's batch-size search may opt out, with `@pytest.mark.unpinned_max_batch`.
  - **Fixtures.** `device` (cpu, and cuda when available) and `dtype` (float32, float64) replace per-test parametrize blocks; a test's own `parametrize` overrides them. `cheap_integrator_params` holds the settings of tests that check wiring rather than accuracy: `rtol=atol=1e-2`, `max_batch=500`, `runtime_checks=True` and `initial_mesh=starting_mesh(TEST_SEED)`. Extend it only by merging into a new dict, `PathIntegrator(**{**cheap_integrator_params, "path_integrand_names": "pvre"})`.
  - **`runtime_checks=True` in tests.** popcornn's default is False, so a NaN or Inf from a potential would reach the gradient silently; tests turn the checks on (except the two that test that default).
  - **Fixed starting mesh.** `starting_mesh(seed)` returns padaquad's own starting mesh for `torch.manual_seed(seed)` (12 points on [0, 1]), drawn from a local generator so torch's global one is untouched. Pass it as `initial_mesh` so padaquad draws no random starting mesh; numerical tests sweep `starting_mesh(0..9)` and set tolerances from the worst case.
  - **Markers.** `slow` (more than about 10 s of CPU time; CI runs `pytest -m "not slow"`, a plain local `pytest` runs everything) and `unpinned_max_batch`.
- **Lint.** ruff runs through pre-commit (#101): every test file starts with `from __future__ import annotations`, `print` is not allowed, and the flake8-pytest-style rules apply. Run `pre-commit run --files <files>`.
- **No iteration-count assertions.** At the examples' tolerances the gradient-norm stopping test is dominated by quadrature noise, so the number of iterations varies several-fold with the seed and with rounding-level changes (padaquad version, GPU). Assert flags, shapes and values with tolerances instead.
- **Versions.** Python ≥ 3.11; `padaquad >= 2.0.0, < 3`. Gauss–Kronrod panels hold only their interior nodes: 7 points for `gk7`, 21 for `gk21`. With `integrate_gradient=False`, `max_batch` must be at least that.

## API facts the old `tests_ongoing` tests got wrong

These all come from the 0.2.0 CHANGELOG ("Changed" and "Removed") and the current code:

- **`optimize_path`.** `Popcornn.optimize_path(*stages)` returns `list[bool]` (one convergence flag per stage). It does not return `(path_output, ts_output)` and does not accept `output_ase_atoms`. Use `get_discrete_path()` and `get_ts()` instead.
- **MLP width.** The MLP path takes `width` (default 128), not `n_embed`.
- **Potential name.** `potential_params` needs `name:`, not `potential:`. `Popcornn._optimize` calls `get_potential(images=..., **potential_params, ...)`, and `get_potential(name, **kwargs)` requires `name`.
- **Removed `Popcornn` argument.** `Popcornn(...)` no longer takes `output_dir`.
- **Integrator settings.** `PathIntegrator` has no `computation` or `sample_type` setting, and `method` takes padaquad rule names (default `gk7`, also `gk21`), not `dopri5`. `PathIntegrator` has no `**kwargs`, so unknown keys raise a `TypeError`. `PathOptimizer` silently drops them (#90).
- **Removed features.** `Metrics`, `growing_string` and `energy_weight` are gone.
- **Repel and LJ cutoff.** `repel` and `lennard_jones` now default to `cutoff=None`; it was 3.0.

## Integrator behavior every numerical test must account for

- **Loose defaults.** `PathIntegrator` defaults to `rtol=1.0, atol=0.0`. padaquad ≥ 1.1 accepts a panel when its error is below `max(atol, rtol·‖I‖)`, where ‖I‖ is the norm of the current integral estimate. With the defaults, a panel is accepted if its error is smaller than the integral itself, which is very loose. In my runs this left errors of about 1e-2 in the gradient (see plan 01). **Every numerical test must set `rtol` and `atol` explicitly.** On a real molecule (T1x with Morse, `tests/test_default_tolerances.py`, skipped for now) the defaults get the gradient's direction nearly right (cosine ≥ 0.9997 over 10 starting meshes) but its size wrong by up to 126%, and after 20 Adam steps the path is 15% of its movement away from a tight-tolerance run (measured 2026-11-27, padaquad 2.0.0; plan 00, Fact 9).
- **Batch size.** Without `max_batch`, padaquad ≥ 2.0.0 does not measure memory: it calls the integrand on 10, 100 and 1,000 points, then grows the batch 10× per pass until it runs out of memory and backs off. The batch it settles on depends on the machine, and a different batch changes results at the rounding level, so every test pins `max_batch` (the conftest guard enforces it). Choose the smallest round value that holds the test's largest pass in one batch (results then equal padaquad's own choice bit for bit), is at least one panel, and, with `integrate_gradient=True`, satisfies `max_batch² × width ≤ 3.2e7` (MLP width; about 2 GiB in float64, since popcornn's per-point gradient needs memory ∝ N², CI_MEMORY_BLOWUP eq. 6, #89). A pinned value bypasses padaquad's out-of-memory retry.
- **Warm start.** `PathIntegrator` passes each solver's previous `IntegrationState` (mesh and batch size) to its next call (padaquad >= 2.0.0; held in `_state` and `_loss_state`), so each solver reuses its previous mesh and batch size between calls. Two calls with the same integrator object are therefore *not* independent. When comparing two settings, build a fresh `PathIntegrator` for each. The first call of each solver starts from `initial_mesh` when it is given (padaquad then draws no random numbers), otherwise from a random mesh drawn from torch's global generator.
- **Loss tolerance.** `track_loss=True` integrates the scalar loss with its own tolerances, `loss_rtol=0.0, loss_atol=1e-4` by default. **`loss_atol` does not guarantee the loss's accuracy for kinked `pvre` integrands** (#97). On the zero-weight sphere path (exact L = 2), over padaquad's starting meshes for seeds 0–9 (the same meshes as `starting_mesh(0..9)`): 2 of 10 missed `loss_atol=1e-4` and `1e-6` (by up to 4.3×), and 3 of 10 stayed off by about 4e-6 even at `loss_atol=1e-8` (about 430×) while padaquad reported convergence. Identical on padaquad 1.3.x, 1.4.1 and 2.0.0. With `integrate_gradient=False`, `loss_atol` is ignored: the loss is the main integral and follows `rtol`/`atol`. A tolerance measured at one starting mesh is therefore not established; sweep several.
- **Cost of `integrate_gradient`.** `integrate_gradient=True` (the default) is much slower than `False` on atomic systems: LJ13 took 21.8 s versus 0.7 s at `rtol=1e-6`. On `hard.xyz` with Morse, it took 13–20 s per iteration. This matches #89. Tests that aren't about the gradient-integration mode itself should use `integrate_gradient=False` to stay fast.

## What `tests_ongoing/` contains (corrected)

- **Benchmark JSONs.** The `**/benchmarks/*.json` files are **not empty**. There are 28 tracked files, about 55 MB in total, holding reference trajectories from the pre-0.2.0 torchpathdiffeq integrator. (My first review wrongly called them 0 bytes because I measured them with `wc -l`, which counts newlines; these JSONs are single lines.) They can't serve as references for the padaquad integrator.
- **Small input to keep.** `configs/xyz/hard.xyz` is the one small input worth copying into `tests/images/` (17 atoms, C5H10O2, 2 frames). Plan 04 uses it.
- **Code changes to schedulers and the TS loss.** The commented-out scheduler and TS-time-loss code in `optimizer.py` was disabled in commit `aaec76a` (#82). The pre-migration version is `popcornn/optimization/path_optimizer.py` at `aaec76a^`, which is the reference for plans 05 and 06.

## Confidence

- **Sure:**
  - Every API fact above, which I read from the current code.
  - The measured timings and accuracies, which come from runs on 2026-11-14 on a Perlmutter login node, CPU, float64 unless noted.
  - **Negative result:** `loss_atol` does not guarantee the loss's accuracy on kinked integrands (seed sweeps on padaquad 1.3.x to 2.0.0). Likely mechanism, a conjecture: Gauss–Kronrod integrates the linear pieces exactly, and for some positions of the corner inside a panel the Gauss and Kronrod estimates agree, so the error estimate collapses.
- **Not tested:**
  - GPU behavior.
