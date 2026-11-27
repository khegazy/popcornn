# 00 — Shared test infrastructure: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Code level:** research-grade (the test suite is kept and shared).

**Checked against:** popcornn `e4a02f4` (#103, release 0.4.0) on branch `onging_tests`, and padaquad 2.0.0, on 2026-11-26. The local editable padaquad is exactly `v2.0.0` with a clean tree, the same version CI installs from PyPI. (pip's metadata in the env still reports 1.4.1, but the code is 2.0.0.) Earlier versions of this plan are summarized in "Revision history".

**Goal:** Every test in `tests/`:
- starts from the same state no matter which tests ran before it;
- finds its data no matter where pytest is started;
- pins `max_batch`;
- uses explicit, justified integrator tolerances;
- integrates from a fixed starting mesh, so padaquad's random draws don't affect it.

On top of that, plans 01–06 get shared fixtures, and a test (written, currently skipped) measures whether `PathIntegrator`'s default tolerances are good enough on a real molecule.

**Tech stack:** Python ≥ 3.11 (CI 3.11 and 3.12), padaquad ≥ 2.0.0 and < 3, pytest 9.0.3, torch 2.8.0 locally (CI's 3.12 job resolves a newer torch), ruff 0.17.4 through pre-commit.

**Spec:** [00_shared_context.md](00_shared_context.md). Background: [CI_MEMORY_BLOWUP.md](../../CI_MEMORY_BLOWUP.md), [PADAQUAD_POPCORNN_TEST_SUMMARY.md](../../PADAQUAD_POPCORNN_TEST_SUMMARY.md) and [PADAQUAD_BATCH_FIT_BRIEF.md](../../PADAQUAD_BATCH_FIT_BRIEF.md), all untracked at the repository root. The last two describe padaquad 1.4's memory benchmark, which 2.0.0 removed.

## Decisions (Kareem, 2026-11-16 to 2026-11-17) and where they are implemented

| # | Decision | Task |
|---|---|---|
| 1 | Set a reasonable `max_batch` for each test, for consistency and to skip padaquad's batch-size search. Report any change in results. | Task 3 (results: Fact 1) |
| 2 | Migrate the copied device/dtype blocks into shared fixtures. | Task 4 |
| 3 | Give every test a suitable, explicit `rtol`/`atol`. Add a test of the default tolerances on a real molecule (Morse on T1x), written now and skipped. | Task 5 (measurements: Fact 9; test written) |
| 4 | A test's starting point must never depend on earlier tests; find the least invasive way. | Task 2 (audit: Fact 7) |
| 5 | Keep the current `slow` plan, with no nightly job. Fix the data paths (in the plan; code on go-ahead). | Task 1 (prototype: Fact 8), Task 3 Step 8 |
| 6 | `PathIntegrator` accepts a fixed starting mesh, and tests use one. | Task 7 (evidence: Fact 10) |
| 7 (G) | (2026-11-26) Tests run with padaquad's `runtime_checks=True`, so a NaN or Inf fails loudly. This applies to tests only: popcornn's default stays False. | Task 4 |

**Why a guard as well as pinning (decision 1), on padaquad 2.0.0.** padaquad 2.0.0 has no memory benchmark. With `max_batch=None` it runs test calls on 10, 100 and 1,000 points, then grows the batch 10× per pass until it runs out of memory (Fact 2). Leaving `max_batch` unset no longer costs time. Three reasons to pin remain:
- **Your consistency rule.**
- **Memory.** popcornn's per-point gradient grows like N² in memory (#89), and an out-of-memory condition on a CPU can kill the process rather than raise.
- **Machine independence.** The batch padaquad finds depends on where it runs out of memory, and a different batch changes results at the rounding level (Fact 1).

The guard (Task 3) enforces the rule for future tests.

## Global constraints

- **Running tests.** Use `~/.conda/envs/popcornn/bin/python -m pytest ...`. After Task 1, this works from any directory.
- **Pytest settings that stay:** `--import-mode=importlib`, `-p no:warnings`, `xfail_strict = true`, `testpaths = ["tests"]`.
- **ruff (#101) applies to `tests/`.** Every new or edited file must pass `pre-commit run --files <files>`. In particular:
  - Every file starts with `from __future__ import annotations`.
  - `print` is not allowed (T20).
  - The legacy `np.random.*` functions are flagged (NPY002).
  - flake8-pytest-style (PT) rules apply.
  - Strings use double quotes.
  - `tests/**` ignores `ARG`, `D`, `E402` and `PTH`.
- **Memory rule for `max_batch`.** With `integrate_gradient=True`, keep `max_batch² × width ≤ 3.2e7`, where `width` is the MLP width.
  - popcornn's per-point gradient still uses `is_grads_batched=True` with an N × N identity (`integrator.py:429-436`, #89). Its memory is $M \approx 8.5\,N^2 w b$ (CI_MEMORY_BLOWUP eq. 6), so the rule keeps one batch at about 2 GiB in float64.
  - A pinned `max_batch` bypasses padaquad's out-of-memory retry, so this rule is the only protection.
- **Panel sizes on padaquad 2.0.0.** Gauss–Kronrod panels hold only their interior nodes: 7 for `gk7` and 21 for `gk21`. With `integrate_gradient=False` (padaquad `take_gradient=True`), `max_batch` must be at least that number, or padaquad raises `ValueError`.
- **Fresh integrators.** Each comparison of two integrator settings builds a fresh `PathIntegrator` (`00`, "Warm start").
- **No iteration-count assertions.** At the example tolerances (`rtol=1.0`), the gradient-norm stopping test is dominated by quadrature noise. The number of iterations varies several-fold with the seed and with rounding-level changes such as the padaquad version or the GPU. No test may assert an iteration count. (This is from your 2026-11-26 measurements: Müller–Brown 48–504 iterations over 10 seeds.)
- **Package code.** Only Task 7 changes package code: `popcornn/tools/integrator.py` and `docs/configuration.md`.
- **Task order.** 1 → 2 → 3 → 4 → 5 → 7 → 6.

## Facts this plan relies on

Re-checked on padaquad 2.0.0 and popcornn `e4a02f4` on 2026-11-26 unless marked.

1. **Pinning `max_batch` doesn't change results when every node fits in one batch.** I compared the three tests that leave it unset against pinned values (scratch script `pin_compare2.py`).

   | Test | Batch padaquad found | Pinned 100 | Pinned 500 | Pinned 1000 |
   |---|---|---|---|---|
   | `test_track_ts_aligned_with_quadrature_mesh` | 231 (273 nodes) | nodes identical; integral, gradient and energies differ by ≤ 2.8e-14 | bit-identical | bit-identical |
   | `test_track_ts_off_yields_none` | 231 (273 nodes) | nodes identical; integral and gradient differ by ≤ 1.4e-14 | bit-identical | bit-identical |
   | `test_optimize_path_returns_ts_image` | (not exposed) | bit-identical | bit-identical | bit-identical |

   - **The differences at 100 are rounding** from summing over several batches.
   - **The padaquad version does change results.** This has nothing to do with batch size or test order. The end-to-end test's `ts_time` with the same seed was 0.28736918540 on padaquad 1.3.x, 0.28735889924 on 1.4.1, and 0.28735897507 on 2.0.0.
2. **padaquad 2.0.0 finds the batch size by growing it, not by benchmarking.**
   - **The search** (`base.py`, the block after "The test calls"; `memory.py:63, 67`): with neither `max_batch` nor `state`, `integrate` calls the integrand on 10, 100 and 1,000 copies of one point, capped at the first pass's size. It starts at the largest size that ran, then grows the batch 10× per pass, and after an out-of-memory error retries at 0.75× its size and stops growing.
   - **The 1.4 benchmark is no longer called.** `_setup_memory_checks` still exists, but only the public `get_max_batch(f=...)` uses it. That also removes the intermittent 60-s benchmark error seen on 2026-11-16.
   - **No time cost any more.** The three unpinned tests took 4.4–8.8 s each unpinned and 4.9–9.5 s pinned at 500, at a load average of about 35. The time saving from pinning is gone; the reasons that remain are in "Decisions".
3. **A guard at the popcornn level works and needs no padaquad internals.** The old guard patched padaquad's `_setup_memory_checks`, which 2.0.0 no longer calls from `integrate`. The new design wraps `PathIntegrator.__init__` and raises `RuntimeError` when `max_batch` is `None`, unless the test opts out with a marker. I prototyped it as a scratch plugin (`guard2.py`) and ran it over the integrating test files. Result: 6 failed, 35 passed, 5 skipped. The 6 are:
   - `test_track_ts_aligned_with_quadrature_mesh` and `test_track_ts_off_yields_none` (no `max_batch`);
   - `test_optimize_path_returns_ts_image` (no `max_batch`);
   - `test_warm_start_passes_previous_state[found]`, which leaves it unset on purpose. Its docstring calls it "the one exception to pinning it in tests";
   - `test_device_and_runtime_checks_reach_padaquad[None]` and `[True]`, which only build the integrator.
4. **Fixture behavior** (checked 2026-11-14 to 2026-11-16; pytest is unchanged at 9.0.3):
   - A test's own `@pytest.mark.parametrize("device", ...)` overrides a `device` fixture from `conftest.py`.
   - `pytest.warns` works under `-p no:warnings`.
   - `--strict-markers` and `-m "not slow"` behave as expected.
   - A session-scoped patch plus a per-test opt-out behaves as Task 3 needs: blocked unless marked, and restored after a marked test.
5. **Duplicated device and dtype blocks** (test files unchanged since 2026-11-16): 15 copies, 6 in `test_images.py`, 5 in `test_potentials.py` and 4 in `test_paths.py`. `test_charge_spin` (`test_images.py:1178`) has only the `device` part.
6. **The tracked loss does not reliably meet `loss_atol` on a kinked integrand.** Zero-weight sphere path, exact L = 2, mesh seeds 0–9, re-measured on 2.0.0 (`seeds2.py`). The values are identical to 1.3.x and 1.4.1.

   | Settings | Seeds over tolerance | Worst \|L − 2\| / tolerance |
   |---|---|---|
   | `loss_atol=1e-4` | 5, 7 | 2.24 |
   | `loss_atol=1e-6` | 1, 9 | 4.29 |
   | `loss_atol=1e-8` | 1, 3, 9 | 429 (\|L − 2\| = 4.3e-6) |
   | `integrate_gradient=False`, `rtol=1e-4` (loss_atol ignored) | none | 0.67 of `rtol`·L |

7. **Order dependence.**
   - **Full-suite audit (2026-11-16).** 16 test cases depend on test order, all in `test_paths.py`: the 12 MLP cases of `test_velocity` (never seeded) and the 4 MLP cases of `test_set_potential` (draw before their reseed at line 575). A per-test seed removes all 16.
   - **The rewritten `test_integrator_samples.py` (#103)** was re-audited on 2026-11-26: nothing flagged, 18 passed.
   - **Redundant seed calls.** Six `torch.manual_seed(0)` calls run before any draw in their test, so the per-test seed duplicates them: `test_integrand.py:20`, `test_integrator_samples.py:25` (the fixture), `test_ts_end_to_end.py:18`, `test_converged_flag.py:45` and `:132`, and `test_paths.py:79`.
   - **Deliberate reseeds that stay:** `test_integrator_samples.py:175, 183, 311`. They reset the random first mesh so two integrations start alike; Task 7 makes them unnecessary.
   - **Other seeds that stay:** the non-zero seeds in `test_integrand.py` and `test_ts_search.py`, and `test_paths.py:575`.
8. **The path fix works from any directory** (prototype 2026-11-16):
   - `pythonpath = ["."]` plus `IMAGES_DIR = Path(__file__).parent / "images"` in `tests/__init__.py` work from the root, from `tests/`, and from elsewhere.
   - `process_images` needs a `str` (`images.py:107`).
   - A `pytest_make_parametrize_id` hook keeps ids as file names.
   - **To replace now:** 37 working-directory-relative strings (15 in `test_potentials.py`, 12 in `test_images.py`, 9 in `test_paths.py`, 1 in `test_default_tolerances.py`). In addition, #103 made `muller_brown_setup` build its path as `Path(__file__).parent / "images" / ...`, which is correct but is a second definition of the image directory. It becomes `IMAGES_DIR`.
9. **Default tolerances on T1x with Morse,** re-measured on 2.0.0 (`default_tol2.py`). The values are identical to 1.4.1 to every reported digit.
   - **Setup.** Gradient over 10 mesh seeds against a tight reference (`gk21`, `rtol=1e-8`, `atol=1e-12`, `max_iter=50`, `integrate_gradient=False`).
   - **The two measures.** The relative error is $\lVert g - g_\text{ref}\rVert / \lVert g_\text{ref}\rVert$, which measures direction and length together. The cosine is $g\cdot g_\text{ref}/(\lVert g\rVert\lVert g_\text{ref}\rVert)$, which measures direction only (0.99 is about 8°).

     | MLP width | Settings | Relative error, median / max | Smallest cosine | Reference self-check |
     |---|---|---|---|---|
     | 8 | default | 0.13 / 1.3 | 0.99972 | 7.8e-4 |
     | 8 | `rtol=1e-4, atol=1e-6` | 0.086 / 0.19 | 0.99998 | |
     | 128 | default | 0.063 / 0.18 | 0.99173 | 1.3e-4 |
     | 128 | `rtol=1e-4, atol=1e-6` | 0.008 / 0.017 | 0.99994 | |

   - **After 20 Adam steps (2026-11-16, on 1.4.1),** the default-tolerance path was 9–23% of the optimization's movement away from the tight run, up to 0.019 Å.
   - **No barrier.** Morse on T1x has its energy maximum at an endpoint.
   - **The written test.** `tests/test_default_tolerances.py`, unskipped on 2.0.0, gives 2 passed and 2 xfailed (strict).
10. **padaquad 2.0.0 uses a supplied starting mesh and draws no random numbers** (`mesh_check2.py`):

    | Call | Random numbers drawn | Starting subintervals |
    |---|---|---|
    | no mesh (control) | yes | 11 |
    | `mesh=` with 6 points | no | 5 |
    | `state=` returned by that call | no | 6 |
    | `mesh=` and `state=` together | `ValueError: Pass mesh= or state=, not both.` | — |

    - **Validation.** padaquad itself rejects a malformed mesh with `ValueError`: wrong shape, not increasing, a single point, or not a tensor. A mesh whose ends lie inside the bounds is extended to them.
    - **What popcornn does today.** popcornn 0.4.0 passes `state=self._state` (`integrator.py:475-477`) and never `mesh=`, so the first call of each solver draws a random mesh.
11. **Other popcornn 0.4.0 and padaquad 2.0.0 changes that matter for tests:**
    - **`runtime_checks` now defaults to False** in popcornn (your choice). A NaN or Inf from a potential then reaches `param.grad` silently. `test_memory_options_leave_results_unchanged[runtime_checks]` shows that turning it on doesn't change results. See Decision G.
    - **`trade_speed_for_memory` defaults to True** and logs advice warnings on the first call of each solver. Tests that count log lines (`test_progress_log_outputs`) pass as they are.
    - **#103 removed the missing `tests/requirements.txt` install from CI.** The `bash -l {0}` shell still drops `-e`.

---

## Task 1: Data paths that don't depend on the working directory

**Files:**
- Modify: `pyproject.toml` (replace `pythonpath = "popcornn/popcornn"`, a directory that doesn't exist, with `pythonpath = ["."]`)
- Modify: `tests/__init__.py` (replace `sys.path.append(".")` with `IMAGES_DIR`)
- Create: `tests/conftest.py`
- Modify: `tests/test_potentials.py`, `tests/test_images.py`, `tests/test_paths.py`, `tests/test_default_tolerances.py` (the 37 strings), and `tests/test_integrator_samples.py` (`muller_brown_setup`)

**Interfaces:**
- Produces:
  - `tests.IMAGES_DIR: pathlib.Path`, imported with `from tests import IMAGES_DIR`.
  - A `pytest_make_parametrize_id` hook in `tests/conftest.py`.

- [ ] **Step 1: Record the baseline.**
  - From the repository root, save `pytest --collect-only -q` to a file.
  - Run `cd tests && ~/.conda/envs/popcornn/bin/python -m pytest -q -x test_images.py`. Expected: `FileNotFoundError`.
- [ ] **Step 2: Add `IMAGES_DIR` and the root path.**
  - `tests/__init__.py` keeps `from __future__ import annotations`, drops `sys.path.append(".")`, and defines `IMAGES_DIR = Path(__file__).parent / "images"`.
  - In `pyproject.toml`, set `pythonpath = ["."]`.
- [ ] **Step 3: Add the id hook** in a new `tests/conftest.py`. `pytest_make_parametrize_id(config, val, argname) -> str | None` returns `Path(val).name` for a string starting with `str(IMAGES_DIR)`, and `None` otherwise.
- [ ] **Step 4: Replace the paths.** Each `"tests/images/<name>"` becomes `str(IMAGES_DIR / "<name>")`, including those inside parametrize lists and `read(...)` calls. `muller_brown_setup` uses `str(IMAGES_DIR / "muller_brown.json")`.
- [ ] **Step 5: Verify.**
  - **Same tests.** The collection count is unchanged; ids change only from `tests/images/<name>` to `<name>`.
  - **Every directory works.** Run from the root, from `tests/`, and from `/tmp`, with the same pass and skip counts each time.
  - **No paths left.** `grep -rn '"tests/images\|Path(__file__).parent / "images"' tests/` finds only `tests/__init__.py`.
  - **Lint.** Run pre-commit on the changed files.
- [ ] **Step 6: Commit.** `test: find test data relative to the tests package, not the working directory`

## Task 2: Every test starts from the same random state

This is the least invasive fix (Fact 7): one autouse fixture in `tests/conftest.py`, plus deleting the six seed calls it makes redundant. Only torch is seeded: no test uses NumPy's or Python's global generator, and ruff's NPY002 pushes NumPy code to local generators.

**Files:**
- Modify: `tests/__init__.py`, `tests/conftest.py`
- Create: `tests/test_conftest.py` (tests of the conftest machinery; Task 3 adds to it)
- Modify: the files of the six redundant seed calls (Fact 7)

**Interfaces:**
- Produces:
  - `TEST_SEED = 0` in `tests/__init__.py`.
  - The autouse fixture `_seed_rng() -> None`, which calls `torch.manual_seed(TEST_SEED)` before each test. That seeds the CPU generator and every CUDA device.

- [ ] **Step 1: Write the failing test** in `tests/test_conftest.py`.

```python
@pytest.mark.parametrize("repeat", [1, 2])
def test_each_test_starts_from_the_same_random_state(repeat):
    """Every test starts from torch.manual_seed(TEST_SEED), whatever ran before.
    The second case fails without the conftest fixture, because the first case
    has already advanced the global generator."""
    expected = torch.rand(4, generator=torch.Generator().manual_seed(TEST_SEED))
    assert torch.equal(torch.rand(4), expected)
```

- [ ] **Step 2: Run it.** Expected: `[2]` FAILS.
- [ ] **Step 3: Add `TEST_SEED` and `_seed_rng`,** with a docstring that gives the reason (decision 4).
- [ ] **Step 4: Delete the six redundant seed calls** (Fact 7). Keep the deliberate reseeds and the non-zero seeds.
- [ ] **Step 5: Verify.**
  - **The new test.** Both cases pass.
  - **The audit.** The scratch audit plugin over the full suite flags nothing.
  - **The full suite.** The same pass and skip counts as before.
- [ ] **Step 6: Commit.** `test: seed torch before every test so no test depends on test order`

## Task 3: Pin `max_batch` in every test, guard against unpinned integrators, and add markers

**Files:**
- Modify: `tests/conftest.py`, `tests/test_conftest.py`, `pyproject.toml`, `.github/workflows/tests.yaml`
- Modify: `tests/test_integrator_samples.py`, `tests/test_ts_end_to_end.py`

**Interfaces:**
- Produces:
  - The marker `unpinned_max_batch`: the test may build a `PathIntegrator` without `max_batch`. It replaces the earlier name `calibrates_batch`, because padaquad 2.0.0 no longer calibrates.
  - The marker `slow`.
  - The autouse fixtures `_forbid_unpinned_max_batch` (session scope) and `_allow_unpinned_max_batch` (function scope).

- [ ] **Step 1: Write the guard tests** in `tests/test_conftest.py`.
  - `test_unpinned_max_batch_is_refused`: `pytest.raises(RuntimeError, match="unpinned_max_batch")` around `PathIntegrator(path_integrand_names="pvre", rtol=1e-2, atol=1e-2)`.
  - `test_unpinned_max_batch_allowed_when_marked`, marked `unpinned_max_batch`: the same constructor call succeeds.
- [ ] **Step 2: Run them.** Expected: the first FAILS with `DID NOT RAISE`.
- [ ] **Step 3: Register the markers in `pyproject.toml`.** Add `"--strict-markers"` to `addopts`, and add `markers`:
  - `slow`: "more than about 10 s of CPU time on one core; CI deselects it".
  - `unpinned_max_batch`: "may build a PathIntegrator with max_batch=None (padaquad then searches for a batch size)".
- [ ] **Step 4: Write the guard in `tests/conftest.py`.** This is the design of Fact 3.
  - **Module level.** Keep `_ORIGINAL_INIT = PathIntegrator.__init__` and `_INIT_SIGNATURE = inspect.signature(_ORIGINAL_INIT)`.
  - **`_refuse_unpinned(self, *args, **kwargs)`** binds the arguments with `_INIT_SIGNATURE`, so `max_batch` is found whether it was passed by position or by name. If it is `None`, it raises `RuntimeError` naming `max_batch` and `@pytest.mark.unpinned_max_batch`; otherwise it calls `_ORIGINAL_INIT`. Apply `functools.wraps(_ORIGINAL_INIT)` to it.
  - **`_forbid_unpinned_max_batch`** (autouse, `scope="session"`) patches `PathIntegrator.__init__` inside `pytest.MonkeyPatch.context()` and yields. Session scope also covers module-scoped fixtures.
  - **`_allow_unpinned_max_batch`** (autouse) restores `_ORIGINAL_INIT` for tests marked `unpinned_max_batch`.
  - **Why `PathIntegrator`, not padaquad.** This catches every unpinned integrator, including those built by `Popcornn` from `integrator_params`, and depends on no private padaquad name.
- [ ] **Step 5: Run the guard tests.** Expected: PASS.
- [ ] **Step 6: Pin `max_batch=500` in the five tests that don't test the search, and mark the one that does.**
  - **Pin:** both `track_ts` tests, `test_optimize_path_returns_ts_image`, and both cases of `test_device_and_runtime_checks_reach_padaquad`. The last two only build the integrator, so the value doesn't matter there.
  - **Mark:** `test_warm_start_passes_previous_state[found]`, via `pytest.param(None, id="found", marks=pytest.mark.unpinned_max_batch)`.
  - **Why 500** (Fact 1): every node of these integrations fits in one batch, so the results are bit-identical to padaquad's own choice. It also meets the memory rule at width 128.
  - **The general rule** for later tests: the smallest round value that holds the test's largest pass in one batch, satisfies the memory rule, and is at least the panel size. If no value can do all three, take the largest the memory rule allows and record the measured difference in the docstring.
- [ ] **Step 7: Check that results are unchanged.** Rerun the scratch comparison (`pin_compare2.py`) for each test whose `max_batch` this task sets. Expected: bit-identical. If any test differs, **stop and report to Kareem** with the size of the difference.
- [ ] **Step 8: Deselect `slow` tests in CI.** In `.github/workflows/tests.yaml`, run `pytest -m "not slow" --cov --cov-report=xml`. Locally, plain `pytest` still runs everything.
- [ ] **Step 9: Run the full suite and pre-commit.** Expected: no failures.
- [ ] **Step 10: Commit.** `test: pin max_batch in every test and refuse unpinned PathIntegrators unless marked`

## Task 4: Shared `device`, `dtype` and `cheap_integrator_params` fixtures, with the 15 copies migrated

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_images.py`, `tests/test_potentials.py`, `tests/test_paths.py` (the 15 blocks)
- Modify: `tests/test_integrator_samples.py`, `tests/test_converged_flag.py`, `tests/test_ts_end_to_end.py`, `tests/test_conftest.py` (the cheap settings)

**Interfaces:**
- Produces:
  - **`device`:** cpu, plus cuda skipped when unavailable, with ids `cpu` and `cuda`.
  - **`dtype`:** float32 and float64, with ids `float32` and `float64`.
  - **`cheap_integrator_params`:** returns a new dict on each call: `{"rtol": 1e-2, "atol": 1e-2, "max_batch": 500, "runtime_checks": True}`.
    - **Why `runtime_checks`** (decision 7): popcornn's default is False, so a NaN or Inf from a potential would otherwise reach the gradient silently. With it on, padaquad raises.
    - Tests extend it only by merging into a new dict: `{**cheap_integrator_params, ...}`.
    - It leaves out `integrate_gradient`. New tests that aren't about the gradient mode add `"integrate_gradient": False`, as `00` asks.

- [ ] **Step 1: Add the three fixtures.** The docstring of `cheap_integrator_params` gives the reason for each value (Task 5a and Task 3, Step 6) and the merge pattern.
- [ ] **Step 2: Migrate the 15 blocks.**
  - Delete both parametrize decorators and the inline CUDA skip from each of the 14 full blocks.
  - In `test_charge_spin`, delete only the device part.
  - Leave `test_uma` and `test_ts_search` alone.
- [ ] **Step 3: Check the migration.** The `--collect-only` count is unchanged, and the pass and skip counts are unchanged. On the login node (A100) the `cuda` cases run.
- [ ] **Step 4: Use the fixture where a test has no reason for its own settings.**
  - **`test_converged_flag.py`.** `_stage` takes `integrator_params` and merges `"path_integrand_names": "pvre"` into it. `_run(cheap_integrator_params, **popcornn_kwargs)` and `test_log_file_fixed_at_construction` pass it on. `max_batch` changes from 100 to 500; Step 5 checks that the assertions still hold.
  - **`test_ts_end_to_end.py` and both `track_ts` tests** merge the fixture in place of their literals.
  - **Settings that stay, each with a test-specific reason:** `_WARM_START_PARAMS` (`max_batch=400` holds every batch, plus `track_loss`), the out-of-memory tests (`max_batch=400`, which the simulated limit of 50 depends on), and `test_memory_options_leave_results_unchanged` (`max_batch=46` spreads a pass over several calls).
  - **The guard tests** merge `"max_batch": None`.
  - **`runtime_checks=True` in every test that integrates** (decision 7), through the fixture or added explicitly to the test-specific settings above and to `COMMON` in `test_default_tolerances.py`.
    - **Exception: `test_memory_options_leave_results_unchanged`.** It compares the default `False` against `True` (its `[runtime_checks]` case), so its base settings keep popcornn's default.
    - **Exception: `test_device_and_runtime_checks_reach_padaquad`.** It checks that the default `False` reaches padaquad.
    - Each exception gets a one-line comment saying why.
- [ ] **Step 5: Run the full suite and pre-commit.**
  - Expected: no failures.
  - **If a test now raises from padaquad's checks** (a NaN or Inf, or an order check), that is a real problem the default was hiding. **Stop and report it** rather than turning the checks off for that test.
  - **Timing.** Note the suite's run time before and after. The checks build output-sized temporaries, so a large slowdown would be worth knowing about.
- [ ] **Step 6: Commit.** `test: share device/dtype/cheap-integrator fixtures in conftest`

## Task 5: Explicit, justified tolerances for every test, and the default-tolerance test

### 5a. Tolerances of the existing integrating tests

Every existing integrating test checks wiring (shapes, flags, finite values, log lines, call arguments, batch sizes, bit-equality between memory options), not numerical accuracy. So `rtol=atol=1e-2` is suitable for all of them:
- **Coarse but stable.** Fact 9 puts its accuracy in the same range as the defaults.
- **Explicit.** These tests no longer depend on `PathIntegrator`'s defaults.
- **Reproducible.** Together with Tasks 2, 3 and 7, each run is bit-reproducible on a given software stack.

| Test | What it checks | Tolerance |
|---|---|---|
| both `track_ts` tests | samples align with nodes; `samples is None` | 1e-2 (cheap) |
| `test_warm_start_passes_previous_state` | the state passed between calls | 1e-2 |
| `test_bounds_relative_to_previous_mesh` | extend/cut against a fresh integration, 5e-2 relative L2 margin (measured ≤ 7.1e-3) | 1e-2: the margin was measured at this tolerance |
| `test_oom_batch_reduction_persists` (3 cases) | batch sizes after simulated out-of-memory errors | 1e-2 |
| `test_memory_options_leave_results_unchanged` (7 cases) | bit-equality between options | 1e-2 |
| `test_device_and_runtime_checks_reach_padaquad` | constructor arguments only | none needed |
| `test_converged_flag.py` (4 tests) | flags with thresholds 1e10 and 1e-12, far from any `grad_norm`; log lines | 1e-2 |
| `test_optimize_path_returns_ts_image` | `ts_time` in [0, 1]; finite | 1e-2 |

- [ ] **Step 1:** Write a short form of this table into the docstring of `cheap_integrator_params`.

**The rule for numerical tests (plans 01–06):** choose `rtol`/`atol` from a sweep over `starting_mesh(0..9)` (Task 7), set the assertion tolerance from the worst case, and write it in a comment.

### 5b. The default-tolerance test

**Status:** written, and skipped on request: [tests/test_default_tolerances.py](../../tests/test_default_tolerances.py). It uses Morse on T1x, an MLP of width 8 and depth 2, and the `pvre` integrand.
- **Thresholds:** cosine ≥ 0.99, gradient relative error ≤ 0.2, path ratio ≤ 0.05. The reference must agree with itself to 2e-3.
- **Unskipped on padaquad 2.0.0:** `test_tight_reference_is_converged` and `test_default_gradient_direction` pass. `test_default_gradient_size` and `test_default_optimization_matches_tight` fail as expected (strict xfail).
- **Checks:** it passes ruff, and pytest collects and skips it.

- [ ] **Step 2: Follow-up edits.**
  - Task 1 converts its path.
  - Task 3 adds `slow`. The test already pins `max_batch=4000` (memory grows linearly with `integrate_gradient=False`), so the guard passes it.
  - Task 7, Step 6 moves it to fixed starting meshes.
  - When the skip is lifted, the xfail reasons name an issue about the defaults.
- [ ] **Step 3: Commit** (not yet committed). `test: compare PathIntegrator default tolerances with a tight reference on T1x (skipped)`

## Task 7: A fixed starting mesh, so tests don't depend on padaquad's random draws

Task 2 fixes every test's starting state. Within a test, padaquad still draws a random starting mesh on each solver's first call, so changes in padaquad move results (Fact 1). padaquad 2.0.0 uses a supplied mesh and then draws nothing, and it validates the mesh itself (Fact 10). This is the only task that changes package code. The new argument defaults to `None`, so users see no change unless they set it.

**Files:**
- Modify: `popcornn/tools/integrator.py`, `docs/configuration.md` (one row next to `max_batch`)
- Modify: `tests/__init__.py`, `tests/conftest.py`, `tests/test_integrator_samples.py`, `tests/test_default_tolerances.py`

**Interfaces:**
- Consumes: `cheap_integrator_params` (Task 4) and `TEST_SEED` (Task 2).
- Produces:
  - **`PathIntegrator(..., initial_mesh: torch.Tensor | Sequence | None = None)`.**
    - **Storage.** It is stored with `torch.as_tensor(initial_mesh, dtype=self.dtype, device=self.device)`, so a config can give it as a list.
    - **Use.** On a solver's first call, when its state is `None`, popcornn passes `mesh=self.initial_mesh` instead of a fresh random mesh. On later calls it passes `state=` as today; padaquad forbids passing both.
    - **Both solvers.** This applies to the main solver (`self._state`) and the loss solver (`self._loss_state`).
    - **Bounds.** popcornn still passes `t_init` and `t_final`, and padaquad extends or cuts the mesh to them.
  - **No validation in popcornn.** padaquad raises a clear `ValueError` for a malformed mesh (Fact 10), so that check stays in one place.
  - **For tests:** `starting_mesh(seed: int) -> torch.Tensor` in `tests/__init__.py`.
    - It returns exactly padaquad's own starting mesh for `torch.manual_seed(seed)`: 3 equal segments with 4 sorted random points each, the first segment shifted to start at 0 and the last to end at 1, giving 12 points (`[12, 1]`, 11 subintervals). (The first implementation built a different 13-point layout; the final review caught it, and a test now checks the match.)
    - It uses a local `torch.Generator().manual_seed(seed)`, so the global generator is never touched.
    - `cheap_integrator_params` adds `"initial_mesh": starting_mesh(TEST_SEED)`.
  - **Why not evenly spaced points?** padaquad's `random_initial_mesh=False` gives evenly spaced points, which can line up with features of the integrand. The sphere's corner at t = 0.5 on a panel boundary integrates exactly and hides the error of Fact 6.

- [ ] **Step 1: Write the failing test** `test_initial_mesh_is_used_without_random_draws` in `tests/test_integrator_samples.py`.
  - Wrap both solvers' `integrate` as `_record_integrate_calls` already does, and also record `kwargs.get("mesh")`.
  - Save the torch random state, then call `integrate_path` twice with `initial_mesh=starting_mesh(0)` and `track_loss=True`.
  - Assert that the random state is unchanged.
  - Assert that each solver's first call got `mesh` equal to `starting_mesh(0)` and `state=None`, and its second call got `mesh=None` and the first call's state.
- [ ] **Step 2: Run it.** Expected: FAIL (`TypeError`: unexpected keyword `initial_mesh`).
- [ ] **Step 3: Add the option** to `PathIntegrator`.
  - **Docstring.** Say that the mesh is used on each solver's first call only, after which the state takes over, and that padaquad validates it.
  - **Config docs.** Add the row in `docs/configuration.md`.
- [ ] **Step 4: Run the test.** Expected: PASS.
- [ ] **Step 5: Use it in the tests.**
  - **Cheap settings.** Add `starting_mesh` and put it in `cheap_integrator_params`.
  - **The test-specific settings** of Task 4 (`_WARM_START_PARAMS`, the out-of-memory and memory-options tests) also get `"initial_mesh": starting_mesh(TEST_SEED)`. Their deliberate reseeds (`test_integrator_samples.py:175, 183, 311`) then have nothing left to reset; delete them.
  - **Verify.** Rerun the full suite and an extended audit that checks the random state around each `integrate_path`. Expected: all pass, and nothing is drawn inside `integrate_path` except in tests that deliberately leave `initial_mesh` unset.
- [ ] **Step 6: Use fixed meshes in `tests/test_default_tolerances.py`.**
  - Replace `torch.manual_seed(mesh_seed)` with `initial_mesh=starting_mesh(mesh_seed)`.
  - Re-measure in a copy without the skip. Expected: 2 passed and 2 xfailed.
  - The new meshes change the measured values, so update the comments next to the thresholds.
- [ ] **Step 7: Run pre-commit, then commit.** `feat(integrator): accept a fixed starting mesh; tests use one`

## Task 6: Update `00_shared_context.md`

- [ ] **Loss tolerance:** add Fact 6.
- [ ] **Memory measurement:** padaquad 2.0.0 finds the batch by growing it (Fact 2). Every test pins `max_batch` by the rule in Task 3, Step 6, and the guard enforces it. Replace "for example 2000" with the memory rule.
- [ ] **Warm start:** padaquad 2.0.0 reuses a mesh only through `state=` (popcornn 0.4.0 does this), and `initial_mesh` fixes the first mesh.
- [ ] **How to run:**
  - pytest works from any directory, and data paths use `IMAGES_DIR`;
  - the per-test seed;
  - the `unpinned_max_batch` and `slow` markers;
  - the `device`, `dtype` and `cheap_integrator_params` fixtures, and the merge pattern;
  - tests run with `runtime_checks=True` (popcornn's default stays False);
  - `starting_mesh(seed)`;
  - ruff and pre-commit;
  - no iteration-count assertions.
- [ ] **Loose defaults:** add Fact 9.
- [ ] **Versions:** Python ≥ 3.11, padaquad ≥ 2.0.0 and < 3, and the panel sizes (gk7 7, gk21 21).
- [ ] **Commit:** `docs(tests): record test conventions and measurements in the shared context`

---

## Consumers: what plans 01–06 must do with this

I have not edited these plans.

| Plan | Edit needed |
|---|---|
| all | Use `IMAGES_DIR`. Rely on the per-test seed. Pin `max_batch` by the Task 3 rule. Pass `initial_mesh` and `runtime_checks=True`. Set tolerances from a sweep over `starting_mesh(0..9)`. Don't assert iteration counts. Recheck any panel-size arithmetic: gk21 has 21 points per panel now, not 23. |
| 01 | `max_batch` ≤ 500 at width 128 with `integrate_gradient=True`, instead of 2000. Re-measure its tolerances. |
| 02 (`02_morse_harmonic_implementation_plan.md`, lines 140 and 151) | Use the `device` fixture. It may keep its local `dtype` parametrize while the float32 strict xfail for #94 is needed. |
| 04 (`04_atomic_end_to_end_implementation_plan.md`, line 130) | Replace `INTEGRATOR_PARAMS` with `{**cheap_integrator_params, "integrate_gradient": False}`. |
| 05, 06 | Use `cheap_integrator_params`, and the `device` fixture for plan 06's CUDA test. |

## Review focus

1. **A pinned `max_batch` that is too large for the MLP width.** The guard doesn't catch it, and a pinned value bypasses padaquad's out-of-memory retry.
2. **Results that depend on the padaquad version.** After Task 7, tests don't depend on padaquad's random draws, but its refinement rules and arithmetic still change between releases (Fact 1: three versions, three `ts_time` values). Assertions need tolerances, not bit-exact values.
3. **Iteration counts.** They are dominated by quadrature noise (Global constraints). A test that asserts one will break on any rounding-level change.
4. **A future test reading data at collection time without `IMAGES_DIR`.** Task 1's grep catches the existing ones; reviewers must catch new ones.
5. **GPU nondeterminism.** Task 2 fixes the starting point, not run-to-run variation in some CUDA kernels. No current test asserts bit-exact GPU results.
6. **An `initial_mesh` that padaquad ignores.** `test_initial_mesh_is_used_without_random_draws` checks the call arguments and the random state.

## Out of scope

- 00's "API facts" section (#90 belongs to plans 05 and 06).
- Copying `hard.xyz` (plan 04).
- The CI shell (`bash -l {0}` drops `-e`). Since #103 removed the failing install, nothing currently fails silently there.
- #89.
- Changing `PathIntegrator`'s defaults, or its stopping rule given the quadrature noise in the gradient norm.

## Decisions still needed

Resolved:
- **A.** Morse on T1x; the test is written and skipped.
- **B.** The thresholds are accepted.
- **C.** padaquad uses a supplied mesh.
- **D.** Plan only for the path fix.
- **E.** Task 7 is approved.
- **F.** #103 moved popcornn to padaquad 2.0.0, and the local checkout matches PyPI.

- **G.** Tests run with `runtime_checks=True`; popcornn's default stays False (Task 4).

Open: none. Implementation starts on your go-ahead.

## Confidence

- **Sure** (measured on padaquad 2.0.0 and popcornn `e4a02f4`):
  - Facts 1, 2, 3, 6, 9 and 10.
  - The audit of the rewritten integrator tests.
  - That padaquad validates a supplied mesh.
- **Less sure:**
  - That a CPU out-of-memory error during padaquad's batch growth kills the process rather than raising. That is from the batch-fit brief, not measured here.
  - Whether Morse on T1x represents real use, since it has no barrier.
  - All timings (load average 25–110).

## Revision history

- **2026-11-14:** First version and adversarial review.
- **2026-11-15:** Re-checked after #98, #99 and #101.
- **2026-11-16:** Rewritten around decisions 1–5, with new measurements (Facts 1, 2, 7, 8 and 9) and Tasks 1, 2 and 5.
- **2026-11-17:**
  - Decisions A–F.
  - `tests/test_default_tolerances.py` written and skipped.
  - Task 7 approved.
- **2026-11-26, after #103 (popcornn 0.4.0 on padaquad 2.0.0):**
  1. **Guard.** It moved from padaquad's `_setup_memory_checks`, which `integrate` no longer calls, to `PathIntegrator.__init__`. The marker is renamed `unpinned_max_batch`, and the set of guarded tests changed: `test_batch_size_measured_once` is gone, the `[found]` case and the device test were added.
  2. **Why pin.** With no time saving left, the reasons are consistency, memory and machine independence.
  3. **Task 7.** It now passes `mesh=` on the first call and `state=` afterwards, and leaves validation to padaquad.
  4. **Re-measured** Facts 1, 6, 9 and 10, and audited the rewritten integrator tests.
  5. **Paths.** The count is 37 strings plus `muller_brown_setup`'s own path.
  6. **Panel sizes and iteration counts.** Panel sizes are now 7 and 21, and no test may assert iteration counts.
  7. **Warning removed.** The local padaquad no longer breaks popcornn.
  8. **Decision G** added, then decided the same day: tests run with `runtime_checks=True`. Two exceptions test the default itself, and Task 4 says to stop and report if a test starts raising.

- **2026-11-27, implemented** on branch `onging_tests` (commits `3bbf668` to the Task 6 commit), in the order 1, 2, 3, 4, 5, 7, 6. Deviations, each recorded as a ruling:
  - **`test_uma`** takes `dtype` from the shared fixture (its values were the same) and keeps its own device list.
  - **Literal settings stay.** The tests with test-specific settings keep literal `rtol`/`atol`, because module-level dicts can't use a fixture.
  - **Reference limit.** A first version raised `test_default_tolerances.py`'s reference-agreement limit to 1e-2. The final review traced the cause to `starting_mesh` not matching padaquad's layout. With that fixed, the accepted 2e-3 holds again (measured 7.8e-4 for the pair the test uses; ≤ 1.6e-3 over meshes 0–5), and it is restored.
  - **Changelog.** An "Unreleased" CHANGELOG entry records `initial_mesh`.

  Measured after Task 7:
  - **Default tolerances.** Cosine ≥ 0.9997, size error median 0.13 and max 1.26 (as in Fact 9), path ratio 0.152.
  - **Bounds-test margin.** Extend ≤ 7.1e-3, cut ≤ 2.1e-6.
  - **Random draws.** No `integrate_path` call in the suite draws random numbers.
  - **Full suite.** 152 passed, 21 skipped.
