# Memory and OOM errors

Popcornn evaluates the path on an **adaptive mesh** of times — denser
where the loss is changing fast, sparser where it's flat. The number
of points in a single quadrature step is therefore not fixed; it
varies from one call to the next, and on big systems with MLIPs, that
variability is what triggers most CUDA out-of-memory errors.

`popcornn/popcornn.py` already sets

```python
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

at import time, which solves most OOM cases. If you're still hitting
OOM, work through the steps below in order.

## 1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

PyTorch's default memory allocator is optimized for one batch size
seen many times. The adaptive integrator doesn't behave that way: it
might evaluate 16 points one iteration, 64 the next. Without
expandable segments, PyTorch fragments memory across these resizes
and runs out even though the live working set fits.

`popcornn` sets this environment variable at the top of
`popcornn/popcornn.py`, so it's already on for the YAML driver and
for any code that does `from popcornn import Popcornn` early. If
you're using popcornn as a library and importing other things first,
make sure to set it before any CUDA allocator calls:

```python
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

at the very top of your entry-point file.

## 2. `max_batch`

If you still hit OOM, hard-cap the per-quadrature-step batch:

```yaml
integrator_params:
  path_integrand_names: pvre_pseudo_huber
  max_batch: 32
```

This passes at most 32 points to the path and potential in one call.
It is slower (more sequential calls), and it limits the memory of each
call, not the total: padaquad evaluates every pending quadrature panel
before accepting or rejecting any. In `integrate_gradient: false` mode
`max_batch` must be at least the number of nodes per panel (7 for
`gk7`, 21 for `gk21`).

With `max_batch` left at `None`, padaquad measures the memory of one
integrand evaluation on the first `integrate_path` call of a
`PathIntegrator` and sizes the batch from it. The measurement is slow,
so it runs once per integrator, not once per optimizer step: padaquad
stores the batch size and popcornn reuses it on every later call.

If a batch still runs out of memory, what happens depends on the
gradient mode:

- `integrate_gradient: true` (the default), and the `track_loss`
  integral: padaquad shrinks the batch and retries, and the smaller
  size is kept for the rest of that stage (it never grows back).
- `integrate_gradient: false`: padaquad backpropagates through each
  batch and cannot retry, so the error is raised. Set `max_batch`
  explicitly for memory-tight runs in this mode.

Across stages built by `Popcornn._optimize_stage` (one fresh integrator
per stage), the stored batch size does **not** carry over — that's
intentional, since different stages typically use different potentials
with different memory profiles.

## 3. Float precision

Switching from `float32` to `float64` doubles the memory footprint,
so:

```yaml
initialization_params:
  dtype: float32       # default
```

is preferred unless you specifically need `float64` for numerical
sensitivity (you almost never do for path optimization).

## 4. Smaller path network

A bigger MLP doesn't just cost more compute — every parameter
contributes to the gradient that gets propagated through the
integrator. Drop `width` in `path_params` if memory is tight:

```yaml
path_params:
  name: mlp
  width: 64    # default is 128
  depth: 2
```

⚠️ **Changing `width` changes the** $\sigma_{\min}(J_\text{path})$
**calibration**, which the shipped `threshold` derivation assumes is
≈ 1. If you halve `width`, the calibration shifts by
$\sqrt{64/128} \approx 0.71$, and `threshold` should be scaled
accordingly:

$$\text{threshold}_{\text{new}} \;=\; \text{threshold}_{\text{shipped}} \cdot \sqrt{W_{\text{new}} / 128}.$$

See [paths](paths.md) for the $\sigma_{\min}$ calibration and
[derivation](derivation.md) for why this rescaling holds.

## 5. CPU offload

You can run the whole optimization on CPU by setting
`device: cpu` in `initialization_params`. Slow, but it never OOMs.
This is mostly only practical for the analytic toy potentials.

## What to do if it's still OOM

If you've worked through 1–5 and still see OOM, file a GitHub issue
with:

- The full traceback, including the actual CUDA allocator message.
- `nvidia-smi` output showing how much memory the GPU actually has.
- The full config file you're running.
- The popcornn and padaquad commit hashes
  (`pip show popcornn padaquad | grep -i 'name\|version'`).
