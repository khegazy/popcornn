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

This forces every quadrature call to evaluate at most 32 points.
Slower (more sequential calls), but bounded memory.

torchpathint also remembers what value worked the last time
`integrate_path` was called on a given `PathIntegrator` instance, so
the OOM-and-halve cycle only fires once per integrator lifetime, not
once per optimizer step. Across stages built by
`Popcornn._optimize` (one fresh integrator per stage), the learned
value does **not** carry over — that's intentional, since different
stages typically use different potentials with different memory
profiles. Multi-stage harnesses that *do* know their later stages
share a potential can thread the value explicitly:

```python
integ1 = PathIntegrator(...)
# ... run stage 1
integ2 = PathIntegrator(..., max_batch=integ1.max_batch)
# ... stage 2 starts with stage 1's learned batch size
```

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
- The popcornn and torchpathint commit hashes
  (`pip show popcornn torchpathint | grep -i 'name\|version'`).
