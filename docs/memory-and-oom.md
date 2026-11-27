# Memory and OOM errors

Popcornn evaluates the path on an **adaptive mesh** of times — denser
where the loss is changing fast, sparser where it's flat. The number
of points in a single quadrature step is therefore not fixed; it
varies from one call to the next, and on big systems with MLIPs, that
variability is what triggers most CUDA out-of-memory errors.

`popcornn/popcornn.py` already sets

```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
It is slower (more sequential calls). In `integrate_gradient: true`
mode (the default), and in the `track_loss` integral, it limits the
memory of each call, not the total: padaquad evaluates every pending
quadrature panel before accepting or splitting any, and how many of
their outputs it holds at once is set by the memory options below. In `integrate_gradient: false` mode padaquad evaluates
at most `max_batch // nodes-per-panel` panels and backpropagates
through them before evaluating more, so `max_batch` also bounds the
working memory; there it must be at least the number of nodes per
panel (7 for `gk7`, 21 for `gk21`), else padaquad raises an error.

With `max_batch` left at `None`, padaquad finds the batch size during
the first `integrate_path` call of a `PathIntegrator`. It calls the
integrand on 10, 100 and 1,000 copies of one point (fewer when the
first pass is smaller) and starts from the largest call that ran, then
lets the batch grow by factors of 10 whenever more points are waiting,
until a call runs out of memory;
that call is retried at 0.75 times its size until it fits, and the
batch size then stops growing. The gradient integral and the
`track_loss` integral each find their own size. padaquad returns the
size with the refined mesh, and popcornn passes both to every later
call, so the search happens once per integrator, not once per
optimizer step. The out-of-memory errors of the search are caught and
logged as warnings; they are expected.

On a CPU, pin `max_batch` for anything larger than the analytic toys:
there, an allocation that does not fit can get the process killed by
the operating system instead of raising an error padaquad can catch.

If a batch runs out of memory later, padaquad shrinks it and retries,
and the smaller size is kept for the rest of that stage (it never
grows back). With `integrate_gradient: false`, an error in a backward
pass makes padaquad restore the gradients and restart that integral
with smaller batches; popcornn passes it the path's parameters so this
works even when a potential uses a custom autograd function.

### Memory options

```yaml
integrator_params:
  trade_speed_for_memory: true   # popcornn default (padaquad's is false)
  evaluation_buffer: 5368709120  # bytes; popcornn default 5 GiB (padaquad's is null)
  result_device: cpu            # default cpu; null keeps them on the GPU
  record_evaluations: true       # default true
```

These four padaquad options decide where and when the integrand's
values are held during an integral. They never change which points are
evaluated, the mesh, or the integral (padaquad user guide, "Devices and
memory"; popcornn's `test_memory_options_leave_results_unchanged`
checks this bit for bit). They matter when the integrand's output is
large. In `integrate_gradient: true` mode (the default) the output *is*
the gradient: one vector of the path's parameter count D per point,
4 bytes per entry in float32. D grows with the number of atoms N; for
the shipped network (`width: 128`, `depth: 2`) it is about
129 · 3N + 256 (fit to the table in [paths](paths.md)). A 1,000-atom
system, as in catalysis when a slab or a large cluster is kept in the
path, has D ≈ 390,000, about 1.5 MB per point, so a pass of a few
thousand points holds several GB. For gg3 (13 atoms, D = 5,287), the
largest shipped example, a pass of about 100 points holds about 2 MB,
so the options make no practical difference for the shipped examples.

- **`trade_speed_for_memory`** (popcornn default `true`). With `false`,
  padaquad keeps every value of a refinement pass on the GPU until the
  pass is evaluated, then computes the quadrature once. With `true`, it
  computes the quadrature once `evaluation_buffer` bytes of values are
  waiting and moves those values to `result_device`, so the GPU holds
  at most the buffer plus one call's values. The freed memory also lets
  the batch-size search settle on a larger `max_batch`. It applies to the gradient integral with
  `integrate_gradient: true` and to the `track_loss` integral; with
  `integrate_gradient: false` it is ignored, because padaquad then
  integrates and backpropagates each call at once.
- **`evaluation_buffer`** (popcornn default 5 GiB, `5368709120`;
  padaquad's is `null`). With `trade_speed_for_memory: true`, how many
  bytes of values may wait on the GPU before their quadrature is
  computed. A pass whose values fit in the buffer is integrated once,
  as with `trade_speed_for_memory: false`, so the buffer costs time
  only for passes larger than it; the shipped examples never fill it.
  `null` computes the quadrature after every call, which holds the
  least memory and costs about 0.2 ms per call on an A100 (0.1 ms on a
  CPU), as measured by padaquad. Lower the buffer when 5 GB of values
  on the GPU, next to the potential's own memory, does not fit. Ignored
  unless `trade_speed_for_memory: true` and `integrate_gradient: true`.
- **`result_device`** (default `cpu`). Where the values of accepted
  panels and the per-panel results are stored. `null` keeps them on the
  GPU, which skips the copy to the host but holds them until the end of
  the integral, and then `trade_speed_for_memory` saves little.
- **`record_evaluations`** (default `true`). Whether the values of
  accepted panels are kept at all. popcornn does not read them, so
  `false` saves their copy and memory, which matters for very large D.
  padaquad then builds the next call's starting mesh with its
  `power_law` merge rule, which may start with more panels, and logs a
  warning once per stage. The values are always recorded while
  `track_ts` is on (including the diagnostic integral at the end of
  each stage), because the transition-state samples are read from them.

**`runtime_checks`** (popcornn default `false`; padaquad's is `true`)
is not a memory option, but it also saves memory: with `true`, padaquad
checks every call's output for NaN or Inf, which builds a temporary
array as large as the call's output, and its own bookkeeping once per
call (padaquad's CHANGELOG reports the finite check running out of
memory on a 40 GB A100 after the call itself fit). With `false`, a NaN
or Inf value is accepted and reaches the gradient, so turn it on to find
where a path or potential produces one.

padaquad logs a warning on the first integral of each stage when
`trade_speed_for_memory` is on with `evaluation_buffer: null`. It is
advice; the run is correct.

Across stages built by `Popcornn._optimize` (one fresh integrator
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
