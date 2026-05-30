"""End-to-end smoke test for the TS-finding wire-up.

Verifies that ``Popcornn.optimize_path`` on Muller-Brown actually
runs ``ts_search`` per-iteration, populates ``path.ts_time``, and
returns a non-None ``ts_image``. This is a small-iteration smoke
test — convergence quality is checked manually by running
``examples/run.py`` against the config.
"""

import torch

from popcornn import Popcornn


def test_optimize_path_returns_ts_image():
    torch.manual_seed(0)
    mep = Popcornn(
        images=[[-0.558, 1.442], [0.623, 0.028]],
        path_params={
            'name': 'mlp',
            'n_embed': 4,
            'depth': 2,
            'activation': 'gelu',
        },
        device='cpu',
        dtype='float64',
    )
    final, ts_image = mep.optimize_path(
        {
            'potential_params': {'name': 'muller_brown'},
            'integrator_params': {
                'path_integrand_names': 'pvre',
                'rtol': 1e-2,
                'atol': 1e-2,
            },
            'optimizer_params': {
                'optimizer': {'name': 'adam', 'lr': 1e-3},
            },
            'num_optimizer_iterations': 10,
        },
        output_ase_atoms=False,
    )
    assert ts_image is not None
    assert mep.path.ts_time is not None
    ts_t = float(mep.path.ts_time)
    assert 0.0 <= ts_t <= 1.0, f"ts_time {ts_t} outside [0, 1]"
    # The saddle predicted on a barely-trained path won't be accurate, but
    # it must at least be a real Tensor with finite contents.
    assert torch.isfinite(mep.path.ts_energy).all()
    assert torch.isfinite(mep.path.ts_force).all()
