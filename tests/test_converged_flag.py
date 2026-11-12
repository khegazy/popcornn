"""Per-stage convergence flags returned by ``Popcornn.optimize_path``.

Each stage contributes one bool: True if the ``threshold`` trigger fired,
False if the stage exhausted ``num_optimizer_iterations`` (always False
when ``threshold`` is None). ``max_batch`` is pinned so padaquad skips its
memory benchmark.
"""

import torch

from popcornn import Popcornn


def _stage(num_optimizer_iterations, threshold=None):
    return {
        'potential_params': {'name': 'muller_brown'},
        'integrator_params': {
            'path_integrand_names': 'pvre',
            'rtol': 1e-2,
            'atol': 1e-2,
            'max_batch': 100,
        },
        'optimizer_params': {
            'optimizer': {'name': 'adam', 'lr': 1e-3},
            'threshold': threshold,
            'patience': 1,
        },
        'num_optimizer_iterations': num_optimizer_iterations,
    }


def test_optimize_path_returns_per_stage_converged_flags():
    torch.manual_seed(0)
    mep = Popcornn(
        images=[[-0.558, 1.442], [0.623, 0.028]],
        path_params={'name': 'mlp', 'width': 8, 'depth': 2, 'activation': 'gelu'},
        device='cpu',
        dtype='float64',
    )
    assert mep.converged is None

    converged = mep.optimize_path(
        _stage(3, threshold=1e10),   # trigger fires on the first iteration
        _stage(2, threshold=1e-12),  # unreachable trigger: iteration cap hit
        _stage(2, threshold=None),   # trigger disabled
    )

    assert converged == [True, False, False]
    assert all(type(flag) is bool for flag in converged)
    assert mep.converged == converged
