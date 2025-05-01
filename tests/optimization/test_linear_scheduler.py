import os
import torch
import numpy as np
from popcornn.tools.scheduler import Linear
from popcornn.tools.tests import test_popcornn_run


def test_linear_scheuler():
    N_steps = 10
    start, end = 0.0, 1.0
    scheduler = Linear(start, end, N_steps)
    comparison = torch.linspace(start, end, N_steps)
    for i in range(N_steps):
        assert np.allclose([scheduler.get_value()], [comparison[i]]),\
            "Linear scheduler valued does not match expected"
        scheduler.step()

    benchmark_path = os.path.join(
        "optimization", "benchmarks"
    )
    for potential in ['wolfe_schlegel', 'morse']:
        name = f"scheduler_linear_{potential}"
        config_path = os.path.join(
            "configs", f"{name}.yaml"
        )
        test_popcornn_run(name, config_path, benchmark_path)