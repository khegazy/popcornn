import os
import torch
import numpy as np
from popcornn.tools.scheduler import Cosine
from popcornn.tools.tests import test_popcornn_run


def test_cosine_scheduler():
    N_steps = 100
    start, end = 0.0, 1.0
    scheduler = Cosine(start, end, N_steps)
    comparison = torch.linspace(0, 1, N_steps)
    comparison = end\
        - (end - start)*(1 + torch.cos(comparison*torch.pi))/2.
    for i in range(N_steps):
        assert np.allclose([scheduler.get_value()], [comparison[i]]),\
            "Cosine scheduler value does not match expected"
        scheduler.step()

    benchmark_path = os.path.join(
        "optimization", "benchmarks"
    )
    for potential in ['wolfe_schlegel', 'morse']:
        name = f"scheduler_cosine_{potential}"
        config_path = os.path.join(
            "configs", f"{name}.yaml"
        )
        test_popcornn_run(name, config_path, benchmark_path)