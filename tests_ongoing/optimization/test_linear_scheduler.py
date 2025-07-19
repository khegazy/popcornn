import os
import torch
import numpy as np
from popcornn import tools
from popcornn.tools.scheduler import Linear
from popcornn.tools.tests import popcornn_run_test, scheduler_test


def test_linear_scheduler():
    N_steps = 100
    start, end = 0.0, 1.0
    def schedule_fxn(start, end, N_steps, device):
        return torch.linspace(start, end, N_steps, device=device)
    
    # Compare scheduler stepping
    scheduler = Linear(start, end, N_steps)
    comparison = schedule_fxn(start, end, N_steps, 'cpu')
    for i in range(N_steps):
        assert np.allclose([scheduler.get_value()], [comparison[i]]),\
            "Linear scheduler value does not match expected"
        scheduler.step()

    benchmark_path = os.path.join(
        "optimization", "benchmarks"
    )
    for potential in ['wolfe_schlegel', 'morse']:
        # Test high level functionality
        name = f"scheduler_linear_{potential}"
        config_path = os.path.join(
            "configs", f"{name}.yaml"
        )
        mep, _, _ = popcornn_run_test(name, config_path, benchmark_path)

        # Test scheduler effects match calculated scheduled weights
        config = tools.import_run_config(config_path)
        scheduler_test(mep.path, config, schedule_fxn, mep.device)