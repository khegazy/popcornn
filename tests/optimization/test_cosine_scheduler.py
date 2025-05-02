import os
import torch
import numpy as np
from popcornn import tools
from popcornn.tools.scheduler import Cosine
from popcornn.tools.tests import popcornn_run_test, scheduler_test


def test_cosine_scheduler():
    N_steps = 100
    start, end = 1.0, 0.0
    def schedule_fxn(start, end, N_steps, device):
        comparison = torch.linspace(0, 1, N_steps, device=device)
        return end\
            - (end - start)*(1 + torch.cos(comparison*torch.pi))/2.
        
    # Compare scheduler stepping
    scheduler = Cosine(start, end, N_steps)
    comparison = schedule_fxn(start, end, N_steps, 'cpu')
    for i in range(N_steps):
        assert np.allclose([scheduler.get_value()], [comparison[i]], atol=1e-6, rtol=1e-5),\
            "Cosine scheduler value does not match expected"
        scheduler.step()

    benchmark_path = os.path.join(
        "optimization", "benchmarks"
    )
    for potential in ['wolfe_schlegel', 'morse']:
        # Test high level functionality
        name = f"scheduler_cosine_{potential}"
        config_path = os.path.join(
            "configs", f"{name}.yaml"
        )
        mep, _, _ = popcornn_run_test(name, config_path, benchmark_path)

        # Test scheduler effects match calculated scheduled weights
        config = tools.import_run_config(config_path)
        scheduler_test(mep.path, config, schedule_fxn, mep.device)