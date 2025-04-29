import os
import torch
import numpy as np
from popcornn import tools
from popcornn.run_optimization import optimize_MEP


POTENTIALS = ['wolfe_schlegel', 'repel', 'morse']#, 'lennard_jones', 'muller_brown', 'schwefel', 'repel', 'harmonic']
POTENTIALS = ['morse']#, 'lennard_jones', 'muller_brown', 'schwefel', 'repel', 'harmonic']

def test_potentials():
    torch.manual_seed(2025)
    np.random.seed(2025)
    
    for pot in POTENTIALS:
        print("STARTING POTENTIAL", pot)
        #config_path = os.path.join("configs", f"{pot}.yaml")
        config_path = os.path.join("configs", f"6445_{pot}.yaml")
        config = tools.import_run_config(config_path)
    
        # Run the optimization
        final_images, ts_image, optimization_results = optimize_MEP(
            save_optimization_freq=None, **config
        )
test_potentials()