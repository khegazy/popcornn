import os
import torch
import numpy as np
from popcornn import Popcornn
from popcornn import tools
from popcornn.run_optimization import optimize_MEP


POTENTIALS = ['wolfe_schlegel', 'repel', 'morse']#, 'lennard_jones', 'muller_brown', 'schwefel', 'repel', 'harmonic']
POTENTIALS = ['morse']#, 'lennard_jones', 'muller_brown', 'schwefel', 'repel', 'harmonic']

def test_potentials():
    torch.manual_seed(2025)
    np.random.seed(2025)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for pot in POTENTIALS:
        print("STARTING POTENTIAL", pot)
        #config_path = os.path.join("configs", f"{pot}.yaml")
        config_path = os.path.join("configs", f"6445_{pot}.yaml")
        config = tools.import_run_config(config_path)
    
        # Run the optimization
        mep = Popcornn(device=device, **config.get('init_params', {}))
        final_images, ts_image = mep.run(*config.get('opt_params', []))
test_potentials()