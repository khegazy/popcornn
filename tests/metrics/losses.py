import os
import json
import torch
import numpy as np
from popcornn import tools
from popcornn import Popcornn
from popcornn.tools.metrics import LOSS_FXNS, Metrics

def test_losses():
    for potential in ['wolfe_schlegel', 'morse']:
        # Setup environment 
        torch.manual_seed(2025)
        np.random.seed(2025)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Get config file
        config_path = os.path.join("configs", f"loss_{potential}.yaml")
        config = tools.import_run_config(config_path)
        opt_params = config['opt_params'][0]
        opt_params['integrator_params']['path_loss_name'] = 'growing_string'
        for envelope in ['gauss', 'poly', 'sine', 'sine-gauss', 'butter']:
            opt_params['integrator_params']['path_loss_params'] = {
                'weight_type' : f"inv_{envelope}",
                'variance_scale' : 2,
                'weight_scale' : 1
            }
            print("ENV", envelope)

            # Run the optimization
            mep = Popcornn(device=device, **config.get('init_params', {}))
            path_output, ts_outpuit = mep.optimize_path(*config.get('opt_params', []), output_ase_atoms=False)
test_losses()
    