import torch
from popcornn.tools import Metrics



def test_ode_functions():
    B = 100
    N_atoms = 13
    time = torch.linspace(0, 1, N_atoms).unsqueeze(-1)
    energy = torch.randn(B, N_atoms)*50 + 1400
    velocity = torch.rand((B, N_atoms, 3))*5 + 3
    force = torch.rand((B, N_atoms, 3))*20 + 10

    ode_results = {}
    for name in Metrics.ode_fxn_names:
        print(name)
        metric = Metrics(device='cpu')
        metric.create_ode_fxn(is_parallel=True, fxn_names=[name])
        ode_results[name] = metric.ode_fxn(
            t=time,
            path=None,
            energy=energy,
            force=force,
            velocity=velocity
        )
    

        
test_ode_functions()