import torch
from popcornn.tools import Metrics



def test_ode_functions():
    T = 100
    N_atoms = 17
    time = torch.linspace(0, 1, T).unsqueeze(-1)
    energy = torch.randn(T, N_atoms)*50 + 1400
    velocity = torch.rand((T, N_atoms*3))*5 + 3
    force = torch.rand((T, N_atoms*3))*20 + 10

    # Test single ode functions while saving energy and force
    for save_E_F in [True, False]:
        ode_results = {}
        for is_parallel in [True, False]:
            for name in Metrics.ode_fxn_names:
                metric = Metrics(device='cpu', save_energy_force=save_E_F)
                metric.create_ode_fxn(
                    is_parallel=is_parallel,
                    fxn_names=[name]
                )
                if is_parallel:
                    result = metric.ode_fxn(
                        eval_time=time,
                        time=time,
                        path=None,
                        energy=energy,
                        force=force,
                        velocity=velocity
                    )
                else:
                    result = metric.ode_fxn(
                        eval_time=time[0],
                        time=time[0].unsqueeze(0),
                        path=None,
                        energy=energy[0],
                        force=force[0],
                        velocity=velocity[0]
                    )

                if name not in ode_results:
                        ode_results[name] = result
                else:
                    compare = ode_results[name] 
                    if not is_parallel:
                        compare = compare[0]
                    assert torch.allclose(compare, result),\
                        f"parallel vs sequential results don't match for {name}"
    
    # Test multiple ode functions without saving energy and force
    fxn_scales = [17.68, 11.45]
    for is_parallel in [True, False]:
        for idx, name1 in enumerate(Metrics.ode_fxn_names):
            for name2 in Metrics.ode_fxn_names[idx+1:]:        
                metric = Metrics(device='cpu', save_energy_force=False)
                metric.create_ode_fxn(
                    is_parallel=is_parallel,
                    fxn_names=[name1, name2],
                    fxn_scales=fxn_scales
                    )
                if is_parallel:
                    result = metric.ode_fxn(
                        eval_time=time,
                        time=time,
                        path=None,
                        energy=energy,
                        force=force,
                        velocity=velocity
                    )
                else:
                    result = metric.ode_fxn(
                        eval_time=time[0],
                        time=time[0].unsqueeze(0),
                        path=None,
                        energy=energy[0],
                        force=force[0],
                        velocity=velocity[0]
                    )

                compare = fxn_scales[0]*ode_results[name1]\
                    + fxn_scales[1]*ode_results[name2]
                if not is_parallel:
                    compare = compare[0]
                assert torch.allclose(compare, result),\
                    f"Multiple weighted metric results don't match for {name1} and {name2}, parallel {is_parallel}"