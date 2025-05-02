import os
import json
import torch
import numpy as np
from popcornn import tools
from popcornn import Popcornn
from popcornn.tools.metrics import Metrics
from popcornn.tools.integrator import ODEintegrator
from popcornn.optimization.path_optimizer import PathOptimizer


def popcornn_run_test(name, config_path, benchmark_path, save_results=False):
    # Setup environment 
    os.makedirs(benchmark_path, exist_ok=True)
    torch.manual_seed(2025)
    np.random.seed(2025)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get config file
    config = tools.import_run_config(config_path)

    # Run the optimization
    mep = Popcornn(device=device, **config.get('init_params', {}))
    path_output, ts_output = mep.run(*config.get('opt_params', []), output_ase_atoms=False)

    # Compare path output with saved benchmarks
    T_atol, T_rtol = 1e-6, 1e-6
    pos_atol, pos_rtol = 1e-4, 1e-4
    V_atol, V_rtol = 1e-4, 1e-4
    E_atol, E_rtol = 1e-4, 1e-5
    F_atol, F_rtol = 1e-4, 1e-5

    path_benchmark_filename = os.path.join(
        benchmark_path, f"{name}_path.json"
    )
    if save_results:
        if path_output.energyterms is None:
            energyterms = None
            forceterms = None
        else:
           energyterms = path_output.energyterms.tolist() 
           forceterms = path_output.forceterms.tolist() 
        with open(path_benchmark_filename, 'w') as file:
            json.dump(
                {
                    "time" : path_output.time.tolist(),
                    "position" : path_output.position.tolist(),
                    "velocity" : path_output.velocity.tolist(),
                    "energy" : path_output.energy.tolist(),
                    "energyterms" : energyterms,
                    "force" : path_output.force.tolist(),
                    "forceterms" : forceterms,
                },
                file
            )
    with open(path_benchmark_filename, 'r') as file:
        path_benchmark = json.load(file)
    
    time_test = torch.allclose(
        path_output.time.cpu().to(torch.float32),
        torch.tensor(path_benchmark['time']),
        atol=T_atol, rtol=T_rtol
    )
    assert time_test, "path output time does not match benchmark"
    position_test = torch.allclose(
        path_output.position.cpu().to(torch.float32),
        torch.tensor(path_benchmark['position']),
        atol=pos_atol, rtol=pos_rtol
    )
    assert position_test, "path output position does not match benchmark"
    velocity_test = torch.allclose(
        path_output.velocity.cpu().to(torch.float32),
        torch.tensor(path_benchmark['velocity']),
        atol=V_atol, rtol=V_rtol
    )
    assert velocity_test, "path output velocity does not match benchmark"
    energy_test = torch.allclose(
        path_output.energy.cpu().to(torch.float32),
        torch.tensor(path_benchmark['energy']),
        atol=E_atol, rtol=E_rtol
    )
    assert energy_test, "path output energy does not match benchmark"
    if path_output.energyterms is not None:
        energyterms_test = torch.allclose(
            path_output.energyterms.cpu().to(torch.float32),
            torch.tensor(path_benchmark['energyterms']),
            atol=E_atol, rtol=E_rtol
        )
        assert energyterms_test, "path output energyterms does not match benchmark"
    force_test = torch.allclose(
        path_output.force.cpu().to(torch.float32),
        torch.tensor(path_benchmark['force']),
        atol=F_atol, rtol=F_rtol
    )
    assert force_test, "path output force does not match benchmark"
    if path_output.forceterms is not None:
        forceterms_test = torch.allclose(
            path_output.forceterms.cpu().to(torch.float32),
            torch.tensor(path_benchmark['forceterms']),
            atol=F_atol, rtol=F_rtol
        )
        assert forceterms_test, "path output forceterms does not match benchmark"


    # Compare TS output with benchmark
    ts_benchmark_filename = os.path.join(
        benchmark_path, f"{name}_TS.json"
    )
    if save_results:
        if ts_output.energyterms is None:
            energyterms = None
            forceterms = None
        else:
           energyterms = ts_output.energyterms.tolist() 
           forceterms = ts_output.forceterms.tolist() 
        with open(ts_benchmark_filename, 'w') as file:
            json.dump(
                {
                    "time" : ts_output.time.tolist(),
                    "position" : ts_output.position.tolist(),
                    "velocity" : ts_output.velocity.tolist(),
                    "energy" : ts_output.energy.tolist(),
                    "energyterms" : energyterms,
                    "force" : ts_output.force.tolist(),
                    "forceterms" : forceterms,
                },
                file
            )
    with open(ts_benchmark_filename, 'r') as file:
        ts_benchmark = json.load(file)

    time_test = torch.allclose(
        ts_output.time.cpu().to(torch.float32),
        torch.tensor(ts_benchmark['time']),
        atol=T_atol, rtol=T_rtol
    )
    assert time_test, "path output time does not match benchmark"
    position_test = torch.allclose(
        ts_output.position.cpu().to(torch.float32),
        torch.tensor(ts_benchmark['position']),
        atol=pos_atol, rtol=pos_rtol
    )
    assert position_test, "path output position does not match benchmark"
    velocity_test = torch.allclose(
        ts_output.velocity.cpu().to(torch.float32),
        torch.tensor(ts_benchmark['velocity']),
        atol=V_atol, rtol=V_rtol
    )
    assert velocity_test, "path output velocity does not match benchmark"
    energy_test = torch.allclose(
        ts_output.energy.cpu().to(torch.float32),
        torch.tensor(ts_benchmark['energy']),
        atol=E_atol, rtol=E_rtol
    )
    assert energy_test, "path output energy does not match benchmark"
    if ts_output.energyterms is not None:
        energyterms_test = torch.allclose(
            ts_output.energyterms.cpu().to(torch.float32),
            torch.tensor(ts_benchmark['energyterms']),
            atol=E_atol, rtol=E_rtol
        )
        assert energyterms_test, "path output energyterms does not match benchmark"
    force_test = torch.allclose(
        ts_output.force.cpu().to(torch.float32),
        torch.tensor(ts_benchmark['force']),
        atol=F_atol, rtol=F_rtol
    )
    assert force_test, "path output force does not match benchmark"
    if ts_output.forceterms is not None:
        forceterms_test = torch.allclose(
            ts_output.forceterms.cpu().to(torch.float32),
            torch.tensor(ts_benchmark['forceterms']),
            atol=F_atol, rtol=F_rtol
        )
        assert forceterms_test, "path output forceterms does not match benchmark"
    
    return mep, path_output, ts_output


def scheduler_test(path, config, schedule_fxn, device):
    # Shortcuts
    config = config['opt_params'][0]
    scheduler_config = config['optimizer_params']['path_ode_schedulers']
    fxn1_name = config['integrator_params']['path_ode_names'][0]
    fxn2_name = config['integrator_params']['path_ode_names'][1]

    # Run optimizer and get integration values with scheduler
    integrator = ODEintegrator(
        **config['integrator_params'], device=device
    )
    optimizer = PathOptimizer(
        path=path,
        **config['optimizer_params'],
        device=device
    )

    time = None
    scheduled_evals = []
    for i in range(scheduler_config[fxn2_name]['last_step']):
        path_integral = optimizer.optimization_step(
            path, integrator, time=time, update_path=False 
        )
        scheduled_evals.append(
            torch.flatten(path_integral.y, start_dim=0, end_dim=1)
        )
        time = torch.concatenate(
            [path_integral.t[:,0,:], torch.tensor([[1]], device=device)],
            dim=0
        )
    scheduled_evals = torch.stack(scheduled_evals)
    
    # Calculate function values and weight with scheduler
    time = torch.flatten(path_integral.t, start_dim=0, end_dim=1)
    metrics = Metrics(device=device)
    fxn1 = getattr(metrics, fxn1_name)
    fxn1_val, _ = fxn1(
        eval_time=time,
        path=path,
        requires_energy=True,
        requires_velocity=True,
        requires_force=True,
    )
    fxn1_scheduled_vals = fxn1_val.unsqueeze(0)*schedule_fxn(
        scheduler_config[fxn1_name]['start_value'],
        scheduler_config[fxn1_name]['end_value'],
        scheduler_config[fxn1_name]['last_step'],
        device=device
    ).unsqueeze(-1).unsqueeze(-1)
    fxn2 = getattr(
        metrics, fxn2_name
    )
    fxn2_val, _ = fxn2(
        eval_time=time,
        path=path,
        requires_energy=True,
        requires_velocity=True,
        requires_force=True
    )
    fxn2_scheduled_vals = fxn2_val.unsqueeze(0)*schedule_fxn(
        scheduler_config[fxn2_name]['start_value'],
        scheduler_config[fxn2_name]['end_value'],
        scheduler_config[fxn2_name]['last_step'],
        device=device
    ).unsqueeze(-1).unsqueeze(-1)
    
    compare_schedule = fxn1_scheduled_vals[:,:,0] + fxn2_scheduled_vals[:,:,0] 
    assert torch.allclose(
        scheduled_evals[:,:,0], compare_schedule, atol=1e-6, rtol=1e-4
    )