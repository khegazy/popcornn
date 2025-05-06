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
    path_output, ts_output = mep.optimize_path(*config.get('opt_params', []), output_ase_atoms=False)

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
        if path_output.energies_decomposed is None:
            energies_decomposed = None
            forces_decomposed = None
        else:
           energies_decomposed = path_output.energies_decomposed.tolist() 
           forces_decomposed = path_output.forces_decomposed.tolist() 
        with open(path_benchmark_filename, 'w') as file:
            json.dump(
                {
                    "time" : path_output.time.tolist(),
                    "positions" : path_output.positions.tolist(),
                    "velocities" : path_output.velocities.tolist(),
                    "energies" : path_output.energies.tolist(),
                    "energies_decomposed" : energies_decomposed,
                    "forces" : path_output.forces.tolist(),
                    "forces_decomposed" : forces_decomposed,
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
        path_output.positions.cpu().to(torch.float32),
        torch.tensor(path_benchmark['positions']),
        atol=pos_atol, rtol=pos_rtol
    )
    assert position_test, "path output position does not match benchmark"
    velocity_test = torch.allclose(
        path_output.velocities.cpu().to(torch.float32),
        torch.tensor(path_benchmark['velocities']),
        atol=V_atol, rtol=V_rtol
    )
    assert velocity_test, "path output velocity does not match benchmark"
    energy_test = torch.allclose(
        path_output.energies.cpu().to(torch.float32),
        torch.tensor(path_benchmark['energies']),
        atol=E_atol, rtol=E_rtol
    )
    assert energy_test, "path output energy does not match benchmark"
    if path_output.energies_decomposed is not None:
        energies_decomposed_test = torch.allclose(
            path_output.energies_decomposed.cpu().to(torch.float32),
            torch.tensor(path_benchmark['energies_decomposed']),
            atol=E_atol, rtol=E_rtol
        )
        assert energies_decomposed_test, "path output energies_decomposed does not match benchmark"
    force_test = torch.allclose(
        path_output.forces.cpu().to(torch.float32),
        torch.tensor(path_benchmark['forces']),
        atol=F_atol, rtol=F_rtol
    )
    assert force_test, "path output force does not match benchmark"
    if path_output.forces_decomposed is not None:
        forces_decomposed_test = torch.allclose(
            path_output.forces_decomposed.cpu().to(torch.float32),
            torch.tensor(path_benchmark['forces_decomposed']),
            atol=F_atol, rtol=F_rtol
        )
        assert forces_decomposed_test, "path output forces_decomposed does not match benchmark"


    # Compare TS output with benchmark
    ts_benchmark_filename = os.path.join(
        benchmark_path, f"{name}_ts.json"
    )
    if save_results:
        if ts_output.energies_decomposed is None:
            energies_decomposed = None
            forces_decomposed = None
        else:
           energies_decomposed = ts_output.energies_decomposed.tolist() 
           forces_decomposed = ts_output.forces_decomposed.tolist() 
        with open(ts_benchmark_filename, 'w') as file:
            json.dump(
                {
                    "time" : ts_output.time.tolist(),
                    "positions" : ts_output.positions.tolist(),
                    "velocities" : ts_output.velocities.tolist(),
                    "energies" : ts_output.energies.tolist(),
                    "energies_decomposed" : energies_decomposed,
                    "forces" : ts_output.forces.tolist(),
                    "forces_decomposed" : forces_decomposed,
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
        ts_output.positions.cpu().to(torch.float32),
        torch.tensor(ts_benchmark['positions']),
        atol=pos_atol, rtol=pos_rtol
    )
    assert position_test, "path output position does not match benchmark"
    velocity_test = torch.allclose(
        ts_output.velocities.cpu().to(torch.float32),
        torch.tensor(ts_benchmark['velocities']),
        atol=V_atol, rtol=V_rtol
    )
    assert velocity_test, "path output velocity does not match benchmark"
    energy_test = torch.allclose(
        ts_output.energies.cpu().to(torch.float32),
        torch.tensor(ts_benchmark['energies']),
        atol=E_atol, rtol=E_rtol
    )
    assert energy_test, "path output energy does not match benchmark"
    if ts_output.energies_decomposed is not None:
        energies_decomposed_test = torch.allclose(
            ts_output.energies_decomposed.cpu().to(torch.float32),
            torch.tensor(ts_benchmark['energies_decomposed']),
            atol=E_atol, rtol=E_rtol
        )
        assert energies_decomposed_test, "path output energies_decomposed does not match benchmark"
    force_test = torch.allclose(
        ts_output.forces.cpu().to(torch.float32),
        torch.tensor(ts_benchmark['forces']),
        atol=F_atol, rtol=F_rtol
    )
    assert force_test, "path output force does not match benchmark"
    if ts_output.forces_decomposed is not None:
        forces_decomposed_test = torch.allclose(
            ts_output.forces_decomposed.cpu().to(torch.float32),
            torch.tensor(ts_benchmark['forces_decomposed']),
            atol=F_atol, rtol=F_rtol
        )
        assert forces_decomposed_test, "path output forces_decomposed does not match benchmark"
    
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
        requires_energies=True,
        requires_velocities=True,
        requires_forces=True,
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
        requires_energies=True,
        requires_velocities=True,
        requires_forces=True
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