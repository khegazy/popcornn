import os
import torch
import numpy as np
from typing import Any
import time as time
from tqdm import tqdm
from ase import Atoms
from dataclasses import dataclass
import json

from popcornn.paths import get_path
from popcornn.optimization import initialize_path
from popcornn.optimization import PathOptimizer
from popcornn.tools import process_images, output_to_atoms
from popcornn.tools import ODEintegrator
from popcornn.potentials import get_potential


@dataclass
class OptimizationOutput():
    times: list
    position: list
    energy: list
    velocity: list
    force: list
    loss: list
    integral: float
    ts_time: list
    ts_geometry: list
    ts_energy: list
    ts_velocity: list
    ts_force: list

    def save(self, file):
        with open(file, 'w') as f:
            json.dump(self.__dict__, f)


def optimize_MEP(
        images: list[Atoms],
        output_dir: str | None = None,
        potential_params: dict[str, Any] = {},
        path_params: dict[str, Any] = {},
        integrator_params: dict[str, Any] = {},
        optimizer_params: dict[str, Any] = {},
        num_optimizer_iterations: int = 1001,
        num_record_points: int = 101,
        device: str = 'cuda',
):
    """
    Run optimization process.

    Args:
        args (NamedTuple): Command line arguments.
        config (NamedTuple): Configuration settings.
        path_config (NamedTuple): Path configuration.
        logger (NamedTuple): Logger settings.
    """
    print("Images", images)
    print("Potential Params", potential_params)
    print("Path Params", path_params)
    print("Integrator Params", integrator_params)
    print("Optimizer Params", optimizer_params)

    torch.manual_seed(42)

    # Create output directories
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_dir = os.path.join(output_dir, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        plot_dir = os.path.join(output_dir, "plots")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

    #####  Process images  #####
    images = process_images(images)
    
    #####  Get chemical potential  #####
    potential = get_potential(**potential_params, images=images, device=device)

    #####  Get path prediction method  #####
    path = get_path(potential=potential, images=images, **path_params, device=device)

    # Randomly initialize the path, otherwise a straight line
    if len(images) > 2:
        path = initialize_path(
            path=path, 
            times=torch.linspace(0, 1, len(images), device=device), 
            init_points=images.points.to(device),
        )

    #####  Path optimization tools  #####
    integrator = ODEintegrator(**integrator_params, device=device)

    # Gradient descent path optimizer
    optimizer = PathOptimizer(path=path, **optimizer_params, device=device)

    ##########################################
    #####  Optimize minimum energy path  ##### 
    ##########################################
    path.train()
    for optim_idx in tqdm(range(num_optimizer_iterations)):
        path.neval = 0
        try:
            path_integral = optimizer.optimization_step(path, integrator)
            neval = path.neval
        except ValueError as e:
            print("ValueError", e)
            raise e

        if output_dir is not None:
            times = path_integral.t.flatten()
            ts_time = path.TS_time
            path_output = path(times, return_velocity=True, return_energy=True, return_force=True)
            ts_output = path(torch.tensor([ts_time]), return_velocity=True, return_energy=True, return_force=True)

            output = OptimizationOutput(
                times=times.tolist(),
                position=path_output.position.tolist(),
                energy=path_output.energy.tolist(),
                velocity=path_output.velocity.tolist(),
                force=path_output.force.tolist(),
                loss=path_integral.y.tolist(),
                integral=path_integral.integral.item(),
                ts_time=ts_time.tolist(),
                ts_geometry=ts_output.position.tolist(),
                ts_energy=ts_output.energy.tolist(),
                ts_velocity=ts_output.velocity.tolist(),
                ts_force=ts_output.force.tolist(),
            )
            output.save(os.path.join(log_dir, f"output_{optim_idx}.json"))            


        if optimizer.converged:
            print(f"Converged at step {optim_idx}")
            break
    path.TS_search(path_integral.t)

    torch.cuda.empty_cache()

    #####  Save optimization output  #####
    time = torch.linspace(path.t_init.item(), path.t_final.item(), num_record_points)
    ts_time = path.TS_time
    path_output = path(time, return_velocity=True, return_energy=True, return_force=True)
    ts_output = path(torch.tensor([ts_time]), return_velocity=True, return_energy=True, return_force=True)
    if issubclass(images.dtype, Atoms):
        images, ts_images = output_to_atoms(path_output, images), output_to_atoms(ts_output, images)
        return images, ts_images[0]
    else:
        # return OptimizationOutput(
        #     path_time=time.tolist(),
        #     path_geometry=path_output.path_geometry.tolist(),
        #     path_energy=path_output.path_energy.tolist(),
        #     path_velocity=path_output.path_velocity.tolist(),
        #     path_force=path_output.path_force.tolist(),
        #     path_loss=path_integral.y.tolist(),
        #     path_integral=path_integral.integral.item(),
        #     path_ts_time=ts_time.tolist(),
        #     path_ts_geometry=ts_output.path_geometry.tolist(),
        #     path_ts_energy=ts_output.path_energy.tolist(),
        #     path_ts_velocity=ts_output.path_velocity.tolist(),
        #     path_ts_force=ts_output.path_force.tolist(),
        # )
        return path_output, ts_output

