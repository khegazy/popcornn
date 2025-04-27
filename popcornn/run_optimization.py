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
    time: list
    reaction_path: list
    energy: list
    velocity: list
    force: list
    loss: list
    integral: float
    ts_time: list
    ts_structure: list
    ts_energy: list
    ts_velocity: list
    ts_force: list

    def save(self, file):
        with open(file, 'w') as f:
            json.dump(self.__dict__, f)


def optimize_MEP(
        images: list[Atoms],
        potential_params: dict[str, Any] = {},
        path_params: dict[str, Any] = {},
        integrator_params: dict[str, Any] = {},
        optimizer_params: dict[str, Any] = {},
        num_optimizer_iterations: int = 1001,
        num_record_points: int = 101,
        save_optimization_freq: int = 10,
        optimization_dir: str | None = "./logs",
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
    if save_optimization_freq is not None:
        os.makedirs(optimization_dir, exist_ok=True)

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
    for optim_idx in tqdm(range(num_optimizer_iterations)):
        path.neval = 0
        try:
            path_integral = optimizer.optimization_step(path, integrator)
            neval = path.neval
        except ValueError as e:
            print("ValueError", e)
            raise e

        if save_optimization_freq is not None:
            if optim_idx % save_optimization_freq == 0:
                time = path_integral.t.flatten()
                ts_time = path.TS_time
                path_output = path(time, return_velocity=True, return_energy=True, return_force=True)
                ts_output = path(ts_time, return_velocity=True, return_energy=True, return_force=True)

                optimization_output = OptimizationOutput(
                    time=time.tolist(),
                    reaction_path=path_output.path_geometry.tolist(),
                    energy=path_output.path_energy.tolist(),
                    velocity=path_output.path_velocity.tolist(),
                    force=path_output.path_force.tolist(),
                    loss=path_integral.y.tolist(),
                    integral=path_integral.integral.item(),
                    ts_time=ts_time.tolist(),
                    ts_structure=ts_output.path_geometry.tolist(),
                    ts_energy=ts_output.path_energy.tolist(),
                    ts_velocity=ts_output.path_velocity.tolist(),
                    ts_force=ts_output.path_force.tolist(),
                )
                
                optimization_output.save(
                    os.path.join(
                        optimization_dir, f"optimization_step-{optim_idx}.json"
                    )
                )            


        if optimizer.converged:
            print(f"Converged at step {optim_idx}")
            break

    torch.cuda.empty_cache()

    #####  Save optimization output  #####
    time = torch.linspace(path.t_init.item(), path.t_final.item(), num_record_points)
    ts_time = path.TS_time
    path_output = path(time, return_velocity=True, return_energy=True, return_force=True)
    ts_output = path(ts_time, return_velocity=True, return_energy=True, return_force=True)
    optimization_results = OptimizationOutput(
        time=time.tolist(),
        reaction_path=path_output.path_geometry.tolist(),
        energy=path_output.path_energy.tolist(),
        velocity=path_output.path_velocity.tolist(),
        force=path_output.path_force.tolist(),
        loss=path_integral.y.tolist(),
        integral=path_integral.integral.item(),
        ts_time=ts_time.tolist(),
        ts_structure=ts_output.path_geometry.tolist(),
        ts_energy=ts_output.path_energy.tolist(),
        ts_velocity=ts_output.path_velocity.tolist(),
        ts_force=ts_output.path_force.tolist(),
    )
 
    if issubclass(images.dtype, Atoms):
        images, ts_images = output_to_atoms(path_output, images), output_to_atoms(ts_output, images)
        return images, ts_images[0], optimization_results
    else:
        return path_output, ts_output, optimization_results

