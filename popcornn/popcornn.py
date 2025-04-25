import os
from copy import deepcopy
import torch
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
    path_time: list
    path_geometry: list
    path_energy: list
    path_velocity: list
    path_force: list
    path_loss: list
    path_integral: float
    path_ts_time: list
    path_ts_geometry: list
    path_ts_energy: list
    path_ts_velocity: list
    path_ts_force: list

    def save(self, file):
        with open(file, 'w') as f:
            json.dump(self.__dict__, f)

class Popcornn:
    """
    Wrapper class for Popcornn optimization.
    """
    def __init__(
            self, 
            images: list[Atoms],
            path_params: dict[str, Any] = {},
            device: str = 'cuda',
            seed: int = 42,
    ):
        print("Images", images)
        print("Potential Params", potential_params)
        print("Path Params", path_params)
        print("Integrator Params", integrator_params)
        print("Optimizer Params", optimizer_params)

        torch.manual_seed(seed)
        torch.cuda.empty_cache()

        # Process images
        self.images = process_images(images)

        # Get path prediction method
        self.path = get_path(potential=self.potential, images=self.images, **deepcopy(path_params), device=device)

        # Randomly initialize the path, otherwise a straight line
        if len(images) > 2:
            self.path = initialize_path(
                path=self.path, 
                times=torch.linspace(self.path.t_init.item(), self.path.t_final.item(), len(self.images), device=device), 
                init_points=self.images.points.to(device),
            )

    
    def run(
            params_list: list[dict], 
            output_dir: str | None = None,
            num_record_points: int = 101,
        ):
        """
        Run the optimization.
        
        Args:
            params_list (list[dict]): 
                List of dictionaries containing the parameters for each optimization run.
                Each dictionary should contain the following keys:
                - potential_params: Parameters for the potential.
                - integrator_params: Parameters for the loss integrator.
                - optimizer_params: Parameters for the path optimizer.
                - num_optimizer_iterations: Number of optimization iterations.
            output_dir (str | None):
                Directory to save the output files. If None, no files will be saved.
            num_record_points (int): 
                Number of points to record along the path when returning and saving the optimized path.
        """
        # Create output directories
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        # Optimize the path
        for i, params in enumerate(params_list):
            path_output, ts_output = self._optimize(
                **params, 
                output_dir=f"{output_dir}/opt_{i}" if output_dir is not None else None, 
                num_record_points=num_record_points,
            )
        
        # Return the optimized path
        return path_output, ts_output

    def _optimize(
            self,
            potential_params: dict[str, Any] = {},
            integrator_params: dict[str, Any] = {},
            optimizer_params: dict[str, Any] = {},
            num_optimizer_iterations: int = 1000,
            output_dir: str | None = None,
            num_record_points: int = 101,
    ):
        """
        Optimize the minimum energy path.
        """
        # Create output directories
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        # Get potential energy function
        potential = get_potential(images=self.images, **deepcopy(potential_params), device=device)
        self.path.set_potential(potential)

        # Path optimization tools
        integrator = ODEintegrator(**deepcopy(integrator_params), device=device)

        # Gradient descent path optimizer
        optimizer = PathOptimizer(path=self.path, **deepcopy(optimizer_params), device=device)

        # Create output directories
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            log_dir = os.path.join(output_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
        
        # Optimize the path
        for optim_idx in tqdm(range(num_optimizer_iterations)):
            try:
                path_integral = optimizer.optimization_step(self.path, integrator)
            except ValueError as e:
                print("ValueError", e)
                raise e

            # Save the path
            if output_dir is not None:
                time = path_integral.t.flatten()
                ts_time = self.path.TS_time
                path_output = self.path(time, return_velocity=True, return_energy=True, return_force=True)
                ts_output = self.path(ts_time, return_velocity=True, return_energy=True, return_force=True)
                
                with open(os.path.join(log_dir, f"output_{optim_idx}.json"), 'w') as file:
                    json.dump(
                        {
                            "path_time": time.tolist(),
                            "path_geometry": path_output.path_geometry.tolist(),
                            "path_energy": path_output.path_energy.tolist(),
                            "path_velocity": path_output.path_velocity.tolist(),
                            "path_force": path_output.path_force.tolist(),
                            "path_loss": path_integral.y.tolist(),
                            "path_integral": path_integral.integral.item(),
                            "path_ts_time": ts_time.tolist(),
                            "path_ts_geometry": ts_output.path_geometry.tolist(),
                            "path_ts_energy": ts_output.path_energy.tolist(),
                            "path_ts_velocity": ts_output.path_velocity.tolist(),
                            "path_ts_force": ts_output.path_force.tolist(),
                        }, 
                        file,
                    ) 

            # Check for convergence
            if optimizer.converged:
                print(f"Converged at step {optim_idx}")
                break
            
        time = torch.linspace(self.path.t_init.item(), self.path.t_final.item(), num_record_points, device=self.path.device)
        ts_time = self.path.TS_time
        path_output = self.path(time, return_velocity=True, return_energy=True, return_force=True)
        ts_output = self.path(ts_time, return_velocity=True, return_energy=True, return_force=True)
        if issubclass(images.dtype, Atoms):
            images, ts_images = output_to_atoms(path_output, images), output_to_atoms(ts_output, images)
            return images, ts_images[0]
        else:
            return path_output, ts_output

