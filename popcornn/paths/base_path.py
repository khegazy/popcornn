import torch
import numpy as np
import scipy as sp
from dataclasses import dataclass
from popcornn.tools import pair_displacement, wrap_points
from popcornn.tools import Images
from popcornn.potentials.base_potential import BasePotential
from typing import Callable, Any
from ase import Atoms
from ase.io import read


@dataclass
class PathOutput():
    """
    Data class representing the output of a path computation.

    Attributes:
    -----------
    path_geometry : torch.Tensor
        The coordinates along the path.
    path_velocity : torch.Tensor, optional
        The velocity along the path (default is None).
    path_energy : torch.Tensor
        The potential energy along the path.
    path_force : torch.Tensor, optional
        The force along the path (default is None).
    times : torch.Tensor
        The times at which the path was evaluated.
    """
    times: torch.Tensor
    path_geometry: torch.Tensor
    path_energy: torch.Tensor = None
    path_velocity: torch.Tensor = None
    path_force: torch.Tensor = None
    path_forceterms: torch.Tensor = None


class BasePath(torch.nn.Module):
    """
    Base class for path representation.

    Attributes:
    -----------
    initial_point : torch.Tensor
        The initial point of the path.
    final_point : torch.Tensor
        The final point of the path.
    potential : PotentialBase
        The potential function.

    Methods:
    --------
    geometric_path(time, y, *args) -> torch.Tensor:
        Compute the geometric path at the given time.

    get_path(times=None, return_velocity=False, return_force=False) -> PathOutput:
        Get the path for the given times.

    forward(t, return_velocity=False, return_force=False) -> PathOutput:
        Compute the path output for the given times.
    """
    initial_point: torch.Tensor
    final_point: torch.Tensor
    potential: BasePotential

    def __init__(
            self,
            potential: BasePotential,
            images: Images,
            device: torch.device = None,
            **kwargs: Any
        ) -> None:
        """
        Initialize the BasePath.

        Parameters:
        -----------
        potential : callable
            The potential function.
        initial_point : torch.Tensor
            The initial point of the path.
        final_point : torch.Tensor
            The final point of the path.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__()
        self.neval = 0
        self.potential = potential
        self.initial_point = images.points[0].to(device)
        self.final_point = images.points[-1].to(device)
        self.vec = images.vec.to(device)
        if images.pbc is not None and images.pbc.any():
            def transform(points):
                return wrap_points(points, images.cell)
            self.transform = transform
        else:
            self.transform = None
        self.device = device
        self.t_init = torch.tensor(
            [[0]], dtype=torch.float64, device=self.device
        )
        self.t_final = torch.tensor(
            [[1]], dtype=torch.float64, device=self.device
        )
        self.TS_time = None
        self.TS_region = None

    def get_geometry(
            self,
            time: torch.Tensor,
            *args: Any
    ) -> torch.Tensor:
        """
        Compute the geometric path at the given time.

        Parameters:
        -----------
        time : torch.Tensor
            The time at which to evaluate the geometric path.
        y : Any
            Placeholder for additional arguments.
        *args : Any
            Additional arguments.

        Returns:
        --------
        torch.Tensor
            The geometric path at the given time.
        """
        raise NotImplementedError()
    
    def forward(
            self,
            t : torch.Tensor = None,
            return_velocity: bool = False,
            return_energy: bool = False,
            return_force: bool = False,
            return_forceterms: bool = False
    ) -> PathOutput:
        """
        Forward pass to compute the path, potential, velocity, and force.

        Parameters:
        -----------
        t : torch.Tensor
            The time tensor at which to evaluate the path.
        return_velocity : bool, optional
            Whether to return velocity along the path (default is False).
        return_force : bool, optional
            Whether to return force along the path (default is False).

        Returns:
        --------
        PathOutput
            An instance of the PathOutput class containing the computed path, potential, velocity, force, and times.
        """
        if t is None:
            t = torch.linspace(self.t_init.item(), self.t_final.item(), 101)
        while len(t.shape) < 2:
            t = torch.unsqueeze(t, -1)
        t = t.to(torch.float64).to(self.device)

        self.neval += t.numel()
        # print(time)
        # if self.neval > 1e5:
        #     raise ValueError("Too many evaluations!")

        path_geometry = self.get_geometry(t)
        if self.transform is not None:
            path_geometry = self.transform(path_geometry)
        if return_energy or return_force or return_forceterms:
            potential_output = self.potential(path_geometry)

        if return_energy:
            if potential_output.energy is not None:
                path_energy = potential_output.energy
            elif potential_output.energy_terms is not None:
                path_energy = potential_output.energy_terms.sum(dim=-1, keepdim=True)
            else:
                raise ValueError("No energy or energy terms found.")
        else:
            path_energy = None

        if return_force:
            if potential_output.force is not None:
                path_force = potential_output.force
            else:
                path_force = -torch.autograd.grad(
                    path_energy,
                    path_geometry,
                    grad_outputs=torch.ones_like(path_energy),
                    create_graph=self.training,
                )[0]
        else:
            path_force = None
        if return_forceterms:
            if potential_output.force_terms is not None:
                path_forceterms = potential_output.force_terms
            else:
                path_forceterms = -torch.vmap(
                    lambda vec: torch.autograd.grad(
                        potential_output.energy_terms.flatten(), path_geometry, grad_outputs=vec, create_graph=self.training
                    )[0],
                )(torch.eye(potential_output.energy_terms.shape[1], device=self.device).repeat(1, potential_output.energy_terms.shape[0])).transpose(0, 1)
        else:
            path_forceterms = None
        if return_velocity:
            # if is_batched:
            #     fxn = lambda t: torch.sum(self.geometric_path(t), axis=0)
            # else:
            #     fxn = lambda t: self.geometric_path(t)
            # velocity = torch.autograd.functional.jacobian(
            #     fxn, t, create_graph=self.training, vectorize=is_batched
            # )
            path_velocity = torch.autograd.functional.jacobian(
                lambda t: torch.sum(self.get_geometry(t), axis=0), t, create_graph=self.training, vectorize=True
            ).transpose(0, 1)[:, :, 0]
        else:
            path_velocity = None

        if return_energy or return_force or return_forceterms:
            del potential_output
        
        return PathOutput(
            times=t,
            path_geometry=path_geometry,
            path_energy=path_energy,
            path_velocity=path_velocity,
            path_force=path_force,
            path_forceterms=path_forceterms,
        )
    
    def find_TS(self, times, energies, forces, topk_E=7, topk_F=16, idx_shift=3, N_interp=10000):
        # Remove repeated evaluations
        unique_mask = torch.all(times[0,1:] - times[0,:-1] > 1e-13, dim=-1)
        unique_mask = torch.concatenate([unique_mask, torch.tensor([True])])
        times = times[:,unique_mask]
        energies = energies[:,unique_mask]
        forces = forces[:,unique_mask]

        if torch.all(torch.abs(times[:-1,-1] - times[1:,0]) < 1e-13):
            times = times[:,:-1]
            energies = energies[:,:-1]
            forces = forces[:,:-1]
        
        # Find highest energy points
        N_C = times.shape[-2]
        energies = energies.flatten()
        forces = torch.flatten(forces, start_dim=0, end_dim=1)
        times = times[:,:,0].flatten()
        _, TS_idxs = torch.topk(energies, topk_E)
        TS_idxs = TS_idxs - (TS_idxs % N_C)
        TS_idxs = torch.unique(TS_idxs, sorted=False)

        # Get time and energy range
        max_min = len(energies) - (2*N_C + 1)*idx_shift
        idxs_min = TS_idxs - idx_shift*N_C
        idxs_min[idxs_min>max_min] = max_min
        idxs_min[idxs_min<0] = 0
        min_max = (2*N_C + 1)*idx_shift
        idxs_max = TS_idxs + idx_shift*(1 + N_C)
        idxs_max[idxs_max<min_max] = min_max
        idxs_max[idxs_max>=len(energies)] = len(energies)
        
        t_interp = times[:(2*N_C + 1)*idx_shift].detach().cpu().numpy()
        F_interp = np.stack(
            [
                forces[idxs_min[i]:idxs_max[i]].detach().cpu().numpy()\
                for i in range(len(idxs_max)) 
            ]
        )
        E_interp = np.stack(
            [
                energies[idxs_min[i]:idxs_max[i]].detach().cpu().numpy()\
                for i in range(len(idxs_max)) 
            ]
        )
        TS_E_interp = sp.interpolate.interp1d(
            t_interp, E_interp, axis=1, kind='cubic'
        )
        TS_F_interp = sp.interpolate.interp1d(
            t_interp, F_interp, axis=1, kind='cubic'
        )
        TS_search = np.linspace(
            t_interp[0] + 1e-12,
            t_interp[-1] - 1e-12,
            N_interp
        )
        TS_E_search = TS_E_interp(TS_search)
        TS_F_search = TS_F_interp(TS_search)
        TS_magF_search = np.linalg.norm(TS_F_search, ord=2, axis=-1).flatten()
        TS_idx = np.argmin(TS_magF_search)
        

        """
        TS_idxs = np.argpartition(TS_E_search.flatten(), -1*topk_F)[-1*topk_F:]
        TS_times = TS_search[TS_idxs % N_interp]

        TS_times = torch.tensor(TS_times) + times[idxs_min[TS_idxs//N_interp]]
        path_output = path(TS_times, return_energy=True, return_force=True)
        TS_idx = torch.argmin(
            torch.linalg.vector_norm(path_output.path_force, ord=2, dim=-1)
        )
        """
        
        idx0 = TS_idx//N_interp
        idx1 = TS_idx % N_interp
        self.TS_time = TS_search[idx1]
        self.TS_time = torch.tensor(self.TS_time, device=self.device) + times[idxs_min[idx0]]
        self.TS_energy = torch.tensor(TS_E_search[idx0, idx1], device=self.device)
        self.TS_force = torch.tensor(TS_F_search[idx0, idx1], device=self.device)
        self.TS_force_mag = torch.linalg.vector_norm(
            self.TS_force, ord=2, dim=-1
        )

        TS_time_scale = t_interp[-1] - t_interp[0]
        self.TS_region = torch.linspace(
            self.TS_time-TS_time_scale/idx_shift,
            self.TS_time+TS_time_scale/idx_shift,
            11,
            device=self.device
        )
        self.TS_time = torch.unsqueeze(self.TS_time, -1)
        #print(self.TS_energy, self.TS_force_mag, TS_magF_search[TS_idx], self.TS_force)
 
