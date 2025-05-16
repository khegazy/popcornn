import torch
import numpy as np
import scipy as sp
from dataclasses import dataclass
from einops import rearrange
from popcornn.tools import Images, wrap_positions
from popcornn.potentials.base_potential import BasePotential, PotentialOutput
from typing import Callable, Any
from ase import Atoms
from ase.io import read


@dataclass
class PathOutput():
    """
    Data class representing the output of a path computation.

    Attributes:
    -----------
    time : torch.Tensor
        The time at which the path was evaluated.
    positions : torch.Tensor
        The coordinates along the path.
    velocities : torch.Tensor, optional
        The velocities along the path (default is None).
    energies : torch.Tensor
        The potential energy along the path.
    forces : torch.Tensor, optional
        The force along the path (default is None).
    """
    time: torch.Tensor
    positions: torch.Tensor
    velocities: torch.Tensor = None
    energies: torch.Tensor = None
    energies_decomposed: torch.Tensor = None
    forces: torch.Tensor = None
    forces_decomposed: torch.Tensor = None

    def __len__(self):
        """
        Return the number of images.
        """
        return len(self.positions)


class BasePath(torch.nn.Module):
    """
    Base class for path representation.

    Attributes:
    -----------
    initial_position : torch.Tensor
        The initial point of the path.
    final_position : torch.Tensor
        The final point of the path.
    potential : PotentialBase
        The potential function.

    Methods:
    --------
    geometric_path(time, y, *args) -> torch.Tensor:
        Compute the geometric path at the given time.

    get_path(time=None, return_velocities=False, return_forces=False) -> PathOutput:
        Get the path for the given time.

    forward(t, return_velocities=False, return_forces=False) -> PathOutput:
        Compute the path output for the given time.
    """
    initial_position: torch.Tensor
    final_position: torch.Tensor

    def __init__(
            self,
            images: Images,
            device: torch.device = None,
            find_ts: bool = True,
        ) -> None:
        """
        Initialize the BasePath.

        Parameters:
        -----------
        initial_position : torch.Tensor
            The initial point of the path.
        final_position : torch.Tensor
            The final point of the path.
        """
        super().__init__()
        self.neval = 0
        self.find_ts = find_ts
        self.potential = None
        self.initial_position = images.positions[0].to(device)
        self.final_position = images.positions[-1].to(device)
        self._inp_reshaped = None
        if images.pbc is not None and images.pbc.any():
            def transform(positions, **kwargs):
                return wrap_positions(positions, images.cell, images.pbc, **kwargs)
            self.transform = transform
        else:
            self.transform = None
        self.fix_positions = (images.tags==0).to(device) if images.tags is not None else None
        self.device = device
        self.t_init = torch.tensor(
            [[0]], dtype=torch.float64, device=self.device
        )
        self.t_final = torch.tensor(
            [[1]], dtype=torch.float64, device=self.device
        )
        self.ts_time = None
        self.ts_region = None

    def set_potential(
            self,
            potential: BasePotential,
    ) -> None:
        """
        Set the potential function.

        Parameters:
        -----------
        potential : BasePotential
            The potential function to be used.
        """
        self.potential = potential

    def get_positions(
            self,
            time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the geometric path at the given time.

        Parameters:
        -----------
        time : torch.Tensor
            The time at which to evaluate the geometric path.

        Returns:
        --------
        torch.Tensor
            The geometric path at the given time.
        """
        raise NotImplementedError()

    
    def calculate_velocities(self, t, create_graph=True):
        return torch.autograd.functional.jacobian(
            lambda t: torch.sum(self.get_positions(t), axis=0),
            t,
            create_graph=create_graph,
            vectorize=True
        ).transpose(0, 1)[:, :, 0] 
    
    def _check_output(
            self,
            potential_output,
            return_energies: bool,
            return_energies_decomposed: bool,
            return_forces: bool,
            return_forces_decomposed: bool,
        ):
        name = type(self.potential).__name__
        if return_energies and potential_output.energies is None:
            raise ValueError(f"Potential {name} cannot calculate energies")
        if return_energies_decomposed and potential_output.energies_decomposed is None:
            raise ValueError(f"Potential {name} cannot calculate energies_decomposed")
        if return_forces and potential_output.forces is None:
            raise ValueError(f"Potential {name} cannot calculate forces")
        if return_forces_decomposed and potential_output.forces_decomposed is None:
            raise ValueError(f"Potential {name} cannot calculate forces_decomposed")
    
    def forward(
            self,
            time : torch.Tensor = None,
            return_velocities: bool = False,
            return_energies: bool = False,
            return_energies_decomposed: bool = False,
            return_forces: bool = False,
            return_forces_decomposed: bool = False,
    ) -> PathOutput:
        """
        Forward pass to compute the path, potential, velocities, and force.

        Parameters:
        -----------
        t : torch.Tensor
            The time tensor at which to evaluate the path.
        return_velocities : bool, optional
            Whether to return velocities along the path (default is False).
        return_forces : bool, optional
            Whether to return force along the path (default is False).

        Returns:
        --------
        PathOutput
            An instance of the PathOutput class containing the computed path, potential, velocities, force, and time.
        """
        time = self._reshape_in(time)
        time = time.to(torch.float64).to(self.device)

        self.neval += time.numel()

        positions = self.get_positions(time)
        if self.transform is not None:
            positions = self.transform(positions)
        if return_energies or return_energies_decomposed or return_forces or return_forces_decomposed:
            potential_output = self.potential(positions) 
            self._check_output(
                potential_output,
                return_energies=return_energies,
                return_energies_decomposed=return_energies_decomposed,
                return_forces=return_forces,
                return_forces_decomposed=return_forces_decomposed
            )
        else:
            potential_output = PotentialOutput()

        if return_velocities:
            velocities = self.calculate_velocities(time)
        else:
            velocities = None

        return PathOutput(
            time=self._reshape_out(time),
            positions=self._reshape_out(positions),
            velocities=self._reshape_out(velocities),
            energies=self._reshape_out(potential_output.energies),
            energies_decomposed=self._reshape_out(potential_output.energies_decomposed),
            forces=self._reshape_out(potential_output.forces),
            forces_decomposed=self._reshape_out(potential_output.forces_decomposed),
        )
    

    def _reshape_in(self, time):
        if time is None:
            time = torch.linspace(self.t_init.item(), self.t_final.item(), 101)
        
        if len(time.shape) == 3:
            self._inp_reshaped = True
            self._inp_shape = time.shape
            time = rearrange(time, 'b c t -> (b c) t')
        elif len(time.shape) == 2:
            self._inp_reshaped = False
            B, C, = None, None
        elif len(time.shape) == 1:
            self._inp_reshaped = False
            B, C, = None, None
            time = torch.unsqueeze(time, -1)
        else:
            raise ValueError(f"Input path time must be of dimensions [B, C, T], [B, T], or [B] where T is the time dimsion and is generally 1: instead got {time.shape}")

        return time


    def _reshape_out(self, result):
        if self._inp_reshaped is None:
            raise RuntimeError("Must call _reshape_in() before _reshape_out()")
        if self._inp_reshaped and result is not None:
            B, C, _ = self._inp_shape
            return rearrange(result, '(b c) d -> b c d', b=B, c=C)
        return result

    
    def ts_search(self, time, energies=None, forces=None, topk_E=7, topk_F=16, idx_shift=4, N_interp=100000):
        # Calculate missing energies and forces
        calc_energies = energies is None or torch.any(torch.isnan(energies))
        calc_forces = forces is None or torch.any(torch.isnan(forces))
        # Calculate energies and forces if too few time points
        N_input_times = time.shape[0]
        if len(time.shape) == 3:
            N_input_times = N_input_times*(time.shape[1] - 1) - 2
        if N_input_times < 11:
            time = torch.reshape(time, (-1, time.shape[-1]))
            time = torch.linspace(time[1,0], time[-2,0], 15)
            calc_energies = True
            calc_forces = True
        
        # Calculate energies and forces if necessary
        if calc_energies or calc_forces:
            path_output = self.forward(
                time, return_energies=calc_energies, return_forces=calc_forces
            )  
            if calc_energies:
                energies = path_output.energies
            if calc_forces:
                forces = path_output.forces
        
        if len(time.shape) == 3:
            # Remove repeated evaluations
            unique_mask = torch.all(time[0,1:] - time[0,:-1] > 1e-13, dim=-1)
            unique_mask = torch.concatenate([unique_mask, torch.tensor([True], device=self.device)])
            time = time[:,unique_mask]
            energies = energies[:,unique_mask]
            forces = forces[:,unique_mask]

            if len(time) > 1 and torch.all(torch.abs(time[:-1,-1] - time[1:,0]) < 1e-13):
                time = time[:,:-1]
                energies = energies[:,:-1]
                forces = forces[:,:-1]

            N_S = time.shape[0] 
            energies = energies.flatten()
            time = time[:,:,0].flatten()
            forces = torch.flatten(forces, start_dim=0, end_dim=1)
            if N_S > 3:
                N_C = N_S
            else:
                N_C = 1
                time = time[1:-1]
                energies = energies[1:-1]
                forces = forces[1:-1]
        else:
            N_C = 1
            idx_shift = idx_shift*5
            energies = energies.flatten()

        # Find highest energy points
        _, ts_idxs = torch.topk(energies, min(len(energies), topk_E))

        # Start at beginning of integration step
        ts_idxs = (ts_idxs//N_C)*N_C
        ts_idxs = torch.unique(ts_idxs, sorted=False)

        # Get time and energy range
        idxs_min = ts_idxs - idx_shift*N_C
        idxs_min[idxs_min<N_C] = N_C
        idxs_max = ts_idxs + idx_shift*(1 + N_C)
        idxs_max[idxs_max>=len(energies)] = len(energies) - N_C
        idx_ranges = {(idxs_min[i].item(), idxs_max[i].item()) for i in range(len(idxs_min))}
        
        interp_ts = []
        interp_Es = []
        interp_Fs = []
        interp_magFs = []
        self.ts_force_mag = torch.tensor([np.inf], device=self.device)
        for imin, imax in idx_ranges:
            t_interp = time[imin:imax].detach().cpu().numpy()
            #print(time.shape, t_interp.shape, energies[imin:imax].shape, forces[imin:imax].shape)
            ts_F_interp = sp.interpolate.interp1d(
                t_interp, forces[imin:imax].detach().cpu().numpy(), axis=0, kind='cubic'
            )
            ts_search = np.linspace(
                t_interp[0] + 1e-12,
                t_interp[-1] - 1e-12,
                N_interp
            )
            interp_F = ts_F_interp(ts_search)
            interp_magF = np.linalg.norm(interp_F, ord=2, axis=-1).flatten()
            ts_idx = np.argmin(interp_magF)
            if interp_magF[ts_idx] < self.ts_force_mag:
                self.ts_time = torch.tensor(ts_search[ts_idx], device=self.device)
                self.ts_force = torch.tensor(interp_F[ts_idx], device=self.device)
                self.ts_force_mag = interp_magF[ts_idx]
                ts_E_interp = sp.interpolate.interp1d(
                    t_interp,
                    energies[imin:imax].detach().cpu().numpy(),
                    kind='cubic'
                )
                interp_E = ts_E_interp(ts_search)
                self.ts_energy = torch.tensor(interp_E[ts_idx], device=self.device)
                ts_time_scale = t_interp[-1] - t_interp[0]

                if False and ts_search[0] < self.orig_ts_time and ts_search[-1] > self.orig_ts_time:
                    print(self.orig_ts_time.detach().cpu().numpy()[0,0], ts_search)
                    oidx = np.argmin(np.abs(self.orig_ts_time.detach().cpu().numpy()[0,0] - ts_search))
                    orig_FM = interp_magF[oidx]

            #interp_ts.append(ts_search)
            #interp_Es.append(ts_E_interp(ts_search))
            #interp_Fs.append(ts_F_interp(ts_search))
            #interp_magFs.append(np.linalg.norm(interp_Fs[-1], ord=2, axis=-1).flatten())
            #print("TS time", ts_search[0], ts_search[-1])
        #print("NEW METHOD", self.ts_time, self.ts_energy, self.ts_force_mag)
        #print("OLD METHOD", self.orig_ts_time, self.orig_ts_energy, orig_FM)
        #if torch.abs(self.ts_time - self.orig_ts_time).flatten()/self.orig_ts_time > 1e-2:
        #    print("WARNING FAILED CONVERGENCE")
        """
        interp_ts = np.array(interp_ts)
        interp_Es = np.array(interp_Es)
        interp_Fs = np.array(interp_Fs)
        interp_magFs = np.array(interp_magFs)
        ts_idx = np.argmin(interp_magFs)

        idx0 = ts_idx//N_interp
        idx1 = ts_idx % N_interp
        self.ts_time = torch.tensor(interp_ts[idx0, idx1], device=self.device)
        print("TS time", interp_ts[:,0], interp_ts[:,-1])
        print("SELECTED TS time", self.ts_time)
        if torch.abs(self.ts_time - self.orig_ts_time).flatten()/self.orig_ts_time > 1e-3:
            asdfas
        self.ts_energy = torch.tensor(interp_ts[idx0, idx1], device=self.device)
        self.ts_force = torch.tensor(interp_Fs[idx0, idx1], device=self.device)
        self.ts_force_mag = torch.linalg.vector_norm(
            self.ts_force, ord=2, dim=-1
        )
        """


        """
        t_interp = time[:(2*N_C + 1)*idx_shift].detach().cpu().numpy()
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
        print("NEW TS")
        print("\tidxs: ", idxs_min, idxs_max)
        #print("\tEs: ", E_interp)
        ts_E_interp = sp.interpolate.interp1d(
            t_interp, E_interp, axis=1, kind='cubic'
        )
        ts_F_interp = sp.interpolate.interp1d(
            t_interp, F_interp, axis=1, kind='cubic'
        )
        ts_search = np.linspace(
            t_interp[0] + 1e-12,
            t_interp[-1] - 1e-12,
            N_interp
        )
        ts_E_search = ts_E_interp(ts_search)
        ts_F_search = ts_F_interp(ts_search)
        ts_magF_search = np.linalg.norm(ts_F_search, ord=2, axis=-1).flatten()
        ts_idx = np.argmin(ts_magF_search)
        

        #ts_idxs = np.argpartition(ts_E_search.flatten(), -1*topk_F)[-1*topk_F:]
        #ts_time = ts_search[ts_idxs % N_interp]

        #ts_time = torch.tensor(ts_time) + time[idxs_min[ts_idxs//N_interp]]
        #path_output = path(ts_time, return_energies=True, return_forces=True)
        #ts_idx = torch.argmin(
        #    torch.linalg.vector_norm(path_output.path_force, ord=2, dim=-1)
        #)
        
        idx0 = ts_idx//N_interp
        idx1 = ts_idx % N_interp
        self.ts_time = ts_search[idx1]
        self.ts_time = torch.tensor(self.ts_time, device=self.device) + time[idxs_min[idx0]]
        print("TS time", ts_search[0] + time[idxs_min], ts_search[-1] + time[idxs_min])
        print("SELECTED TS time", self.ts_time)
        self.ts_energy = torch.tensor(ts_E_search[idx0, idx1], device=self.device)
        self.ts_force = torch.tensor(ts_F_search[idx0, idx1], device=self.device)
        self.ts_force_mag = torch.linalg.vector_norm(
            self.ts_force, ord=2, dim=-1
        )
        """

        #ts_time_scale = t_interp[-1] - t_interp[0]
        self.ts_region = torch.linspace(
            self.ts_time-ts_time_scale/idx_shift,
            self.ts_time+ts_time_scale/idx_shift,
            11,
            device=self.device
        )
        #self.ts_time = torch.unsqueeze(self.ts_time, -1)
        #self.ts_time = torch.unsqueeze(self.ts_time, -1)
        #print(self.ts_energy, self.ts_force_mag, ts_magF_search[ts_idx], self.ts_force)
 