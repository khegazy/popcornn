import numpy as np
import torch
import ase
from ase.calculators.singlepoint import SinglePointCalculator
# from popcornn.paths.base_path import PathOutput
# from popcornn.tools.preprocess import Images


def pair_displacement(
        initial_atoms: ase.Atoms, 
        final_atoms: ase.Atoms,
    ) -> np.ndarray:
    """
    Pair displacement between two Atoms objects.

    Parameters:
    -----------
    initial_atoms : ase.Atoms
        Initial Atoms object.
    final_atoms : ase.Atoms
        Final Atoms object.

    Returns:
    --------
    np.ndarray
        Pair displacement.
    """
    assert len(initial_atoms) == len(final_atoms), "Initial and final atoms must have the same number of atoms."
    pair = initial_atoms + final_atoms
    vec = pair.get_distances(
        [i for i in range(len(initial_atoms))],
        [i + len(initial_atoms) for i in range(len(initial_atoms))],
        mic=True,
        vector=True,
    )
    return vec

def output_to_atoms(output, ref_images):
    """
    Convert output to ase.Atoms.
    
    Parameters:
    -----------
    output : paths.PathOutput
        Path output.
    ref_images : tools.Images
        Reference images.

    Returns:
    --------
    list[ase.Atoms]
        List of Atoms objects.
    """
    images = []
    n_atoms = len(ref_images.numbers)
    for positions, energy, velocities, forces in zip(output.position, output.energy, output.velocity, output.force):
        atoms = ase.Atoms(
            numbers=ref_images.numbers.detach().cpu().numpy(),
            positions=positions.detach().cpu().numpy().reshape(n_atoms, 3),
            velocities=velocities.detach().cpu().numpy().reshape(n_atoms, 3),
            pbc=ref_images.pbc.detach().cpu().numpy(),
            cell=ref_images.cell.detach().cpu().numpy(),
            tags=ref_images.tags.detach().cpu().numpy(),
        )
        calc = SinglePointCalculator(
            atoms,
            energy=energy.detach().cpu().numpy().item(),
            forces=forces.detach().cpu().numpy().reshape(n_atoms, 3),
        )
        atoms.calc = calc
        images.append(atoms)
    return images

def wrap_points(
        points: torch.Tensor,
        cell: torch.Tensor,
        center: float = 0
    ) -> torch.Tensor:
    """
    PyTorch implementation of ase.geometry.wrap_positions function.
    Assume periodic boundary conditions for all dimensions.
    """

    fractional = torch.linalg.solve(cell.T, points.view(*points.shape[:-1], -1, 3).transpose(-1, -2)).transpose(-1, -2)

    # fractional[..., :, self.pbc] %= 1.0
    fractional = (fractional + center) % 1.0 - center    # TODO: Modify this to handle partially true PBCs

    return torch.matmul(fractional, cell).view(*points.shape)