import numpy as np
import torch
import ase
from ase.calculators.singlepoint import SinglePointCalculator
# from popcornn.paths.base_path import PathOutput
# from popcornn.tools.preprocess import Images


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
    n_atoms = len(ref_images.atomic_numbers)
    for i in range(len(output)):
        atoms = ase.Atoms(
            numbers=ref_images.atomic_numbers.detach().cpu().numpy(),
            positions=output.positions[i].detach().cpu().numpy().reshape(n_atoms, 3),
            velocities=output.velocities[i].detach().cpu().numpy().reshape(n_atoms, 3) if output.velocities is not None else None,
            pbc=ref_images.pbc.detach().cpu().numpy(),
            cell=ref_images.cell.detach().cpu().numpy(),
            constraint=ase.constraints.FixAtoms(mask=ref_images.fix_positions.detach().cpu().numpy()),
            tags=ref_images.tags.detach().cpu().numpy(),
        )
        calc = SinglePointCalculator(
            atoms,
            energy=output.energies[i].detach().cpu().numpy().item() if output.energies is not None else None,
            forces=output.forces[i].detach().cpu().numpy().reshape(n_atoms, 3) if output.forces is not None else None,
        )
        atoms.calc = calc
        images.append(atoms)
    return images

def wrap_positions(
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        center: torch.Tensor = 0.5,
    ) -> torch.Tensor:
    """
    PyTorch implementation of ase.geometry.wrap_positions function.
    This function is also used for interatomic distances under minimum image convention.

    Parameters:
    -----------
    positions: float tensor of shape (n, 3)
        Positions of the atoms
    cell: float tensor of shape (3, 3)
        Unit cell vectors.
    pbc: one or 3 bool
        For each axis in the unit cell decides whether the positions
        will be moved along this axis.
    center: float tensor of shape (3,)
        The positons in fractional coordinates that the new positions
        will be nearest possible to.
    """

    if not isinstance(center, torch.Tensor):
        center = torch.ones(cell.shape[0], dtype=cell.dtype, device=cell.device) * center

    if not isinstance(pbc, torch.Tensor):
        pbc = torch.ones(cell.shape[0], dtype=torch.bool, device=cell.device) * pbc
    shift = center - 0.5

    # Don't change coordinates when pbc is False
    shift[~pbc] = 0.0

    assert cell[pbc].any(dim=1).all(), (cell, pbc)

    fractional = torch.linalg.solve(cell.T, positions.view(-1, cell.shape[-1]).T).T - shift

    fractional[:, pbc] = fractional[:, pbc] % 1.0 - shift[pbc]

    return torch.matmul(fractional, cell).view(*positions.shape)

def radius_graph(
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        cutoff: float,
        n_data: int,
        n_atoms: int,
    ) -> torch.Tensor:
    """
    Create a graph of atom pairs within a cutoff distance.

    Parameters:
    -----------
    positions: float tensor of shape (n_data * n_atoms, 3)
        Positions of the atoms.
    cell: float tensor of shape (3, 3)
        Unit cell vectors.
    pbc: one or 3 bool
        For each axis in the unit cell decides whether the positions
        will be moved along this axis.
    cutoff: float
        Cutoff distance for the graph.
    n_data: int
        Number of data points.
    n_atoms: int
        Number of atoms in each data point.

    Returns:
    --------
    torch.Tensor
        Graph of atom pairs within the cutoff distance.
    """
    # Create all-pairs distance matrix
    edge_index = torch.triu_indices(n_atoms, n_atoms, offset=1)
    edge_index = edge_index[:, None, :] + torch.arange(n_atoms, device=positions.device)[None, :, None] * n_data
    edge_index = edge_index.view(2, -1)
    disp = positions[:, edge_index[0]] - positions[:, edge_index[1]]

    # Apply periodic boundary conditions if cell is provided
    if pbc.any():
        disp = wrap_positions(disp, cell, pbc, center=1.0)

    # Create graph based on cutoff distance
    edge_index = edge_index[:, disp.norm(dim=-1) < cutoff]

    return edge_index