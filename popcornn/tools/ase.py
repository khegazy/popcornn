import numpy as np
import torch
import ase
from ase import Atoms
from ase.geometry import find_mic
from ase.calculators.singlepoint import SinglePointCalculator
# fairchem is lazy-imported in radius_graph_fairchem to keep `import popcornn`
# off the Ray Serve / FSDP / wandb cold-import path (~5s on login, ~35s on compute).
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
            info={'charge': ref_images.charge.item(), 'spin': ref_images.spin.item()},
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

def unwrap_atoms(
        images: list[Atoms],
    ) -> list[Atoms]:
    """
    Unwrap atoms in a list of ASE Atoms objects.
    This function ensures that the positions of atoms in consecutive images
    are consistent by applying the minimum image convention (MIC).
    """
    for i in range(len(images) - 1):
        positions_i = images[i].get_positions()
        positions_f = images[i + 1].get_positions()
        cell = images[i].get_cell()
        pbc = images[i].get_pbc()

        # Unwrap positions
        diff = find_mic(positions_f - positions_i, cell=cell, pbc=pbc)[0]
        images[i + 1].set_positions(positions_i + diff)
    return images

def radius_graph(
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        cutoff: float = None,
        max_neighbors: int = None,
        method: str = "mic",
    ) -> torch.Tensor:
    """
    Create a graph of atom pairs within a cutoff distance.

    Parameters:
    -----------
    positions: float tensor of shape (n_data, n_atoms, 3)
        Positions of the atoms.
    cell: float tensor of shape (3, 3)
        Unit cell vectors.
    pbc: one or 3 bool
        For each axis in the unit cell decides whether the positions
        will be moved along this axis.
    cutoff: float
        Cutoff distance for the graph.
    max_neighbors: int
        Maximum number of neighbors for each atom. If -1, no limit is applied.
    method: str
        Method to use for creating the graph. Options are "mic" and "fairchem".

    Returns:
    --------
    torch.Tensor
        Graph of atom pairs within the cutoff distance.
    """
    if method == "mic":
        return radius_graph_mic(positions, cell, pbc, cutoff, max_neighbors)
    elif method == "fairchem":
        return radius_graph_fairchem(positions, cell, pbc, cutoff, max_neighbors)
    else:
        raise ValueError(f"Invalid method: {method}. Options are 'mic' and 'fairchem'.")

def radius_graph_mic(
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        cutoff: float = None,
        max_neighbors: int = None,
    ) -> torch.Tensor:
    """
    Create a graph of atom pairs within a cutoff distance.

    Parameters:
    -----------
    positions: float tensor of shape (n_data, n_atoms, 3)
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
    device = positions.device
    dtype = positions.dtype
    n_data, n_atoms, _ = positions.shape

    # Create all-pairs distance matrix
    edge_index = torch.triu_indices(n_atoms, n_atoms, offset=1, device=device)
    edge_index = edge_index[:, None, :] + torch.arange(n_data, device=device)[None, :, None] * n_atoms
    edge_index = edge_index.view(2, -1)
    positions_flat = positions.view(-1, 3)
    disp = positions_flat[edge_index[0]] - positions_flat[edge_index[1]]

    # Apply periodic boundary conditions if cell is provided
    if pbc.any():
        disp = wrap_positions(disp, cell, pbc, center=1.0)

    # Create graph based on cutoff distance
    dist = torch.norm(disp, dim=-1)
    if cutoff is not None:
        mask = dist < cutoff
        edge_index = edge_index[:, mask]
        edge_distance = dist[mask]
        edge_distance_vec = disp[mask]
    else:
        edge_index = edge_index
        edge_distance = dist
        edge_distance_vec = disp

    return {
        'edge_index': edge_index,
        'edge_distance': edge_distance,
        'edge_distance_vec': edge_distance_vec,
    }

def radius_graph_fairchem(
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        cutoff: float,
        max_neighbors: int,
    ) -> torch.Tensor:
    """
    Create a graph of atom pairs within a cutoff distance.

    Parameters:
    -----------
    positions: float tensor of shape (n_data, n_atoms, 3)
        Positions of the atoms.
    cell: float tensor of shape (3, 3)
        Unit cell vectors.
    pbc: bool tensor of shape (3,)
        For each axis in the unit cell decides whether the positions
        will be moved along this axis.
    cutoff: float
        Cutoff distance for the graph.
    max_neighbors: int
        Maximum number of neighbors for each atom. If -1, no limit is applied.

    Returns:
    --------
    graph_dict: dict
        Dictionary containing the graph information.
        - 'edge_index': int tensor of shape (2, n_edges)
            Indices of the edges in the graph.
        - 'edge_distance': float tensor of shape (n_edges,)
            Distances between the atoms connected by the edges.
        - 'edge_distance_vec': float tensor of shape (n_edges, 3)
            Vectors representing the distances between the atoms connected by the edges.
        - 'cell_offsets': float tensor of shape (n_edges, 3)
            Offsets of the cell vectors for each edge.
        - 'offset_distances': float tensor of shape (n_edges, 3)
            Distances between the atoms connected by the edges, including the cell offsets.
        - 'neighbors': int tensor of shape (n_graphs,)
            Number of neighbors for each atom.
    """
    from fairchem.core.graph.compute import generate_graph
    from fairchem.core.datasets import data_list_collater
    from fairchem.core.datasets.atomic_data import AtomicData

    device = positions.device
    dtype = positions.dtype
    n_data, n_atoms, _ = positions.shape

    data_list = []
    for pos in positions:
        data = AtomicData(
            pos=pos.to(dtype=dtype),
            atomic_numbers=torch.zeros(n_atoms, device=device, dtype=torch.long),
            cell=cell.unsqueeze(0).to(dtype=dtype),
            pbc=pbc.unsqueeze(0),
            natoms=torch.tensor([n_atoms], device=device, dtype=torch.long),
            edge_index=torch.empty((2, 0), device=device, dtype=torch.long),
            cell_offsets=torch.empty((0, 3), device=device, dtype=dtype),
            nedges=torch.tensor([0], device=device, dtype=torch.long),
            charge=torch.tensor([0], device=device, dtype=torch.long),
            spin=torch.tensor([0], device=device, dtype=torch.long),
            fixed=torch.zeros(n_atoms, device=device, dtype=torch.long),
            tags=torch.zeros(n_atoms, device=device, dtype=torch.long),
        )
        data_list.append(data)
    batch = data_list_collater(data_list, otf_graph=True)

    # TODO: remove this when fairchem supports torch.float64
    # batch.pos = batch.pos.to(dtype=dtype)
    # batch.cell = batch.cell.to(dtype=dtype)
    # batch.cell_offsets = batch.cell_offsets.to(dtype=dtype)

    graph_dict = generate_graph(
        batch,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        enforce_max_neighbors_strictly=False,
        radius_pbc_version=2,
        pbc=batch.pbc,
    )

    return graph_dict
