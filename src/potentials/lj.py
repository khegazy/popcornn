from typing import List, Tuple
import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones


def lj(
        atomic_symbols: List[str],
        atomic_coords: np.ndarray,
        epsilon_value: float = 1.0,
        sigma_value: float = 1.0,
) -> Tuple[float, np.ndarray]:
    """
    Compute the energy and forces of a system using the Lennard-Jones potential.

    Parameters:
        atomic_symbols (List[str]): List of atomic symbols.
        atomic_coords (np.ndarray): Atomic coordinates of shape (n_atoms, 3).
        epsilon_value (float): Epsilon value for the Lennard-Jones potential.
        sigma_value (float): Sigma value for the Lennard-Jones potential.

    Returns:
        Tuple[float, np.ndarray]: Energy and forces. Forces are of shape (n_atoms, 3).
    """
    atoms = Atoms(
        symbols=atomic_symbols,
        positions=atomic_coords
    )
    lj_calculator = LennardJones(
        epsilon=epsilon_value,
        sigma=sigma_value
    )
    atoms.set_calculator(lj_calculator)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    return energy, forces


if __name__ == '__main__':
    symbols = ['O', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H']
    coordinates = np.array([
        [ 0.47317182,  1.50497424,  0.51737247],
        [-0.73634452,  1.24725771, -0.16507229],
        [-1.25216459, -0.16253110,  0.10778120],
        [-0.18407060, -1.19505481, -0.26112652],
        [ 1.14246848, -0.83297208,  0.41134402],
        [ 1.50018054,  0.62109656,  0.11850918],
        [-0.58658675,  1.38609475, -1.25100717],
        [-1.44693886,  2.00520297,  0.17754664],
        [-1.49410391, -0.24838334,  1.17510388],
        [-2.17852897, -0.33371534, -0.45438129],
        [-0.04865446, -1.20515598, -1.35165087],
        [-0.50298914, -2.20262608,  0.02942530],
        [ 1.05150989, -0.96116858,  1.49779346],
        [ 1.95157576, -1.49001491,  0.06910250],
        [ 2.39649147,  0.92912583,  0.66491561],
        [ 1.69896569,  0.74626212, -0.96120110],
    ])

    energy, forces = lj(symbols, coordinates)
    print(f'Energy: {energy}')
    print(f'Forces:\n {forces}')
