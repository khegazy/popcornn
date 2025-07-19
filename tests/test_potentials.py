import pytest
import numpy as np
import torch
import ase
from ase.io import read
from ase.calculators.lj import LennardJones
from ase.calculators.calculator import Calculator
from ase.neighborlist import NeighborList
from ase.data import covalent_radii
from ase.mep import interpolate
from fairchem.core import pretrained_mlip, FAIRChemCalculator

from popcornn.tools import process_images
from popcornn.paths import get_path
from popcornn.potentials import get_potential


@pytest.mark.parametrize(
    'dtype',
    [torch.float32, torch.float64],
)
@pytest.mark.parametrize(
    'device',
    [torch.device('cpu'), torch.device('cuda')]
)
def test_sphere(dtype, device):
    if device.type == 'cuda' and not torch.cuda.is_available():
        pytest.skip(reason='CUDA is not available, skipping test.')
        
    images = process_images('tests/images/sphere.json', device=device, dtype=dtype)
    path = get_path('linear', images=images, device=device, dtype=dtype)
    potential = get_potential('sphere', images=images, device=device, dtype=dtype)

    potential_output = potential(path(torch.tensor([0.0, 0.5, 1.0], requires_grad=True, device=device, dtype=dtype)).positions)
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device.type == device.type
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(potential_output.energies,
        torch.tensor([[2.0], [1.0], [2.0]], device=device, dtype=dtype),
        atol=1e-5
    )
    assert potential_output.energies.grad_fn is not None
    assert potential_output.energies_decomposed is None
    assert potential_output.forces.shape == (3, 2)
    assert potential_output.forces.device.type == device.type
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(potential_output.forces,
        torch.tensor([[2.0, -2.0], [0.0, -2.0], [-2.0, -2.0]], device=device, dtype=dtype),
        atol=1e-5
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed is None


@pytest.mark.parametrize(
    'dtype',
    [torch.float32, torch.float64],
)
@pytest.mark.parametrize(
    'device',
    [torch.device('cpu'), torch.device('cuda')]
)
def test_muller_brown(dtype, device):
    if device.type == 'cuda' and not torch.cuda.is_available():
        pytest.skip(reason='CUDA is not available, skipping test.')
        
    images = process_images('tests/images/muller_brown.json', device=device, dtype=dtype)
    path = get_path('linear', images=images, device=device, dtype=dtype)
    potential = get_potential('muller_brown', images=images, device=device, dtype=dtype)

    potential_output = potential(path(torch.tensor([0.0, 0.5, 1.0], requires_grad=True, device=device, dtype=dtype)).positions)
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device.type == device.type
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(potential_output.energies,
        torch.tensor([[-146.69948920058778], [3.225535000493114], [-80.76774924875772]], device=device, dtype=dtype),
        atol=1e-5
    )
    assert potential_output.energies.grad_fn is not None
    assert potential_output.energies_decomposed is None
    assert potential_output.forces.shape == (3, 2)
    assert potential_output.forces.device.type == device.type
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(potential_output.forces,
        torch.tensor(
            [
                [0.0001873538288492682, -0.20449389471367851], 
                [-63.037220140249005, -57.82076869131242], 
                [-0.048943927961211386, -0.44871341799094555]
            ], 
            device=device, dtype=dtype
        ),
        atol=1e-3
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed is None


@pytest.mark.parametrize(
    'dtype',
    [torch.float32, torch.float64],
)
@pytest.mark.parametrize(
    'device',
    [torch.device('cpu'), torch.device('cuda')]
)
def test_wolfe(dtype, device):
    if device.type == 'cuda' and not torch.cuda.is_available():
        pytest.skip(reason='CUDA is not available, skipping test.')
        
    images = process_images('tests/images/wolfe.json', device=device, dtype=dtype)
    path = get_path('linear', images=images, device=device, dtype=dtype)
    potential = get_potential('wolfe_schlegel',images=images, device=device, dtype=dtype)

    potential_output = potential(path(torch.tensor([0.0, 0.5, 1.0], requires_grad=True, device=device, dtype=dtype)).positions)
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device.type == device.type
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(potential_output.energies, 
        torch.tensor([[-64.81812976863], [-0.04301175469875001], [-66.45448705023]], device=device, dtype=dtype),
        atol=1e-5
    )
    assert potential_output.energies.grad_fn is not None
    assert potential_output.energies_decomposed is None
    assert potential_output.forces.shape == (3, 2)
    assert potential_output.forces.device.type == device.type
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(potential_output.forces,
        torch.tensor(
            [
                [0.0032145200000005536, 0.04517024000000619], 
                [-2.614820315, -1.194996355], 
                [-0.0003081600000115481, -0.06473332000001358]
            ], 
            device=device, dtype=dtype
        ),
        atol=1e-3
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed is None


@pytest.mark.parametrize(
    'dtype',
    [torch.float32, torch.float64],
)
@pytest.mark.parametrize(
    'device',
    [torch.device('cpu'), torch.device('cuda')]
)
def test_lennard_jones(dtype, device):
    if device.type == 'cuda' and not torch.cuda.is_available():
        pytest.skip(reason='CUDA is not available, skipping test.')
        
    raw_images = [read('tests/images/LJ13.xyz', index=i) for i in (0, 1, 1)]
    interpolate(raw_images)
    for image in raw_images:
        image.calc = LennardJones()
    images = process_images('tests/images/LJ13.xyz', device=device, dtype=dtype)
    path = get_path('linear', images=images, device=device, dtype=dtype)
    potential = get_potential('lennard_jones', images=images, device=device, dtype=dtype)
    potential_output = potential(path(torch.tensor([0.0, 0.5, 1.0], requires_grad=True, device=device, dtype=dtype)).positions)
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device.type == device.type
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(potential_output.energies,
        torch.tensor([[image.get_potential_energy()] for image in raw_images], device=device, dtype=dtype),
        atol=1e-5
    )
    assert potential_output.energies.grad_fn is not None
    assert potential_output.energies_decomposed.shape == (3, 156)
    assert potential_output.energies_decomposed.device.type == device.type
    assert potential_output.energies_decomposed.dtype == dtype
    assert torch.allclose(potential_output.energies_decomposed.sum(dim=-1, keepdim=True), potential_output.energies, atol=1e-5)
    assert potential_output.energies_decomposed.grad_fn is not None
    assert potential_output.forces.shape == (3, 39)
    assert potential_output.forces.device.type == device.type
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(potential_output.forces,
        torch.tensor([image.get_forces().flatten() for image in raw_images], device=device, dtype=dtype),
        atol=1e-3
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed.shape == (3, 156, 39)
    assert potential_output.forces_decomposed.device.type == device.type
    assert potential_output.forces_decomposed.dtype == dtype
    assert torch.allclose(potential_output.forces_decomposed.sum(dim=-2, keepdim=False), potential_output.forces, atol=1e-5)
    assert potential_output.forces_decomposed.grad_fn is not None

    raw_images = [read('tests/images/LJ35.xyz', index=i) for i in (0, 1, 1)]
    interpolate(raw_images, mic=True)
    for image in raw_images:
        image.calc = LennardJones()
    images = process_images('tests/images/LJ35.xyz', device=device, dtype=dtype)
    path = get_path('linear', images=images, device=device, dtype=dtype)
    potential = get_potential('lennard_jones', images=images, device=device, dtype=dtype)
    potential_output = potential(path(torch.tensor([0.0, 0.5, 1.0], requires_grad=True, device=device, dtype=dtype)).positions)
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device.type == device.type
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(potential_output.energies,
        torch.tensor([[image.get_potential_energy()] for image in raw_images], device=device, dtype=dtype),
        atol=1e-5
    )
    assert potential_output.energies.grad_fn is not None
    assert potential_output.energies_decomposed.shape == (3, 4352)
    assert potential_output.energies_decomposed.device.type == device.type
    assert potential_output.energies_decomposed.dtype == dtype
    assert torch.allclose(potential_output.energies_decomposed.sum(dim=-1, keepdim=True), potential_output.energies, atol=1e-5)
    assert potential_output.energies_decomposed.grad_fn is not None
    assert potential_output.forces.shape == (3, 105)
    assert potential_output.forces.device.type == device.type
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(potential_output.forces,
        torch.tensor([image.get_forces().flatten() for image in raw_images], device=device, dtype=dtype),
        atol=1e-3
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed.shape == (3, 4352, 105)
    assert potential_output.forces_decomposed.device.type == device.type
    assert potential_output.forces_decomposed.dtype == dtype
    assert torch.allclose(potential_output.forces_decomposed.sum(dim=-2, keepdim=False), potential_output.forces, atol=1e-5)
    assert potential_output.forces_decomposed.grad_fn is not None


@pytest.mark.parametrize(
    'dtype',
    [torch.float32, torch.float64],
)
@pytest.mark.parametrize(
    'device',
    [torch.device('cpu'), torch.device('cuda')]
)
def test_repel(dtype, device):
    if device.type == 'cuda' and not torch.cuda.is_available():
        pytest.skip(reason='CUDA is not available, skipping test.')
        
    class RepelCalculator(Calculator):
        '''
        Revised from ASE LennardJones Calculator to use the repulsive potential
        '''
        implemented_properties = ['energy', 'forces']
        def __init__(self, alpha=1.7, beta=0.01, cutoff=3.0, **kwargs):
            super().__init__(**kwargs)
            self.alpha = alpha
            self.beta = beta
            self.rc = cutoff
            self.nl = None

        def calculate(self, atoms=None, properties=None, system_changes=None):
            Calculator.calculate(self, atoms, properties, system_changes)

            natoms = len(self.atoms)
            alpha = self.alpha
            beta = self.beta
            rc = self.rc
            if self.nl is None or 'numbers' in system_changes:
                self.nl = NeighborList([rc / 2] * natoms, self_interaction=False, bothways=True)
            self.nl.update(self.atoms)
            positions = self.atoms.positions
            cell = self.atoms.cell

            energies = np.zeros(natoms)
            forces = np.zeros((natoms, 3))
            for ii in range(natoms):
                neighbors, offsets = self.nl.get_neighbors(ii)
                cells = np.dot(offsets, cell)
                distance_vectors = positions[neighbors] + cells - positions[ii]
                r = np.linalg.norm(distance_vectors, axis=1)
                r0 = covalent_radii[self.atoms.numbers[ii]] + covalent_radii[self.atoms.numbers[neighbors]]
                pairwise_energies = np.exp(-alpha * (r - r0) / r0) + beta * r0 / r
                pairwise_energies[r > rc] = 0.0
                pairwise_forces = - np.exp(-alpha * (r - r0) / r0) * alpha / r0 / r - beta * r0 / r ** 3
                pairwise_forces[r > rc] = 0.0
                pairwise_energies -= (np.exp(-alpha * (rc - r0) / r0) + beta * r0 / rc) * (r < rc)
                pairwise_forces = pairwise_forces[:, np.newaxis] * distance_vectors
                energies[ii] += 0.5 * pairwise_energies.sum()
                forces[ii] += pairwise_forces.sum(axis=0)
            energy = energies.sum()
            self.results['energy'] = energy
            self.results['forces'] = forces

    raw_images = [read('tests/images/LJ13.xyz', index=i) for i in (0, 1, 1)]
    interpolate(raw_images)
    for image in raw_images:
        image.calc = RepelCalculator()
    images = process_images('tests/images/LJ13.xyz', device=device, dtype=dtype)
    path = get_path('linear', images=images, device=device, dtype=dtype)
    potential = get_potential('repel', images=images, device=device, dtype=dtype)
    potential_output = potential(path(torch.tensor([0.0, 0.5, 1.0], requires_grad=True, device=device, dtype=dtype)).positions)
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device.type == device.type
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(potential_output.energies,
        torch.tensor([[image.get_potential_energy()] for image in raw_images], device=device, dtype=dtype),
        atol=1e-5
    )
    assert potential_output.energies.grad_fn is not None
    assert potential_output.energies_decomposed.shape == (3, 156)
    assert potential_output.energies_decomposed.device.type == device.type
    assert potential_output.energies_decomposed.dtype == dtype
    assert torch.allclose(potential_output.energies_decomposed.sum(dim=-1, keepdim=True), potential_output.energies, atol=1e-5)
    assert potential_output.energies_decomposed.grad_fn is not None
    assert potential_output.forces.shape == (3, 39)
    assert potential_output.forces.device.type == device.type
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(potential_output.forces,
        torch.tensor([image.get_forces().flatten() for image in raw_images], device=device, dtype=dtype),
        atol=1e-3
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed.shape == (3, 156, 39)
    assert potential_output.forces_decomposed.device.type == device.type
    assert potential_output.forces_decomposed.dtype == dtype
    assert torch.allclose(potential_output.forces_decomposed.sum(dim=-2, keepdim=False), potential_output.forces, atol=1e-5)
    assert potential_output.forces_decomposed.grad_fn is not None

    raw_images = [read('tests/images/LJ35.xyz', index=i) for i in (0, 1, 1)]
    interpolate(raw_images, mic=True)
    for image in raw_images:
        image.calc = RepelCalculator()
    images = process_images('tests/images/LJ35.xyz', device=device, dtype=dtype)
    path = get_path('linear', images=images, device=device, dtype=dtype)
    potential = get_potential('repel', images=images, device=device, dtype=dtype)
    potential_output = potential(path(torch.tensor([0.0, 0.5, 1.0], requires_grad=True, device=device, dtype=dtype)).positions)
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device.type == device.type
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(potential_output.energies,
        torch.tensor([[image.get_potential_energy()] for image in raw_images], device=device, dtype=dtype),
        atol=1e-5
    )
    assert potential_output.energies.grad_fn is not None
    assert potential_output.energies_decomposed.shape == (3, 4352)
    assert potential_output.energies_decomposed.device.type == device.type
    assert potential_output.energies_decomposed.dtype == dtype
    assert torch.allclose(potential_output.energies_decomposed.sum(dim=-1, keepdim=True), potential_output.energies, atol=1e-5)
    assert potential_output.energies_decomposed.grad_fn is not None
    assert potential_output.forces.shape == (3, 105)
    assert potential_output.forces.device.type == device.type
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(potential_output.forces,
        torch.tensor([image.get_forces().flatten() for image in raw_images], device=device, dtype=dtype),
        atol=1e-3
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed.shape == (3, 4352, 105)
    assert potential_output.forces_decomposed.device.type == device.type
    assert potential_output.forces_decomposed.dtype == dtype
    assert torch.allclose(potential_output.forces_decomposed.sum(dim=-2, keepdim=False), potential_output.forces, atol=1e-5)
    assert potential_output.forces_decomposed.grad_fn is not None


@pytest.mark.parametrize(
    'dtype',
    [torch.float32, torch.float64],
)
@pytest.mark.parametrize(
    'device',
    [torch.device('cpu'), torch.device('cuda')]
)
def test_uma(dtype, device):
    pytest.skip(reason='Give github an UMA license')
    if device.type == 'cuda' and not torch.cuda.is_available():
        pytest.skip(reason='CUDA is not available, skipping test.')
        
    if dtype == torch.float64:
        pytest.skip("UMA potential is not supported for float64 due to precision issues.")

    raw_images = [read('tests/images/T1x.xyz', index=i) for i in (0, 1, 1)]
    interpolate(raw_images)
    for image in raw_images:
        image.calc = FAIRChemCalculator(
            pretrained_mlip.get_predict_unit('uma-s-1', device=device.type),
            task_name='omol'
        )
    images = process_images('tests/images/T1x.xyz', device=device, dtype=dtype)
    path = get_path('linear', images=images, device=device, dtype=dtype)
    potential = get_potential('uma', model_name='uma-s-1', task_name='omol', images=images, device=device, dtype=dtype)
    potential_output = potential(path(torch.tensor([0.0, 0.5, 1.0], requires_grad=True, device=device, dtype=dtype)).positions)
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device.type == device.type
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(potential_output.energies,
        torch.tensor([[image.get_potential_energy()] for image in raw_images], device=device, dtype=dtype),
        atol=1e-5
    )
    assert potential_output.energies.grad_fn is not None
    assert potential_output.energies_decomposed is None
    assert potential_output.forces.shape == (3, 39)
    assert potential_output.forces.device.type == device.type
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(potential_output.forces,
        torch.tensor([image.get_forces().flatten() for image in raw_images], device=device, dtype=dtype),
        atol=1e-3 if device.type == 'cpu' else 1e-2
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed is None

    raw_images = [read('tests/images/OC20NEB.xyz', index=i) for i in (0, 1, 1)]
    interpolate(raw_images)
    for image in raw_images:
        image.calc = FAIRChemCalculator(
            pretrained_mlip.get_predict_unit('uma-s-1', device=device.type),
            task_name='oc20'
        )
    images = process_images('tests/images/OC20NEB.xyz', device=device, dtype=dtype)
    path = get_path('linear', images=images, device=device, dtype=dtype)
    potential = get_potential('uma', model_name='uma-s-1', task_name='oc20', images=images, device=device, dtype=dtype)
    potential_output = potential(path(torch.tensor([0.0, 0.5, 1.0], requires_grad=True, device=device, dtype=dtype)).positions)
    assert potential_output.energies.shape == (3, 1)
    assert potential_output.energies.device.type == device.type
    assert potential_output.energies.dtype == dtype
    assert torch.allclose(potential_output.energies,
        torch.tensor([[image.get_potential_energy()] for image in raw_images], device=device, dtype=dtype),
        atol=1e-5
    )
    assert potential_output.energies.grad_fn is not None
    assert potential_output.energies_decomposed is None
    assert potential_output.forces.shape == (3, 42)
    assert potential_output.forces.device.type == device.type
    assert potential_output.forces.dtype == dtype
    assert torch.allclose(potential_output.forces * ~images.fix_positions,
        torch.tensor([image.get_forces().flatten() for image in raw_images], device=device, dtype=dtype),
        atol=1e-3 if device.type == 'cpu' else 1e-2
    )
    assert potential_output.forces.grad_fn is not None
    assert potential_output.forces_decomposed is None