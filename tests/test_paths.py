import pytest
import torch

from popcornn.tools import process_images
from popcornn.paths import get_path
from popcornn.potentials import get_potential


@pytest.mark.parametrize(
    'dtype',
    [torch.float32, torch.float64]
)
@pytest.mark.parametrize(
    'device',
    [torch.device('cpu'), torch.device('cuda')]
)
def test_linear(dtype, device):
    if device.type == 'cuda' and not torch.cuda.is_available():
        pytest.skip(reason='CUDA is not available, skipping test.')

    images = process_images('tests/images/muller_brown.json', device=device, dtype=dtype)
    path = get_path('linear', images=images, device=device, dtype=dtype)
    assert path.transform is None

    path_output = path()
    assert path_output.time.shape == (101, 1)
    assert path_output.time.device.type == device.type
    assert path_output.time.dtype == dtype
    assert torch.allclose(path_output.time, torch.linspace(0, 1, 101, device=device, dtype=dtype).view(-1, 1))
    assert path_output.positions.shape == (101, 2)
    assert path_output.positions.device.type == device.type
    assert path_output.positions.dtype == dtype
    assert torch.allclose(path_output.positions, 
        torch.stack([torch.linspace(-0.558, -0.050, 101, device=device, dtype=dtype), torch.linspace(1.442, 0.467, 101, device=device, dtype=dtype)], dim=1), 
        atol=1e-5
    )

    path_output = path(torch.linspace(0, 1, 11, device=device, dtype=dtype))
    assert path_output.time.shape == (11, 1)
    assert path_output.time.device.type == device.type
    assert path_output.time.dtype == dtype
    assert torch.allclose(path_output.time, torch.linspace(0, 1, 11, device=device, dtype=dtype).view(-1, 1))
    assert path_output.positions.shape == (11, 2)
    assert path_output.positions.device.type == device.type
    assert path_output.positions.dtype == dtype
    assert torch.allclose(path_output.positions, 
        torch.stack([torch.linspace(-0.558, -0.050, 11, device=device, dtype=dtype), torch.linspace(1.442, 0.467, 11, device=device, dtype=dtype)], dim=1), 
        atol=1e-5
    )


@pytest.mark.parametrize(
    'dtype',
    [torch.float32, torch.float64]
)
@pytest.mark.parametrize(
    'device',
    [torch.device('cpu'), torch.device('cuda')]
)
def test_mlp(dtype, device):
    if device.type == 'cuda' and not torch.cuda.is_available():
        pytest.skip(reason='CUDA is not available, skipping test.')

    torch.manual_seed(0)  # For reproducibility
    images = process_images('tests/images/muller_brown.json', device=device, dtype=dtype)
    path = get_path('mlp', images=images, device=device, dtype=dtype)

    path_output = path()
    assert path_output.time.shape == (101, 1)
    assert path_output.time.device.type == device.type
    assert path_output.time.dtype == dtype
    assert torch.allclose(path_output.time, torch.linspace(0, 1, 101, device=device, dtype=dtype).view(-1, 1))
    assert path_output.positions.shape == (101, 2)
    assert path_output.positions.device.type == device.type
    assert path_output.positions.dtype == dtype
    assert torch.allclose(path_output.positions[0], torch.tensor([-0.558, 1.442], device=device, dtype=dtype), atol=1e-5)
    assert torch.allclose(path_output.positions[-1], torch.tensor([-0.050, 0.467], device=device, dtype=dtype), atol=1e-5)
    assert not torch.allclose(path_output.positions[50], torch.tensor([-0.304, 0.9545], device=device, dtype=dtype), atol=1e-5)

    path_output = path(torch.linspace(0, 1, 11, device=device, dtype=dtype))
    assert path_output.time.shape == (11, 1)
    assert path_output.time.device.type == device.type
    assert path_output.time.dtype == dtype
    assert torch.allclose(path_output.time, torch.linspace(0, 1, 11, device=device, dtype=dtype).view(-1, 1))
    assert path_output.positions.shape == (11, 2)
    assert path_output.positions.device.type == device.type
    assert path_output.positions.dtype == dtype
    assert torch.allclose(path_output.positions[0], torch.tensor([-0.558, 1.442], device=device, dtype=dtype), atol=1e-5)
    assert torch.allclose(path_output.positions[-1], torch.tensor([-0.050, 0.467], device=device, dtype=dtype), atol=1e-5)
    assert not torch.allclose(path_output.positions[5], torch.tensor([-0.304, 0.9545], device=device, dtype=dtype), atol=1e-5)

    path = get_path('mlp', images=images, device=torch.device('cpu'), dtype=dtype)
    assert path.mlp.__repr__() == (
        'Sequential(\n'
        '  (0): Linear(in_features=1, out_features=2, bias=True)\n'
        '  (1): GELU(approximate=\'none\')\n'
        '  (2): Linear(in_features=2, out_features=2, bias=True)\n'
        ')'
    )

    path = get_path('mlp', n_embed=1, depth=2, activation='gelu', images=images, device=torch.device('cpu'), dtype=dtype)
    assert path.mlp.__repr__() == (
        'Sequential(\n'
        '  (0): Linear(in_features=1, out_features=2, bias=True)\n'
        '  (1): GELU(approximate=\'none\')\n'
        '  (2): Linear(in_features=2, out_features=2, bias=True)\n'
        ')'
    )

    path = get_path('mlp', n_embed=32, depth=3, activation='elu', images=images, device=torch.device('cpu'), dtype=dtype)
    assert path.mlp.__repr__() == (
        'Sequential(\n'
        '  (0): Linear(in_features=1, out_features=64, bias=True)\n'
        '  (1): ELU(alpha=1.0)\n'
        '  (2): Linear(in_features=64, out_features=64, bias=True)\n'
        '  (3): ELU(alpha=1.0)\n'
        '  (4): Linear(in_features=64, out_features=2, bias=True)\n'
        ')'
    )


# TODO: Implement test for input reshape
@pytest.mark.skip(reason='Input reshape tests are not implemented yet.')
def test_input():
    pass


# TODO: Implement test for output reshape
@pytest.mark.skip(reason='Output reshape tests are not implemented yet.')
def test_output():
    pass


def test_unwrap():
    images = process_images('tests/images/LJ35.xyz', unwrap_positions=True, device=torch.device('cpu'), dtype=torch.float32)
    path = get_path('linear', images=images, device=torch.device('cpu'), dtype=torch.float32)
    assert path.transform is not None
    assert torch.allclose(
        path(torch.tensor([0.5])).positions,
        torch.tensor(
            [[
                 5.61231020e-01,  3.24026880e-01,  4.58243210e-01,
                -2.06421651e-17,  6.48053770e-01,  3.20770249e+00,
                 5.61231020e-01,  3.24026880e-01,  2.29121606e+00,
                -5.61231020e-01,  1.62013441e+00,  1.37472964e+00,
                -5.11837599e-17,  1.29610753e+00,  4.58243210e-01,
                -5.61231020e-01,  1.62013441e+00,  3.20770249e+00,
                -5.11837599e-17,  1.29610753e+00,  2.29121606e+00,
                 2.80615511e-01,  1.62013442e-01,  1.37472964e+00,
                -5.61231020e-01,  2.26818818e+00,  4.58243210e-01,
                -1.12246205e+00,  2.59221506e+00,  3.20770249e+00,
                -5.61231020e-01,  2.26818818e+00,  2.29121606e+00,
                 1.12246205e+00,  6.48053770e-01,  1.37472964e+00,
                 1.68369307e+00,  3.24026880e-01,  4.58243210e-01,
                 1.12246205e+00,  6.48053770e-01,  3.20770249e+00,
                 1.68369307e+00,  3.24026880e-01,  2.29121606e+00,
                 5.61231020e-01,  1.62013441e+00,  1.37472964e+00,
                 1.12246205e+00,  1.29610753e+00,  4.58243210e-01,
                 5.61231020e-01,  1.62013441e+00,  3.20770249e+00,
                 1.12246205e+00,  1.29610753e+00,  2.29121606e+00,
                -1.02367520e-16,  2.59221506e+00,  1.37472964e+00,
                 5.61231020e-01,  2.26818818e+00,  4.58243210e-01,
                -1.02367520e-16,  2.59221506e+00,  3.20770249e+00,
                 5.61231020e-01,  2.26818818e+00,  2.29121606e+00,
                 2.24492410e+00,  6.48053770e-01,  1.37472964e+00,
                 2.80615512e+00,  3.24026880e-01,  4.58243210e-01,
                 2.24492410e+00,  6.48053770e-01,  3.20770249e+00,
                 2.80615512e+00,  3.24026880e-01,  2.29121606e+00,
                 1.68369307e+00,  1.62013441e+00,  1.37472964e+00,
                 2.24492410e+00,  1.29610753e+00,  4.58243210e-01,
                 1.68369307e+00,  1.62013441e+00,  3.20770249e+00,
                 2.24492410e+00,  1.29610753e+00,  2.29121606e+00,
                 1.12246205e+00,  2.59221506e+00,  1.37472964e+00,
                 1.68369307e+00,  2.26818818e+00,  4.58243210e-01,
                 1.12246205e+00,  2.59221506e+00,  3.20770249e+00,
                 1.68369307e+00,  2.26818818e+00,  2.29121606e+00
            ]]
        ),
        atol=1e-5
    )

    images = process_images('tests/images/LJ35.xyz', unwrap_positions=False, device=torch.device('cpu'), dtype=torch.float32)
    path = get_path('linear', images=images, device=torch.device('cpu'), dtype=torch.float32)
    assert path.transform is not None
    assert torch.allclose(
        path(torch.tensor([0.5])).positions,
        torch.tensor(
            [[
                 5.61231020e-01,  3.24026880e-01,  4.58243210e-01,
                -2.06421651e-17,  6.48053770e-01,  3.20770249e+00,
                 5.61231020e-01,  3.24026880e-01,  2.29121606e+00,
                -5.61231020e-01,  1.62013441e+00,  1.37472964e+00,
                -5.11837599e-17,  1.29610753e+00,  4.58243210e-01,
                -5.61231020e-01,  1.62013441e+00,  3.20770249e+00,
                -5.11837599e-17,  1.29610753e+00,  2.29121606e+00,
                -5.61231025e-01,  1.62013441e+00,  1.37472964e+00,
                -5.61231020e-01,  2.26818818e+00,  4.58243210e-01,
                -1.12246205e+00,  2.59221506e+00,  3.20770249e+00,
                -5.61231020e-01,  2.26818818e+00,  2.29121606e+00,
                 1.12246205e+00,  6.48053770e-01,  1.37472964e+00,
                 1.68369307e+00,  3.24026880e-01,  4.58243210e-01,
                 1.12246205e+00,  6.48053770e-01,  3.20770249e+00,
                 1.68369307e+00,  3.24026880e-01,  2.29121606e+00,
                 5.61231020e-01,  1.62013441e+00,  1.37472964e+00,
                 1.12246205e+00,  1.29610753e+00,  4.58243210e-01,
                 5.61231020e-01,  1.62013441e+00,  3.20770249e+00,
                 1.12246205e+00,  1.29610753e+00,  2.29121606e+00,
                -1.02367520e-16,  2.59221506e+00,  1.37472964e+00,
                 5.61231020e-01,  2.26818818e+00,  4.58243210e-01,
                -1.02367520e-16,  2.59221506e+00,  3.20770249e+00,
                 5.61231020e-01,  2.26818818e+00,  2.29121606e+00,
                 2.24492410e+00,  6.48053770e-01,  1.37472964e+00,
                 2.80615512e+00,  3.24026880e-01,  4.58243210e-01,
                 2.24492410e+00,  6.48053770e-01,  3.20770249e+00,
                 2.80615512e+00,  3.24026880e-01,  2.29121606e+00,
                 1.68369307e+00,  1.62013441e+00,  1.37472964e+00,
                 2.24492410e+00,  1.29610753e+00,  4.58243210e-01,
                 1.68369307e+00,  1.62013441e+00,  3.20770249e+00,
                 2.24492410e+00,  1.29610753e+00,  2.29121606e+00,
                 1.12246205e+00,  2.59221506e+00,  1.37472964e+00,
                 1.68369307e+00,  2.26818818e+00,  4.58243210e-01,
                 1.12246205e+00,  2.59221506e+00,  3.20770249e+00,
                 1.68369307e+00,  2.26818818e+00,  2.29121606e+00
            ]]
        ),
        atol=1e-5
    )


@pytest.mark.parametrize(
    'raw_images',
    ['tests/images/muller_brown.json', 'tests/images/LJ13.xyz', 'tests/images/LJ35.xyz']
)
@pytest.mark.parametrize(
    'path_name',
    ['linear', 'mlp']
)
@pytest.mark.parametrize(
    'dtype',
    [torch.float32, torch.float64]
)
@pytest.mark.parametrize(
    'device',
    [torch.device('cpu'), torch.device('cuda')]
)
def test_velocity(raw_images, path_name, dtype, device):
    if device.type == 'cuda' and not torch.cuda.is_available():
        pytest.skip(reason='CUDA is not available, skipping test.')
        
    images = process_images(raw_images, device=device, dtype=dtype)
    path = get_path(path_name, images=images, device=device, dtype=dtype)
    velocity = path(torch.tensor([0.5], device=device, dtype=dtype), return_velocities=True).velocities
    assert velocity is not None
    finite_difference = path(torch.tensor([0.5 - 1e-3, 0.5 + 1e-3], device=device, dtype=dtype)).positions.diff(dim=0) / (2 * 1e-3)
    assert torch.allclose(velocity, finite_difference, atol=1e-3)


@pytest.mark.parametrize(
    'path_name',
    ['linear', 'mlp']
)
@pytest.mark.parametrize(
    'dtype',
    [torch.float32, torch.float64]
)
@pytest.mark.parametrize(
    'device',
    [torch.device('cpu'), torch.device('cuda')]
)
def test_set_potential(path_name, dtype, device):
    if device.type == 'cuda' and not torch.cuda.is_available():
        pytest.skip(reason='CUDA is not available, skipping test.')
        
    images = process_images('tests/images/muller_brown.json', device=device, dtype=dtype)
    path = get_path(path_name, images=images, device=device, dtype=dtype)
    assert path.potential is None
    with pytest.raises(AssertionError, match='Potential must be set by \'set_potential\' before calling \'forward\''):
        path(torch.tensor([0.5], device=device, dtype=dtype), return_energies=True)
    with pytest.raises(AssertionError, match='Potential must be set by \'set_potential\' before calling \'forward\''):
        path(torch.tensor([0.5], device=device, dtype=dtype), return_energies_decomposed=True)
    with pytest.raises(AssertionError, match='Potential must be set by \'set_potential\' before calling \'forward\''):
        path(torch.tensor([0.5], device=device, dtype=dtype), return_forces=True)
    with pytest.raises(AssertionError, match='Potential must be set by \'set_potential\' before calling \'forward\''):
        path(torch.tensor([0.5], device=device, dtype=dtype), return_forces_decomposed=True)
    potential = get_potential('muller_brown', images=images, device=device, dtype=dtype)
    path.set_potential(potential)
    assert path.potential is not None
    path_output = path(torch.tensor([0.5], requires_grad=True, device=device, dtype=dtype), return_energies=True)
    assert path_output.energies is not None
    assert path_output.energies.shape == (1, 1)
    assert path_output.energies.device.type == device.type
    assert path_output.energies.dtype == dtype
    assert torch.allclose(
        path_output.energies,
        potential(path(torch.tensor([0.5], requires_grad=True, device=device, dtype=dtype)).positions).energies,
        atol=1e-5
    )
    assert path_output.energies.requires_grad is True
    with pytest.raises(ValueError, match='Potential MullerBrown cannot calculate energies_decomposed'):
        path(torch.tensor([0.5], requires_grad=True, device=device, dtype=dtype), return_energies_decomposed=True)
    path_output = path(torch.tensor([0.5], requires_grad=True, device=device, dtype=dtype), return_forces=True)
    assert path_output.forces is not None
    assert path_output.forces.shape == (1, 2)
    assert path_output.forces.device.type == device.type
    assert path_output.forces.dtype == dtype
    assert torch.allclose(
        path_output.forces,
        potential(path(torch.tensor([0.5], requires_grad=True, device=device, dtype=dtype)).positions).forces,
        atol=1e-5
    )
    assert path_output.forces.requires_grad is True
    with pytest.raises(ValueError, match='Potential MullerBrown cannot calculate forces_decomposed'):
        path(torch.tensor([0.5], requires_grad=True, device=device, dtype=dtype), return_forces_decomposed=True)

    torch.manual_seed(0)  # For reproducibility
    images = process_images('tests/images/LJ35.xyz', device=device, dtype=dtype)
    path = get_path(path_name, images=images, device=device, dtype=dtype)
    potential = get_potential('lennard_jones', images=images, device=device, dtype=dtype)
    path.set_potential(potential)
    assert path.potential is not None
    path_output = path(torch.tensor([0.5], requires_grad=True, device=device, dtype=dtype), return_energies=True)
    assert path_output.energies is not None
    assert path_output.energies.shape == (1, 1)
    assert path_output.energies.device.type == device.type
    assert path_output.energies.dtype == dtype
    assert torch.allclose(
        path_output.energies,
        potential(path(torch.tensor([0.5], requires_grad=True, device=device, dtype=dtype)).positions).energies,
        atol=1e-5
    )
    assert path_output.energies.requires_grad is True
    path_output = path(torch.tensor([0.5], requires_grad=True, device=device, dtype=dtype), return_energies_decomposed=True)
    assert path_output.energies_decomposed is not None
    # assert path_output.energies_decomposed.shape == (1, 4316)
    assert (
        path_output.energies_decomposed.ndim == 2 
        and path_output.energies_decomposed.shape[0] == 1 
        and 4000 < path_output.energies_decomposed.shape[1] < 4500
    )
    assert path_output.energies_decomposed.device.type == device.type
    assert path_output.energies_decomposed.dtype == dtype
    assert torch.allclose(
        path_output.energies_decomposed,
        potential(path(torch.tensor([0.5], requires_grad=True, device=device, dtype=dtype)).positions).energies_decomposed,
        atol=1e-5
    )
    assert path_output.energies_decomposed.grad_fn is not None
    path_output = path(torch.tensor([0.5], requires_grad=True, device=device, dtype=dtype), return_forces=True)
    assert path_output.forces is not None
    assert path_output.forces.shape == (1, 105)
    assert path_output.forces.device.type == device.type
    assert path_output.forces.dtype == dtype
    assert torch.allclose(
        path_output.forces,
        potential(path(torch.tensor([0.5], requires_grad=True, device=device, dtype=dtype)).positions).forces,
        atol=1e-5
    )
    assert path_output.forces.requires_grad is True
    path_output = path(torch.tensor([0.5], requires_grad=True, device=device, dtype=dtype), return_forces_decomposed=True)
    assert path_output.forces_decomposed is not None
    # assert path_output.forces_decomposed.shape == (1, 4316, 105)
    assert (
        path_output.forces_decomposed.ndim == 3
        and path_output.forces_decomposed.shape[0] == 1
        and 4000 < path_output.forces_decomposed.shape[1] < 4500
        and path_output.forces_decomposed.shape[2] == 105
    )
    assert path_output.forces_decomposed.device.type == device.type
    assert path_output.forces_decomposed.dtype == dtype
    assert torch.allclose(
        path_output.forces_decomposed,
        potential(path(torch.tensor([0.5], requires_grad=True, device=device, dtype=dtype)).positions).forces_decomposed,
        atol=1e-5
    )
    assert path_output.forces_decomposed.grad_fn is not None
