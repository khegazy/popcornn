import torch
import numpy as np
from popcornn.tools.images import Images
from popcornn.potentials.base_potential import BasePotential
from popcornn.paths.base_path import BasePath

def test_ts_search():
    # Setup environment
    torch.manual_seed(2025)
    np.random.seed(2025)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create initial/end points
    x_init = torch.tensor([-1.5, 1, -0.5], device=device)
    x_final = torch.tensor([-0.25, 0.25, 1], device=device)
    images = Images(
        dtype=x_init.dtype,
        positions=torch.stack([x_init, x_final]),
        vec=torch.stack([x_init, x_final])
    )
    base_path = BasePath(images=images, device=device)
    
    # Create simple path where sum of coordinates is from -1 to 1
    def path(t):
        x = x_init + (x_final - x_init)*t
        return x

    # Potential is inspired by Legendre functions
    def legendre_like(n, x):
        x = torch.sum(x, dim=-1, keepdim=True)
        if n == 2:
            E = -1*(3*x**2 - 1)/2.
            max_E = 0.5
            time = 0.0
        elif n == 3:
            E = (5*x**3 - 2*x**2 - 3*x)/2.
            max_E = 8./27
            time = -1./3
        elif n == 4:
            E = (35*x**4 - 30*x**2 + 3)/8. - 0.75*x**2
            max_E = 0.375
            time = 0
        return E, max_E, time

    # Evaluate TS search 
    for l in [2, 3, 4]:
        time = torch.linspace(
            0, 1, 14, device=device, requires_grad=True
        ).unsqueeze(-1)
        positions = path(time)
        energies, ts_energy_truth, ts_time_truth = legendre_like(l, positions)
        ts_time_truth = 0.5 + ts_time_truth/2. 
        force = BasePotential.calculate_conservative_forces(energies, positions)
        base_path.ts_search(time[:,0], energies[:,0], force)

        ts_position = path(torch.tensor([[base_path.ts_time]], device=device))
        ts_energy, _, _ = legendre_like(l, ts_position)
        ts_energy = ts_energy[0]
        
        assert np.isclose(
            ts_time_truth, base_path.ts_time.cpu().item(), atol=1e4, rtol=1e-4
        ),\
        f"Did not match TS times for legendre {l}, got {base_path.ts_time.item()}, expected {ts_time_truth}"
        assert np.isclose(ts_energy_truth, ts_energy.cpu().item()),\
        f"Did not match TS energy for legendre {l}, got {ts_energy.item()}, expected {ts_energy_truth}"
        assert base_path.ts_force_mag < 1e-3,\
        f"Did not find sufficiently small TS gradient magnitude for legendre {l}, got {base_path.ts_force_mag}"