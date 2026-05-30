import torch
from torchani.units import HARTREE_TO_EV

from .base_potential import BasePotential, PotentialOutput

class AniPotential(BasePotential):
    """
    Wrapper around TorchANI — small-molecule MLIP. Energies are
    converted Hartree → eV; forces come from autograd through the
    energy.
    """

    def __init__(self, model_path, **kwargs):
        """
        Parameters
        ----------
        model_path : str
            Path to a saved ANI checkpoint.
        """
        super().__init__(**kwargs)
        self.model = self.load_model(model_path)
        self.n_eval = 0

    
    def forward(self, positions):
        data = self.data_formatter(positions)
        pred = self.model(data)
        self.n_eval += 1
        energies = pred.energies.view(*positions.shape[:-1], 1) * HARTREE_TO_EV
        forces = self.calculate_conservative_forces(energies, positions)
        return PotentialOutput(energies=energies, forces=forces)
        

    def load_model(self, model_path):
        # calc = ANICalculator(model_path)
        # model = calc.model
        model = torch.load(model_path, weights_only=False, map_location=self.device)
        model.eval()
        model.requires_grad_(False)
        return model
    
    def data_formatter(self, pos):
        n_atoms = self.n_atoms
        n_data = pos.numel() // (n_atoms * 3)
        z = self.atomic_numbers.repeat(n_data, 1)
        pos = pos.view(n_data, n_atoms, 3)
        return (z, pos)
