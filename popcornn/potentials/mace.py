import torch
from torch_geometric.data import Data
from mace.calculators import mace_off

from .base_potential import BasePotential, PotentialOutput

class MacePotential(BasePotential):
    def __init__(self, model_path, **kwargs):
        """
        Constructor for NewtonNetPotential

        Parameters
        ----------
        model_path: str or list of str
            path to the model. eg. '5k/models/best_model_state.tar'
        settings_path: str or list of str
            path to the .yml setting path. eg. '5k/run_scripts/config_h2.yml'
        device: 
            device to run model. eg. 'cpu', ['cuda:0', 'cuda:1']
        kwargs
        """
        super().__init__(**kwargs)
        self.model = self.load_model(model_path)
        self.n_eval = 0

    
    def forward(self, points):
        data = self.data_formatter(points)
        pred = self.model(data.z, data.disp, data.edge_index, data.batch)
        self.n_eval += 1
        energy_terms = pred.energy
        # force = pred.gradient_force
        energy_terms = energy_terms.view(-1, self.n_atoms)
        return PotentialOutput(energy_terms=energy_terms)
        # force = force.view(*points.shape)
        # return PotentialOutput(energy=energy, force=force)
        

    def load_model(self, model_path):
        model = mace_off().models[0]
        state_dict = torch.load(model_path).get('state_dict')
        state_dict = {k.replace('potential.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        model.requires_grad_(False)
        return model
    
    def data_formatter(self, pos):
        config = data.config_from_atoms(atoms, charges_key=self.charges_key)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max, heads=self.heads
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(self.device)
        
        n_atoms = self.n_atoms
        n_data = pos.numel() // (n_atoms * 3)
        z = self.numbers.repeat(n_data)
        pos = pos.view(n_data * n_atoms, 3)
        lattice = torch.ones(1, device=self.device) * torch.inf
        batch = torch.arange(n_data, device=self.device).repeat_interleave(n_atoms)
        data = Data(pos=pos, z=z, lattice=lattice, batch=batch)
        
        return self.transform(data)
