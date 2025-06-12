
import torch
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from fairchem.core.datasets import data_list_collater
from fairchem.core.datasets.atomic_data import AtomicData

from .base_potential import BasePotential, PotentialOutput

class UMAPotential(BasePotential):
    def __init__(self, model_name, task_name, **kwargs):
        """
        Constructor for UMA Potential

        Parameters
        ----------
        model_path: str
            path to the model. eg. 'weights/uma/model.pt'
        dataset: str
            dataset name. eg. 'oc20'
        """
        super().__init__(**kwargs)
        self.task_name = task_name
        self.predictor = self.load_model(model_name, task_name)
        self.n_eval = 0


    def forward(self, positions):
        data = self.data_formatter(positions)
        pred = self.predictor.predict(data)
        self.n_eval += 1
        energies = pred['energy'].unsqueeze(-1).to(dtype=self.dtype)
        forces = pred['forces'].view(*positions.shape)
        return PotentialOutput(energies=energies, forces=forces)


    def load_model(self, model_name, task_name):
        predictor = pretrained_mlip.get_predict_unit(model_name=model_name, device=self.device)
        calc = FAIRChemCalculator(predictor, task_name=task_name)
        calc.predictor.model.module.output_heads['energyandforcehead'].head.training = True
        return calc.predictor

    def data_formatter(self, positions):
        positions = positions.view(*positions.shape[:-1], self.n_atoms, 3)
        data_list = []
        for pos in positions:
            data = AtomicData(
                pos=pos,
                atomic_numbers=self.atomic_numbers.long(),
                cell=self.cell.unsqueeze(0),
                pbc=self.pbc.unsqueeze(0),
                natoms=torch.tensor([self.n_atoms], device=self.device, dtype=torch.long),
                edge_index=torch.empty((2, 0), device=self.device, dtype=torch.long),
                cell_offsets=torch.empty((0, 3), device=self.device, dtype=self.dtype),
                nedges=torch.tensor([0], device=self.device, dtype=torch.long),
                charge=self.charge.unsqueeze(0),
                spin=self.spin.unsqueeze(0),
                # fixed=self.fix_positions.long(),
                fixed=torch.zeros(self.n_atoms, device=self.device, dtype=torch.long),  # default fixed positions
                tags=self.tags.long(),
            )
            data.dataset = self.task_name
            data_list.append(data)
        batch = data_list_collater(data_list, otf_graph=True)
        
        return batch