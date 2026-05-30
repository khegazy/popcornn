
import torch
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from fairchem.core.datasets import data_list_collater
from fairchem.core.datasets.atomic_data import AtomicData

from .base_potential import BasePotential, PotentialOutput

class UMAPotential(BasePotential):
    """
    Wrapper around Meta's UMA (universal MLIP).

    Loads a pretrained UMA checkpoint via ``fairchem-core`` and
    exposes the energy/forces head through popcornn's
    ``PotentialOutput`` interface. Needs a HuggingFace token plus the
    model weights — see ``docs/potentials.md`` for setup.
    """

    def __init__(self, model_name, task_name, **kwargs):
        """
        Parameters
        ----------
        model_name : str
            UMA checkpoint identifier, e.g. ``"uma-s-1p1"``.
        task_name : str
            UMA task, e.g. ``"omol"``.
        """
        super().__init__(**kwargs)
        self.task_name = task_name
        self.predictor = self.load_model(model_name, task_name)
        self.n_eval = 0


    def forward(self, positions):
        data = self.data_formatter(positions)
        pred = self.predictor.predict(data)
        self.n_eval += 1
        energies = (
            pred['energy'].unsqueeze(-1)
            .to(dtype=self.dtype)  # https://github.com/facebookresearch/fairchem/issues/1317
        )
        forces = pred['forces'].view(*positions.shape)
        return PotentialOutput(energies=energies, forces=forces)


    def load_model(self, model_name, task_name):
        # Propagate self.dtype to fairchem's model via InferenceSettings.
        # Default is float32; without this, fp64 mep ends up with float32
        # UMA weights and float32 forces despite self.dtype=float64.
        inference_settings = InferenceSettings(base_precision_dtype=self.dtype)
        predictor = pretrained_mlip.get_predict_unit(
            model_name=model_name, device=self.device.type,
            inference_settings=inference_settings,
        )
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
                fixed=torch.zeros_like(self.atomic_numbers, dtype=torch.long),
                tags=self.tags.long(),
            )
            data.dataset = self.task_name
            data_list.append(data)
        batch = data_list_collater(data_list, otf_graph=True)
        
        return batch