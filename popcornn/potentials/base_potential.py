import torch
from torch import nn
from dataclasses import dataclass

@dataclass
class PotentialOutput():
    """
    Data class representing the output of a path computation.

    Attributes:
    -----------
    energy : torch.Tensor
        The potential energy of the path.
    force : torch.Tensor, optional
        The force along the path.
    """
    energy: torch.Tensor = None
    force: torch.Tensor = None
    energyterms: torch.Tensor = None
    forceterms: torch.Tensor = None



class BasePotential(nn.Module):
    def __init__(self, images, is_conservative=True, device='cpu', add_azimuthal_dof=False, add_translation_dof=False, **kwargs) -> None:
        super().__init__()
        self.is_conservative = is_conservative
        self.atomic_numbers = images.atomic_numbers.to(device) if images.atomic_numbers is not None else None
        self.n_atoms = len(images.atomic_numbers) if images.atomic_numbers is not None else None
        self.pbc = images.pbc.to(device) if images.pbc is not None else None
        self.cell = images.cell.to(device) if images.cell is not None else None
        self.tag = images.tags.to(device) if images.tags is not None else None
        self.point_option = 0
        self.point_arg = 0
        if add_azimuthal_dof:
            self.point_option = 1
            self.point_arg = add_azimuthal_dof
        elif add_translation_dof:
            self.point_option = 2
        self.device = device
        
        # Put model in eval mode
        self.eval()

    @staticmethod 
    def calculate_conservative_force(energy, position, create_graph=True):
        return -torch.autograd.grad(
            energy,
            position,
            grad_outputs=torch.ones_like(energy),
            create_graph=create_graph,
        )[0]
    
    def calculate_conservative_forceterms(self, energyterms, position, create_graph=True):
        self._forceterm_fxn = torch.vmap(
            lambda vec: torch.autograd.grad(
                energyterms.flatten(), 
                position,
                grad_outputs=vec,
                create_graph=create_graph,
            )[0],
        )
        inp_vec = torch.eye(
            energyterms.shape[1], device=self.device
        ).repeat(1, energyterms.shape[0])
        return -1*self._forceterm_fxn(inp_vec).transpose(0, 1)

    def forward(
            self,
            positions: torch.Tensor
    ) -> PotentialOutput:
        raise NotImplementedError
