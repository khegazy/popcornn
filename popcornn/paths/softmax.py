import torch
from torch import nn
import math

from .base_path import BasePath
from .linear import LinearPath


class SquaredSoftmax(nn.Module):
    def __init__(self, sharpening=1):
        super().__init__()
        self.sharpening = sharpening
 
    def forward(self, x):
        return torch.softmax(-(x * self.sharpening)**2, dim=-1)


class SoftMaxPath(BasePath):
    def __init__(
        self,
        width: int = 128,
        # depth: int = 2,
        sharpening: float = 1,
        base: BasePath = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
       
        self.mlp = nn.Sequential(
            nn.Linear(1, width, dtype=self.dtype, bias=True),
            SquaredSoftmax(sharpening=sharpening),
            nn.Linear(width, self.final_position.shape[-1], dtype=self.dtype, bias=True)
        )
        # with torch.no_grad():
        #     self.mlp[0].bias.mul_(self.mlp[0].weight.squeeze(-1))
        self.mlp.to(self.device)
        self.neval = 0

        self.base = base if base is not None else LinearPath(**kwargs)
        
        print("Number of trainable parameters in MLP:", sum(p.numel() for p in self.parameters() if p.requires_grad))
        print(self.mlp)

    def get_positions(self, time: float):
        """
        Evaluate the path at ``time``.

        Parameters
        ----------
        time : torch.Tensor
            Times in [0, 1]; shape ``[N, 1]``.

        Returns
        -------
        torch.Tensor
            Positions of shape ``[N, D]``.
        """
        # (1 - time) * time pins the endpoints — at t=0 and t=1 the
        # learned correction vanishes so the path matches reactant /
        # product exactly regardless of MLP weights.
        # MLP input is rescaled to [-1, 1] (t' = 2t - 1) so pre-activations
        # at midpath come from bias only (z = b), giving a symmetric σ_min(t)
        # profile and a system-independent fresh-init σ_min calibration.
        mlp_out = self.mlp(2 * time - 1) * (1 - time) * time
        if self.fix_positions is not None:
            mlp_out[:, self.fix_positions] = 0.0
        base_out = self.base.get_positions(time)
        out = base_out + mlp_out
        return out