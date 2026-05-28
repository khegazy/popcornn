import math

import torch
from torch import nn

from .base_path import BasePath
from .linear import LinearPath


class SoftOneHotPath(BasePath):
    """
    Path whose learned correction has a normalized Gaussian-RBF basis,
    parameterized directly by per-unit centers ``c`` and a shared
    bandwidth ``τ``.

    The configuration at time ``t`` is

    .. math::
        x(t) = x_\\text{base}(t) + (1 - t)\\, t \\cdot W_\\text{out}\\, h(2t - 1)

    where the basis is

    .. math::
        h_i(z) = \\frac{\\exp(-(z - c_i)^2 / \\tau)}
                       {\\sum_j \\exp(-(z - c_j)^2 / \\tau)}

    on the rescaled time ``z = 2t - 1 \\in [-1, 1]``.

    Why this parameterization (rather than ``Linear(2, n)`` over ``[z, z²]``)
    --------------------------------------------------------------------
    The Gaussian-RBF family has exactly ``n + 1`` geometric degrees of
    freedom (``n`` centers + 1 shared bandwidth). The previous
    ``Linear(2, n)`` formulation exposed ``3n`` trainable scalars,
    over-parameterizing the family by ``2n - 1`` null directions and —
    worse — letting each unit's ``z²`` coefficient drift independently.
    Two failure modes followed: ``z²`` weights drift to zero (the unit's
    logit becomes linear in ``z``, so the unit wins on a half-line
    rather than a localized interval), or flip sign (the bump becomes
    a bowl: the unit wins at the extremes ``z = ±1`` instead of near
    ``c_i``). Both reorganize the basis discontinuously and produce the
    classic "loss good early, oscillates later" instability.

    Hard-coding ``-(z - c)² / τ`` removes both modes: there is no
    ``z²`` coefficient to drift, and every parameter moves a meaningful
    geometric DOF.

    Init: centers ``c_i = linspace(-1, 1, n)``, bandwidth set so the
    effective Gaussian width is

    .. math::
        \\sigma_\\text{eff} = \\sqrt{\\tau / 2} = k\\, d,
        \\quad d = 2 / (n - 1).

    ``k`` is the number of center-spacings within one ``σ_eff``;
    ``k = 1`` gives the partition-of-unity setting (adjacent bumps
    meet at half-height). Bandwidth is stored as ``log_tau`` so it
    stays strictly positive under unconstrained gradient updates.

    Parameters
    ----------
    n : int, default=128
        Number of basis neurons / bump centers.
    k : float, default=1.0
        Initial effective width: ``σ_eff / d``. ``k`` ↗ ⇒ broader
        bumps, more overlap. Only sets init; the bandwidth trains.
    base : BasePath, optional
        Base path the soft-one-hot correction adds onto. Defaults to
        ``LinearPath``.

    Notes
    -----
    To lock the basis geometry and only train the head, after
    construction set

    .. code-block:: python

        path.c.requires_grad_(False)
        path.log_tau.requires_grad_(False)
    """
    def __init__(
        self,
        n: int = 128,
        k: float = 1.0,
        base: BasePath = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n = int(n)
        self.k = float(k)

        D = self.final_position.shape[-1]
        d = 2.0 / (self.n - 1)
        tau_init = 2.0 * (self.k * d) ** 2

        self.c = nn.Parameter(
            torch.linspace(-1, 1, self.n, dtype=self.dtype)
        )
        self.log_tau = nn.Parameter(
            torch.tensor(math.log(tau_init), dtype=self.dtype)
        )
        self.out = nn.Linear(self.n, D, dtype=self.dtype, bias=True)

        self.to(self.device)
        self.neval = 0

        self.base = base if base is not None else LinearPath(**kwargs)

        print(
            "Number of trainable parameters in SoftOneHotPath:",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )
        print(f"  c: [{self.n}], log_tau: scalar (tau={tau_init:.3e})")
        print(self.out)

    def get_positions(self, time):
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
        # (1 - time) * time pins both endpoints — at t=0 and t=1 the
        # learned correction vanishes so the path matches reactant /
        # product exactly regardless of basis/head weights.
        # Input is rescaled to z = 2t - 1 ∈ [-1, 1] so the Gaussian
        # centers c_i = linspace(-1, 1, n) span the time domain exactly.
        z = 2 * time - 1                                 # [N, 1] in [-1, 1]
        tau = self.log_tau.exp()
        # Broadcast subtraction: z [N, 1] - c [n] -> [N, n].
        logits = -((z - self.c) ** 2) / tau              # [N, n]
        h = torch.softmax(logits, dim=-1)                # [N, n]
        mlp_out = self.out(h) * (1 - time) * time        # [N, D]
        if self.fix_positions is not None:
            mlp_out[:, self.fix_positions] = 0.0
        base_out = self.base.get_positions(time)
        return base_out + mlp_out
