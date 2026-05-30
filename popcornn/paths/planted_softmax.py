import torch
from torch import nn

from .base_path import BasePath
from .linear import LinearPath


class PlantedSoftmaxPath(BasePath):
    """
    Softmax basis path with all ``D`` units planted on the upper convex
    hull of ``{(w_i, b_i)}`` at init.

    The configuration at time ``t`` is

    .. math::
        x(t) = x_\\text{base}(t) + (1 - t)\\, t \\cdot W_\\text{out}\\,
               \\mathrm{softmax}\\bigl(\\kappa\\,(w\\,z + b)\\bigr),\\quad
        z = 2t - 1 \\in [-1, 1].

    Architecture is identical to ``MLPpath(activation='softmax', depth=2)``.
    The structural change is the **input-layer init**: instead of PyTorch's
    default ``w_i, b_i ~ Uniform(-1, 1)`` (independent), we set

    .. math::
        w_i = -1 + 2\\,\\frac{i - 1}{D - 1},\\qquad b_i = -\\tfrac{1}{2} w_i^2,
        \\qquad i = 1,\\dots,D.

    The points ``(w_i, b_i)`` lie on the parabolic arc ``b = -w^2 / 2``,
    which is the Legendre dual of ``φ(z) = z^2/2``. Every line
    ``z ↦ w_i z + b_i`` is a vertex of the upper envelope; the crossover
    between adjacent units ``i`` and ``i+1`` is

    .. math::
        \\tau_{i,i+1} = \\frac{w_i + w_{i+1}}{2} = -1 + \\frac{2i - 1}{D - 1}.

    All ``D`` units fire on equal-length sub-intervals of ``z ∈ [-1, 1]``,
    saturating PR(Σ_h) → ``D - 1`` at high ``κ``. Spacing follows the
    ``linspace`` precedent of ``SoftOneHotPath``; output-layer randomness
    is preserved.

    Why this lifts the rank ceiling
    --------------------------------
    With PyTorch default init only the ``O(\\ln D)`` units on the upper
    hull of the random cloud ever win — most are interior points and
    never fire. The participation-ratio (effective rank) of
    ``Σ_h = Cov_z(h)`` is capped at ``~ ln D``, independent of nominal
    width ``D``. Planting on the parabolic arc forces hull membership of
    all ``D`` lines and lifts the ceiling to ``D - 1``.

    After init, ``w`` and ``b`` train freely.

    Saturation scale of ``κ``
    -------------------------
    With planted init the typical adjacent-vertex slope difference is
    ``Δ_adj ≈ 2/D``; PR(Σ_h) approaches its ``D - 1`` ceiling at
    ``κ⋆ ≈ 2D / Δ_adj ≈ D²``. At ``κ = 1`` the basis is near-uniform
    at init; gradient descent on ``w`` and ``b`` can amplify effective
    sharpness during training.

    Parameters
    ----------
    width : int, default=128
        Hidden width ``D``.
    kappa : float, default=1.0
        Softmax temperature.
    base : BasePath, optional
        Base path the correction adds onto. Defaults to ``LinearPath``.
    """
    def __init__(
        self,
        width: int = 128,
        kappa: float = 1.0,
        base: BasePath = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.width = int(width)
        self.kappa = float(kappa)

        D_out = self.final_position.shape[-1]
        self.basis = nn.Linear(1, self.width, dtype=self.dtype, bias=True)
        self.out = nn.Linear(self.width, D_out, dtype=self.dtype, bias=True)

        # Plant all D units on the upper hull via the parabolic-arc map
        # (w, b) = (w, -w²/2) — the Legendre dual of φ(z)=z²/2 — with
        # w_i = linspace(-1, 1, D). Each line z ↦ w_i z + b_i is a hull
        # vertex; adjacent crossovers are equally spaced in [-1, 1].
        with torch.no_grad():
            w = torch.linspace(-1.0, 1.0, self.width, dtype=self.dtype)
            self.basis.weight.copy_(w.view(self.width, 1))
            self.basis.bias.copy_(-0.5 * w * w)

        self.to(self.device)
        self.neval = 0

        self.base = base if base is not None else LinearPath(**kwargs)

        print(
            "Number of trainable parameters in PlantedSoftmaxPath:",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )
        print(self.basis)
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
        # Rescale to z = 2t - 1 ∈ [-1, 1] so the planted crossovers
        # τ_i ∈ [-1, 1] span the time domain exactly.
        z = 2 * time - 1                       # [N, 1] in [-1, 1]
        logits = self.basis(z)                 # [N, width]
        if self.kappa != 1.0:
            logits = self.kappa * logits
        h = torch.softmax(logits, dim=-1)      # [N, width]
        mlp_out = self.out(h) * (1 - time) * time
        if self.fix_positions is not None:
            mlp_out[:, self.fix_positions] = 0.0
        base_out = self.base.get_positions(time)
        return base_out + mlp_out
