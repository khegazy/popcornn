import torch
from torch import nn

from .base_path import BasePath
from .linear import LinearPath


# Bounded-output activations need σ(κz) (no outer 1/κ) to preserve their
# ±C saturation at large |z|. Unbounded/linear-asymptote activations need
# (1/κ)·σ(κz) so the κz·slope wing matches the original σ(z) ≈ z slope.
_BOUNDED_ACTIVATION_TYPES = (
    nn.Tanh, nn.Sigmoid, nn.Hardtanh, nn.Softsign,
    nn.Threshold, nn.Hardsigmoid,
)


class TemperedSoftmax(nn.Module):
    """``softmax(τ·z)`` along the last (hidden) dim.

    Competitive normalization: ``∑ᵢ hᵢ(t) = 1``. As τ grows, the highest-
    scoring neuron at each t dominates → time-localized basis (one neuron
    "wins" each t-region). At τ=1 the basis is nearly uniform; at τ ≳ 10
    on Kaiming-default bias init it becomes peaky.

    Use via ``MLPpath(activation='softmax', kappa=τ)`` — the existing
    ``kappa`` knob is reinterpreted as softmax temperature.

    σ_min note: the b₂ output-bias block alone gives σ_min(J_path, t=0.5)
    ≥ (1-t)·t · 1 = 0.25, so σ_min is bias-floored at ~0.25 across all τ
    (validated 2026-05-25, 1% match across MB/LJ13/rost50 × τ∈{1,3,10,30}).
    Thresholds derive from σ ≈ 0.25, NOT from ‖h‖ scaling.
    """
    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = float(tau)

    def forward(self, z):
        if self.tau == 1.0:
            return torch.softmax(z, dim=-1)
        return torch.softmax(self.tau * z, dim=-1)

    def extra_repr(self):
        return f'tau={self.tau}'
    
class SquaredSoftmax(nn.Module):
    """``softmax(τ·z)`` along the last (hidden) dim.

    Competitive normalization: ``∑ᵢ hᵢ(t) = 1``. As τ grows, the highest-
    scoring neuron at each t dominates → time-localized basis (one neuron
    "wins" each t-region). At τ=1 the basis is nearly uniform; at τ ≳ 10
    on Kaiming-default bias init it becomes peaky.

    Use via ``MLPpath(activation='softmax', kappa=τ)`` — the existing
    ``kappa`` knob is reinterpreted as softmax temperature.

    σ_min note: the b₂ output-bias block alone gives σ_min(J_path, t=0.5)
    ≥ (1-t)·t · 1 = 0.25, so σ_min is bias-floored at ~0.25 across all τ
    (validated 2026-05-25, 1% match across MB/LJ13/rost50 × τ∈{1,3,10,30}).
    Thresholds derive from σ ≈ 0.25, NOT from ‖h‖ scaling.
    """
    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = float(tau)

    def forward(self, z):
        if self.tau == 1.0:
            return torch.softmax(- z ** 2, dim=-1)
        return torch.softmax(- self.tau * z ** 2, dim=-1)

    def extra_repr(self):
        return f'tau={self.tau}'


class SharpActivation(nn.Module):
    """Sharpens a base activation ``σ`` while preserving its large-|z|
    behavior. Two forms, picked by whether ``σ`` saturates to a constant
    or grows linearly:

    - **Bounded** σ (tanh, sigmoid, …): ``Sharp_κ[σ](z) = σ(κ·z)`` —
      preserves the ±C plateau; slope at z=0 becomes ``κ·σ'(0)``.
    - **Unbounded** σ with linear wing (GELU, SiLU, ReLU, Softplus, …):
      ``Sharp_κ[σ](z) = (1/κ)·σ(κ·z)`` — preserves the asymptotic
      slope ``σ(z) ≈ z`` at large +z; transition zone shrinks by 1/κ.

    Auto-detects boundedness from the base class; pass ``bounded`` to
    override.

    Both forms narrow the transition zone to ~1/κ, giving the MLP finer
    t-resolution. The default Kaiming init for W₁, b₁ is *not* touched, so
    switching points -b/W stay at default distribution.

    Used on gg-series NN paths (2026-05-22, F11): unbounded form with
    GELU at κ=10 collapses the multi-crossing IBP-cancellation trap on
    rxn 1929 (fmax 0.211 → 0.035). For tanh see ``project_popcornn_gg1929``
    — bounded form needed; broken form ``(1/κ)·tanh(κz)`` shrinks the
    saturated magnitude to ±1/κ and pins the path to the linear base.
    """
    def __init__(self, base_activation: nn.Module, kappa: float, bounded: bool = None):
        super().__init__()
        self.base = base_activation
        self.kappa = float(kappa)
        if bounded is None:
            bounded = isinstance(base_activation, _BOUNDED_ACTIVATION_TYPES)
        self.bounded = bool(bounded)

    def forward(self, z):
        if self.kappa == 1.0:
            return self.base(z)
        if self.bounded:
            return self.base(self.kappa * z)
        return self.base(self.kappa * z) / self.kappa

    def extra_repr(self):
        return f'kappa={self.kappa}, bounded={self.bounded}'


class MLPpath(BasePath):
    """
    Neural-network path: linear base + learned correction.

    The configuration at time ``t`` is

    .. math::
        x(t) = x_\\text{base}(t) + (1 - t)\\, t \\cdot \\text{MLP}(2t - 1;\\, \\theta)

    Two design choices keep the path well-behaved:

    1. ``(1 - t) * t`` envelope pins both endpoints. Whatever the MLP
       outputs, the path equals the base path (reactant at ``t=0``,
       product at ``t=1``).
    2. MLP input is rescaled to ``t' = 2t - 1`` so the input domain is
       centered (``t' ∈ [-1, 1]``). At the midpath ``t=0.5``, the MLP
       pre-activation ``z = w·t' + b`` reduces to ``z = b`` (bias only),
       giving the smallest and most-Gaussian pre-activation distribution
       at the point of physical interest. This makes σ_min(J_path) at the
       saddle a clean, system-independent function of ``width`` (see the
       ``width`` parameter docstring).

    Parameters
    ----------
    width : int, default=128
        Hidden layer width ``M``. Default chosen so

        .. math::
            \\sigma_\\text{min}(J_\\text{path}, t=\\tfrac{1}{2}, \\text{fresh init}) \\approx 1

        which makes the standard threshold derivation
        ``thr_2 = δ · 2 · σ_min · fmax_target`` *system-independent*.
    depth : int, default=2
        Number of ``Linear`` layers (default: input + hidden + output).
    activation : str, default="gelu"
        Any nonlinearity in ``torch.nn`` (case-insensitive), OR the special
        value ``"softmax"`` for a competitive-normalized basis via
        ``TemperedSoftmax`` (requires ``depth=2``). For ``softmax``, the
        ``kappa`` knob is reinterpreted as softmax temperature τ.
    kappa : float, default=1.0
        Elementwise activations: sharpening factor.  ``kappa=1`` is the
        unmodified activation. ``kappa>1`` wraps it with
        ``Sharp_κ[σ](z) = (1/κ)·σ(κ·z)`` (see ``SharpActivation``),
        shrinking the transition zone by ``1/κ`` without touching weights.
        Empirically (2026-05-22 F11 sweep on gg-series chem-13 NN paths)
        ``kappa = 10`` with the default GELU collapses the multi-crossing
        IBP-cancellation trap that pop_off_saddle rxns get stuck in.
        ``kappa ≥ 100`` is too sharp and destabilizes the optimizer.

        Softmax activation: temperature τ. ``kappa=1`` (default) gives
        near-uniform basis; ``kappa ≳ 10`` gives time-localized "one
        neuron per t" basis.
    base : BasePath, optional
        Base path the MLP corrects. Defaults to ``LinearPath``.
    """
    def __init__(
        self,
        width: int = 128,
        depth: int = 2,
        activation: str = "gelu",
        kappa: float = 1.0,
        base: BasePath = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if activation.lower() == 'softmax':
            # Special case: competitive-normalized basis. ``kappa`` is
            # reinterpreted as softmax temperature τ. See ``TemperedSoftmax``.
            if depth != 2:
                raise ValueError(
                    f"activation='softmax' requires depth=2 (softmax in interior "
                    f"layers loses the partition-of-unity geometry); got depth={depth}"
                )
            self.activation = SquaredSoftmax(tau=kappa)
        else:
            activation_dict = {key.lower(): key for key in dir(nn) if not key.startswith('_')}
            name = activation_dict[activation.lower()]
            activation_class = getattr(nn, name)
            base_activation = activation_class()
            if kappa != 1.0:
                self.activation = SharpActivation(base_activation, kappa)
            else:
                self.activation = base_activation
        self.kappa = float(kappa)
        # Hidden width is `width` (default 128). System-independent.
        input_sizes = [1] + [width]*(depth - 1)
        output_sizes = input_sizes[1:] + [self.final_position.shape[-1]]
        self.layers = [
            nn.Linear(
                input_sizes[i//2], output_sizes[i//2], dtype=self.dtype, bias=True
            ) if i%2 == 0 else self.activation\
            for i in range(depth*2 - 1)
        ]
       
        self.mlp = nn.Sequential(*self.layers)
        self.mlp.to(self.device)

        # self.embed = nn.Sequential(
        #     nn.Linear(1, width, bias=True, dtype=self.dtype, device=self.device),
        #     self.activation,
        #     nn.Linear(width, width, bias=True, dtype=self.dtype, device=self.device),
        # )
        # self.out = nn.Linear(width, self.final_position.shape[-1], bias=False, dtype=self.dtype, device=self.device)

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
        out = self.mlp(2 * time - 1) * (1 - time) * time
        # mlp_out = (
        #     self.mlp(2 * time - 1) 
        #     - self.mlp(torch.tensor([1], device=time.device, dtype=self.dtype)) * time
        #     - self.mlp(torch.tensor([-1], device=time.device, dtype=self.dtype)) * (1 - time)
        # )
        # embed = self.embed(2 * time - 1) * (1 - time) * time
        # out = self.out(embed)
        if self.fix_positions is not None:
            out[:, self.fix_positions] = 0.0
        base_out = self.base.get_positions(time)
        out = base_out + out
        return out

# class ResNetLayer(nn.Module):
#     def __init__(
#         self,
#         output_size: int,
#         n_embed: int,
#         activation: str = "selu",
#     ):
#         super().__init__()
#         self.activation = activation_dict[activation]
#         self.layer = nn.Sequential(
#             nn.Linear(output_size, output_size * n_embed, dtype=torch.float64, bias=True),
#             self.activation,
#             nn.Linear(output_size * n_embed, output_size, dtype=torch.float64, bias=True),
#         )
#         # self.layer = SwiGLU(output_size, output_size)

#     def forward(self, x):
#         return x + self.layer(x)
    
# class SwiGLU(nn.Module):
#     def __init__(self, in_features, out_features):
#         super().__init__()
#         self.fc1 = nn.Linear(in_features, out_features)
#         self.fc2 = nn.Linear(in_features, out_features)
#         self.activation = nn.SiLU()

#     def forward(self, x):
#         return self.fc1(x) * self.activation(self.fc2(x))
