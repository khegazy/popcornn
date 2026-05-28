import torch
from dataclasses import dataclass
from einops import rearrange
from popcornn.tools import Images, SamplesCache, wrap_positions
from popcornn.potentials.base_potential import BasePotential, PotentialOutput


@dataclass
class PathOutput():
    """
    Data class representing the output of a path computation.

    Attributes:
    -----------
    time : torch.Tensor
        The time at which the path was evaluated.
    positions : torch.Tensor
        The coordinates along the path.
    velocities : torch.Tensor, optional
        The velocities along the path (default is None).
    energies : torch.Tensor
        The potential energy along the path.
    forces : torch.Tensor, optional
        The force along the path (default is None).
    """
    time: torch.Tensor
    positions: torch.Tensor
    velocities: torch.Tensor = None
    energies: torch.Tensor = None
    energies_decomposed: torch.Tensor = None
    forces: torch.Tensor = None
    forces_decomposed: torch.Tensor = None

    def __len__(self):
        """
        Return the number of images.
        """
        return len(self.positions)


class BasePath(torch.nn.Module):
    """
    Base class for differentiable path representations.

    A path is a smooth mapping ``t -> x(t)`` from ``t in [0, 1]`` to a
    configuration, with ``x(0)`` pinned at the reactant and ``x(1)`` at
    the product. Subclasses implement ``get_positions``; this base
    class wires up

    - velocity computation via autograd (``calculate_velocities``),
    - periodic-cell wrapping (when ``images.pbc`` is set),
    - fixed-atom masking,
    - the ``forward`` interface that popcornn's optimizer drives,
    - the input/output reshaping that lets the integrator pass either
      ``[B, T]`` or ``[B, C, T]`` time tensors.

    Subclasses must populate trainable ``torch.nn.Parameter``\\s.
    """
    initial_position: torch.Tensor
    final_position: torch.Tensor

    def __init__(
            self,
            images: Images,
            device: torch.device,
            dtype: torch.dtype,
            find_ts: bool = True,
        ) -> None:
        """
        Initialize the path.

        Parameters
        ----------
        images : Images
            Processed images. The first frame's positions become
            ``self.initial_position``, the last frame's become
            ``self.final_position``. Periodic-cell info, fixed-atom
            masks, and tags are pulled from here.
        device : torch.device
        dtype : torch.dtype
        find_ts : bool, default=True
            Whether the optimization loop should run ``ts_search`` each
            iteration and populate ``ts_time`` / ``ts_energy`` /
            ``ts_force`` / ``ts_force_mag``. Set False to skip the
            sign-change + fresh-eval step entirely.
        """
        super().__init__()
        self.neval = 0
        self.find_ts = find_ts
        self.potential = None
        self.initial_position = images.positions[0]
        self.final_position = images.positions[-1]
        self._inp_reshaped = None
        if images.pbc is not None and images.pbc.any():
            def transform(positions, **kwargs):
                return wrap_positions(positions, images.cell, images.pbc, **kwargs)
            self.transform = transform
        else:
            self.transform = None
        self.fix_positions = images.fix_positions
        self.device = device
        self.dtype = dtype
        self.t_init = torch.tensor(
            [[0]], dtype=self.dtype, device=self.device
        )
        self.t_final = torch.tensor(
            [[1]], dtype=self.dtype, device=self.device
        )
        self.ts_time = None
        self.n_crossings = None

    def set_potential(
            self,
            potential: BasePotential,
    ) -> None:
        """
        Attach a potential to evaluate energies/forces along the path.

        Each optimization stage constructs its own potential and calls
        this; the path holds onto the most recently set one.
        """
        # TODO: assigning potential as a submodule causes its parameters
        # to leak into path.state_dict() — e.g., a NewtonNet potential
        # adds ~40 keys (potential.model.embedding_layer..., interaction_layers...,
        # scalers..., etc.) that bloat MLP checkpoints and force callers to
        # use load_state_dict(..., strict=False). The path's "saveable"
        # parameters are just the MLP weights, not the (typically frozen,
        # externally-loaded) potential network. Exclude potential parameters
        # from state_dict — either register as non-module attribute
        # (object.__setattr__ or store via a wrapper) or override
        # _save_to_state_dict / state_dict to skip the 'potential.' prefix.
        self.potential = potential

    def get_positions(
            self,
            time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the geometric path at ``time``. Subclasses must override.

        Parameters
        ----------
        time : torch.Tensor
            Times in [0, 1]; shape ``[N, 1]``.

        Returns
        -------
        torch.Tensor
            Positions of shape ``[N, D]``.
        """
        raise NotImplementedError()


    def calculate_velocities(self, t, create_graph=True):
        """
        Compute path velocities via autograd.

        Differentiates ``get_positions`` with respect to ``t``. The
        ``torch.sum`` over the leading axis is a vectorization trick:
        summing collapses the per-time outputs so a single jacobian
        call returns ``dx_i/dt_i`` for every i in one pass.
        """
        return torch.autograd.functional.jacobian(
            lambda t: torch.sum(self.get_positions(t), axis=0),
            t,
            create_graph=create_graph,
            vectorize=True
        ).transpose(0, 1)[:, :, 0]
    
    def _check_output(
            self,
            potential_output,
            return_energies: bool,
            return_energies_decomposed: bool,
            return_forces: bool,
            return_forces_decomposed: bool,
        ):
        """
        Raise if the attached potential can't produce a requested field.

        Toy potentials skip force-decomposition; some MLIP wrappers
        skip energy-decomposition. Catch the missing field at the
        first call site rather than later in the loss layer.
        """
        name = type(self.potential).__name__
        if return_energies and potential_output.energies is None:
            raise ValueError(f"Potential {name} cannot calculate energies")
        if return_energies_decomposed and potential_output.energies_decomposed is None:
            raise ValueError(f"Potential {name} cannot calculate energies_decomposed")
        if return_forces and potential_output.forces is None:
            raise ValueError(f"Potential {name} cannot calculate forces")
        if return_forces_decomposed and potential_output.forces_decomposed is None:
            raise ValueError(f"Potential {name} cannot calculate forces_decomposed")
    
    def forward(
            self,
            time : torch.Tensor = None,
            return_velocities: bool = False,
            return_energies: bool = False,
            return_energies_decomposed: bool = False,
            return_forces: bool = False,
            return_forces_decomposed: bool = False,
    ) -> PathOutput:
        """
        Evaluate the path (and optionally the potential) at given times.

        Parameters
        ----------
        time : torch.Tensor, optional
            Times in [0, 1]. Accepts shape ``[B]``, ``[B, T]`` or
            ``[B, C, T]``; the input shape is restored on the output.
            ``None`` defaults to 101 points linearly spaced over
            ``[t_init, t_final]``.
        return_velocities : bool, default=False
        return_energies : bool, default=False
        return_energies_decomposed : bool, default=False
        return_forces : bool, default=False
        return_forces_decomposed : bool, default=False
            Each toggles whether to populate the corresponding field
            on the returned ``PathOutput``. Disabled-by-default to
            avoid paying for autograd / potential evaluations the
            caller doesn't need.

        Returns
        -------
        PathOutput
            With ``time`` and ``positions`` always populated; other
            fields populated when the matching flag is set.
        """
        time = self._reshape_in(time)

        self.neval += time.numel()

        positions = self.get_positions(time)
        if self.transform is not None:
            positions = self.transform(positions)
        if return_energies or return_energies_decomposed or return_forces or return_forces_decomposed:
            assert self.potential is not None, "Potential must be set by \'set_potential\' before calling \'forward\'"
            potential_output = self.potential(positions) 
            self._check_output(
                potential_output,
                return_energies=return_energies,
                return_energies_decomposed=return_energies_decomposed,
                return_forces=return_forces,
                return_forces_decomposed=return_forces_decomposed
            )
        else:
            potential_output = PotentialOutput()

        if return_velocities:
            velocities = self.calculate_velocities(time)
        else:
            velocities = None

        return PathOutput(
            time=self._reshape_out(time),
            positions=self._reshape_out(positions),
            velocities=self._reshape_out(velocities),
            energies=self._reshape_out(potential_output.energies),
            energies_decomposed=self._reshape_out(potential_output.energies_decomposed),
            forces=self._reshape_out(potential_output.forces),
            forces_decomposed=self._reshape_out(potential_output.forces_decomposed),
        )
    

    def _reshape_in(self, time):
        """
        Flatten an arbitrary-shape time tensor to ``[N, T]`` for batched
        evaluation. ``_reshape_out`` undoes the flatten on the way out.

        The integrator passes ``[B, C, T]`` (batch x quadrature-channel
        x time) but downstream layers want a flat batch dim, so cache
        the input shape, rearrange, and remember to invert.
        """
        if time is None:
            time = torch.linspace(self.t_init.item(), self.t_final.item(), 101, device=self.device, dtype=self.dtype)
        
        if len(time.shape) == 3:
            self._inp_reshaped = True
            self._inp_shape = time.shape
            time = rearrange(time, 'b c t -> (b c) t')
        elif len(time.shape) == 2:
            self._inp_reshaped = False
            B, C, = None, None
        elif len(time.shape) == 1:
            self._inp_reshaped = False
            B, C, = None, None
            time = torch.unsqueeze(time, -1)
        else:
            raise ValueError(f"Input path time must be of dimensions [B, C, T], [B, T], or [B] where T is the time dimsion and is generally 1: instead got {time.shape}")

        return time


    def _reshape_out(self, result):
        """Restore the original shape stashed by ``_reshape_in``."""
        if self._inp_reshaped is None:
            raise RuntimeError("Must call _reshape_in() before _reshape_out()")
        if self._inp_reshaped and result is not None:
            B, C, _ = self._inp_shape
            return rearrange(result, '(b c) d -> b c d', b=B, c=C)
        return result

    
    def ts_search(
        self,
        samples: SamplesCache,
    ):
        """
        Locate the predicted transition state on the current path.

        Linearly interpolates ``t`` at **every** interior sign change of
        ``dE/dt`` on the cached quadrature samples, appends the two
        endpoints (``t=0`` and ``t=1``) to the candidate set, re-evaluates
        the path at all candidates in one batched forward, and picks the
        candidate with the highest model-truth ``energy``. Returns the
        winning candidate's fresh ``E`` and ``F`` and the corresponding
        barrier ``E_TS - E_reactant``.

        Why every interior crossing: on wiggly paths (high-capacity MLP,
        adversarial quadrature, multi-basin landscapes) there are multiple
        ``dE/dt = 0`` brackets and the cache-based ranking (mean of
        bracket-endpoint energies) can mis-rank them — picking a
        non-rate-limiting local max, or worse, a ``- → +`` crossing
        that's actually a local *minimum* of E. Evaluating model-truth
        ``E`` at each candidate and taking the argmax filters both
        failure modes.

        Why endpoints are included as candidates: a **barrierless** path
        has no interior maximum — the highest-energy point lies at one
        of the endpoints (the higher-lying reactant or product minimum
        for an exoergic / endoergic step). The endpoint candidate
        ``t=0`` or ``t=1`` correctly wins the argmax in that case, and
        the barrier reduces to ``max(E_R, E_P) - E_R`` — either zero
        (exoergic) or ``E_P - E_R`` (endoergic).

        Parameters
        ----------
        samples : SamplesCache
            Per-quadrature-point ``(time, energies, dE/dt)`` from
            ``PathIntegrator.integrate_path(save_samples=True)``.

        Notes
        -----
        Sets ``self.ts_time``, ``self.ts_energy``, ``self.ts_force``,
        ``self.ts_force_mag`` (per-atom fmax = ``max_i ‖F_i‖_2`` for
        atomistic systems; vector L2 norm for toy potentials with no
        ``n_atoms`` set), and ``self.barrier`` (``E_TS - E_reactant``,
        non-negative).

        The K candidates are batched into a single ``forward`` call —
        cost is one batched MLIP eval, not K sequential evals. On
        stiff paths ``K = 3`` (two endpoints + one interior crossing).
        On pathological wiggly paths ``K`` can be 20+; the batched call
        must fit in GPU memory. If it OOMs, chunk the candidates
        yourself before calling ``ts_search``.
        """
        time = samples.time
        dEdt = samples.dEdt
        N = time.shape[0]

        # Interior sign-change brackets. ``dE/dt`` touches zero at the
        # endpoints (reactant/product minima), so a global sign search
        # would falsely bracket the first/last segment; restrict to
        # interior segments [i, i+1] with i ∈ [1, N-3].
        #
        # Use ``<= 0`` rather than ``< 0`` so a sample that lands exactly
        # on the saddle (``dE/dt = 0``) is still bracketed; the linear
        # interp then degenerates to ``t = t_zero`` cleanly. Exclude the
        # both-zero case to avoid a 0/0 in the interp formula.
        interior_candidates = None
        if N >= 4:
            d_int = dEdt[1:N-1]
            both_zero = (d_int[:-1] == 0) & (d_int[1:] == 0)
            sign_change = ((d_int[:-1] * d_int[1:]) <= 0) & ~both_zero
            if sign_change.any():
                # Linear-interp t at each bracketed zero crossing.
                brackets = sign_change.nonzero(as_tuple=False).flatten()
                # ``brackets`` indexes into the interior segments; map
                # back to absolute indices into ``time`` / ``dEdt``.
                i_abs = brackets + 1
                t_i = time[i_abs]
                t_j = time[i_abs + 1]
                d_i = dEdt[i_abs]
                d_j = dEdt[i_abs + 1]
                # t_TS_k = t_i_k - d_i_k * (t_j_k - t_i_k) / (d_j_k - d_i_k)
                interior_candidates = t_i - d_i * (t_j - t_i) / (d_j - d_i)

        self.n_crossings = (
            int(interior_candidates.numel()) if interior_candidates is not None else 0
        )

        # Always include the endpoints as candidates so a barrierless
        # path can pick the higher-energy reactant or product as TS.
        # Put t=0 first so cand_E[0] is unambiguously the reactant energy.
        endpoint_candidates = torch.tensor(
            [0.0, 1.0], device=self.device, dtype=self.dtype,
        )
        if interior_candidates is not None:
            candidate_times = torch.cat(
                [endpoint_candidates,
                 interior_candidates.to(device=self.device, dtype=self.dtype)],
                dim=0,
            )
        else:
            candidate_times = endpoint_candidates

        # Batched fresh forward at every candidate. Picks the model-truth
        # global-max-E candidate — the rate-limiting saddle when several
        # brackets exist, or the higher-energy endpoint for a barrierless
        # path.
        cand_out = self.forward(
            candidate_times,
            return_velocities=False,
            return_energies=True,
            return_forces=True,
        )
        energies = cand_out.energies.detach().cpu()
        forces = cand_out.forces.detach().cpu()
        winner = int(torch.argmax(energies).item())

        self.ts_time = candidate_times[winner]
        self.ts_energy = energies[winner]
        self.ts_force = forces[winner]
        self.barrier = self.ts_energy - energies[0]