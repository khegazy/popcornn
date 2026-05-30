"""Path integrand classes and helpers.

One ``PathIntegrand`` ABC plus nine concrete subclasses that each compute a
per-time integrand value from a dict of cached path quantities. Free
functions own variable resolution (``resolve_variables``), term construction
(``build_integrand_terms``), and the weighted-sum loop
(``evaluate_integrand_sum``) so that integrand classes themselves stay pure
and never reach into the path object.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import torch


class PathIntegrand(ABC):
    """One per-point integrand value.

    Subclasses declare the cache keys they consume in ``requires`` and
    implement ``evaluate(variables)``. Variable resolution is the harness's
    job (``resolve_variables``); integrands never touch the path.
    """

    requires: tuple[str, ...] = ()

    @abstractmethod
    def evaluate(self, variables: dict) -> torch.Tensor: ...


class Geodesic(PathIntegrand):
    """``‖F_decomposed · v‖₂``. Use with ``repel`` for geodesic interpolation."""

    requires = ('forces_decomposed', 'velocities')

    def evaluate(self, variables):
        projection = torch.einsum(
            'bki,bi->bk',
            variables['forces_decomposed'],
            variables['velocities'],
        )
        return torch.linalg.norm(projection, dim=-1, keepdim=True)


class VRE(PathIntegrand):
    """``‖F‖ · ‖v‖`` — magnitudes-only product. Pair with pVRE for angular error."""

    requires = ('forces', 'velocities')

    def evaluate(self, variables):
        F = torch.linalg.norm(variables['forces'], dim=-1, keepdim=True)
        V = torch.linalg.norm(variables['velocities'], dim=-1, keepdim=True)
        return F * V


class pVRE(PathIntegrand):
    """``|v · F|``. Drives F⟂path (saddle condition); default TS-search loss.

    The sign-driven gradient keeps pushing as the path converges, so this
    snaps onto the saddle ridge precisely — but its kink at v·F=0 makes the
    integrand expensive to quadrature. Pair with ``pVRESquared`` as a
    warm-up stage when integration cost matters; see
    ``examples/configs/muller_brown.yaml``.
    """

    requires = ('forces', 'velocities')

    def evaluate(self, variables):
        overlap = torch.sum(
            variables['velocities'] * variables['forces'],
            dim=-1,
            keepdim=True,
        )
        return torch.abs(overlap)


class pVREMag(PathIntegrand):
    """``‖v ⊙ F‖₂`` — per-component product, then norm."""

    requires = ('forces', 'velocities')

    def evaluate(self, variables):
        return torch.linalg.norm(
            variables['velocities'] * variables['forces'],
            dim=-1,
            keepdim=True,
        )


class pVRESquared(PathIntegrand):
    """``(v · F)²`` — smooth pVRE; C^∞ gradient for cleaner adaptive quadrature.

    Same saddle-condition physics as ``pVRE``
    (zero iff ``v ⊥ F``), but no kink at the zero crossings — ``|s|`` makes
    ``∂L/∂θ ∝ sign(s)`` jump in t at every crossing, while ``s²`` keeps
    ``∂L/∂θ ∝ s`` smooth there. The integrator quadratures ``∂L/∂θ`` along
    the path, so removing those jumps lets adaptive Gauss–Kronrod hit its
    design convergence rate instead of refining indefinitely around the
    discontinuities.
    """

    requires = ('forces', 'velocities')

    def evaluate(self, variables):
        overlap = torch.sum(
            variables['velocities'] * variables['forces'],
            dim=-1,
            keepdim=True,
        )
        return overlap ** 2


class pVREHuber(PathIntegrand):
    """Huber on ``s = v · F``.

    ``½·s²`` for ``|s| ≤ δ``; ``δ·(|s| − ½·δ)`` otherwise. C¹-smooth at
    the seam (gradient ``δ·sign(s)`` on the linear arm matches ``s`` on
    the quadratic side at ``|s|=δ``), so adaptive Gauss–Kronrod doesn't
    refine indefinitely the way ``pvre``'s ``|s|`` kink forces it to,
    while the linear arms keep ``∂L/∂θ`` constant-magnitude far from the
    saddle — unlike ``pvre_squared``, whose ``∂L/∂θ ∝ s`` plateaus once
    the path is close. Sweeps δ to interpolate one-shot between the two:
    δ → ∞ recovers ``½·pvre_squared``; δ → 0 recovers ``δ·pvre``.
    """

    requires = ('forces', 'velocities')

    def __init__(self, delta=1.0):
        if delta <= 0:
            raise ValueError(f"pvre_huber delta must be positive, got {delta}.")
        self.delta = float(delta)

    def evaluate(self, variables):
        overlap = torch.sum(
            variables['velocities'] * variables['forces'],
            dim=-1,
            keepdim=True,
        )
        abs_overlap = overlap.abs()
        quadratic = 0.5 * overlap ** 2
        linear = self.delta * (abs_overlap - 0.5 * self.delta)
        return torch.where(abs_overlap <= self.delta, quadratic, linear)


class pVREPseudoHuber(PathIntegrand):
    """Pseudo-Huber on ``s = v · F``: ``δ²·(√(1 + (s/δ)²) − 1)``.

    Same δ-controlled interpolation between ``pvre_squared`` and ``pvre``
    as ``pvre_huber`` (δ → ∞ ⇒ ``½·s²``; δ → 0 ⇒ ``δ·|s|``), but C^∞
    everywhere — there is no piecewise seam at ``|s|=δ``, just a smooth
    transition. The integrand ``∂L/∂s = s / √(1 + (s/δ)²)`` is bounded
    by ``±δ`` and analytic, so adaptive Gauss–Kronrod converges at its
    design rate without the local subdivision Huber's slope corner at
    ``|s|=δ`` provokes.
    """

    requires = ('forces', 'velocities')

    def __init__(self, delta=1.0):
        if delta <= 0:
            raise ValueError(f"pvre_pseudo_huber delta must be positive, got {delta}.")
        self.delta = float(delta)

    def evaluate(self, variables):
        overlap = torch.sum(
            variables['velocities'] * variables['forces'],
            dim=-1,
            keepdim=True,
        )
        return self.delta ** 2 * (torch.sqrt(1.0 + (overlap / self.delta) ** 2) - 1.0)


class Energy(PathIntegrand):
    """Raw potential energy."""

    requires = ('energies',)

    def evaluate(self, variables):
        return variables['energies']


class EnergyMean(PathIntegrand):
    """Mean energy across the trailing dim."""

    requires = ('energies',)

    def evaluate(self, variables):
        return torch.mean(variables['energies'], dim=-1, keepdim=True)


class VREError(PathIntegrand):
    """``VRE - pVRE``. Force-velocity angular mismatch; → 0 on a true MEP."""

    requires = ('forces', 'velocities')

    def __init__(self):
        self._pvre = pVRE()
        self._vre = VRE()

    def evaluate(self, variables):
        return self._vre.evaluate(variables) - self._pvre.evaluate(variables)


class ForceMagnitude(PathIntegrand):
    """``‖F‖₂``. Useful as a TS-time loss."""

    requires = ('forces',)

    def evaluate(self, variables):
        return torch.linalg.norm(variables['forces'], dim=-1, keepdim=True)


PATH_INTEGRANDS: dict[str, type[PathIntegrand]] = {
    'geodesic': Geodesic,
    'pvre': pVRE,
    'pvre_mag': pVREMag,
    'pvre_squared': pVRESquared,
    'pvre_huber': pVREHuber,
    'pvre_pseudo_huber': pVREPseudoHuber,
    'vre': VRE,
    'vre_error': VREError,
    'E': Energy,
    'E_mean': EnergyMean,
    'F_mag': ForceMagnitude,
}


@dataclass
class IntegrandTerm:
    """One ``(name, integrand, scale)`` entry in an integrand sum."""

    name: str
    integrand: PathIntegrand
    scale: float


def build_integrand_terms(names, scales=None, kwargs=None) -> list[IntegrandTerm]:
    """Look ``names`` up in ``PATH_INTEGRANDS``, instantiate, pair with scales.

    Parameters
    ----------
    names : str or list[str]
        Registry keys.
    scales : float, list[float], torch.Tensor, or None
        Per-term weights; defaults to all ones.
    kwargs : dict[str, dict] or None
        Per-term constructor kwargs, keyed by name. ``{}`` or ``None`` falls
        back to a no-arg constructor for every term, which is what every
        unparameterized integrand wants. Only the integrands that take
        ``__init__`` arguments (e.g. ``pvre_huber``'s ``delta``) need an
        entry here; all other names ignore ``kwargs`` even when present.

    Raises
    ------
    ValueError
        On unknown names, duplicate names, scale-length mismatch, or
        ``kwargs`` keys that don't appear in ``names``.
    """
    if names is None or (isinstance(names, (list, tuple)) and len(names) == 0):
        raise ValueError("Must supply at least one integrand name.")
    if isinstance(names, str):
        names = [names]

    if scales is None:
        scales = [1.0] * len(names)
    elif isinstance(scales, torch.Tensor):
        scales = scales.tolist()
    elif not isinstance(scales, Iterable):
        scales = [scales]

    if len(names) != len(scales):
        raise ValueError(
            f"Number of integrand names ({len(names)}) does not match "
            f"number of scales ({len(scales)})."
        )

    if kwargs is None:
        kwargs = {}
    stray = set(kwargs) - set(names)
    if stray:
        raise ValueError(
            f"path_integrand_kwargs has keys not in names: {sorted(stray)}; "
            f"names = {names}."
        )

    seen: set[str] = set()
    terms: list[IntegrandTerm] = []
    for name, scale in zip(names, scales):
        if name not in PATH_INTEGRANDS:
            raise ValueError(
                f"Unknown integrand {name!r}; choose from {sorted(PATH_INTEGRANDS)}."
            )
        if name in seen:
            raise ValueError(f"Cannot use the same integrand twice: {name!r}.")
        seen.add(name)
        terms.append(IntegrandTerm(
            name=name,
            integrand=PATH_INTEGRANDS[name](**kwargs.get(name, {})),
            scale=float(scale),
        ))
    return terms


# TODO: Remove this function and move its logic into the integrator class, which is the only caller.
def resolve_variables(
    eval_time,
    path,
    requires,
    *,
    time=None,
    positions=None,
    velocities=None,
    energies=None,
    energies_decomposed=None,
    forces=None,
    forces_decomposed=None,
):
    """Return the cached path quantities the caller's integrands need.

    Reuses prior values when ``time`` matches ``eval_time``; otherwise
    re-walks the path once. Quantities not asked for are still threaded
    through if already available.
    """
    requires = set(requires)

    needs_velocities = 'velocities' in requires
    needs_energies = 'energies' in requires and energies is None
    needs_energies_dec = 'energies_decomposed' in requires and energies_decomposed is None
    needs_forces = 'forces' in requires and forces is None
    needs_forces_dec = 'forces_decomposed' in requires and forces_decomposed is None

    missing_any_energy = needs_energies or needs_energies_dec
    missing_any_force = needs_forces or needs_forces_dec

    time_match = (
        time is not None
        and time.shape == eval_time.shape
        and torch.allclose(time, eval_time, atol=1e-10)
    )

    if not time_match or missing_any_energy or missing_any_force:
        path_output = path(
            eval_time,
            return_velocities=needs_velocities,
            return_energies=needs_energies,
            return_energies_decomposed=needs_energies_dec,
            return_forces=needs_forces,
            return_forces_decomposed=needs_forces_dec,
        )
        time = eval_time
        if path_output.velocities is not None:
            velocities = path_output.velocities
        if path_output.energies is not None:
            energies = path_output.energies
        if path_output.energies_decomposed is not None:
            energies_decomposed = path_output.energies_decomposed
        if path_output.forces is not None:
            forces = path_output.forces
        if path_output.forces_decomposed is not None:
            forces_decomposed = path_output.forces_decomposed
    elif needs_velocities and velocities is None:
        velocities = path.calculate_velocities(time)

    return {
        'time': time,
        'positions': positions,
        'velocities': velocities,
        'energies': energies,
        'energies_decomposed': energies_decomposed,
        'forces': forces,
        'forces_decomposed': forces_decomposed,
    }


# TODO: Remove this function and move its logic into the integrator class, which is the only caller. A sum of integrands should have its own class that handles this logic internally.
def evaluate_integrand_sum(
    terms,
    eval_time,
    path,
    *,
    cache=None,
    also_resolve=(),
):
    """Resolve variables once, then return ``Σ scale_i · integrand_i(variables)``.

    Parameters
    ----------
    also_resolve : iterable of str, optional
        Extra cache keys to force-resolve in addition to those declared
        by the terms. Used by ``PathIntegrator`` to capture
        ``('energies', 'forces')`` for transition-state finding even when
        the active integrand only requires forces.

    Returns
    -------
    (loss, variables) : tuple
        ``loss`` is the weighted integrand sum. ``variables`` is the
        resolved cache, suitable for re-passing as ``cache=`` on the next
        call to skip path re-evaluation when ``eval_time`` matches.
    """
    requires = {r for term in terms for r in term.integrand.requires}
    requires |= set(also_resolve)
    variables = resolve_variables(eval_time, path, requires, **(cache or {}))

    total = sum(term.scale * term.integrand.evaluate(variables) for term in terms)

    return total, variables
