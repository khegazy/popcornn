import torch
from popcornn.tools import (
    PATH_INTEGRANDS,
    build_integrand_terms,
    evaluate_integrand_sum,
)


def test_single_integrand_outputs_match_per_class():
    """Single-term ``evaluate_integrand_sum`` reproduces the per-class
    ``PathIntegrand.evaluate`` output (sanity-checks the registry
    plumbing and the cache helper). Skips ``geodesic`` since it needs
    forces_decomposed which isn't part of this synthetic harness."""
    T = 100
    N_atoms = 17
    torch.manual_seed(0)
    time = torch.linspace(0, 1, T).unsqueeze(-1)
    energies = torch.randn(T, N_atoms) * 50 + 1400
    velocities = torch.rand((T, N_atoms * 3)) * 5 + 3
    forces = torch.rand((T, N_atoms * 3)) * 20 + 10
    cache = {
        'time': time,
        'energies': energies,
        'velocities': velocities,
        'forces': forces,
    }

    for name, cls in PATH_INTEGRANDS.items():
        if name == 'geodesic':
            continue
        terms = build_integrand_terms([name])
        composite, _ = evaluate_integrand_sum(terms, time, path=None, cache=cache)
        direct = cls().evaluate(cache)
        assert torch.allclose(composite, direct), \
            f"{name}: composite sum diverged from direct evaluate"


def test_weighted_sum_matches_linear_combination():
    """Two-term weighted sum equals scale_a * a + scale_b * b."""
    T = 100
    N_atoms = 17
    torch.manual_seed(1)
    time = torch.linspace(0, 1, T).unsqueeze(-1)
    energies = torch.randn(T, N_atoms) * 50 + 1400
    velocities = torch.rand((T, N_atoms * 3)) * 5 + 3
    forces = torch.rand((T, N_atoms * 3)) * 20 + 10
    cache = {
        'time': time,
        'energies': energies,
        'velocities': velocities,
        'forces': forces,
    }

    scales = [17.68, 11.45]
    pairs = [
        ('pvre', 'vre'),
        ('pvre', 'F_mag'),
        ('E_mean', 'F_mag'),
    ]
    for name_a, name_b in pairs:
        terms = build_integrand_terms([name_a, name_b], scales)
        combined, _ = evaluate_integrand_sum(terms, time, path=None, cache=cache)
        direct = (
            scales[0] * PATH_INTEGRANDS[name_a]().evaluate(cache)
            + scales[1] * PATH_INTEGRANDS[name_b]().evaluate(cache)
        )
        assert torch.allclose(combined, direct), \
            f"weighted sum {name_a} + {name_b} did not match"


def test_also_resolve_returns_requested_fields():
    """``also_resolve`` forces extra fields into the resolved variables dict
    even when no integrand declares them in ``requires``."""
    T = 50
    N_atoms = 5
    torch.manual_seed(2)
    time = torch.linspace(0, 1, T).unsqueeze(-1)
    energies = torch.randn(T, N_atoms)
    velocities = torch.rand(T, N_atoms * 3)
    forces = torch.rand(T, N_atoms * 3)
    cache = {
        'time': time,
        'energies': energies,
        'velocities': velocities,
        'forces': forces,
    }

    # F_mag only declares 'forces'; without also_resolve, energies wouldn't
    # be required. With it, the resolver still threads them through.
    terms = build_integrand_terms(['F_mag'])
    _, variables = evaluate_integrand_sum(
        terms, time, path=None, cache=cache, also_resolve=('energies',),
    )
    assert variables['energies'] is not None
    assert torch.equal(variables['energies'], energies)


def test_pvre_squared_equals_pvre_squared():
    """``pvre_squared`` integrand value is the elementwise square of
    ``pvre``. This is the defining property of the smooth variant."""
    T = 100
    N_atoms = 17
    torch.manual_seed(3)
    velocities = torch.randn(T, N_atoms * 3, dtype=torch.float64)
    forces = torch.randn(T, N_atoms * 3, dtype=torch.float64)
    cache = {'velocities': velocities, 'forces': forces}
    pvre = PATH_INTEGRANDS['pvre']().evaluate(cache)
    pvre_sq = PATH_INTEGRANDS['pvre_squared']().evaluate(cache)
    assert torch.allclose(pvre_sq, pvre ** 2)
    assert (pvre_sq >= 0).all()


def test_pvre_huber_matches_manual():
    """``pvre_huber.evaluate`` reproduces the Huber formula on s = v·F:
    quadratic basin for ``|s| ≤ δ``, linear arms outside."""
    T = 200
    N_atoms = 7
    delta = 0.5
    torch.manual_seed(4)
    velocities = torch.randn(T, N_atoms * 3, dtype=torch.float64)
    forces = torch.randn(T, N_atoms * 3, dtype=torch.float64)
    cache = {'velocities': velocities, 'forces': forces}

    huber = PATH_INTEGRANDS['pvre_huber'](delta=delta).evaluate(cache)

    overlap = torch.sum(velocities * forces, dim=-1, keepdim=True)
    abs_overlap = overlap.abs()
    expected = torch.where(
        abs_overlap <= delta,
        0.5 * overlap ** 2,
        delta * (abs_overlap - 0.5 * delta),
    )
    assert torch.allclose(huber, expected)
    # Spans both regimes for this random sample.
    assert (abs_overlap <= delta).any() and (abs_overlap > delta).any()


def test_pvre_huber_limit_behaviors():
    """δ → ∞ collapses to ½·pvre_squared; δ → 0 collapses to δ·pvre."""
    T = 100
    N_atoms = 11
    torch.manual_seed(5)
    velocities = torch.randn(T, N_atoms * 3, dtype=torch.float64)
    forces = torch.randn(T, N_atoms * 3, dtype=torch.float64)
    cache = {'velocities': velocities, 'forces': forces}

    pvre_sq = PATH_INTEGRANDS['pvre_squared']().evaluate(cache)
    pvre = PATH_INTEGRANDS['pvre']().evaluate(cache)

    huber_large = PATH_INTEGRANDS['pvre_huber'](delta=1e6).evaluate(cache)
    assert torch.allclose(huber_large, 0.5 * pvre_sq)

    delta_small = 1e-6
    huber_small = PATH_INTEGRANDS['pvre_huber'](delta=delta_small).evaluate(cache)
    # All |s| ≫ δ_small here, so the linear arm dominates.
    expected = delta_small * (pvre - 0.5 * delta_small)
    assert torch.allclose(huber_small, expected)


def test_pvre_huber_continuity_at_seam():
    """Evaluation is continuous and the derivative matches at |s|=δ.

    Build a synthetic ``v · F = s`` exactly (single dof, set v = s, F = 1)
    so the seam can be probed precisely. Compare the integrand values at
    s = δ ± ε via finite differences against the analytic gradient
    (s on the quadratic side, δ·sign(s) on the linear arm) at the same
    point — they should agree to FD accuracy.
    """
    delta = 0.5
    eps = 1e-6
    huber = PATH_INTEGRANDS['pvre_huber'](delta=delta)

    def loss_at(s):
        v = torch.tensor([[s]], dtype=torch.float64)
        f = torch.tensor([[1.0]], dtype=torch.float64)
        return huber.evaluate({'velocities': v, 'forces': f}).item()

    # Continuity: value at s=δ from both sides agrees with the seam value.
    assert abs(loss_at(delta - eps) - loss_at(delta)) < 10 * eps
    assert abs(loss_at(delta + eps) - loss_at(delta)) < 10 * eps

    # Derivative at the seam (analytic: δ from both sides since dquad/ds = s = δ).
    fd_left = (loss_at(delta) - loss_at(delta - eps)) / eps
    fd_right = (loss_at(delta + eps) - loss_at(delta)) / eps
    assert abs(fd_left - delta) < 1e-3
    assert abs(fd_right - delta) < 1e-3


def test_pvre_pseudo_huber_matches_manual():
    """``pvre_pseudo_huber.evaluate`` reproduces ``δ²·(√(1+(s/δ)²) − 1)``."""
    T = 200
    N_atoms = 7
    delta = 0.5
    torch.manual_seed(40)
    velocities = torch.randn(T, N_atoms * 3, dtype=torch.float64)
    forces = torch.randn(T, N_atoms * 3, dtype=torch.float64)
    cache = {'velocities': velocities, 'forces': forces}

    pseudo = PATH_INTEGRANDS['pvre_pseudo_huber'](delta=delta).evaluate(cache)

    overlap = torch.sum(velocities * forces, dim=-1, keepdim=True)
    expected = delta ** 2 * (torch.sqrt(1.0 + (overlap / delta) ** 2) - 1.0)
    assert torch.allclose(pseudo, expected)
    assert (pseudo >= 0).all()


def test_pvre_pseudo_huber_limit_behaviors():
    """δ → ∞ collapses to ½·pvre_squared; δ → 0 collapses to δ·pvre."""
    T = 100
    N_atoms = 11
    torch.manual_seed(50)
    velocities = torch.randn(T, N_atoms * 3, dtype=torch.float64)
    forces = torch.randn(T, N_atoms * 3, dtype=torch.float64)
    cache = {'velocities': velocities, 'forces': forces}

    pvre_sq = PATH_INTEGRANDS['pvre_squared']().evaluate(cache)
    pvre = PATH_INTEGRANDS['pvre']().evaluate(cache)

    # δ → ∞: Taylor gives L = s²/2 − s⁴/(8δ²) + …. Pick δ where the
    # analytic remainder dominates over float64 cancellation in √(1+x)−1.
    delta_large = 1e2
    pseudo_large = PATH_INTEGRANDS['pvre_pseudo_huber'](delta=delta_large).evaluate(cache)
    # Remainder bound: |L − s²/2| ≤ s⁴ / (8 δ²) ≤ 5 · max(s²)² / δ².
    bound = 5.0 * pvre_sq.max() ** 2 / delta_large ** 2
    assert (pseudo_large - 0.5 * pvre_sq).abs().max() < bound

    # δ → 0 with all |s| ≫ δ: √(1+(s/δ)²) ≈ |s|/δ, so L ≈ δ·|s| − δ².
    delta_small = 1e-6
    pseudo_small = PATH_INTEGRANDS['pvre_pseudo_huber'](delta=delta_small).evaluate(cache)
    expected = delta_small * pvre - delta_small ** 2
    assert torch.allclose(pseudo_small, expected, rtol=1e-3, atol=1e-12)


def test_pvre_pseudo_huber_is_smooth():
    """The pseudo-Huber gradient is C^∞ — finite-difference derivatives
    around |s|=δ agree on both sides, unlike Huber which has a slope corner."""
    delta = 0.5
    eps = 1e-6
    pseudo = PATH_INTEGRANDS['pvre_pseudo_huber'](delta=delta)

    def loss_at(s):
        v = torch.tensor([[s]], dtype=torch.float64)
        f = torch.tensor([[1.0]], dtype=torch.float64)
        return pseudo.evaluate({'velocities': v, 'forces': f}).item()

    # Analytic ∂L/∂s at s=δ is δ / √2.
    expected_grad = delta / (1.0 + 1.0) ** 0.5
    fd_left = (loss_at(delta) - loss_at(delta - eps)) / eps
    fd_right = (loss_at(delta + eps) - loss_at(delta)) / eps
    assert abs(fd_left - expected_grad) < 1e-3
    assert abs(fd_right - expected_grad) < 1e-3
    # Smoothness signature: left and right slopes coincide at the seam.
    assert abs(fd_left - fd_right) < 1e-5


def test_path_integrand_kwargs_reach_constructor():
    """``path_integrand_kwargs`` plumbed through ``build_integrand_terms``
    reaches the integrand constructor, and the term evaluates with the
    supplied parameter end-to-end via ``evaluate_integrand_sum``."""
    T = 50
    N_atoms = 5
    delta = 0.01
    torch.manual_seed(6)
    time = torch.linspace(0, 1, T).unsqueeze(-1)
    velocities = torch.randn(T, N_atoms * 3, dtype=torch.float64)
    forces = torch.randn(T, N_atoms * 3, dtype=torch.float64)
    cache = {'time': time, 'velocities': velocities, 'forces': forces}

    terms = build_integrand_terms(
        ['pvre_huber'],
        kwargs={'pvre_huber': {'delta': delta}},
    )
    assert terms[0].integrand.delta == delta

    composite, _ = evaluate_integrand_sum(terms, time, path=None, cache=cache)
    direct = PATH_INTEGRANDS['pvre_huber'](delta=delta).evaluate(cache)
    assert torch.allclose(composite, direct)

    # Sanity: a different δ produces a different value on the same data.
    other = PATH_INTEGRANDS['pvre_huber'](delta=delta * 100).evaluate(cache)
    assert not torch.allclose(composite, other)


def test_stray_kwargs_key_raises():
    """A name in ``kwargs`` that isn't in ``names`` is a typo, not silent."""
    try:
        build_integrand_terms(['pvre'], kwargs={'pvre_huber': {'delta': 1.0}})
    except ValueError as e:
        assert 'pvre_huber' in str(e)
    else:
        raise AssertionError("Expected ValueError for stray kwargs key")


def test_unknown_name_raises():
    try:
        build_integrand_terms(['not_a_real_integrand'])
    except ValueError as e:
        assert 'Unknown integrand' in str(e)
    else:
        raise AssertionError("Expected ValueError for unknown integrand name")


def test_duplicate_name_raises():
    try:
        build_integrand_terms(['F_mag', 'F_mag'])
    except ValueError as e:
        assert 'twice' in str(e)
    else:
        raise AssertionError("Expected ValueError for duplicate integrand name")
