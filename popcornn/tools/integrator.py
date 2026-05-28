from dataclasses import dataclass

import torch

from torchpathint import path_integral

from .integrand import build_integrand_terms, evaluate_integrand_sum


@dataclass(frozen=True)
class SamplesCache:
    """Per-quadrature-point energy + dE/dt samples harvested during integration.

    Shape contract:
    - ``time``: ``[N*K]``, flattened from ``IntegralOutput.t``'s ``[N, K]``.
    - ``energies``: ``[N*K, E]`` (typically ``E=1``; potentials may emit a
      decomposed energy tensor).
    - ``dEdt``: ``[N*K]``, scalar ``dE/dt = ∇E·ẋ = -(F·v).sum(-1)``
      precomputed inside the integrator from the same forces and
      velocities that were already resolved for the loss integrand —
      cached as a scalar so the consumer doesn't carry F (shape ``[D]``)
      around just to recompute the projection.

    The integrator populates this object from the same evaluations that
    produced the gradient integral, so consuming it for transition-state
    finding adds zero extra path-forward calls.
    """

    time: torch.Tensor
    energies: torch.Tensor
    dEdt: torch.Tensor


class PathIntegrator:
    """
    Adaptive-quadrature path integrator.

    Wraps ``torchpathint.path_integral`` with the parts of the popcornn API
    the optimizer uses:

    - per-iteration ``integrand_scales`` updates so schedulers take effect,
    - direct scatter of ``∂L/∂θ`` into ``path.parameters().grad``
      (no separate ``.backward()`` call),
    - a ``grad_integral`` alias on the returned object for the flat
      ``[D]`` integrated gradient, and ``grad_norm`` / ``grad_norm_2``
      fields holding ``‖∫∇L dt‖_∞`` (monitoring only since 2026-05-15)
      and ``‖∫∇L dt‖_2`` (used by the convergence check),
    - an optional detached pass that integrates the loss itself, exposed
      as a ``loss`` field for monitoring,
    - an optional ``save_samples`` mode that captures per-quadrature-point
      ``(t, energies, forces)`` for transition-state finding without any
      extra path evaluations.
    """

    def __init__(
            self,
            method='gk7',
            path_integrand_names=None,
            path_integrand_scales=None,
            path_integrand_kwargs=None,
            rtol=1.0,
            atol=0.0,
            norm='2',
            max_batch=None,
            max_iter=5,
            track_loss=False,
            loss_rtol=0.0,
            loss_atol=0.0001,
            track_ts=False,
            device=None,
            dtype=None,
        ):
        """
        Parameters
        ----------
        method : str, default="gk7"
            torchpathint quadrature rule. ``gk7`` is the adaptive
            Gauss–Kronrod 7-point rule; chosen as the popcornn default
            after the 2026-05-12 integrator sweep (gg3 NN pseudo-Huber)
            showed it's 30% faster than ``gk21`` at identical TS quality.
            Use ``gk21`` / ``gk31`` for tighter integration if the path
            is unusually rough, or ``gl<n>`` for non-adaptive Gauss–Legendre.
        path_integrand_names : str or list of str, optional
            Per-point integrand (or list of them) to integrate. Looked up
            by name in ``PATH_INTEGRANDS``. See ``docs/loss-functions.md``.
        path_integrand_scales : float or list, optional
            Weighting per term when ``path_integrand_names`` is a list.
        path_integrand_kwargs : dict[str, dict], optional
            Per-term constructor kwargs, keyed by integrand name. Only
            parameterized integrands (e.g. ``pvre_huber``'s ``delta``)
            need an entry; unparameterized terms ignore ``kwargs`` even
            when present.
        rtol, atol : float
            Adaptive-quadrature tolerances on the gradient integral.
        norm : str, default='2'
            Vector norm to use for error estimation. ``"2"`` / ``"max"`` select L2 / L∞. Only relevant for adaptive Gauss-Kronrod.
        max_batch : int, optional
            Hard cap on the number of quadrature points evaluated in
            one batch. ``None`` lets torchpathint auto-size. After each
            ``integrate_path`` the learned size sticks: if torchpathint
            had to halve to recover from a CUDA OOM, the smaller value
            is reused on the next call so the OOM-and-halve cycle only
            fires once per integrator lifetime, not once per optimizer
            step.

            Scope is the ``PathIntegrator`` instance. Within a single
            stage / optimizer loop, persistence is automatic. Across
            stages built by ``Popcornn._optimize_stage`` (one fresh
            integrator per stage) it does **not** carry over — that's
            intentional, since different stages typically use different
            potentials with different memory profiles. Multi-stage
            harnesses that *do* know their later stages share a potential
            can thread the value explicitly::

                integ1 = PathIntegrator(...)
                for it in range(N1):
                    optr1.optimization_step(path, integ1)
                integ2 = PathIntegrator(..., max_batch=integ1.max_batch)
                for it in range(N2):
                    optr2.optimization_step(path, integ2)
        track_loss : bool, default=False
            Run a separate detached integral of the scalar loss
            ``∫L(t) dt`` for monitoring. Costs an extra pass; when
            enabled the result is attached as ``IntegralOutput.loss``.
        loss_rtol, loss_atol : float, optional
            Tolerances for the detached loss integral. Default to
            ``rtol``/``atol``.
        device : torch.device
        dtype : torch.dtype
        """
        self.method = method
        self.atol = atol
        self.rtol = rtol
        self.norm = norm
        # Loss integral is debug-only; let it run at its own (looser by
        # default) tolerance rather than paying for gradient-grade accuracy.
        self.track_loss = track_loss
        self.loss_rtol = loss_rtol
        self.loss_atol = loss_atol
        self.track_ts = track_ts
        self.max_batch = max_batch
        self.max_iter = max_iter
        self.device = device
        self.dtype = dtype
        self.N_integrals = 0
        self.integral_output = None

        if path_integrand_names is None:
            self._terms = []
        else:
            self._terms = build_integrand_terms(
                path_integrand_names,
                path_integrand_scales,
                path_integrand_kwargs,
            )

    def update_integrand_scales(self, **kwargs):
        """Replace one or more per-term scales. Used by schedulers to ramp
        terms up/down between iterations."""
        for name, scale in kwargs.items():
            for i, term in enumerate(self._terms):
                if term.name == name:
                    term.scale = float(scale)
                    break
            else:
                raise KeyError(
                    f"No integrand named {name!r} in this integrator "
                    f"(have: {[t.name for t in self._terms]})."
                )

    def integrate_path(
            self,
            path,
            integrand_scales=None,
            t_init=torch.tensor([0.]),
            t_final=torch.tensor([1.]),
        ):
        """
        Integrate the gradient of the loss along the path.

        Sets ``param.grad`` for each path parameter to the integrated
        gradient ``∫₀¹ ∂L/∂θ dt`` (accumulated, so multiple calls
        between ``optimizer.zero_grad()`` compose). Also returns the
        underlying ``IntegralOutput`` enriched with popcornn-level
        fields:

        - ``.grad_integral``: alias for the flat ``[D]`` integrated
          gradient (same tensor as ``.integral`` from torchpathint;
          named for clarity at popcornn call sites).
        - ``.grad_norm``: ``‖∫∇L dt‖_∞`` (monitoring only since 2026-05-15).
        - ``.grad_norm_2``: ``‖∫∇L dt‖_2`` (convergence trigger).
        - ``.loss``: scalar ``∫L(t) dt`` when ``track_loss=True``.
        - ``.samples``: ``SamplesCache`` (or ``None``) holding
          per-quadrature-point ``(t, E, dE/dt)`` when ``save_samples=True``.

        Parameters
        ----------
        path : BasePath
            Holds the trainable parameters and potential.
        integrand_scales : dict, optional
            Updated values for per-term scales (from schedulers).
        t_init, t_final : torch.Tensor
            Integration bounds, in [0, 1]. Squeezed to 0-d for
            torchpathint.
        """
        if integrand_scales:
            self.update_integrand_scales(**integrand_scales)

        # torchpathint requires 0-d bounds; popcornn historically passed 1-d.
        t_init_0d = torch.as_tensor(t_init, dtype=self.dtype, device=self.device).squeeze()
        t_final_0d = torch.as_tensor(t_final, dtype=self.dtype, device=self.device).squeeze()

        # Filter out frozen params (requires_grad=False). When a potential
        # is attached via set_potential, it's registered as an nn.Module
        # submodule of the path, so its params appear in path.parameters().
        # Potentials that freeze their weights (e.g. NewtonNet's
        # `model.requires_grad_(False)`) would otherwise trip
        # autograd.grad's "differentiated Tensors does not require grad".
        params = [p for p in path.parameters() if p.requires_grad]
        sizes = [p.numel() for p in params]

        # Side-buffer for transition-state-finding samples. Each entry is a
        # ``(t_chunk, E_chunk, dEdt_chunk)`` triplet captured inside ``f``
        # and later reassembled in IntegralOutput.t.flatten() order via
        # byte-keyed lookup (the same t tensor is passed to f and indexed
        # into accepted_t by torchpathint, so byte equality holds).
        # dE/dt is computed inside ``f`` as -(F·v).sum(-1) — forces and
        # velocities have already been resolved for the loss integrand,
        # so we reuse them rather than re-evaluating, and store only the
        # scalar so the consumer doesn't carry [D]-shaped forces around.
        sample_buffer: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        also_resolve = ('energies', 'forces', 'velocities') if self.track_ts else ()

        def f(t_flat):
            l, variables = evaluate_integrand_sum(
                self._terms,
                t_flat.unsqueeze(-1),
                path,
                also_resolve=also_resolve,
            )
            if self.track_ts:
                dEdt = -(variables['forces'] * variables['velocities']).sum(dim=-1)
                sample_buffer.append((
                    t_flat.detach().cpu(),
                    variables['energies'].detach().cpu(),
                    dEdt.detach().cpu(),
                ))
            l_per_t = l.reshape(t_flat.shape[0], -1).sum(dim=-1)  # [N], graph live
            n = l_per_t.shape[0]
            grad_out = torch.eye(n, device=l_per_t.device, dtype=l_per_t.dtype)
            grads = torch.autograd.grad(
                outputs=l_per_t,
                inputs=params,
                grad_outputs=grad_out,
                is_grads_batched=True,
            )
            return torch.cat([g.reshape(n, -1) for g in grads], dim=-1)  # [N, D]

        # full_output gates whether torchpathint populates .t and .y on the
        # returned IntegralOutput. Required when something downstream reads
        # them: _stitch_samples (save_samples=True) or popcornn._optimize's
        # per-iter JSON dump (caller sets self.full_output=True). Off by
        # default to avoid the diagnostic-buffer overhead.
        integral_output = path_integral(
            f,
            t_init_0d,
            t_final_0d,
            method=self.method,
            atol=self.atol,
            rtol=self.rtol,
            norm=self.norm,
            max_batch=self.max_batch,
            max_iter=self.max_iter,
            device=self.device,
            dtype=self.dtype,
        )
        # Persist any OOM-driven shrink across calls. torchpathint returns
        # the value of max_batch that survived the call (= input if no OOM,
        # smaller otherwise); reusing it means the halve fires once per
        # integrator, not once per optimizer step.
        self.max_batch = integral_output.max_batch

        # ``integral_output.integral`` is torchpathint's generic name for the
        # integrated function value — here it's the flat [D] gradient. Alias
        # to ``.grad_integral`` so popcornn-internal call sites read clearly
        # without disambiguating against the loss-integral pass below.
        integral_output.grad_integral = integral_output.integral

        # Scatter the [D] integrated gradient into param.grad. Accumulate so
        # multiple integrate_path calls between optimizer.zero_grad() compose.
        offset = 0
        flat = integral_output.grad_integral.detach()
        for p, k in zip(params, sizes):
            chunk = flat[offset:offset + k].reshape(p.shape)
            p.grad = chunk if p.grad is None else p.grad + chunk
            offset += k

        # No scalar loss graph in this design. Surface ‖∫∇L dt‖_∞ (per-component
        # max) as the convergence signal consumed by PathOptimizer's threshold
        # check. L∞ is closer to MLP-size-independent than L2 — the latter scales
        # with √D and forces the threshold to be retuned per parameter count.
        # Also expose L2 for monitoring (cheap on the same flat tensor).
        if self.norm == '2':
            integral_output.grad_norm = flat.norm()
        elif self.norm == 'max':
            integral_output.grad_norm = flat.abs().max()
        else:
            raise ValueError(f"Unsupported norm {self.norm!r} (choose '2' or 'max').")

        if self.track_ts:
            integral_output.samples = self._stitch_samples(sample_buffer, integral_output.t)
        else:
            integral_output.samples = None

        if self.track_loss:
            def fval(t_flat):
                l, _ = evaluate_integrand_sum(
                    self._terms,
                    t_flat.unsqueeze(-1),
                    path,
                )
                return l.reshape(t_flat.shape[0], -1).detach()
            loss_output = path_integral(
                fval,
                t_init_0d,
                t_final_0d,
                method=self.method,
                atol=self.loss_atol,
                rtol=self.loss_rtol,
                norm=self.norm,
                max_batch=self.max_batch,
                device=self.device,
                dtype=self.dtype,
            )
            integral_output.loss = loss_output.integral.detach()
            # Loss pass may shrink further (it has no autograd graph, but
            # the integrand magnitudes can drive different K's per interval).
            self.max_batch = loss_output.max_batch

        self.integral_output = integral_output
        self.N_integrals += 1
        return integral_output

    def _stitch_samples(self, sample_buffer, accepted_t):
        """Flatten the per-call ``(t, E, dEdt)`` buffer into a single
        ``SamplesCache`` containing **every** captured evaluation —
        including rejected-refinement points that did not make it into
        the accepted-quadrature subset.

        Previously this filtered by byte-key to ``accepted_t``; the
        rejected refinement evals were discarded. Keeping them lets
        downstream consumers (TS search, diagnostic plots) see the full
        sampling pattern the integrator explored.

        ``accepted_t`` is no longer consumed but is kept in the signature
        for call-site compatibility. Output is sorted by ``time`` so
        consumers can scan in order without an extra ``argsort``.
        """
        if not sample_buffer:
            empty = torch.empty(0, device=self.device, dtype=self.dtype)
            return SamplesCache(time=empty, energies=empty, dEdt=empty)
        time = torch.cat([t_chunk for t_chunk, _, _ in sample_buffer], dim=0).to(self.device).flatten()
        energies = torch.cat([e_chunk for _, e_chunk, _ in sample_buffer], dim=0).to(self.device)
        dEdt = torch.cat([dedt_chunk for _, _, dedt_chunk in sample_buffer], dim=0).to(self.device)
        order = torch.argsort(time)
        return SamplesCache(time=time[order], energies=energies[order], dEdt=dEdt[order])
