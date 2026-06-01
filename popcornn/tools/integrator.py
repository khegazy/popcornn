from dataclasses import dataclass

import torch

from padaquad import adaptive_quadrature

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

    Wraps a ``padaquad`` adaptive-quadrature solver with the parts of the
    popcornn API the optimizer uses:

    - the solver is built **once** in ``__init__`` and reused across every
      ``integrate_path`` call (padaquad is stateful, unlike a free-function
      integrator), so warm-start and memory benchmarking persist,
    - **warm-start**: the optimized (pruned + refined) ``mesh_optimal`` from
      the previous integration is fed back as the next call's initial mesh
      via the ``mesh=`` argument, which skips most adaptive-refinement cost
      when the integrand changes only slightly between optimizer steps,
    - per-iteration ``integrand_scales`` updates so schedulers take effect,
    - direct scatter of ``∂L/∂θ`` into ``path.parameters().grad``
      (no separate ``.backward()`` call — we integrate the gradient itself
      with ``take_gradient=False`` and write the result into ``.grad``),
    - a ``grad_integral`` alias on the returned object for the flat
      ``[D]`` integrated gradient, and a ``grad_norm`` field holding
      ``‖∫∇L dt‖`` (L2 or L∞ per ``norm``) used by the convergence check,
    - an optional detached pass that integrates the loss itself (on a
      second solver with its own tolerances), exposed as a ``loss`` field
      for monitoring,
    - an optional ``track_ts`` mode where the integrand emits
      per-quadrature-point ``(energies, dE/dt)`` as padaquad
      ``tracked_variables``, reassembled into a ``SamplesCache`` for
      transition-state finding without any extra path evaluations.
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
            padaquad quadrature rule. ``gk7`` is the adaptive
            Gauss–Kronrod 7-point rule; chosen as the popcornn default
            after the 2026-05-12 integrator sweep (gg3 NN pseudo-Huber)
            showed it's 30% faster than ``gk21`` at identical TS quality.
            Use ``gk21`` / ``gk31`` for tighter integration if the path
            is unusually rough; padaquad also offers ``gk15`` and
            Clenshaw–Curtis (``cc*``) / Runge–Kutta rules.
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
            Vector norm used to reduce the flat ``[D]`` gradient integral
            to the scalar ``grad_norm`` convergence signal. ``"2"`` / ``"max"``
            select L2 / L∞. (padaquad's *internal* adaptive-error norm is a
            fixed RMS across the integrand's output dimensions and is not
            configurable here.)
        max_batch : int, optional
            Hard cap on the number of quadrature points evaluated in
            one batch. ``None`` lets padaquad size the batch from a
            one-time memory benchmark. To keep that benchmark from
            re-running every iteration (padaquad re-benchmarks whenever
            ``id(f)`` changes, and our integrand closure is rebuilt each
            call), the value is snapshotted after the first
            ``integrate_path`` via the solver's memory estimate and reused
            thereafter — so the benchmark fires once per integrator
            lifetime, not once per optimizer step.

            Caveat: the benchmark measures the integrand's *output*
            footprint, not its autograd-graph *peak* (the batched-Jacobian
            ``autograd.grad`` allocates more transiently than it returns),
            and unlike the old backend padaquad does not auto-shrink on a
            CUDA OOM — it raises. For memory-tight GPU runs set
            ``max_batch`` explicitly.

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
        max_iter : int, default=5
            Maximum adaptive-refinement depth, passed to padaquad as
            ``max_adaptive_splits``: the most times any single quadrature panel
            may be split, after which it is accepted even if it still fails the
            error tolerance. (padaquad has no global iteration count; this is a
            per-panel recursion-depth cap, so the same integer bounds local
            refinement at ~2^max_iter panels rather than total passes.) This
            bounds padaquad's otherwise-uncapped refinement: a *finite*
            integrand whose error never falls below ``atol + rtol·|integral|``
            (e.g. a near-zero integral against ``atol=0``) would otherwise
            split every panel forever, burning unbounded compute. The default
            of 5 leaves well-behaved integrals untouched (production runs
            converge well before depth 5; verified identical grad_norm at
            5/10/20 on Müller-Brown). Note this does NOT rescue an integrand
            that returns NaN/Inf (a NaN error ratio is neither accepted nor
            split); such degenerate inputs must be avoided upstream.
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

        # padaquad wants a device *string* ('cuda'/'cpu'), not a torch.device,
        # and rejects dtype=None in _set_dtype — coalesce to float64 (the per-call
        # set_dtype_by_input then matches the actual mesh-bound dtype anyway).
        solver_device = self.device.type if isinstance(self.device, torch.device) else self.device
        solver_dtype = self.dtype if self.dtype is not None else torch.float64

        # Build the adaptive-quadrature solver ONCE; reused across every
        # integrate_path call so warm-start (mesh_optimal) and the memory
        # benchmark persist for the integrator's lifetime.
        self._solver = adaptive_quadrature(
            sampling_type='uniform',          # str -> steps.ADAPTIVE_UNIFORM
            method=self.method,
            atol=self.atol,
            rtol=self.rtol,
            max_batch=self.max_batch,
            max_adaptive_splits=self.max_iter,  # per-panel split-depth cap; bounds refinement
            error_calc_idx=None,              # RMS error over all gradient components
            dtype=solver_dtype,
            device=solver_device,
        )
        # Warm-start cache: previous run's optimized mesh, fed back as the
        # next call's initial mesh. None on the first call -> fresh random mesh.
        self._mesh_optimal = None

        # Detached loss integral runs on its own solver so it can use its own
        # (looser) tolerances and keep an independent warm-start mesh.
        if self.track_loss:
            self._loss_solver = adaptive_quadrature(
                sampling_type='uniform',
                method=self.method,
                atol=self.loss_atol,
                rtol=self.loss_rtol,
                max_batch=self.max_batch,
                max_adaptive_splits=self.max_iter,
                error_calc_idx=None,
                dtype=solver_dtype,
                device=solver_device,
            )
            self._loss_mesh_optimal = None

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
          gradient (same tensor as padaquad's ``.integral``; named for
          clarity at popcornn call sites).
        - ``.grad_norm``: ``‖∫∇L dt‖`` (L2 or L∞ per ``norm``); the
          convergence trigger read by ``PathOptimizer``.
        - ``.loss``: scalar ``∫L(t) dt`` when ``track_loss=True``, else
          ``None``.
        - ``.samples``: ``SamplesCache`` (or ``None``) holding
          per-quadrature-point ``(t, E, dE/dt)`` when ``track_ts=True``.

        Parameters
        ----------
        path : BasePath
            Holds the trainable parameters and potential.
        integrand_scales : dict, optional
            Updated values for per-term scales (from schedulers).
        t_init, t_final : torch.Tensor
            Integration bounds, in [0, 1]. Passed to padaquad as 1-D
            ``[1]`` mesh bounds.
        """
        if integrand_scales:
            self.update_integrand_scales(**integrand_scales)

        # padaquad wants 1-D ``[T]`` mesh bounds (T=1 here), not 0-d. Their
        # dtype also drives the solver's working precision (set_dtype_by_input).
        t_init_1d = torch.as_tensor(t_init, dtype=self.dtype, device=self.device).reshape(1)
        t_final_1d = torch.as_tensor(t_final, dtype=self.dtype, device=self.device).reshape(1)

        # Filter out frozen params (requires_grad=False). When a potential
        # is attached via set_potential, it's registered as an nn.Module
        # submodule of the path, so its params appear in path.parameters().
        # Potentials that freeze their weights (e.g. NewtonNet's
        # `model.requires_grad_(False)`) would otherwise trip
        # autograd.grad's "differentiated Tensors does not require grad".
        params = [p for p in path.parameters() if p.requires_grad]
        sizes = [p.numel() for p in params]

        also_resolve = ('energies', 'forces', 'velocities') if self.track_ts else ()

        # Gradient-of-loss integrand: returns the per-point batched Jacobian
        # ``∂L/∂θ`` as [N, D]. padaquad integrates it (take_gradient=False, no
        # .backward()) so ``result.integral`` is the flat [D] gradient we
        # scatter into param.grad below. When track_ts is on, the integrand
        # also emits ``(energies, dE/dt)`` as padaquad tracked_variables —
        # evaluated at every node, returned (detached) at the accepted nodes,
        # and reassembled into a SamplesCache. dE/dt = -(F·v).sum(-1) reuses
        # the forces/velocities already resolved for the loss integrand.
        # padaquad passes nodes as [N, 1] (already 2-D) — no unsqueeze.
        def f(t):
            l, variables = evaluate_integrand_sum(
                self._terms,
                t,
                path,
                also_resolve=also_resolve,
            )
            l_per_t = l.reshape(t.shape[0], -1).sum(dim=-1)  # [N], graph live
            n = l_per_t.shape[0]
            grad_out = torch.eye(n, device=l_per_t.device, dtype=l_per_t.dtype)
            grads = torch.autograd.grad(
                outputs=l_per_t,
                inputs=params,
                grad_outputs=grad_out,
                is_grads_batched=True,
            )
            jac = torch.cat([g.reshape(n, -1) for g in grads], dim=-1)  # [N, D]
            if self.track_ts:
                dEdt = -(variables['forces'] * variables['velocities']).sum(dim=-1)  # [N]
                return jac, (variables['energies'], dEdt)
            return jac, None

        # Whether padaquad must benchmark the integrand's memory this call.
        # The closure ``f`` is rebuilt every call, so padaquad's id(f)-keyed
        # cache never hits; passing an explicit ``max_batch`` suppresses the
        # benchmark. On the first call (max_batch is None) we let it run, then
        # snapshot the sizing so subsequent calls skip it — benchmarking once
        # per integrator lifetime rather than once per optimizer step.
        snapshot_max_batch = self.max_batch is None

        result = self._solver.integrate(
            f,
            mesh=self._mesh_optimal,          # None on first call -> fresh random mesh
            mesh_init=t_init_1d,
            mesh_final=t_final_1d,
            take_gradient=False,              # we scatter ∂L/∂θ ourselves; no .backward()
            max_batch=self.max_batch,
        )
        # Warm-start the next call from this run's pruned + refined mesh.
        self._mesh_optimal = result.mesh_optimal
        if snapshot_max_batch:
            self.max_batch = self._solver._get_max_f_evals(self._solver.total_mem_usage)

        integral_output = result
        # padaquad's ``.integral`` is the integrated function value — here the
        # flat [D] gradient. Alias to ``.grad_integral`` so popcornn-internal
        # call sites read clearly against the loss-integral pass below.
        integral_output.grad_integral = result.integral

        # Scatter the [D] integrated gradient into param.grad. Accumulate so
        # multiple integrate_path calls between optimizer.zero_grad() compose.
        offset = 0
        flat = integral_output.grad_integral.detach()
        for p, k in zip(params, sizes):
            chunk = flat[offset:offset + k].reshape(p.shape)
            p.grad = chunk if p.grad is None else p.grad + chunk
            offset += k

        # Surface ‖∫∇L dt‖ as the convergence signal consumed by
        # PathOptimizer's threshold check. L∞ is closer to MLP-size-independent
        # than L2 — the latter scales with √D and forces the threshold to be
        # retuned per parameter count.
        if self.norm == '2':
            integral_output.grad_norm = flat.norm()
        elif self.norm == 'max':
            integral_output.grad_norm = flat.abs().max()
        else:
            raise ValueError(f"Unsupported norm {self.norm!r} (choose '2' or 'max').")

        if self.track_ts:
            integral_output.samples = self._samples_from_result(result)
        else:
            integral_output.samples = None

        # padaquad pre-fills ``result.loss`` from its default loss_fxn (= the
        # integral, i.e. the [D] gradient here), which is meaningless to
        # popcornn. Overwrite: None unless track_loss runs a real loss pass.
        if self.track_loss:
            def fval(t):
                l, _ = evaluate_integrand_sum(self._terms, t, path)
                return l.reshape(t.shape[0], -1).detach()  # [N, 1], bare tensor
            loss_result = self._loss_solver.integrate(
                fval,
                mesh=self._loss_mesh_optimal,
                mesh_init=t_init_1d,
                mesh_final=t_final_1d,
                take_gradient=False,
                max_batch=self.max_batch,     # int by now (snapshotted on the grad pass)
            )
            self._loss_mesh_optimal = loss_result.mesh_optimal
            integral_output.loss = loss_result.integral.detach()  # [1]
        else:
            integral_output.loss = None

        self.integral_output = integral_output
        self.N_integrals += 1
        return integral_output

    def _samples_from_result(self, result):
        """Build a ``SamplesCache`` from padaquad's accepted-node output.

        ``result.nodes`` (``[N, C, T]``, T=1) gives the per-quadrature-point
        times; ``result.tracked_variables`` carries the ``(energies, dE/dt)``
        the integrand emitted, returned detached at the accepted nodes and
        aligned with ``nodes``. Flatten the ``[N, C, ...]`` layout to a flat
        sample list and sort by time so consumers (TS search) can scan in
        order without an extra ``argsort``.

        Note: unlike the old byte-keyed stitch, these are the *accepted* mesh
        nodes only (rejected-refinement evals are not returned by padaquad).
        """
        if result.tracked_variables is None or result.nodes.shape[0] == 0:
            empty = torch.empty(0, device=self.device, dtype=self.dtype)
            return SamplesCache(time=empty, energies=empty, dEdt=empty)
        energies, dEdt = result.tracked_variables          # [N, C, E], [N, C]
        time = result.nodes[..., 0].reshape(-1).to(self.device)              # [N*C]
        energies = energies.reshape(-1, energies.shape[-1]).to(self.device)  # [N*C, E]
        dEdt = dEdt.reshape(-1).to(self.device)                              # [N*C]
        order = torch.argsort(time)
        return SamplesCache(time=time[order], energies=energies[order], dEdt=dEdt[order])
