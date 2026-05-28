import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.optim import lr_scheduler
from popcornn.tools import scheduler
from popcornn.tools.scheduler import get_schedulers

from popcornn.tools.integrand import build_integrand_terms, evaluate_integrand_sum


def _update_term_scales(terms, scales: dict) -> None:
    """In-place: set each term's scale from `scales[term.name]` if present."""
    for name, scale in scales.items():
        for term in terms:
            if term.name == name:
                term.scale = float(scale)
                break


class PathOptimizer():
    """
    Single-stage path optimizer.

    Owns the underlying ``torch.optim`` optimizer, the (optional) LR
    scheduler, all per-loss-term schedulers, the convergence-trigger
    state, and the optional transition-state loss machinery. One
    instance per ``Popcornn._optimize`` stage.
    """

    def __init__(
            self,
            path,
            optimizer=None,
            track_ts=False,
            lr_scheduler=None,
            # path_integrand_schedulers=None,
            # ts_time_loss_names=None,
            # ts_time_loss_scales=torch.ones(1),
            # ts_time_loss_schedulers=None,
            threshold=None,
            patience=1,
            device='cpu',
            dtype=None,
            **config
        ):
        """
        Parameters
        ----------
        path : BasePath
            The path whose parameters this optimizer steps.
        optimizer : dict
            ``{"name": <torch.optim class>, ...kwargs}``. Required.
        track_ts : bool, optional
            Whether to track the transition state during optimization.
        lr_scheduler : dict, optional
            ``{"name": <torch.optim.lr_scheduler class>, ...kwargs}``.
        path_integrand_schedulers : dict, optional
            Schedules for per-integrand-term scales.
        ts_time_loss_names, ts_time_loss_scales, ts_time_loss_schedulers : optional
            Loss applied at the predicted TS time.
        threshold : float, optional
            Convergence trigger on ``‖∫∇L dt‖_2``. ``None`` disables.
        patience : int, default=5
            Number of consecutive iterations the trigger must hold.
        device : str
        dtype : torch.dtype
        **config
            Reserved for forward compatibility.
        """
        super().__init__()

        self.track_ts = track_ts
        self.device=device
        self.dtype=dtype
        self.iteration = 0
        self.threshold = threshold
        self.patience = patience
        self._below_threshold_count = 0

        ####  Initialize transition state loss information  #####
        # self.has_ts_time_loss = ts_time_loss_names is not None
        # self.has_ts_loss = self.has_ts_time_loss
        # if self.has_ts_loss:
        #     if self.track_ts is None or self.track_ts:
        #         self.track_ts = True
        #     else:
        #         raise ValueError("Cannot have transition state losses and set track_ts=False")
        # elif self.track_ts is None:
        #     # No explicit override and no TS-loss to force the issue —
        #     # let the path decide. BasePath's own default is True.
        #     self.find_ts = path.find_ts

        # self.ts_time_loss_names = ts_time_loss_names
        # self.ts_time_loss_scales = ts_time_loss_scales
        # if self.has_ts_time_loss:
        #     self.ts_time_terms = build_integrand_terms(
        #         self.ts_time_loss_names, self.ts_time_loss_scales
        #     )

        #####  Initialize schedulers  #####
        # self.integrand_schedulers = get_schedulers(path_integrand_schedulers)
        # self.ts_time_loss_schedulers = get_schedulers(ts_time_loss_schedulers)
        
        #####  Initialize optimizer  #####
        self.path = path
        if optimizer is not None:
            self.set_optimizer(**optimizer)
        else:
            raise ValueError("Must specify optimizer parameters (dict) with key 'optimizer'")

        #####  Initialize learning rate scheduler  #####
        if lr_scheduler is not None:
            self.set_lr_scheduler(**lr_scheduler)
        else:
            self.lr_scheduler = None
        self.converged = False

    def set_optimizer(self, name, **config):
        """
        Build and attach a ``torch.optim`` optimizer by class name.

        ``name`` matches against ``dir(torch.optim)`` case-insensitively
        — i.e. ``"adam"`` finds ``torch.optim.Adam``, ``"lbfgs"`` finds
        ``torch.optim.LBFGS``. Extra kwargs forward to the constructor.
        """
        optimizer_dict = {key.lower(): key for key in dir(optim) if not key.startswith('_')}
        name = optimizer_dict[name.lower()]
        optimizer_class = getattr(optim, name)
        self.optimizer = optimizer_class(self.path.parameters(), **config)

    def set_lr_scheduler(self, name, **config):
        """
        Attach a ``torch.optim.lr_scheduler`` by class name. Same
        case-insensitive name resolution as ``set_optimizer``.
        """
        scheduler_dict = {key.lower(): key for key in dir(lr_scheduler) if not key.startswith('_')}
        assert name.lower() in scheduler_dict,\
            f"Scheduler class does not support '{name}', either add this functionality or select from {list(scheduler_dict.keys())}"
        name = scheduler_dict[name.lower()]
        scheduler_class = getattr(lr_scheduler, name)
        self.lr_scheduler = scheduler_class(self.optimizer, **config)

    
    def optimization_step(
            self,
            path,
            integrator,
            t_init=torch.tensor([0.]),
            t_final=torch.tensor([1.]),
            update_path=True
        ):
        """
        Run one optimizer step.

        Steps:

        1. Read the current scheduled values for all per-loss scales.
        2. Hand the path + loss to ``integrator.integrate_path``,
           which scatters ``∂L/∂θ`` into ``path.parameters().grad``.
        3. Run ``ts_search`` on the integrator's sample cache, then add
           gradients from any TS-time loss.
        4. Step Adam, all schedulers, and the LR scheduler.
        5. Update the convergence-trigger counter.

        Returns
        -------
        IntegralOutput
            The integrator output, with ``.grad_norm_2`` set to the
            ``‖∫∇L dt‖_2`` value used for convergence checks (switched
            from ``.grad_norm`` / ``‖·‖_∞`` on 2026-05-15 — see memory
            ``project_popcornn_g2_trigger_switch``). ``.grad_norm``
            still holds the ``‖·‖_∞`` for monitoring. See
            ``PathIntegrator.integrate_path`` for the full set of
            popcornn-level fields attached.
        """
        self.optimizer.zero_grad()
        t_init = t_init.to(self.dtype).to(self.device)
        t_final = t_final.to(self.dtype).to(self.device)
        # integrand_scales = {
        #     name : schd.get_value() for name, schd in self.integrand_schedulers.items()
        # }

        # if self.has_ts_loss:
        #     ts_time_loss_scales = {
        #         name : schd.get_value() for name, schd in self.ts_time_loss_schedulers.items()
        #     }
        integral_output = integrator.integrate_path(
            path,
            # integrand_scales=integrand_scales,
            t_init=t_init,
            t_final=t_final,
        )
        # integrate_path scatters dL/dθ into path.parameters().grad directly;
        # no .backward() call here.

        #####  Transition State  #####
        # ts_search reads the per-quadrature-point (t, E, dE/dt) cache the
        # integrator collected during the gradient pass, brackets the
        # interior sign change of dE/dt, linearly interpolates t_TS, and
        # does one fresh path.forward at t_TS to get model-truth E/F.
        if self.track_ts and integral_output.samples is not None:
            path.ts_search(integral_output.samples)

        # Evaluate transition state losses
        # if self.find_ts and path.ts_time is not None:
        #     if self.has_ts_time_loss:
        #         _update_term_scales(self.ts_time_terms, ts_time_loss_scales)
        #         ts_time_loss, _ = evaluate_integrand_sum(
        #             self.ts_time_terms,
        #             torch.tensor([[path.ts_time]]),
        #             path,
        #         )
        #         ts_time_loss[:, 0].backward()

        # Convergence: ‖∫∇L dt‖_2 below threshold for `patience` consecutive
        # iterations. Patience guards against single-step dips driven by
        # adaptive-quadrature error wiggling around the threshold.
        if self.threshold is not None:
            if integral_output.grad_norm.item() < self.threshold:
                self._below_threshold_count += 1
                if self._below_threshold_count >= self.patience:
                    self.converged = True
            else:
                self._below_threshold_count = 0

        #####  Update Optimization  #####
        # Path update step
        if not self.converged and update_path:
            self.optimizer.step()
        # Update schedulers
        # for name, sched in self.integrand_schedulers.items():
        #     sched.step()
        # if self.has_ts_loss:
        #     for name, sched in self.ts_time_loss_schedulers.items():
        #         sched.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.iteration = self.iteration + 1

        return integral_output

    