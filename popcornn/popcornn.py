import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # TODO: move this to the integrator
import json
import time as time
from typing import Any

import torch
from ase import Atoms

from popcornn.paths import get_path
from popcornn.optimization import initialize_path
from popcornn.optimization import PathOptimizer
from popcornn.tools import process_images, output_to_atoms
from popcornn.tools import PathIntegrator
from popcornn.potentials import get_potential


class _StageLogger:
    """Per-stage sparse-table logger + wall-time tracker.

    Streams a header at stage start, sparse per-iter rows, a
    convergence-trigger announcement, and a stage-end summary to
    stdout. When ``log_path`` is provided, also writes one
    JSONL row per iteration with the full scalar metric set —
    flushed each row so a killed run leaves a partial-but-valid
    file.
    """

    def __init__(self, stage_idx, integrand_terms, lr, threshold,
                 n_iter, n_params, log_path=None):
        self.stage_idx = stage_idx
        self.n_iter = n_iter
        terms_str = (
            " + ".join(f"{t.name}×{t.scale:g}" for t in integrand_terms)
            or "<none>"
        )
        thr = "—" if threshold is None else f"{threshold:.1e}"
        print(
            f"── stage {stage_idx} ──  integrand: {terms_str}   "
            f"lr={lr:.1e}   threshold={thr}   "
            f"n_iter={n_iter}   n_params={n_params}"
        )
        header = (
            f"  {'iter':>6s}    {'step_s':>8s}  "
            f"  {'loss':>12s}    {'grad':>12s}  "
            f"  {'barrier':>8s}    {'force':>8s}  "
            f"  {'ts_time':>8s}    {'crossings':>8s}  "
        )
        print(header)
        self._t_start = time.perf_counter()

        self._metrics_fh = None
        if log_path is not None:
            parent = os.path.dirname(log_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._metrics_fh = open(log_path, 'w')

    def row(self, it, loss, grad_norm, step_s, barrier=None, ts_force=None, ts_time=None, n_crossings=None):
        loss_str = "—" if loss is None else f"{loss:12.4e}"
        barrier_str = "—" if barrier is None else f"{barrier:8.4f}"
        force_str = "—" if ts_force is None else f"{float(ts_force.norm().item()):8.4f}"
        ts_time_str = "—" if ts_time is None else f"{ts_time:8.2f}"
        crossings_str = "—" if n_crossings is None else f"{n_crossings:>8d}"
        line = (
            f"  {it:>6d}    {step_s:8.4f}  "
            f"  {loss_str:>12s}    {grad_norm:12.4e}  "
            f"  {barrier_str:>8s}    {force_str:>8s}  "
            f"  {ts_time_str:>8s}    {crossings_str:>8s}  "
        )
        print(line, flush=True)

    def metrics(self, **fields):
        """Append one JSONL row with scalar metrics. No-op when no
        ``log_path`` was configured."""
        if self._metrics_fh is None:
            return
        self._metrics_fh.write(json.dumps(fields) + "\n")
        self._metrics_fh.flush()

    def wall_s(self):
        return time.perf_counter() - self._t_start

    def converged(self, it, g_2, threshold, patience):
        print(
            f"converged at iter {it}  "
            f"(loss_grad_norm={g_2:.4e} < threshold={threshold:.4e} "
            f"for patience={patience})",
            flush=True,
        )

    def end(self, it, barrier, ts_force):
        """Print the stage-end summary line.

        Reports per-stage wall, per-iter wall, the post-convergence
        barrier (``E_TS - E_reactant``), and ``|F| = ‖ts_force‖`` at the
        predicted TS. The caller (``Popcornn._optimize``) runs one final
        ``integrator.integrate_path`` with ``track_loss`` / ``track_ts``
        on right before this call, so ``barrier`` and ``ts_force`` are
        always populated.
        """
        wall = self.wall_s()
        s_iter = wall / (it + 1)
        line = (
            f"stage {self.stage_idx} done  iters={it + 1}  "
            f"time={wall:.1f}s  time/iter={s_iter:.4f}s  "
            f"barrier={float(barrier):.4f}  "
            f"ts_force_norm={float(ts_force.norm().item()):.4f}"
        )
        print(line)

    def close(self):
        if self._metrics_fh is not None:
            self._metrics_fh.close()
            self._metrics_fh = None


class Popcornn:
    """
    High-level driver for popcornn reaction-path optimization.

    Wraps the path representation, image processing, and
    multi-stage optimization loop. The typical lifecycle is

    1. Construct with the reactant/product/intermediate images and a
       ``path_params`` dict that picks the path representation.
    2. Call ``optimize_path`` with one or more stage-config dicts.
    3. Receive the optimized path frames and (when TS extraction is
       active) a single predicted transition-state frame.
    """
    def __init__(
            self, 
            images: list[Atoms],
            unwrap_positions: bool = True,
            path_params: dict[str, Any] = {},
            track_loss: bool = False,
            track_ts: bool = False,
            num_record_points: int = 101,
            device: str | None = None,
            dtype: str = "float32",
            seed: int | None = 0,
            log_path: str | None = None,
    ):
        """
        Initialize the Popcornn class.

        Args:
            images (list[Atoms]): List of ASE Atoms objects representing the images.
            unwrap_positions (bool): Whether to unwrap the positions of the images. Default is True.
            path_params (dict[str, Any]): Parameters for the path prediction method.
            track_loss (bool): Whether to track the loss during optimization, which requires additional forward passes and may slow down optimization. Default is False.
            track_ts (bool): Whether to track the transition state during optimization, which requires additional forward passes and may slow down optimization. Default is False.
            num_record_points (int): Number of points to record along the optimized path when calling get_discrete_path. Default is 101.
            device (str | None): Device to use for optimization. If None, will use 'cuda' if available, otherwise 'cpu'.
            dtype (str): Data type to use for optimization. Can be 'float32' or 'float64'.
            seed (int | None): Random seed for reproducibility. If None, no seed is set.
        """
        # Set device
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        self.device = device

        # Set dtype
        if dtype == "float32":
            self.dtype = torch.float32
        elif dtype == "float64":
            self.dtype = torch.float64
        else:
            raise ValueError(f"Invalid dtype: {dtype}. Use 'float32' or 'float64'.")

        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)

        # Process images
        self.images = process_images(images, unwrap_positions=unwrap_positions, device=self.device, dtype=self.dtype)

        # Get path prediction method
        self.path = get_path(images=self.images, **path_params, device=self.device, dtype=self.dtype)

        # Fit the path to the intermediate images, otherwise a straight line
        if len(self.images) > 2:  #TODO: make this spline or bezier instead of fitting
            self.path = initialize_path(
                path=self.path, 
                times=torch.linspace(self.path.t_init.item(), self.path.t_final.item(), len(self.images), device=self.device, dtype=self.dtype),
                init_points=self.images.positions,
            )

        # Track loss and TS during optimization when requested. This adds overhead from extra forward passes, so only enable when needed for debugging or analysis.
        self.track_loss = track_loss
        self.track_ts = track_ts

        # Number of points to record along the optimized path
        self.num_record_points = num_record_points

        self.log_path = log_path


    def optimize_path(
            self,
            *optimization_params: list[dict],
    ):
        """
        Run a chain of optimization stages on ``self.path`` in place.

        Each entry of ``optimization_params`` is one stage. The path's
        trainable parameters persist across stages, so a typical pattern
        is a cheap ``repel`` + ``geodesic`` clash-resolution stage
        followed by an MLIP-driven stage targeting the transition state.
        After all stages finish, retrieve the path via
        ``get_discrete_path()`` and the TS via ``get_ts()``.

        Parameters
        ----------
        *optimization_params : dict
            One dict per stage. Recognized keys:

            ``potential_params``
                Forwarded to ``get_potential``.
            ``integrator_params``
                Forwarded to ``PathIntegrator``.
            ``optimizer_params``
                Forwarded to ``PathOptimizer``.
            ``num_optimizer_iterations``
                Hard cap on Adam steps for this stage.
        metrics_log_path : str, optional
            File path for a per-iteration JSONL scalar-metrics log.
            One row per iteration; flushed each row so a killed run
            leaves a partial-but-valid file.
        """
        # Optimize the path. When output_dir is set but the caller didn't
        # specify a metrics path, default to <output_dir>/metrics so the
        # lightweight scalar log lands next to the heavy per-iter JSONs.
        # if metrics_log_path is None and self.output_dir is not None:
        #     metrics_log_path = os.path.join(self.output_dir, 'metrics')
        # if metrics_log_path is not None:
        #     os.makedirs(metrics_log_path, exist_ok=True)
        for i, params in enumerate(optimization_params):
            # if self.output_dir is not None:
            #     output_dir = f"{self.output_dir}/opt_{i}"
            # else:
            #     output_dir = None
            # if metrics_log_path is not None:
            #     stage_metrics_path = os.path.join(metrics_log_path, f"opt_{i}.jsonl")
            # else:
            #     stage_metrics_path = None

            self._optimize(
                **params,
                # output_dir=output_dir,
                # output_ase_atoms=output_ase_atoms,
                stage_idx=i,
            )


    def get_discrete_path(self, num_record_points=None, output_ase_atoms=True):
        """
        Get a discrete path of points along the optimized path.

        ``num_record_points`` defaults to ``self.num_record_points``
        (set by the constructor).
        """
        if num_record_points is None:
            num_record_points = self.num_record_points
        # Evaluate points along the optimized path and return
        time = torch.linspace(self.path.t_init.item(), self.path.t_final.item(), num_record_points, device=self.device, dtype=self.dtype)
        path_output = self.path(time)
        if issubclass(self.images.image_type, Atoms) and output_ase_atoms:
            images = output_to_atoms(path_output, self.images)
            return images
        else:
            images = path_output.positions.tolist()
            return images

    def get_ts(self, output_ase_atoms=True):
        """
        Get the predicted transition state frame.
        """
        # Get the last integrator
        time = torch.tensor([self.path.ts_time], device=self.device, dtype=self.dtype)
        path_output = self.path(time, return_velocities=True, return_energies=True, return_forces=True)
        if issubclass(self.images.image_type, Atoms) and output_ase_atoms:
            image = output_to_atoms(path_output, self.images)[0]
            return image
        else:
            image = {
                "ts_time": time.tolist(),
                "ts_positions": path_output.positions.tolist(),
                "ts_energies": path_output.energies.tolist(),
                "ts_velocities": path_output.velocities.tolist(),
                "ts_forces": path_output.forces.tolist(),
            }
            return image

    def _optimize(
            self,
            potential_params: dict[str, Any] = {},
            integrator_params: dict[str, Any] = {},
            optimizer_params: dict[str, Any] = {},
            num_optimizer_iterations: int = 1000,
            # output_dir: str | None = None,
            # output_ase_atoms: bool = True,
            stage_idx: int = 0,
    ):
        """
        Run a single optimization stage.

        Builds the potential, integrator, and optimizer for this stage,
        then steps Adam until either the convergence trigger fires or
        ``num_optimizer_iterations`` is reached. After the loop, runs
        one diagnostic ``integrate_path`` with ``track_loss`` /
        ``track_ts`` on so the stage-end log line carries barrier and
        ``|F|`` at the predicted TS.

        Parameters
        ----------
        potential_params : dict
            Forwarded to ``get_potential``. ``name`` is required.
        integrator_params : dict
            Forwarded to ``PathIntegrator``.
        optimizer_params : dict
            Forwarded to ``PathOptimizer``. ``threshold`` controls the
            convergence trigger.
        num_optimizer_iterations : int, default=1000
            Iteration cap.
        stage_idx : int, default=0
            Index of this stage in the parent ``optimize_path`` chain;
            used only for the stdout progress header.
        metrics_log_path : str, optional
            If set, write one JSONL row per iteration to this exact
            file path with scalar metrics (iter, loss, grad_norm, lr,
            step_s, wall_s, converged, barrier, ts_force_norm).
            Flushed after each row so a killed run leaves a partial-
            but-valid file.
        """
        # Create output directories
        # if output_dir is not None:
        #     os.makedirs(output_dir, exist_ok=True)

        # Get potential energy function
        potential = get_potential(images=self.images, **potential_params, device=self.device, dtype=self.dtype)
        self.path.set_potential(potential)

        # Path optimization tools
        integrator = PathIntegrator(**integrator_params, track_loss=self.track_loss, track_ts=self.track_ts, device=self.device, dtype=self.dtype)

        # Gradient descent path optimizer
        optimizer = PathOptimizer(path=self.path, **optimizer_params, track_ts=self.track_ts, device=self.device, dtype=self.dtype)

        # The per-iter big JSON dump (gated on output_dir) reads the quadrature
        # nodes/values off the returned result (padaquad: .nodes / .y).
        # if output_dir is not None:
        #     integrator.full_output = True

        # Create output directories
        # if output_dir is not None:
        #     os.makedirs(output_dir, exist_ok=True)
        #     log_dir = os.path.join(output_dir, "logs")
        #     os.makedirs(log_dir, exist_ok=True)
        
        # Per-stage progress logger: header now, sparse rows during the loop,
        # convergence/stage-end summary on exit. Optional JSONL written
        # via the same logger when metrics_log_path is set.
        logger = _StageLogger(
            stage_idx=stage_idx,
            integrand_terms=integrator._terms,
            lr=optimizer.optimizer.param_groups[0]['lr'],
            threshold=optimizer.threshold,
            n_iter=num_optimizer_iterations,
            n_params=sum(p.numel() for p in self.path.parameters() if p.requires_grad),
            log_path=f"{self.log_path}/stage_{stage_idx}_metrics.jsonl" if self.log_path is not None else None,
        )

        # Safe defaults for the degenerate num_optimizer_iterations=0 case
        # — the loop body never runs, but the post-loop diagnostic block
        # still references optim_idx and integral_output.
        optim_idx = -1
        integral_output = None

        # Optimize the path
        for optim_idx in range(num_optimizer_iterations):
            lr = optimizer.optimizer.param_groups[0]['lr']
            t_step = time.perf_counter()
            try:
                integral_output = optimizer.optimization_step(self.path, integrator)
            except ValueError as e:
                print("ValueError", e)
                raise e
            step_s = time.perf_counter() - t_step

            loss_attr = getattr(integral_output, 'loss', None)
            loss_v = float(loss_attr[0].item()) if loss_attr is not None else None
            grad_norm = integral_output.grad_norm.item()

            # Per-iter TS diagnostics (only when track_ts triggered path.ts_search,
            # which sets path.barrier alongside path.ts_force).
            barrier = None
            ts_force = None
            ts_time = None
            n_crossings = None
            if self.track_ts:
                b = getattr(self.path, 'barrier', None)
                barrier = b.item() if b is not None else None
                ts_force = getattr(self.path, 'ts_force', None)
                ts_time = getattr(self.path, 'ts_time', None)
                n_crossings = getattr(self.path, 'n_crossings', None)

            logger.row(optim_idx, loss_v, grad_norm, step_s, barrier=barrier, ts_force=ts_force, ts_time=ts_time, n_crossings=n_crossings)

            logger.metrics(
                iter=optim_idx,
                loss=loss_v,
                grad_norm=grad_norm,
                lr=lr,
                step_s=step_s,
                wall_s=logger.wall_s(),
                converged=bool(optimizer.converged),
                barrier=barrier,
                ts_force_norm=(float(ts_force.norm().item()) if ts_force is not None else None),
                n_crossings=n_crossings,
            )

            # Save the path
            # if output_dir is not None:
            #     t_grid = integral_output.t.flatten()
            #     path_output = self.path(t_grid, return_velocities=True, return_energies=True, return_forces=True)
            #     if self.path.ts_time is not None:
            #         ts_time = torch.tensor([self.path.ts_time], device=self.device, dtype=self.dtype)
            #         ts_output = self.path(ts_time, return_velocities=True, return_energies=True, return_forces=True)
            #         ts_record = {
            #             "ts_time": ts_time.tolist(),
            #             "ts_positions": ts_output.positions.tolist(),
            #             "ts_energies": ts_output.energies.tolist(),
            #             "ts_velocities": ts_output.velocities.tolist(),
            #             "ts_forces": ts_output.forces.tolist(),
            #         }
            #     else:
            #         ts_record = {
            #             "ts_time": None,
            #             "ts_positions": None,
            #             "ts_energies": None,
            #             "ts_velocities": None,
            #             "ts_forces": None,
            #         }

            #     record = {
            #         "time": t_grid.tolist(),
            #         "positions": path_output.positions.tolist(),
            #         "energies": path_output.energies.tolist(),
            #         "velocities": path_output.velocities.tolist(),
            #         "forces": path_output.forces.tolist(),
            #         "loss_evals": integral_output.y.tolist(),
            #         "grad_norm": integral_output.grad_norm.item(),
            #         "grad_norm_2": integral_output.grad_norm_2.item(),
            #         **ts_record,
            #     }
            #     loss = getattr(integral_output, 'loss', None)
            #     if loss is not None:
            #         record["loss"] = loss.tolist()
            #     with open(os.path.join(log_dir, f"output_{optim_idx}.json"), 'w') as file:
            #         json.dump(record, file)

            # Check for convergence
            if optimizer.converged:
                logger.converged(
                    optim_idx,
                    integral_output.grad_norm.item(),
                    optimizer.threshold,
                    optimizer.patience,
                )
                break

        # One diagnostic integration with track_loss + track_ts on
        # so the stage-end line carries loss, barrier, and |F| at the TS.
        # Cheap: one extra integrate call per stage, no opt step.
        integrator.track_loss = True
        integrator.track_ts = True
        integral_output = integrator.integrate_path(self.path)
        self.path.ts_search(integral_output.samples)
        final_barrier = self.path.barrier.item()
        final_ts_force = self.path.ts_force

        logger.end(
            optim_idx,
            barrier=final_barrier,
            ts_force=final_ts_force,
        )
        logger.close()

