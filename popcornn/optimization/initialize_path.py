from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# ``initialize_path`` stops once ``loss_init`` (mean-squared displacement over all
# image coordinates) is below this. Tests import it rather than copying the value.
INIT_LOSS_THRESHOLD = 1e-8


def randomly_initialize_path(
    path: torch.tensor,
    n_points: int,
    order_points: bool = False,
    seed: int | None = None,
) -> torch.Tensor | np.ndarray:
    """
    Initialize the path through randomly drawn intermediate points.

    Samples ``n_points`` random points uniformly in the bounding box
    spanned by reactant and product (per-dimension), optionally
    sorted along the first axis (or along every axis when
    ``order_points`` is True), then fits the path to pass through
    them.

    Parameters
    ----------
    path : BasePath
        The path to fit.
    n_points : int
        Number of intermediate points to sample.
    order_points : bool, default=False
        If True, sort the random points along every dimension so the
        sampled chain is monotonic in each coordinate.
    seed : int, default=1910
        RNG seed.

    Returns
    -------
    BasePath
        The fitted path.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # times = rnd.uniform(shape=(n_points, 1), minval=0.1, maxval=0.9)
    times = torch.unsqueeze(
        torch.linspace(0, 1, n_points + 2, device=path.device)[1:-1], -1
    )
    times.requires_grad = False

    n_dims = len(path.initial_position)
    rnd_dims = []
    for idx in range(n_dims):
        min_val = torch.min(
            torch.tensor(
                [path.initial_position[idx], path.final_position[idx]],
                device=path.device,
            )
        ).item()
        max_val = torch.max(
            torch.tensor(
                [path.initial_position[idx], path.final_position[idx]],
                device=path.device,
            )
        ).item()
        # rnd_vals = rnd.uniform(size=(n_points, 1), low=min_val, high=max_val)
        rnd_vals = (
            torch.rand(n_points, 1, device=path.device) * (max_val - min_val) + min_val
        )
        if order_points or idx == 0:
            # if path.initial_position[idx] > path.final_position[idx]:
            #     rnd_dims.append(-1*np.sort(-1*rnd_vals, axis=0))
            # else:
            #     rnd_dims.append(np.sort(rnd_vals, axis=0))
            descending = path.initial_position[idx] > path.final_position[idx]
            rnd_dims.append(
                torch.sort(rnd_vals, axis=0, descending=descending.item()).values
            )
        else:
            rnd_dims.append(rnd_vals)
    # rnd_dims = torch.tensor(
    #     np.concatenate(rnd_dims, axis=-1), requires_grad=False, device=path.device
    # )
    rnd_dims = torch.cat(rnd_dims, dim=-1)

    return initialize_path(path, times, rnd_dims)


def loss_init(
    path: torch.tensor, times: torch.tensor, positions: torch.tensor
) -> torch.Tensor:
    """
    Mean-squared displacement between path predictions and target points.

    Used as the fitting loss in ``initialize_path``. For periodic
    systems, the displacement is wrapped under the minimum-image
    convention before squaring.
    """
    preds = path(times).positions
    assert preds.shape == positions.shape, (
        f"Shapes do not match: {preds.shape} != {positions.shape}"
    )
    disp = positions - preds
    if path.transform is not None:
        disp = path.transform(disp, center=1.0)
    return torch.mean(disp**2)


def initialize_path(
    path: torch.tensor,
    times: torch.tensor,
    init_points: torch.tensor,
    lr: float = 0.001,
    max_steps: int = 10000,  # 5000
) -> torch.tensor:
    """
    Fit the path's parameters so it passes through given points at given times.

    Run before optimization whenever the user supplies more than two
    images: the intermediate frames become target points, and a
    quick Adam loop minimizes ``loss_init`` until the path
    interpolates them.

    Parameters
    ----------
    path : BasePath
        Path to fit. Must have trainable parameters.
    times : torch.Tensor
        Target times in [0, 1]; shape ``[N, 1]``.
    init_points : torch.Tensor
        Target positions; shape ``[N, D]``.
    lr : float, default=0.001
        Adam learning rate for the fit.
    max_steps : int, default=10000
        Cap on Adam steps; 0 only evaluates the loss. The loss is
        checked before each step and the fit stops as soon as it is
        below ``INIT_LOSS_THRESHOLD``, so the returned path meets the
        threshold whenever the fit converges. If the cap is reached
        first, a WARNING is logged and the path is returned as is.

    Returns
    -------
    BasePath
        The fitted path (same object, mutated in place).

    Raises
    ------
    ValueError
        If ``max_steps`` is negative.
    """
    if max_steps < 0:
        msg = f"max_steps must be >= 0, got {max_steps}"
        raise ValueError(msg)
    logger.info("Beginning path initialization")
    optimizer = torch.optim.Adam(path.parameters(), lr=lr)
    for n_steps in range(max_steps + 1):
        optimizer.zero_grad()
        loss = loss_init(path, times, init_points)
        # Checked before stepping, so the path returned is the one that met it.
        if loss.item() < INIT_LOSS_THRESHOLD:
            break
        if n_steps == max_steps:
            logger.warning(
                f"Path initialization did not converge after {max_steps} iterations: "
                f"loss {loss.item():.3e} is not below {INIT_LOSS_THRESHOLD:g}"
            )
            break
        loss.backward()
        optimizer.step()
        if n_steps % 1000 == 0:
            logger.debug(f"\tIteration {n_steps}: Loss {loss.item():.4}")

    # n_steps is the number of Adam steps taken.
    logger.info(f"Finished path initialization after {n_steps} iterations")
    return path
