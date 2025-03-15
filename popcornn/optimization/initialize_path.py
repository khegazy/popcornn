import torch
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from typing import Union


def randomly_initialize_path(
        path: torch.tensor,
        n_points: int,
        order_points: bool = False,
        seed: int = 1910
) -> Union[torch.Tensor, np.ndarray]:
    """
    Randomly initialize the path.

    Parameters:
    -----------
    path : torch.Tensor
        The path object.
    n_points : int
        Number of points.
    order_points : bool, optional
        Whether to order points (default is False).
    seed : int, optional
        Random seed (default is 1910).

    Returns:
    --------
    Union[torch.Tensor, np.ndarray]
        Initialized path.
    """
    #times = rnd.uniform(shape=(n_points, 1), minval=0.1, maxval=0.9)
    times = torch.unsqueeze(torch.linspace(0, 1, n_points+2, device=path.device)[1:-1], -1)
    times.requires_grad = False

    n_dims = len(path.initial_point)
    rnd_dims = []
    for idx in range(n_dims):
        min_val = torch.min(
            torch.tensor([path.initial_point[idx], path.final_point[idx]], device=path.device)
        ).item()
        max_val = torch.max(
            torch.tensor([path.initial_point[idx], path.final_point[idx]], device=path.device)
        ).item()
        print("MIN MAX", min_val, max_val)
        # rnd_vals = rnd.uniform(size=(n_points, 1), low=min_val, high=max_val)
        rnd_vals = torch.rand(n_points, 1, device=path.device) * (max_val - min_val) + min_val
        if order_points or idx == 0:
            # if path.initial_point[idx] > path.final_point[idx]:
            #     rnd_dims.append(-1*np.sort(-1*rnd_vals, axis=0))
            # else:
            #     rnd_dims.append(np.sort(rnd_vals, axis=0))
            descending = path.initial_point[idx] > path.final_point[idx]
            rnd_dims.append(torch.sort(rnd_vals, axis=0, descending=descending.item()).values)
        else:
            rnd_dims.append(rnd_vals)
    print(len(rnd_dims), rnd_dims[0].shape)
    # rnd_dims = torch.tensor(
    #     np.concatenate(rnd_dims, axis=-1), requires_grad=False, device=path.device
    # )
    rnd_dims = torch.cat(rnd_dims, dim=-1)

    return initialize_path(path, times, rnd_dims)


def loss_init(
        path: torch.tensor,
        times: torch.tensor,
        points: torch.tensor
) -> torch.Tensor:
    """
    Initialize the loss.

    Parameters:
    -----------
    path : torch.Tensor
        The path object.
    times : torch.Tensor
        Times.
    points : torch.Tensor
        Points.

    Returns:
    --------
    torch.Tensor
        Loss value.
    """
    preds = path(times).path_geometry
    assert preds.shape == points.shape, f"Shapes do not match: {preds.shape} != {points.shape}"
    disp = points - preds
    if path.transform is not None:
        disp = path.transform(disp, center=0.5)
    return torch.mean(disp ** 2)
    
    


def initialize_path(
        path: torch.tensor,
        times: torch.tensor,
        init_points: torch.tensor,
        lr: float = 0.001,
        max_steps: int = 10000 # 5000
) -> torch.tensor:
    """
    Initialize the path.

    Parameters:
    -----------
    path : torch.Tensor
        The path object.
    times : torch.Tensor
        Times.
    init_points : torch.Tensor
        Initial points.
    lr : float, optional
        Learning rate (default is 0.001).
    max_steps : int, optional
        Maximum number of steps (default is 5000).

    Returns:
    --------
    torch.Tensor
        Initialized path.
    """
    print("INFO: Beginning path initialization")
    loss, prev_loss = torch.tensor([2e-10]), torch.tensor([1e-10])
    print(path.named_parameters())
    optimizer = torch.optim.Adam(path.parameters(), lr=lr)
    idx, rel_error = 0, 100
    # while (idx < 1500 or loss > 1e-8) and idx < max_steps:
    for idx in range(max_steps):
        optimizer.zero_grad()

        prev_loss = loss.item()
        loss = loss_init(path, times, init_points)

        loss.backward()
        optimizer.step()
        rel_error = np.abs(prev_loss - loss.item())/prev_loss
        # idx = idx + 1
        if idx % 1000 == 0:
            print(f"\tIteration {idx}: Loss {loss:.4} | Relative Error {rel_error:.5}")
        #     fig, ax = plt.subplots()
        #     path_output = path.get_path()
        #     geometric_path = path_output.geometric_path.detach().cpu().numpy()
        #     ax.plot(init_points[:,0], init_points[:,1], 'ob')
        #     ax.plot(geometric_path[:,0], geometric_path[:,1], '-k')
        #     fig.savefig(f"./plots/initialization/init_path_{idx}.png")
        # if rel_error < 1e-8:
        #     break
        if loss.item() < 1e-8:
            break
    else:
        # raise ValueError(f"INFO: Maximum number of steps reached: {max_steps}")
        pass

        #print(prev_loss, loss, jnp.abs(prev_loss - loss)/prev_loss)

    print(f"INFO: Finished path initialization after {idx} iterations")
    # fig, ax = plt.subplots()
    # path_output = path.get_path()
    # geometric_path = path_output.geometric_path.detach().numpy()
    # ax.plot(init_points[:,0], init_points[:,1], 'ob')
    # ax.plot(geometric_path[:,0], geometric_path[:,1], '-k')
    # fig.savefig("./plots/init_path.png")

    return path
