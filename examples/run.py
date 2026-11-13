import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import read, write
from popcornn import Popcornn
from popcornn.tools import build_default_arg_parser, import_run_config


if __name__ == "__main__":
    ###############################
    #####  Setup environment  #####
    ###############################

    # Import configuration files
    args = build_default_arg_parser().parse_args()
    config = import_run_config(args.config)
    
    # Run the optimization
    my_path = Popcornn(**config.get('initialization_params', {}))  # Initialize Popcornn with parameter dictionary
    my_path.optimize_path(*config.get('optimization_params', []))  # Run the optimization with a list of parameter dictionaries
    
    # Write the final images, or plot them on the contours of a 2D potential
    final_images = my_path.get_discrete_path()
    ts_image = my_path.get_ts()
    if isinstance(ts_image, Atoms):
        write('popcornn_path.xyz', final_images)
        write('popcornn_ts.xyz', ts_image)
    else:
        path, ts = np.array(final_images), np.array(ts_image['ts_positions'])
        pad = np.ptp(path, axis=0).max() / 3
        x, y = np.meshgrid(*[np.linspace(lo - pad, hi + pad, 256) for lo, hi in zip(path.min(axis=0), path.max(axis=0))])
        grid = torch.tensor(np.stack([x.ravel(), y.ravel()], axis=-1), device=my_path.device, dtype=my_path.dtype, requires_grad=True)
        energies = my_path.path.potential(grid).energies.detach().cpu().numpy().reshape(x.shape)
        levels = np.arange(energies.min(), energies.max(), (ts_image['ts_energies'][0][0] - energies.min()) / 5)  # five contours below the TS
        plt.contour(x, y, energies, levels=levels, colors='gray', linestyles='solid', linewidths=1)
        plt.plot(path[:, 0], path[:, 1], color='tab:red', linewidth=2)
        plt.scatter(ts[:, 0], ts[:, 1], marker='x', color='k', label='TS', zorder=3)
        plt.gca().set(aspect='equal', xticks=[], yticks=[])
        plt.legend(loc='upper left')
        plt.savefig('popcornn_path.pdf', bbox_inches='tight')
