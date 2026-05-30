import os
import json
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
    
    # Write the final images
    final_images = my_path.get_discrete_path()
    ts_image = my_path.get_ts()
    if isinstance(final_images, list) and isinstance(final_images[0], Atoms):
        write('popcornn_path.xyz', final_images)
    else:
        with open('popcornn_path.json', 'w') as f:
            json.dump(final_images, f)
    if isinstance(ts_image, Atoms):
        write('popcornn_ts.xyz', ts_image)
    else:
        with open('popcornn_ts.json', 'w') as f:
            json.dump(ts_image, f)
