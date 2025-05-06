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
    mep = Popcornn(**config.get('init_params', {}))  # Initialize Popcornn with parameter dictionary
    final_images, ts_image = mep.optimize_path(*config.get('opt_params', []))  # Run the optimization with a list of parameter dictionaries
    
    # Write the final images
    if isinstance(final_images, list) and isinstance(final_images[0], Atoms):
        write('configs/popcornn.xyz', final_images)
    if isinstance(ts_image, Atoms):
        write('configs/popcornn_ts.xyz', ts_image)
