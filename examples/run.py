from ase import Atoms
from ase.io import read, write
from popcornn import tools, optimize_MEP


if __name__ == "__main__":
    ###############################
    #####  Setup environment  #####
    ###############################

    # Import configuration files
    args = tools.build_default_arg_parser().parse_args()
    config = tools.import_run_config(args.config)
    
    # Run the optimization
    final_images, ts_image = optimize_MEP(**config)
    
    # Write the final images
    if isinstance(final_images, list) and isinstance(final_images[0], Atoms):
        write('configs/popcornn.xyz', final_images)
    if isinstance(ts_image, Atoms):
        write('configs/popcornn_ts.xyz', ts_image)
