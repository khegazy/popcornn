# Popcornn
Path Optimization with a Continuous Representation Neural Network for reaction path with machine learning interatomic potentials

## Installation and Dependencies
We recommend using conda environment to install dependencies of this library. Please install (or load) conda and then proceed with the following commands:
```
conda create --name popcornn python=3.12
conda activate popcornn

pip install popcornn
```
A few different popular machine learning potentials have been interfaced with Popcornn, such as CHGNet, EScAIP, LEFTNet, MACE, [NewtonNet](https://github.com/THGLab/NewtonNet), Orb, [UMA](https://github.com/facebookresearch/fairchem), etc. Please refer to the respective codespaces for the installation guides.

## Quick Start
You can find several run files inside the `example` directory that rely on the implemented modules in the Popcornn library. We provide a simple run script. The run scripts need to be accompanied with a yaml config file. You can run an example optimization script with the following command in the `example` directory:
```
python run.py --config configs/rxn0003.yaml
```
Everything there is to be specified is in the config file.

## Set up your own Popcornn
The config file is read in the run script as a dictionary, so you can also directly specify the configs in your own python script, giving you more handles on the inputs and outputs.

### Initialize the path
The first step for you would be to specify the endpoints of the reaction you are working on:
```
from ase.io import read

images = read('configs/rxn0003.xyz', index=':')
for image in images:
    image.info = {"charge": 0, "spin": 1}  # if required by the MLIP, set the total charge and multiplicity
```
It can be a list of ASE Atoms, or if it's a string, we can also read `xyz` or `traj` files. If there are more than 2 frames provided, the path will be first fitted to go through the intermediate frames, but they are not fixed. Note that the reactant and product should be index-mapped, rotationally/translationally aligned, and ideally unwrapped. By default, we unwrap the product according to the minimum image convention with respect to the reactant, but if the cell is small and some atoms are expected to move more than half a cell, you should unwrap the frames manually and disable `unwrap_positions` in `path_params` (see below).

Next, you can set up the path using the images:
```
from popcornn import Popcornn

mep = Popcornn(images=images, path_params={'name': 'mlp', 'n_embed': 1, 'depth': 2})
```
Optional initialization parameters for Popcornn include `num_record_points` for the number of frames to be recorded after optimization, `output_dir` for optional debug outputs, `device`, `dtype`, and `seed`. For simpler reactions, `depth` of 2 helps limit the complexity of the reaction, while more complicated reactions may require a deeper path neural network.

### Optimize the path
Machine learning potentials are vulnerable to unphysical, out-of-distribution configurations, and it's important to resolve atom clashing as an interpolation step. Luckily, you can do both the interpolation and the optimization with Popcornn! In general, therefore, you need multiple optimizations by providing multiple `opt_params`, each with a different potential, integral loss, and optimizer:
```
final_images, ts_image = mep.optimize_path(
    {
        'potential_params': {'potential': 'repel'},
        'integrator_params': {'path_ode_names': 'geodesic'},
        'optimizer_params': {'optimizer': {'name': 'adam', 'lr': 1.0e-1}},
        'num_optimizer_iterations': 1000,
    },
    {
        'potential_params': {'potential': 'uma', 'model_name': 'uma-s-1', 'task_name': 'omol'},
        'integrator_params': {'path_ode_names': 'projected_variable_reaction_energy', 'rtol': 1.0e-5, 'atol': 1.0e-7},
        'optimizer_params': {'optimizer': {'name': 'adam', 'lr': 1.0e-3}},
        'num_optimizer_iterations': 1000,
    },
)
```
Finally, after optimization, you can save the optimized path as a list of Atoms for visualization and further optimization:
```
from ase.io import write

write('popcornn.xyz', final_images)
write('popcornn_ts.xyz', ts_image)
```
In this example, you should get a barrier of ~3.6 eV.
