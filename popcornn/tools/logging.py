import os
import numpy as np
from .visualize import plot_path


class logging():
    def __init__(self, *args, **kwargs):
        return
    
    def training_logger(self, step, val):
        step_string = ("step: " + str(step)).ljust(15)
        val_string = "val: " + str(val)
        print(step_string, val_string)
    
    def optimization_step(
            self,
            step,
            path,
            potential,
            loss,
            plot=True,
            plot_dir='./',
            geo_paths=None,
            pes_paths=None,
            add_azimuthal_dof=False,
            add_translation_dof=False
        ):
        print(f"Step {step} | Loss: {loss.item():.7}")
        path_output = path.get_path(return_velocities=True, return_forces=True)
        if geo_paths is not None:
            geo_paths.append(path_output.geometric_path)
        if pes_paths is not None:
            pes_paths.append(path_output.potential_path)
        if plot:
            plot_path(
                path_output.geometric_path.detach().to('cpu').numpy(),
                f"test_plot_{step:03d}",
                pes_fxn=potential,
                plot_min_max=(-2, 2, -2, 2),
                levels=np.arange(-100, 100, 5),
                plot_dir=plot_dir,
                add_translation_dof=add_translation_dof,
                add_azimuthal_dof=add_azimuthal_dof
            )
        return path_output