import time
import torch
import torch.distributed as dist
import numpy as np
from dataclasses import dataclass
from enum import Enum

from torchdiffeq import odeint
from torchpathdiffeq import SerialAdaptiveStepsizeSolver, get_parallel_RK_solver 

from .metrics import Metrics, get_loss_fxn

@dataclass
class IntegralOutput():
    integral: torch.Tensor
    times: torch.Tensor
    geometries: torch.Tensor

class ODEintegrator(Metrics):
    def __init__(
            self,
            method='dopri5',
            rtol=1e-7,
            atol=1e-9,
            computation='parallel',
            sample_type='uniform',
            remove_cut=0.1,
            path_loss_name=None,
            path_loss_params={},
            path_ode_names=None,
            path_ode_scales=torch.ones(1),
            path_ode_energy_idx=1,
            path_ode_force_idx=2,
            process=None,
            max_batch=None,
            is_multiprocess=False,
            is_load_balance=False,
            n_added_evals=3,
            device=None,
        ):
        super().__init__(device, save_energy_force=True)
        self.max_batch = max_batch
        self.is_multiprocess = is_multiprocess
        self.is_load_balance = is_load_balance
        self.process = process
        self.N_integrals = 0
        self.path_ode_energy_idx = path_ode_energy_idx
        self.path_ode_force_idx = path_ode_force_idx
        
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.integral_output = None
        self.add_y_arg = False

        if computation == 'serial':
            self.is_parallel = False
            self._integrator = SerialAdaptiveStepsizeSolver(
                method=self.method,
                atol=atol,
                rtol=rtol,
                t_init=torch.tensor([0], dtype=torch.float64),
                t_final=torch.tensor([1], dtype=torch.float64),
                device=device,
            )
            if self.is_load_balance:
                self.balance_load = self._serial_load_balance
        elif computation == 'parallel':
            self.sample_type = sample_type
            self.is_parallel = True
            self.remove_cut = remove_cut
            self._integrator = get_parallel_RK_solver(
                self.sample_type,
                method=self.method,
                atol=self.atol,
                rtol=self.rtol,
                remove_cut=self.remove_cut,
                max_path_change=None,
                y0=torch.tensor([0], dtype=torch.float, device=device),
                t_init=torch.tensor([0], dtype=torch.float64),
                t_final=torch.tensor([1], dtype=torch.float64),
                error_calc_idx=0,
                device=device,
            )
        else:
            raise ValueError(f"integrator argument must be either 'parallel' or 'serial', not {computation}.")
        
        if self.is_multiprocess:
            if self.process is None or not self.process.is_distributed:
                raise ValueError("Must run program in distributed mode with multiprocess integrator.")
            self.inner_path_integral = self.path_integral
            self.integrator = self.multiprocess_path_integral
            self.run_time = torch.tensor([1], requires_grad=False)# = np.ones(self.process.world_size)
            if self.is_load_balance:
                self.mp_times = torch.linspace(
                    0, 1, self.process.world_size+1, requires_grad=False
                )

        #####  Build loss funtion to integrate path over  #####
        ### Setup ode_fxn
        if path_ode_names is None:
            self.eval_fxns = None
            self.eval_fxn_scales = None
            self.ode_fxn = None
        else:
            self.create_ode_fxn(
                self.is_parallel, path_ode_names, path_ode_scales
            )

        ### Setup loss_fxn
        self.loss_name = path_loss_name
        self.loss_fxn = get_loss_fxn(path_loss_name, **path_loss_params)

    
    def integrator(
            self,
            path,
            ode_fxn_scales={},
            loss_scales={},
            t_init=torch.tensor([0], dtype=torch.float64),
            t_final=torch.tensor([1], dtype=torch.float64),
            times=None,
            iteration=None
        ):

        self.update_ode_fxn_scales(**ode_fxn_scales)
        self.loss_fxn.update_parameters(**loss_scales)
        
        if times is None:
            if self.integral_output is None:
                times = None
            else:
                times = self.integral_output.t_optimal
        integral_output = self._integrator.integrate(
            ode_fxn=self.ode_fxn,
            loss_fxn=self.loss_fxn,
            t=times,
            t_init=t_init,
            t_final=t_final,
            ode_args=(path,),
            max_batch=self.max_batch
        )
        integral_output.integral = integral_output.integral[0]
        self.integral_output = integral_output
        self.loss_fxn.update_parameters(integral_output=self.integral_output)
        self.N_integrals = self.N_integrals + 1
        return integral_output


    def path_integral(
            self,
            path,
            ode_fxn_scales={},
            loss_scales={},
            t_init=torch.tensor([0.], dtype=torch.float64),
            t_final=torch.tensor([1.], dtype=torch.float64),
            record_evals=False
        ):
        # Check scales names

        if record_evals:
            path.begin_time_recording()
        
        integral_output = self.integrator(
            path=path,
            ode_fxn_scales=ode_fxn_scales,
            loss_scales=loss_scales,
            t_init=t_init,
            t_final=t_final
        )

        if record_evals:
            time_record, geo_record = path.get_eval_record()
            path.end_eval_recording()
        else:
            time_record = None
            geo_record = None
        
        return integral_output
        