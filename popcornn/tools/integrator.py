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
            computation='parallel',
            path_loss_name=None,
            path_loss_params={},
            path_ode_names=None,
            path_ode_scales=None,
            device=None,
            dtype=None,
            **kwargs
        ):
        super().__init__(device, save_energy_force=True)

        # Check Parameters
        assert computation =='parallel' or computation == 'serial',\
            f"Computation must be 'parallel' or 'serial', instead got {computation}"
        self.N_integrals = 0
        self.device = device
        self.dtype = dtype
        
        #self.rtol = rtol
        #self.atol = atol
        self.integral_output = None
        self.add_y_arg = False

        #####  Setup torchpathdiffeq integrator and parallel compute  #####
        self._setup_integrator_parallism(method, computation, **kwargs)

        #####  Build loss funtion to integrate path over  #####
        ### Setup ode_fxn
        if path_ode_names is None:
            self.eval_fxns = None
            self.eval_fxn_scales = None
            self.ode_fxn = None
        else:
            self.create_ode_fxn(
                computation == 'parallel', 
                path_ode_names,
                path_ode_scales
            )

        ### Setup loss_fxn
        self.loss_name = path_loss_name
        self.loss_fxn = get_loss_fxn(path_loss_name, **path_loss_params)


    def _setup_integrator_parallism(
            self,
            method,
            computation,
            rtol=1e-6,
            atol=1e-7,
            sample_type='uniform',
            remove_cut=0.1,
            path_ode_energy_idx=1,
            path_ode_force_idx=2,
            max_batch=None,
            process=None,
            is_multiprocess=False,
            is_load_balance=False,
            **kwargs
        ):
        
        self.path_ode_energy_idx = path_ode_energy_idx
        self.path_ode_force_idx = path_ode_force_idx
        if computation == 'serial':
            self._integrator = SerialAdaptiveStepsizeSolver(
                method=self.method,
                atol=atol,
                rtol=rtol,
                t_init=torch.tensor([0], device=self.device, dtype=self.dtype),
                t_final=torch.tensor([1], device=self.device, dtype=self.dtype),
                device=self.device,
                **kwargs
            )
            if self.is_load_balance:
                self.balance_load = self._serial_load_balance
        elif computation == 'parallel':
            self._integrator = get_parallel_RK_solver(
                sample_type,
                method=method,
                atol=atol,
                rtol=rtol,
                remove_cut=remove_cut,
                max_path_change=None,
                y0=torch.tensor([0], device=self.device, dtype=self.dtype),
                t_init=torch.tensor([0], device=self.device, dtype=self.dtype),
                t_final=torch.tensor([1], device=self.device, dtype=self.dtype),
                max_batch=max_batch,
                error_calc_idx=0,
                device=self.device,
                **kwargs
            )
        else:
            raise ValueError(f"integrator argument must be either 'parallel' or 'serial', not {computation}.")
        
        if is_multiprocess:
            if self.process is None or not self.process.is_distributed:
                raise ValueError("Must run program in distributed mode with multiprocess integrator.")
            self.inner_path_integral = self.path_integral
            self.integrator = self.multiprocess_path_integral
            self.run_time = torch.tensor([1], requires_grad=False)# = np.ones(self.process.world_size)
            if self.is_load_balance:
                self.mp_times = torch.linspace(
                    0, 1, self.process.world_size+1, requires_grad=False
                )

    def integrate_path(
            self,
            path,
            ode_fxn_scales={},
            loss_scales={},
            t_init=torch.tensor([0]),
            t_final=torch.tensor([1]),
            times=None,
        ):
        # Update loss parameters
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
            #max_batch=self.max_batch
        )
        integral_output.integral = integral_output.integral[0]
        self.integral_output = integral_output
        self.loss_fxn.update_parameters(integral_output=self.integral_output)
        self.N_integrals = self.N_integrals + 1
        return integral_output