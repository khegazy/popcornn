import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.nn.functional import interpolate
from popcornn.tools import scheduler
from popcornn.tools.scheduler import get_schedulers

from popcornn.tools import Metrics


class PathOptimizer():
    def __init__(
            self,
            path,
            optimizer=None,
            find_ts=None,
            lr_scheduler=None,
            path_loss_schedulers=None,
            path_ode_schedulers=None,
            ts_time_loss_names=None,
            ts_time_loss_scales=torch.ones(1),
            ts_time_loss_schedulers=None,
            ts_region_loss_names=None,
            ts_region_loss_scales=torch.ones(1),
            ts_region_loss_schedulers=None,
            device='cpu',
            **config
        ):
        super().__init__()
        
        self.find_ts = find_ts
        self.device=device
        self.iteration = 0
        
        ####  Initialize transition state loss information  #####
        self.has_ts_time_loss = ts_time_loss_names is not None
        self.has_ts_region_loss = ts_region_loss_names is not None
        self.has_ts_loss = self.has_ts_time_loss or self.has_ts_region_loss
        if self.has_ts_loss:
            if self.find_ts is None or self.find_ts:
                self.find_ts = True
            else:
                raise ValueError("Cannot have transition state losses and set find_ts=False")
        
        self.ts_time_loss_names = ts_time_loss_names
        self.ts_time_loss_scales = ts_time_loss_scales
        if self.has_ts_time_loss:
            self.ts_time_metrics = Metrics(device)
            self.ts_time_metrics.create_ode_fxn(
                True, self.ts_time_loss_names, self.ts_time_loss_scales
            )
        
        self.ts_region_loss_names = ts_region_loss_names
        self.ts_region_loss_scales = ts_region_loss_scales
        if self.has_ts_region_loss:
            self.ts_region_metrics = Metrics(device)
            self.ts_region_metrics.create_ode_fxn(
                True, self.ts_region_loss_names, self.ts_region_loss_scales
            )
        
        #####  Initialize schedulers  #####
        self.ode_fxn_schedulers = get_schedulers(path_ode_schedulers)
        self.path_loss_schedulers = get_schedulers(path_loss_schedulers)
        self.ts_time_loss_schedulers = get_schedulers(ts_time_loss_schedulers)
        self.ts_region_loss_schedulers = get_schedulers(ts_region_loss_schedulers)
        
        #####  Initialize optimizer  #####
        self.path = path
        if optimizer is not None:
            self.set_optimizer(**optimizer)
        else:
            raise ValueError("Must specify optimizer parameters (dict) with key 'optimizer'")

        #####  Initialize learning rate scheduler  #####
        if lr_scheduler is not None:
            self.set_lr_scheduler(**lr_scheduler)
        else:
            self.lr_scheduler = None
        self.converged = False

    def set_optimizer(self, name, **config):
        """
        Set the optimizer for the path optimizer.
        """
        optimizer_dict = {key.lower(): key for key in dir(optim) if not key.startswith('_')}
        name = optimizer_dict[name.lower()]
        optimizer_class = getattr(optim, name)
        self.optimizer = optimizer_class(self.path.parameters(), **config)

    def set_lr_scheduler(self, name, **config):
        """
        Set the learning rate scheduler for the optimizer.
        """
        scheduler_dict = {key.lower(): key for key in dir(lr_scheduler) if not key.startswith('_')}
        name = scheduler_dict[name.lower()]
        scheduler_class = getattr(lr_scheduler, name)
        self.lr_scheduler = scheduler_class(self.optimizer, **config)

    
    def optimization_step(
            self,
            path,
            integrator,
            t_init=torch.tensor([0.], dtype=torch.float64),
            t_final=torch.tensor([1.], dtype=torch.float64),
            time=None,
            update_path=True
        ):
        self.optimizer.zero_grad()
        t_init = t_init.to(torch.float64).to(self.device)
        t_final = t_final.to(torch.float64).to(self.device)
        ode_fxn_scales = {
            name : schd.get_value() for name, schd in self.ode_fxn_schedulers.items()
        }
        path_loss_scales = {
            name : schd.get_value() for name, schd in self.path_loss_schedulers.items()
        }
        path_loss_scales['iteration'] = self.iteration,
        
        if self.has_ts_loss:
            ts_time_loss_scales = {
                name : schd.get_value() for name, schd in self.ts_time_loss_schedulers.items()
            }
            ts_region_loss_scales = {
                name : schd.get_value() for name, schd in self.ts_region_loss_schedulers.items()
            }
        path_integral = integrator.integrate_path(
            path,
            ode_fxn_scales=ode_fxn_scales,
            loss_scales=path_loss_scales,
            t_init=t_init,
            t_final=t_final,
            times=time
        )
        if not path_integral.gradient_taken:
            path_integral.loss.backward()
            # (path_integral.integral**2).backward()
        
        #####  Transition State  #####
        # Find transition state 
        path.ts_search(
            path_integral.t,
            path_integral.y[:,:,integrator.path_ode_energy_idx],
            path_integral.y[:,:,integrator.path_ode_force_idx:],
            evaluate_ts=False
        )

        # Evaluate transition state losses
        if self.find_ts and path.ts_time is not None:
            if self.has_ts_time_loss:
                self.ts_time_metrics.update_ode_fxn_scales(**ts_time_loss_scales)
                ts_time_loss = self.ts_time_metrics.ode_fxn(
                    torch.tensor([[path.ts_time]]), path
                )[:,0]
                ts_time_loss.backward()
            if self.has_ts_region_loss:
                self.ts_region_metrics.update_ode_fxn_scales(
                    **ts_region_loss_scales
                )
                ts_region_loss = self.ts_region_metrics.ode_fxn(
                    path.ts_region[:,None], path
                )[:,0]
                ts_region_loss.backward()

        #####  Update Optimization  #####
        # Path update step
        if update_path:
            self.optimizer.step()
        # Update schedulers
        for name, sched in self.ode_fxn_schedulers.items():
            sched.step() 
        for name, sched in self.path_loss_schedulers.items():
            sched.step()
        if self.has_ts_loss:
            for name, sched in self.ts_time_loss_schedulers.items():
                sched.step() 
            for name, sched in self.ts_region_loss_schedulers.items():
                sched.step()
        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(path_integral.loss.item())
                if all([last_lr <= min_lr for last_lr, min_lr in zip(self.lr_scheduler.get_last_lr(), self.lr_scheduler.min_lrs)]):
                    self.converged = True
            else:
                self.lr_scheduler.step()
        self.iteration = self.iteration + 1
        
        return path_integral

    