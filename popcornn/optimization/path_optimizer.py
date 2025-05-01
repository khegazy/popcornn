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
            lr_scheduler=None,
            path_loss_schedulers=None,
            path_ode_schedulers=None,
            TS_time_loss_names=None,
            TS_time_loss_scales=torch.ones(1),
            TS_time_loss_schedulers=None,
            TS_region_loss_names=None,
            TS_region_loss_scales=torch.ones(1),
            TS_region_loss_schedulers=None,
            device='cpu',
            **config
        ):
        super().__init__()
        
        self.device=device
        self.iteration = 0
        
        ####  Initialize loss information  #####
        self.has_TS_time_loss = TS_time_loss_names is not None
        self.has_TS_region_loss = TS_region_loss_names is not None
        self.has_TS_loss = self.has_TS_time_loss or self.has_TS_region_loss
        
        self.TS_time_loss_names = TS_time_loss_names
        self.TS_time_loss_scales = TS_time_loss_scales
        if self.has_TS_time_loss:
            self.TS_time_metrics = Metrics(device)
            self.TS_time_metrics.create_ode_fxn(
                True, self.TS_time_loss_names, self.TS_time_loss_scales
            )
        
        self.TS_region_loss_names = TS_region_loss_names
        self.TS_region_loss_scales = TS_region_loss_scales
        if self.has_TS_region_loss:
            self.TS_region_metrics = Metrics(device)
            self.TS_region_metrics.create_ode_fxn(
                True, self.TS_region_loss_names, self.TS_region_loss_scales
            )
        
        #####  Initialize schedulers  #####
        self.ode_fxn_schedulers = get_schedulers(path_ode_schedulers)
        self.path_loss_schedulers = get_schedulers(path_loss_schedulers)
        self.TS_time_loss_schedulers = get_schedulers(TS_time_loss_schedulers)
        self.TS_region_loss_schedulers = get_schedulers(TS_region_loss_schedulers)
        
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


    """
    def set_scheduler(self, name, **config):
        name = name.lower()
        if name not in scheduler_dict:
            raise ValueError(f"Cannot handle scheduler type {name}, either add it to scheduler_dict or use {list(scheduler_dict.keys())}")
        self.scheduler = scheduler_dict[name](self.optimizer, **config)

    def set_loss_scheduler(self, **kwargs):
        self.loss_scheduler = {}
        for key, value in kwargs.items():
            name = value.pop('name').lower()
            if name not in loss_scheduler_dict:
                raise ValueError(f"Cannot handle loss scheduler type {name}, either add it to loss_scheduler_dict or use {list(loss_scheduler_dict.keys())}")
            if name == "reduce_on_plateau" or name == "increase_on_plateau":
                self.loss_scheduler[key] = loss_scheduler_dict[name](lr_scheduler=self.scheduler, **value)
            else:
                self.loss_scheduler[key] = loss_scheduler_dict[name](**value)
    """
    
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
        TS_time_loss_scales = {
            name : schd.get_value() for name, schd in self.TS_time_loss_schedulers.items()
        }
        TS_region_loss_scales = {
            name : schd.get_value() for name, schd in self.TS_region_loss_schedulers.items()
        }
        path_integral = integrator.integrate_path(
            path, #self.path_loss_name, self.path_loss_scales,
            ode_fxn_scales=ode_fxn_scales,
            loss_scales=path_loss_scales,
            t_init=t_init,
            t_final=t_final,
            times=time
        )
        #print("POST OP T", path_integral.t.shape)
        if not path_integral.gradient_taken:
            path_integral.loss.backward()
            # (path_integral.integral**2).backward()
        #####  Find Transition State  ##### 
        """
        path.TS_search_orig(
            path_integral.t,
            path_integral.y[:,:,integrator.path_ode_energy_idx],
            path_integral.y[:,:,integrator.path_ode_force_idx:],
        )
        """
        path.TS_search(
            path_integral.t,
            path_integral.y[:,:,integrator.path_ode_energy_idx],
            path_integral.y[:,:,integrator.path_ode_force_idx:],
        )

        #############  Testing TS Loss ############
        # Evaluate TS loss functions
        if self.has_TS_loss and path.TS_time is not None:
            if self.has_TS_time_loss:
                self.TS_time_metrics.update_ode_fxn_scales(**TS_time_loss_scales)
                TS_time_loss = self.TS_time_metrics.ode_fxn(
                    torch.tensor([[path.TS_time]]), path
                )[:,0]
                TS_time_loss.backward()
            if self.has_TS_region_loss:
                self.TS_region_metrics.update_ode_fxn_scales(
                    **TS_region_loss_scales
                )
                TS_region_loss = self.TS_region_metrics.ode_fxn(
                    path.TS_region[:,None], path
                )[:,0]
                TS_region_loss.backward()
        ###########################################

        if update_path:
            self.optimizer.step()
        for name, sched in self.ode_fxn_schedulers.items():
            sched.step() 
        for name, sched in self.path_loss_schedulers.items():
            sched.step() 
        for name, sched in self.TS_time_loss_schedulers.items():
            sched.step() 
        for name, sched in self.TS_region_loss_schedulers.items():
            sched.step()
        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(path_integral.loss.item())
                if all([last_lr <= min_lr for last_lr, min_lr in zip(self.lr_scheduler.get_last_lr(), self.lr_scheduler.min_lrs)]):
                    self.converged = True
            else:
                self.lr_scheduler.step()
        

                ##############
        self.iteration = self.iteration + 1
        return path_integral
    
    def _TS_max_E(self):
        raise NotImplementedError

    