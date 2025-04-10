from typing import Any
import torch
import matplotlib.pyplot as plt


class LossBase():
    def __init__(self, weight_scale=None) -> None:
        self.weight_scale = weight_scale
        self.iteration = None
        self.t_midpoint = None
    
    def update_parameters(self, **kwargs):
        if 'weight' in kwargs:
            self.weight_scale = kwargs['weight']
        if 'iteration' in kwargs:
            self.iteration = torch.tensor([kwargs['iteration']])
        # Find the center of the path in time
        if 'integral_output' in kwargs:
            self.t_midpoint = kwargs['integral_output'].t_optimal[:,0]
            if len(self.t_midpoint) % 2 == 1:
                self.t_midpoint = self.t_midpoint[len(self.t_midpoint)//2]
            else:
                t_idx = len(self.t_midpoint)//2
                self.t_midpoint = self.t_midpoint[t_idx-1] + self.t_midpoint[t_idx]
                self.t_midpoint = self.t_midpoint/2.

    def _check_parameters(self, weight_scale=None, **kwargs):
        assert self.weight_scale is not None or weight_scale is not None,\
            "Must provide 'weight_scale' to update_parameters or loss call."
        self.weight_scale = self.weight_scale if weight_scale is None else weight_scale
    
    def get_weights(self, integral_output):
        raise NotImplementedError
    
    def __call__(self, integral_output, **kwargs) -> Any:
        self._check_parameters(**kwargs)
        weights = self.get_weights(
            torch.mean(integral_output.t[:,:,0], dim=1),
            integral_output.t_init,
            integral_output.t_final,
        )
        """
        print("WEIGHTS", self.iteration, weights)
        print(torch.mean(integral_output.t[:,:,0], dim=1))
        fig, ax = plt.subplots()
        ax.set_title(str(self.t_midpoint))
        ax.plot(t_mean, weights)
        ax.plot([0,1], [0,0], ':k')
        ax.set_ylim(-0.1, 1.05)
        fig.savefig(f"test_weights_{self.iteration[0]}.png")
        """

        return integral_output.y0\
            + torch.sum(weights*integral_output.sum_steps[:,0])



class PathIntegral(LossBase):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, integral_output, **kwargs):
        return integral_output.integral[0]


class EnergyWeight(LossBase):
    def __init__(self) -> None:
        super().__init__()
    
    def get_weights(self, integral_output):
        return torch.mean(integral_output.y[1], dim=1)


class GrowingString(LossBase):
    def __init__(self, weight_type='inv_sine', time_scale=10, envelope_scale=1000, **kwargs) -> None:
        super().__init__()
        self.iteration = torch.zeros(1)
        self.time_scale = time_scale
        self.envelope_scale = envelope_scale
        self.t_midpoint = 0.5

        idx1 = weight_type.find("_")
        #idx2 = weight_type.find("_", idx1 + 1)
        envelope_key = weight_type[idx1+1:]
        #envelope_key = weight_type[idx1+1:idx2]
        if envelope_key == 'gauss':
            self.envelope_fxn = self._guass_envelope
        elif envelope_key == 'poly':
            self.order = 1 if 'order' not in kwargs else kwargs['order']
            self.envelope_fxn = self._poly_envolope
        elif envelope_key == 'sine':
            self.envelope_fxn = self._sine_envelope
        elif envelope_key == 'sine-gauss' or envelope_key == 'gauss-sine':
            self.envelope_fxn = self._sine_gauss_envelope
        elif envelope_key == 'butter':
            self.order = 8 if 'order' not in kwargs else kwargs['order']
            self._butter_envelope
        else:
            raise ValueError(f"Cannot make envelope type {envelope_key}")
        """
        decay_key = weight_type[idx2+1:]
        if decay_key == 'exp':
            def decay_fxn(iteration, time_scale):
                return self.envelope_scale*torch.exp(-1*iteration*time_scale)
        else:
            raise ValueError(f"Cannot make decay type {decay_key}")
        """

        fxn_key = weight_type[:idx1]
        if fxn_key == 'inv':
            self.get_weights = self._inv_weights
        else:
            raise ValueError(f"Cannot make weight function type {fxn_key}")
    
        
    def update_parameters(self, **kwargs):
        super().update_parameters(**kwargs)    
        #assert 'variance' in kwargs, "Must provide 'variance' to update_parameters."
        if 'variance' in kwargs:
            self.variance_scale = kwargs['variance']
        if 'order' in kwargs:
            self.order = kwargs['order']

    def _inv_weights(self, t, t_init, t_final):
        envelope = self.envelope_fxn(t, t_init, t_final)
        return 1./(1 + self.weight_scale*envelope)
    
    def _guass_envelope(self, t, t_init, t_final):
        mask = t < self.t_midpoint
        # Left side
        t_left = t[mask]
        if len(t_left) > 0:
            left = torch.exp(-1/(self.variance_scale + 1e-10)\
                *((self.t_midpoint - t_left)*4/(t_init - self.t_midpoint))**2
            )
            t_left = (t_left - t_left[0])/(self.t_midpoint - t_left[0])
            left = left - (left[0] - t_left*left[0])
        else:
            left = None
        # Right side
        t_right = t[torch.logical_not(mask)]
        if len(t_right) > 0:
            right = torch.exp(-1/(self.variance_scale + 1e-10)\
                *((self.t_midpoint - t_right)*4\
                /(t_final - self.t_midpoint))**2)
            t_right = (t_right - t_right[-1])/(self.t_midpoint - t_right[-1])
            right = right - (right[-1] - t_right*right[-1])
        else:
            right = None
        
        if left is None:
            return right
        elif right is None:
            return left
        else:
            return torch.concatenate([left, right])
    
    def _sine_envelope(self, t, t_init, t_final):
        mask = t < self.t_midpoint
        # Left side
        t_left = t[mask]
        if len(t_left) > 0:
            left = (1 - torch.cos(
                (t_left - t_init)*torch.pi/((self.t_midpoint - t_init))
            ))/2.
        else:
            left = None
        # Right side
        t_right = t[torch.logical_not(mask)]
        if len(t_right) > 0:
            right = (1 + torch.cos(
                (t[torch.logical_not(mask)] - self.t_midpoint)\
                    *torch.pi/((t_final - self.t_midpoint))
            ))/2.
        else:
            right = None

        if left is None:
            return right
        elif right is None:
            return left
        else:
            return torch.concatenate([left, right])

    def _poly_envolope(self, t, t_init, t_final):
        mask = t < self.t_midpoint
        # Left side
        t_left = t[mask]
        if len(t_left) > 0: 
            left = torch.abs(
                (t_left - t_init)/((self.t_midpoint - t_init))
            )**self.order
        else:
            left = None
        # Right side
        t_right = t[torch.logical_not(mask)]
        if len(t_right) > 0:
            right = torch.abs((t[torch.logical_not(mask)] - t_final)\
                /(t_final - self.t_midpoint))**self.order
        else:
            right = None
        
        if left is None:
            return right
        elif right is None:
            return left
        else:
            return torch.abs(torch.concatenate([left, right]))

    def _sine_gauss_envelope(self, t, t_init, t_final):
        guass_envelope = self._guass_envelope(t, t_init, t_final)
        sine_envelope = self._sine_envelope(t, t_init, t_final)
        return guass_envelope*sine_envelope


    def _butter_envelope(self, t, t_init, t_final):
        mask = t < self.t_midpoint
        # Left side
        t_left = t[mask]
        if len(t_left) > 0: 
            dt = self.t_midpoint - t_left
            left = 1./torch.sqrt(1 + (dt*2/(self.t_midpoint - t_init))**self.order)
        else:
            left = None
        # Right side
        t_right = t[torch.logical_not(mask)]
        if len(t_right) > 0:
            dt = t_right - self.t_midpoint
            right = 1./torch.sqrt(1 + (dt*2/(self.t_midpoint - t_init))**self.order)
        else:
            right = None

        if left is None:
            return right
        elif right is None:
            return left
        else:
            return torch.concatenate([left, right])
    
   
loss_fxns = {
    'path_integral' : PathIntegral,
    'integral' : PathIntegral,
    'energy_weight' : EnergyWeight,
    'growing_string' : GrowingString
}

def get_loss_fxn(name, **kwargs):
    if name is None:
        return loss_fxns['path_integral']()
    assert name in loss_fxns, f"Cannot find loss {name}, must select from {list(loss_fxns.keys())}"
    return loss_fxns[name](**kwargs)
        


class Metrics():
    def __init__(self, device):
        self.device = device
        self.ode_fxn = None
        self._ode_fxn_scales = None
        self._ode_fxns = None

    def create_ode_fxn(self, is_parallel, fxn_names, fxn_scales=1.0):
        # Parse and check input
        assert fxn_names is not None or len(fxn_names) != 0
        if isinstance(fxn_names, str):
            fxn_names = [fxn_names]
        if isinstance(fxn_scales, (int, float)):
            fxn_scales = [fxn_scales]
        assert len(fxn_names) == len(fxn_scales), f"The number of metric function names {fxn_names} does not match the number of scales {fxn_scales}"

        for fname in fxn_names:
            if fname not in dir(self):
                metric_fxns = [
                    attr for attr in dir(Metrics)\
                        if attr[0] != '_' and callable(getattr(Metrics, attr))
                ]
                raise ValueError(f"Can only integrate metric functions, either add a new function to the Metrics class or use one of the following:\n\t{metric_fxns}")
        self._ode_fxns = [getattr(self, fname) for fname in fxn_names]
        self._ode_fxn_scales = {
            fxn.__name__ : scale for fxn, scale in zip(self._ode_fxns, fxn_scales)
        }

        if is_parallel:
            self.ode_fxn = self._parallel_ode_fxn
        else:
            self.ode_fxn = self._serial_ode_fxn


    def _parallel_ode_fxn(self, t, path, **kwargs):
        loss = 0
        variables = [torch.tensor([[torch.nan]], device=self.device) for i in range(3)]
        for fxn in self._ode_fxns:
            scale = self._ode_fxn_scales[fxn.__name__]
            ode_output = fxn(path=path, t=t, **kwargs)
            variables = [
                out if out is not None else var\
                    for var, out in zip(variables, ode_output[1:])
            ]
            loss = loss + scale*ode_output[0]
        return torch.concatenate([loss] + variables, dim=-1)


    def _serial_ode_fxn(self, t, path, **kwargs):
        loss = 0
        t = t.reshape(1, -1)
        for fxn in self._ode_fxns:
            scale = self._ode_fxn_scales[fxn.__name__]
            loss = loss + scale*fxn(path=path, t=t, **kwargs)[0]
        print("Combine other variables, see _parallel_ode_fxn")
        raise NotImplementedError
        return loss
    
    
    def update_ode_fxn_scales(self, **kwargs):
        for name, scale in kwargs.items():
            assert name in self._ode_fxn_scales
            self._ode_fxn_scales[name] = scale


    def _parse_input(
            self,
            geo_val=None,
            velocity=None,
            pes_val=None,
            force=None,
            path=None,
            t=None,
            path_output=None,
            requires_velocity=False,
            requires_energy=False,
            requires_force=False,
            requires_forceterms=False,
            fxn_name=None
            ):
        inp_velocity = velocity is not None or not requires_velocity
        inp_force = force is not None or not requires_force
        use_input = geo_val is not None and pes_val is not None
        use_input = use_input and inp_velocity and inp_force
        if use_input:
            return geo_val, velocity, pes_val, force
        
        if path_output is not None and path is not None:
            raise ValueError("Cannot call metric functions with both path != None and path_output != None")
        
        if path_output is not None:
            pout_velocity = path_output.velocity is not None or not requires_velocity
            pout_force = path_output.force is not None or not requires_force
            if not pout_velocity or not pout_velocity:
                message = f"When calling {fxn_name} and providing path_output the "
                if not pout_velocity:
                    message += "velocity "
                    if not pout_force:
                        message += "and force "
                else:
                    message += "force "
                raise ValueError(message + "must be provided in the PathOutput.")
            return path_output.geometric_path, path_output.velocity,\
                path_output.potential_path, path_output.force
        
        if path is not None:
            if t is None:
                raise ValueError("Must specify evaluation times for path when using path argument")
            path_output = path(t, return_velocity=requires_velocity, return_energy=requires_energy, return_force=requires_force, return_forceterms=requires_forceterms)
            return path_output.path_geometry, path_output.path_velocity, path_output.path_energy, path_output.path_force, path_output.path_forceterms
        
        message = f"Cannot parse input arguments to {fxn_name}, please use one of the following options\n"
        message += f"\t1) Provide geometric_path and potential path, and if needed velocity and/or force\n"
        message += f"\t2) Provide a PathOutput class\n"
        message += f"\t3) Provide the path calculator and the time(s) to be evaluated"
        raise ValueError(message)

    def E_geo(self, **kwargs):
        kwargs['requires_forceterms'] = True
        kwargs['requires_energy'] = True
        kwargs['requires_velocity'] = True
        kwargs['fxn_name'] = self.E_vre.__name__

        path_geometry, path_velocity, path_energy, path_force, path_forceterms = self._parse_input(**kwargs)
        
        Egeo = torch.linalg.norm(torch.einsum('bki,bi->bk', path_forceterms, path_velocity), dim=-1, keepdim=True)
        return Egeo, path_energy

    def E_vre(self, **kwargs):
        kwargs['requires_force'] = True
        kwargs['requires_energy'] = True
        kwargs['requires_velocity'] = True
        kwargs['fxn_name'] = self.E_vre.__name__

        path_geometry, path_velocity, path_energy, path_force, path_forceterms = self._parse_input(**kwargs)
        
        Evre = torch.linalg.norm(path_force, dim=-1, keepdim=True) * torch.linalg.norm(path_velocity, dim=-1, keepdim=True)
        return Evre, path_energy, path_force, path_velocity

    def E_pvre(self, **kwargs):
        kwargs['requires_force'] = True
        kwargs['requires_energy'] = True
        kwargs['requires_velocity'] = True
        kwargs['fxn_name'] = self.E_pvre.__name__

        path_geometry, path_velocity, path_energy, path_force, path_forceterms = self._parse_input(**kwargs)

        Epvre = torch.abs(torch.sum(path_velocity*path_force, dim=-1, keepdim=True))
        # Epvre = torch.abs(torch.sum(torch.einsum('bki,bi->bk', path_force, path_velocity), dim=-1, keepdim=True))

        return Epvre, path_energy #, path_force, path_velocity

    def E_pvre_vre(self, **kwargs):
        kwargs['requires_force'] = True
        kwargs['requires_energy'] = True
        kwargs['requires_velocity'] = True
        kwargs['fxn_name'] = self.E_pvre_vre.__name__
        
        path_geometry, path_velocity, path_energy, path_force, path_forceterms = self._parse_input(**kwargs)

        Evre = torch.linalg.norm(path_force, dim=-1, keepdim=True) * torch.linalg.norm(path_velocity, dim=-1, keepdim=True)
        Epvre = torch.abs(torch.sum(path_velocity*path_force, dim=-1, keepdim=True))
        return self.parameters['vre_scale'] * Evre + self.parameters['pvre_scale'] * Epvre

    def E_pvre_vre(self, **kwargs):
        kwargs['requires_force'] = True
        kwargs['requires_velocity'] = True
        kwargs['fxn_name'] = self.E_pvre_vre.__name__

        path_geometry, path_velocity, path_energy, path_force, path_forceterms = self._parse_input(**kwargs)

        vre = self.E_vre(force=force, velocity=velocity, **kwargs)
        pvre = self.E_pvre(force=force, velocity=velocity, **kwargs)
        return self.parameters['vre_scale']*vre + self.parameters['pvre_scale']*pvre


    def E_pvre_mag(self, **kwargs):
        kwargs['requires_force'] = True
        kwargs['requires_velocity'] = True
        kwargs['fxn_name'] = self.E_pvre.__name__

        path_geometry, path_velocity, path_energy, path_force, path_forceterms = self._parse_input(**kwargs)
        
        return torch.linalg.norm(path_velocity*path_force), path_energy, path_force, path_velocity

    
    def E(self, **kwargs):
        kwargs['requires_energy'] = True
        kwargs['fxn_name'] = self.E.__name__

        path_geometry, path_velocity, path_energy, path_force, path_forceterms = self._parse_input(**kwargs)

        return path_energy, path_energy, path_force, path_velocity


    def E_mean(self, **kwargs):
        kwargs['requires_energy'] = True
        kwargs['fxn_name'] = self.E_mean.__name__

        path_geometry, path_velocity, path_energy, path_force, path_forceterms = self._parse_input(**kwargs)
        mean_E = torch.mean(path_energy, dim=0, keepdim=True)
        return mean_E, mean_E, path_force, path_velocity



    def vre(self, **kwargs):
        kwargs['requires_force'] = True
        kwargs['requires_velocity'] = True
        kwargs['fxn_name'] = self.E_pvre.__name__
        path_geometry, path_velocity, path_energy, path_force, path_forceterms = self._parse_input(**kwargs)
        
        e_pvre = self.E_pvre(
            geo_val=path_geometry, velocity=path_velocity, pes_val=path_energy, force=path_force
        )
        e_vre = self.E_vre(
            geo_val=path_geometry, velocity=path_velocity, pes_val=path_energy, force=path_force
        )
        return e_vre - e_pvre, path_energy, path_force, path_velocity

    
    def F_mag(self, **kwargs):
        kwargs['requires_force'] = True
        kwargs['requires_energy'] = True
        kwargs['fxn_name'] = self.F_mag.__name__

        path_geometry, path_velocity, path_energy, path_force, path_forceterms = self._parse_input(**kwargs)

        return torch.linalg.norm(path_force, dim=-1, keepdim=True), path_energy, path_force, path_velocity