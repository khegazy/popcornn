from typing import Any
from collections import defaultdict
import torch


class LossBase():
    def __init__(self, weight_scale=None) -> None:
        self.weight_scale = weight_scale
        self.iteration = None
        self.time_midpoint = None
    
    def update_parameters(self, **kwargs):
        if 'weight' in kwargs:
            self.weight_scale = kwargs['weight']
        if 'iteration' in kwargs:
            self.iteration = torch.tensor([kwargs['iteration']])
        # Find the center of the path in time
        if 'integral_output' in kwargs:
            self.time_midpoint = kwargs['integral_output'].t_optimal[:,0]
            t_idx = len(self.time_midpoint)//2
            if len(self.time_midpoint) % 2 == 1:
                self.time_midpoint = self.time_midpoint[t_idx]
            else:
                self.time_midpoint = self.time_midpoint[t_idx-1] + self.time_midpoint[t_idx]
                self.time_midpoint = self.time_midpoint/2.

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
        ax.set_title(str(self.time_midpoint))
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
        self.time_midpoint = 0.5

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

    def _inv_weights(self, time, time_init, time_final):
        envelope = self.envelope_fxn(time, time_init, time_final)
        return 1./(1 + self.weight_scale*envelope)
    
    def _guass_envelope(self, time, time_init, time_final):
        mask = time < self.time_midpoint
        # Left side
        time_left = time[mask]
        if len(time_left) > 0:
            left = torch.exp(-1/(self.variance_scale + 1e-10)\
                *((self.time_midpoint - time_left)*4\
                /(time_init - self.time_midpoint))**2
            )
            time_left = (time_left - time_left[0])\
                /(self.time_midpoint - time_left[0])
            left = left - (left[0] - time_left*left[0])
        else:
            left = None
        # Right side
        time_right = time[torch.logical_not(mask)]
        if len(time_right) > 0:
            right = torch.exp(-1/(self.variance_scale + 1e-10)\
                *((self.time_midpoint - time_right)*4\
                /(time_final - self.time_midpoint))**2)
            time_right = (time_right - time_right[-1])\
                /(self.time_midpoint - time_right[-1])
            right = right - (right[-1] - time_right*right[-1])
        else:
            right = None
        
        if left is None:
            return right
        elif right is None:
            return left
        else:
            return torch.concatenate([left, right])
    
    def _sine_envelope(self, time, time_init, time_final):
        mask = time < self.time_midpoint
        # Left side
        time_left = time[mask]
        if len(time_left) > 0:
            left = (1 - torch.cos(
                (time_left - time_init)*torch.pi/((self.time_midpoint - time_init))
            ))/2.
        else:
            left = None
        # Right side
        time_right = time[torch.logical_not(mask)]
        if len(time_right) > 0:
            right = (1 + torch.cos(
                (time[torch.logical_not(mask)] - self.time_midpoint)\
                    *torch.pi/((time_final - self.time_midpoint))
            ))/2.
        else:
            right = None

        if left is None:
            return right
        elif right is None:
            return left
        else:
            return torch.concatenate([left, right])

    def _poly_envolope(self, time, time_init, time_final):
        mask = time < self.time_midpoint
        # Left side
        time_left = time[mask]
        if len(time_left) > 0: 
            left = torch.abs(
                (time_left - time_init)/((self.time_midpoint - time_init))
            )**self.order
        else:
            left = None
        # Right side
        time_right = time[torch.logical_not(mask)]
        if len(time_right) > 0:
            right = torch.abs((time[torch.logical_not(mask)] - time_final)\
                /(time_final - self.time_midpoint))**self.order
        else:
            right = None
        
        if left is None:
            return right
        elif right is None:
            return left
        else:
            return torch.abs(torch.concatenate([left, right]))

    def _sine_gauss_envelope(self, time, time_init, time_final):
        guass_envelope = self._guass_envelope(time, time_init, time_final)
        sine_envelope = self._sine_envelope(time, time_init, time_final)
        return guass_envelope*sine_envelope


    def _butter_envelope(self, time, time_init, time_final):
        mask = time < self.time_midpoint
        # Left side
        time_left = time[mask]
        if len(time_left) > 0: 
            dt = self.time_midpoint - time_left
            left = 1./torch.sqrt(1 + (dt*2/(self.time_midpoint - time_init))**self.order)
        else:
            left = None
        # Right side
        time_right = time[torch.logical_not(mask)]
        if len(time_right) > 0:
            dt = time_right - self.time_midpoint
            right = 1./torch.sqrt(1 + (dt*2/(self.time_midpoint - time_init))**self.order)
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
    def __init__(self, device, save_energy_force=False):
        self.save_energy_force = save_energy_force
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
        
        self._get_required_variables()


    def _get_required_variables(self):
        assert self._ode_fxns is not None
        self.required_variables = defaultdict(lambda : False)
        for fxn in self._ode_fxns:
            for var in fxn(get_required_variables=True):
                self.required_variables[f"requires_{var}"] = True
    
    def add_required_variable(self, variable_name):
        self.required_variables[variable_name] = True

    def _parallel_ode_fxn(self, eval_time, path, **kwargs):
        loss = 0
        variables = {}
        for fxn in self._ode_fxns:
            scale = self._ode_fxn_scales[fxn.__name__]
            ode_loss, ode_variables = fxn(
                eval_time=eval_time,
                path=path,
                **self.required_variables,
                **variables,
                **kwargs
            )
            variables.update(ode_variables)
            loss = loss + scale*ode_loss
        
        if self.save_energy_force:
            nans = torch.stack([torch.tensor([torch.nan], device=self.device)]*len(variables['time']))
            keep_variables = [
                variables[name] if name in variables and variables[name] is not None else nans\
                    for name in ['energy', 'force']
            ]
            
            loss = torch.concatenate([loss] + keep_variables, dim=-1)

        del variables
        return loss

    def _serial_ode_fxn(self, time, path, **kwargs):
        loss = 0
        time = time.reshape(1, -1)
        for fxn in self._ode_fxns:
            scale = self._ode_fxn_scales[fxn.__name__]
            loss = loss + scale*fxn(path=path, time=time, **kwargs)[0]
        print("Combine other variables, see _parallel_ode_fxn")
        raise NotImplementedError
        return loss
    
    
    def update_ode_fxn_scales(self, **kwargs):
        for name, scale in kwargs.items():
            assert name in self._ode_fxn_scales
            self._ode_fxn_scales[name] = scale


    def _parse_input(
            self,
            eval_time,
            path,
            time=None,
            position=None,
            velocity=None,
            energy=None,
            energyterms=None,
            force=None,
            forceterms=None,
            requires_velocity=False,
            requires_energy=False,
            requires_energyterms=False,
            requires_force=False,
            requires_forceterms=False,
        ):
        
        # Do input and previous time match
        time_match = time is not None\
            and (time.shape == eval_time.shape\
                 and torch.allclose(time, eval_time, atol=1e-10)
            )

        # Is energy missing and required 
        requires_energy = requires_energy and energy is None
        requires_energyterms = requires_energyterms and energyterms is None
        missing_any_energy = requires_energy or requires_energyterms
        
        # Is force missing and required 
        requires_force = requires_force and force is None
        requires_forceterms = requires_forceterms and forceterms is None
        missing_any_force = requires_force or requires_forceterms

        # We must evaluate path if time do not match, or, force or energy is missing
        path_output = None
        if not time_match or missing_any_energy or missing_any_force:
            path_output = path(
                eval_time,
                return_velocity=requires_velocity,
                return_energy=requires_energy, 
                return_energyterms=requires_energyterms, 
                return_force=requires_force,
                return_forceterms=requires_forceterms
            )
            time = eval_time
            velocity = velocity if path_output.velocity is None\
                else path_output.velocity
            energy = energy if path_output.energy is None\
                else path_output.energy
            energyterms = energyterms if path_output.energyterms is None\
                else path_output.energyterms
            force = force if path_output.force is None\
                else path_output.force
            forceterms = forceterms if path_output.forceterms is None\
                else path_output.force_terms

        else:
           # Calculate velocity if missing and required
            if requires_velocity and velocity is None:
                velocity = path.calculate_velocity(time)
                requires_velocity = False
            
        return {
            'time' : time,
            'position' : position,
            'velocity' : velocity,
            'energy' : energy,
            'energyterms' : energyterms,
            'force' : force,
            'forceterms' : forceterms
        }


    def E_geo(self, get_required_variables=False, **kwargs):
        if get_required_variables:
            return ('forceterms', 'velocity')
        variables = self._parse_input(**kwargs)
        
        projection = torch.einsum(
            'bki,bi->bk',
            variables['forceterms'],
            variables['velocity']
        )
        Egeo = torch.linalg.norm(projection, dim=-1, keepdim=True)
        return Egeo, variables


    def E_vre(self, get_required_variables=False, **kwargs):
        if get_required_variables:
            return ('force', 'velocity')
        variables = self._parse_input(**kwargs)
        
        F = torch.linalg.norm(variables['force'], dim=-1, keepdim=True)
        V = torch.linalg.norm(variables['velocity'], dim=-1, keepdim=True)
        Evre = F*V
        return Evre, variables


    def E_pvre(self, get_required_variables=False, **kwargs):
        if get_required_variables:
            return ('force', 'velocity') 
        variables = self._parse_input(**kwargs)

        overlap = torch.sum(
            variables['velocity']*variables['force'],
            dim=-1,
            keepdim=True
        )
        Epvre = torch.abs(overlap)
        return Epvre, variables


    def E_pvre_mag(self, get_required_variables=False, **kwargs):
        if get_required_variables:
            return ('force', 'velocity') 
        variables = self._parse_input(**kwargs)
        
        Epvre_mag = torch.linalg.norm(variables['velocity']*variables['force'])
        return Epvre_mag, variables

    
    def E(self, get_required_variables=False, **kwargs):
        if get_required_variables:
            return ('energy') 
        variables = self._parse_input(**kwargs)
        
        return variables['energy'], variables


    def E_mean(self, get_required_variables=False, **kwargs):
        if get_required_variables:
            return ('energy',) 
        variables = self._parse_input(**kwargs)
        
        mean_E = torch.mean(variables['energy'], dim=0, keepdim=True)
        return mean_E, variables


    def vre(self, get_required_variables=False, **kwargs):
        if get_required_variables:
            return (
                *self.E_pvre(get_required_variables=True),
                *self.E_vre(get_required_variables=True)
            ) 
        variables = self._parse_input(**kwargs)
        
        Epvre = self.E_pvre(**variables)
        Evre = self.E_vre(**variables)
        return Evre - Epvre, variables

    
    def F_mag(self, get_required_variables=False, **kwargs):
        if get_required_variables:
            return ('force',)
        variables = self._parse_input(**kwargs)

        Fmag = torch.linalg.norm(variables['force'], dim=-1, keepdim=True)
        return Fmag, variables