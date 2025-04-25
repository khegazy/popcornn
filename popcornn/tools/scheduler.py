import math
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SchedulerBase:
    def __init__(self, value=1.0, last_epoch=-1):
        self.value = value
        self.last_epoch = last_epoch
        self.last_epoch += 1

    def step(self):
        self.last_epoch += 1
    
    def get_value(self):
        if hasattr(self, '_get_closed_form'):
            return self._get_closed_form()
        else:
            return self.value

class Linear(SchedulerBase):
    def __init__(self, start_factor, end_factor, total_iters, **kwargs):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(**kwargs)
    
    def _get_closed_form(self):
        return self.value * (self.start_factor + (self.end_factor - self.start_factor) * min(self.last_epoch, self.total_iters) / self.total_iters)
    
class Cosine(SchedulerBase):
    def __init__(self, start_factor, end_factor, total_iters, **kwargs):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(**kwargs)
    
    def _get_closed_form(self):
        # return self.eta_min + (self.value - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
        return self.value * (self.end_factor + (self.start_factor - self.end_factor) * (1 + math.cos(math.pi * self.last_epoch / self.total_iters)) / 2)
    

SCHEDULER_DICT = {
    'linear' : Linear,
    'cosine' : Cosine,
    # 'reduce_on_plateau' : ReduceOnPlateau,
    # 'increase_on_plateau' : IncreaseOnPlateau 
}

def get_schedulers(scheduler_params):
    schedulers = {}
    if scheduler_params is None:
        return schedulers
    
    for name, param_dict in scheduler_params.items():
        assert 'name' in param_dict, f"Must specify name of scheduler: {list(SCHEDULER_DICT.keys())}"
        assert param_dict['name'].lower() in SCHEDULER_DICT,\
            f"Cannot find scheduler {param_dict['name']}, options are {list(SCHEDULER_DICT.keys())}"
        sched_name = param_dict.pop('name').lower()
        schedulers[name] = SCHEDULER_DICT[sched_name](**param_dict)
    return schedulers