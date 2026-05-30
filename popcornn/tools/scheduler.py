import numpy as np

class SchedulerBase:
    """
    Step-counted scheduler base class.

    Subclasses implement ``_get_closed_form`` to produce a value as a
    function of the current step. The optimizer steps one of these per
    iteration; ``get_value`` returns the current scheduled value (used
    to multiply ``path_integrand_scales`` etc.).
    """

    def __init__(self, value=1.0, current_step=-1):
        """
        Parameters
        ----------
        value : float, default=1.0
            Multiplied into the scheduled output.
        current_step : int, default=-1
            Internal step counter. Pre-increments once on
            construction so the first ``step()`` lands on step 1.
        """
        self.value = value
        self.current_step = current_step
        self.current_step += 1

    def step(self):
        """Advance the step counter by one. Call once per iteration."""
        self.current_step += 1

    def get_value(self):
        """
        Return the current scheduled value. Falls back to ``value``
        when the subclass doesn't define a closed form.
        """
        if hasattr(self, '_get_closed_form'):
            return self._get_closed_form()
        else:
            return self.value

class Linear(SchedulerBase):
    """Linear interpolation from ``start_value`` to ``end_value`` over ``last_step`` steps."""

    def __init__(self, start_value, end_value, last_step, **kwargs):
        self.start_value = start_value
        self.end_value = end_value
        self.last_step = last_step - 1
        self.delta_value = (self.end_value - self.start_value) / self.last_step
        super().__init__(**kwargs)
    
    def _get_closed_form(self):
        step = min(self.current_step, self.last_step)
        update = self.start_value + self.delta_value * step
        return self.value * update 
    
class Cosine(SchedulerBase):
    """Half-cosine anneal from ``start_value`` to ``end_value`` over ``last_step`` steps."""

    def __init__(self, start_value, end_value, last_step, **kwargs):
        self.start_value = start_value
        self.end_value = end_value
        self.last_step = last_step - 1
        self.delta = self.end_value - self.start_value
        self.freq = np.pi/self.last_step
        super().__init__(**kwargs)
    
    def _get_closed_form(self):
        step = min(self.current_step, self.last_step)
        return self.value * (self.end_value - self.delta * (1 + np.cos(step * self.freq )) / 2)
    

SCHEDULER_DICT = {
    'linear' : Linear,
    'cosine' : Cosine,
}

def get_schedulers(scheduler_params):
    """
    Build a dict of name-keyed schedulers from a config dict.

    ``scheduler_params`` looks like ``{"<term-name>": {"name": "cosine",
    "start_value": ..., ...}}``. ``None`` returns an empty dict, which
    is the no-scheduling case.
    """
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