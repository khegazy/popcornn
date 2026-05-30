import torch

class MinimaUpdate():
    """
    Plain SGD wrapper for relaxing a point to a local minimum of a
    potential. Used to refresh reactant/product positions when the
    user-supplied geometries aren't quite at minima of the potential
    being optimized against.
    """

    def __init__(self, potential, n_steps=10000, step_size=1e-2):
        """
        Parameters
        ----------
        potential : callable
            ``potential(positions)`` -> scalar energy.
        n_steps : int, default=10000
            Number of SGD steps per minimum.
        step_size : float, default=1e-2
            SGD learning rate.
        """
        self.potential = potential
        self.step_size = step_size
        self.n_steps = n_steps

    def find_minima(self, initial_positions=[]):
        """Relax each point in ``initial_positions`` and return the relaxed list."""
        self.minima = [
            self.find_minimum(torch.tensor(point)) for point in initial_positions
        ]
        return self.minima

    def find_minimum(self, point, log_frequency=1000):
        """
        Relax a single point to a local minimum.

        Parameters
        ----------
        point : torch.Tensor
            Starting configuration. A leading batch dim is added if
            missing and stripped before returning.
        log_frequency : int, default=1000
            Reserved; the in-loop logging hook is currently disabled.
        """
        # Adding batch dimension if point is a single point
        unsqueeze = False
        if len(point.shape) == 1:
            point = point.unsqueeze(0)
            unsqueeze = True
        
        point.requires_grad = True
        optimizer = torch.optim.SGD([point], lr=self.step_size)
        print(f"computing minima ... {point}")
        for step in range(self.n_steps):
            energy = torch.sum(self.potential(point))
            energy.backward()
            optimizer.step()
            #if step % log_frequency == 0:
            #    self.training_logger(step, self.potential(point))
        point.requires_grad = False
        
        if unsqueeze:
            point = point[0]   
        return point