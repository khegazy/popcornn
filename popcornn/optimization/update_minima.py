import torch

class MinimaUpdate():
    def __init__(self, potential, n_steps=10000, step_size=1e-2):
        self.potential = potential
        self.step_size = step_size
        self.n_steps = n_steps

    def find_minima(self, initial_positions=[]):
        self.minima = [
            self.find_minimum(torch.tensor(point)) for point in initial_positions
        ]
        return self.minima

    def find_minimum(self, point, log_frequency=1000):
        """
        loop for finding minima
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