from .base_path import BasePath
from popcornn.tools import Images, wrap_positions


class LinearPath(BasePath):
    """
    Straight-line interpolation between reactant and product.

    No trainable parameters. Used as the base path that ``MLPpath``
    adds a learned correction on top of, and occasionally as a sanity
    baseline.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vec = self.final_position - self.initial_position

    def get_positions(self, time: float):
        """
        Evaluate the linear path at ``time``.

        Parameters
        ----------
        time : torch.Tensor
            Times in [0, 1]; shape ``[N, 1]``.

        Returns
        -------
        torch.Tensor
            Positions of shape ``[N, D]`` where ``D`` is the
            configuration dimensionality.
        """
        return self.initial_position + time * self.vec