from .mlp import MLPpath
from .linear import LinearPath
from .soft_one_hot import SoftOneHotPath
from .planted_softmax import PlantedSoftmaxPath
from .softmax import SoftMaxPath

path_dict = {
    "mlp" : MLPpath,
    "linear" : LinearPath,
    "soft_one_hot" : SoftOneHotPath,
    "planted_softmax" : PlantedSoftmaxPath,
    "softmax" : SoftMaxPath
}

def get_path(name, **config):
    """
    Construct a path representation by name.

    Parameters
    ----------
    name : str
        Key in ``path_dict``. Case-insensitive. Currently ``"mlp"``,
        ``"linear"``, or ``"soft_one_hot"``.
    **config
        Forwarded to the path class. ``BasePath`` requires ``images``,
        ``device``, ``dtype``; subclasses add their own (e.g. MLP takes
        ``width``, ``depth``, ``activation``).

    Returns
    -------
    BasePath
        Instantiated path. Has no potential set yet — call
        ``path.set_potential(...)`` before evaluating energies/forces.
    """
    name = name.lower()
    if name not in path_dict:
        raise ValueError(f"Cannot get path {name}, can only handle paths {path_dict.keys()}")
    path = path_dict[name](**config)

    return path
