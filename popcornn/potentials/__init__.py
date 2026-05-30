

def get_potential(name, **kwargs):
    """
    Construct a potential by name.

    Imports the wrapper class lazily so installing one MLIP doesn't
    force you to install all of them.

    Parameters
    ----------
    name : str
        Case-insensitive. See ``docs/potentials.md`` for the full list
        and per-potential extra keys.
    **kwargs
        Forwarded to the potential class. ``BasePotential`` requires
        ``images``, ``device``, ``dtype``; subclasses add their own
        (e.g. UMA needs ``model_name`` and ``task_name``, MACE needs
        ``model_path``).

    Returns
    -------
    BasePotential
    """
    name = name.lower()
    if name == "wolfe_schlegel":
        from .wolfe_schlegel import WolfeSchlegel
        return WolfeSchlegel(**kwargs)
    elif name == "muller_brown":
        from .muller_brown import MullerBrown
        return MullerBrown(**kwargs)
    elif name == "schwefel":
        from .schwefel import Schwefel
        return Schwefel(**kwargs)
    elif name == "constant":
        from .constant import Constant
        return Constant(**kwargs)
    elif name == "sphere":
        from .sphere import Sphere
        return Sphere(**kwargs)
    elif name == "lennard_jones":
        from .lennard_jones import LennardJones
        return LennardJones(**kwargs)
    elif name == "repel":
        from .repel import RepelPotential
        return RepelPotential(**kwargs)
    elif name == "morse":
        from .morse import MorsePotential
        return MorsePotential(**kwargs)
    elif name == "harmonic":
        from .harmonic import HarmonicPotential
        return HarmonicPotential(**kwargs)
    elif name == "mace":
        from .mace import MacePotential
        return MacePotential(**kwargs)
    elif name == "newtonnet":
        from .newtonnet import NewtonNetPotential
        return NewtonNetPotential(**kwargs)
    elif name == "ani":
        from .ani import AniPotential
        return AniPotential(**kwargs)
    elif name == "leftnet" or name == 'left':
        from .leftnet import LeftNetPotential
        return LeftNetPotential(**kwargs)
    elif name == "orb":
        from .orb import OrbPotential
        return OrbPotential(**kwargs)
    elif name == "escaip":
        from .escaip import EScAIPPotential
        return EScAIPPotential(**kwargs)
    elif name == "uma":
        from .uma import UMAPotential
        return UMAPotential(**kwargs)
    else:
        raise ValueError(f"Cannot handle potential type {name}")
