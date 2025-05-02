from .mlp import MLPpath
from .linear import LinearPath

path_dict = {
    "mlp" : MLPpath,
    "linear" : LinearPath,
}

def get_path(name, **config):
    name = name.lower()
    if name not in path_dict:
        raise ValueError(f"Cannot get path {name}, can only handle paths {path_dict.keys()}")
    path = path_dict[name](**config)
    
    return path 
