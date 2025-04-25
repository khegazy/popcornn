from .mlp import MLPpath
from .mlp_dist import MLPDISTpath
from .conv import CONVpath
from .attention import Attentionpath
from .b_spline import BSpline
from .elastic_band import ElasticBand
from .bezier import Bezier

path_dict = {
    "elastic_band" : ElasticBand,
    "mlp" : MLPpath,
    "mlp_dist" : MLPDISTpath,
    "conv" : CONVpath,
    "attention" : Attentionpath,
    "bspline" : BSpline,
    "bezier" : Bezier,
}

def get_path(name, **config):
    name = name.lower()
    if name not in path_dict:
        raise ValueError(f"Cannot get path {name}, can only handle paths {path_dict.keys()}")
    path = path_dict[name](**config)
    
    return path 
