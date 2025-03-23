from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone

def parameters(net_path):
    params = TrackerParams()
    params.debug = 0
    params.visualization = False
    params.use_gpu = True
    params.net = NetWithBackbone(net_path=net_path,
                                 use_gpu=params.use_gpu)
    return params

# def parameters():
#     params = TrackerParams()
#     params.debug = 0
#     params.visualization = False
#     params.use_gpu = True
#     params.net = NetWithBackbone(net_path='',
#                                  use_gpu=params.use_gpu)
#     return params