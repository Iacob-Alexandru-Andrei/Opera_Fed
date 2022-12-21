import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# import numpy as np
# from copy import deepcopy
# import torch


# class PowerPropMixin:
#     def __init__(self, *args, **kwargs):
#         self.power_prop_alpha = kwargs.pop("power_prop_alpha", 2.0)
#         super().__init__(*args, **kwargs)

#     def forward(self, *args, **kwargs):

#         with torch.no_grad():
#             state_dict = self.state_dict()
#             for key in state_dict.keys():
#                 state_dict[key] *= torch.abs(state_dict[key]).float_power(
#                     self.power_prop_alpha - 1
#                 )

#         res = super().forward(*args, **kwargs)

#         with torch.no_grad():
#             state_dict = self.state_dict()
#             for key in state_dict.keys():
#                 state_dict[key] /= torch.abs(state_dict[key]).float_power(
#                     (self.power_prop_alpha - 1) / self.power_prop_alpha
#                 )

#         return res
