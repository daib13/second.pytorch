"""
EPointPillars fork from PointPillars.
Code written by Bin Dai, 2021
Licensed under MIT License [see LICENSE].
"""

import torch 
from torch import nn 
from torch.nn import functional as F 

from second.pytorch.utils import get_paddings_indicator 
from torchplus.nn import Empty 
from torchplus.tools import change_default_args

class EPFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False,
                 p=4):
        """
        Equivariant Pillar Feature Layer
        The Equivariant Pillar Feature Layer could be composed of a series of these layers. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        :param p: <int>. The size of the symmetry group.
        """

        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if use_norm:
            BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty 
            Linear = change_default_args(bias=True)(nn.Linear)
        
        self.linear = Linear(in_channels, self.units)
        self.norm = BatchNorm1d(self.units)