import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class OnlySplit(nn.Module):
    def __init__(self, in_channel, cfg):
        super(OnlySplit, self).__init__()
        self.out_channel = cfg[1]
        self.n_cluster = cfg[2]
        self.conv_inner = self.make_layer_fsplit(in_channel, self.out_channel, self.n_cluster)
    def forward(self, x):
        x_ = torch.chunk(x, self.n_cluster, 1)
        x = [self.conv_inner[i](x_[i]) for i in range(self.n_cluster)]
        x = torch.cat(x, 1)
        return x
    def make_layer_fsplit(self, in_channel, out_channel, num_clusters):
        layers = []
        for i in range(num_clusters):
            conv2d = nn.Conv2d(in_channel/num_clusters, out_channel/num_clusters, 3, padding=1)
            layers += [conv2d]
        return nn.ModuleList(layers)