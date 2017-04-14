import torch
import torch.nn as nn
import torch.functional as F
import random
from torch.autograd import Variable
import numpy as np

class featSplit(nn.Module):
    def __init__(self, in_channel, cfg):
        super(featSplit, self).__init__()
        self.nClusters = cfg[2]
        self.out_channel = cfg[1]
        self.in_channel = in_channel
        self.nIdx = self.out_channel / (self.nClusters + 1) / self.nClusters
        self.conv_inner = self.make_layer_fsplit(self.in_channel, self.out_channel, self.nClusters)
        self.conv_outter = nn.Conv2d(self.out_channel/(self.nClusters + 1), self.out_channel/(self.nClusters + 1), 3, padding=1)
        index = range(self.in_channel / self.nClusters)
        random.shuffle(index)
        self.index = torch.from_numpy(np.asarray(index[:self.nIdx]))

    def forward(self, x):
        x_ = torch.chunk(x, self.nClusters, 1)
        x = [self.conv_inner[i](x_[i]) for i in range(self.nClusters)]
        global_x = torch.cat([torch.index_select(c, 1, Variable(self.index.cuda())) for c in x_], 1)
        x.append(self.conv_outter(global_x))
        x = torch.cat(x, 1)
        return x

    def make_layer_fsplit(self, in_channel, out_channel, num_clusters):
        layers = []
        for i in range(num_clusters):
            conv2d = nn.Conv2d(in_channel/num_clusters, out_channel/(num_clusters + 1), 3, padding=1)
            layers += [conv2d]
        return nn.ModuleList(layers)