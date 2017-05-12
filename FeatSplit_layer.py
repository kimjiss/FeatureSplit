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
        self.conv_outer = nn.Conv2d(self.out_channel/(self.nClusters + 1), self.out_channel/(self.nClusters + 1), 3, padding=1)
        index = range(self.in_channel / self.nClusters)
        random.shuffle(index)
        self.index = torch.from_numpy(np.asarray(index[:self.nIdx]))

    def forward(self, x):
        x_ = torch.chunk(x, self.nClusters, 1)
        x = [self.conv_inner[i](x_[i]) for i in range(self.nClusters)]
        global_x = torch.cat([torch.index_select(c, 1, Variable(self.index.cuda())) for c in x_], 1)
        x.append(self.conv_outer(global_x))
        x = torch.cat(x, 1)
        return x

    def make_layer_fsplit(self, in_channel, out_channel, num_clusters):
        layers = []
        for i in range(num_clusters):
            conv2d = nn.Conv2d(in_channel/num_clusters, out_channel/(num_clusters + 1), 3, padding=1)
            layers += [conv2d]
        return nn.ModuleList(layers)

class OnlySplit(nn.Module):
    def __init__(self, in_channel, cfg):
        super(OnlySplit, self).__init__()
        self.out_channel = cfg[1]
        self.n_cluster = cfg[2]
        self.conv_inner = self.make_layer_fsplit(in_channel, self.out_channel, self.n_cluster)
        print self.n_cluster
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

class OnlySplit_global(nn.Module):
    def __init__(self, in_channel, cfg):
        super(OnlySplit, self).__init__()
        self.out_channel = cfg[1]
        self.n_cluster = cfg[2]
        self.global_num = cfg[3]
        self.conv_inner = self.make_layer_fsplit(in_channel, self.out_channel, self.n_cluster)
        self.conv_outer = nn.Conv2d(self.in_channel, self.global_num * self.n_cluster,
                                    3, padding=1)

    def forward(self, x):
        x_ = torch.chunk(x, self.n_cluster, 1)
        x = [self.conv_inner[i](x_[i]) for i in range(self.n_cluster)]
        x = torch.cat(x, 1)
        return x

    def make_layer_fsplit(self, in_channel, out_channel, num_clusters):
        layers = []
        for i in range(num_clusters):
            conv2d = nn.Conv2d(in_channel / num_clusters, out_channel / num_clusters - self.global_num, 3, padding=1)
            layers += [conv2d]
        return nn.ModuleList(layers)

class shuffleSplit(nn.Module):
    def __init__(self, in_channel, cfg, batch_size, image_size):
        super(shuffleSplit, self).__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.in_channel = in_channel
        self.out_channel = cfg[1]
        self.n_cluster = cfg[2]
        self.conv_filters = self.make_layer_fsplit(in_channel, self.out_channel, self.n_cluster)
        self.mask = []
        for i in range(self.n_cluster):
            mask = np.zeros([batch_size, in_channel, image_size, image_size], dtype=np.float32)
            for j in range(in_channel / self.n_cluster):
                mask[:, j * self.n_cluster + i, :, :] = 1
            self.mask.append(torch.from_numpy(mask))
        # print self.mask[0].shape, self.mask[1].shape, self.mask[2].shape, self.mask[3].shape
        # print self.mask[0][3, 0, 1, 1], self.mask[0][3, 1, 1, 1], self.mask[0][3, 2, 1, 1]

    def forward(self, x):
        if self.mask[0].size(0) != x.size(0):
            self.mask = []
            for i in range(self.n_cluster):
                mask = np.zeros([x.size(0), x.size(1), x.size(2), x.size(3)], dtype=np.float32)
                for j in range(x.size(1) / self.n_cluster):
                    mask[:, j * self.n_cluster + i, :, :] = 1
                self.mask.append(torch.from_numpy(mask))
        x = [self.conv_filters[f](torch.mul(x, Variable(self.mask[f].cuda()))) for f in range(self.n_cluster)]
        x = torch.cat(x, 1)
        return x

    def make_layer_fsplit(self, in_channel, out_channel, num_clusters):
        layers = []
        for i in range(num_clusters):
            conv2d = nn.Conv2d(in_channel, out_channel/num_clusters, 3, padding=1)
            layers += [conv2d]
        return nn.ModuleList(layers)

class shuffleSplit_global(nn.Module):
    def __init__(self, in_channel, cfg, batch_size, image_size):
        super(shuffleSplit_global, self).__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.in_channel = in_channel
        self.out_channel = cfg[1]
        self.n_cluster = cfg[3]
        self.global_num = cfg[2]
        self.conv_filters = self.make_layer_fsplit(in_channel, self.out_channel, self.n_cluster)
        self.conv_outer = nn.Conv2d(self.in_channel, self.global_num * self.n_cluster,
                                     3, padding=1)
        print self.global_num
        self.mask = []
        for i in range(self.n_cluster):
            mask = np.zeros([batch_size, in_channel, image_size, image_size], dtype=np.float32)
            for j in range(in_channel / self.n_cluster):
                mask[:, j * self.n_cluster + i, :, :] = 1
            self.mask.append(torch.from_numpy(mask))
        # print self.mask[0].shape, self.mask[1].shape, self.mask[2].shape, self.mask[3].shape
        # print self.mask[0][3, 0, 1, 1], self.mask[0][3, 1, 1, 1], self.mask[0][3, 2, 1, 1]

    def forward(self, x):
        if self.mask[0].size(0) != x.size(0):
            self.mask = []
            for i in range(self.n_cluster):
                mask = np.zeros([x.size(0), x.size(1), x.size(2), x.size(3)], dtype=np.float32)
                for j in range(x.size(1) / self.n_cluster):
                    mask[:, j * self.n_cluster + i, :, :] = 1
                self.mask.append(torch.from_numpy(mask))
        x_ = [self.conv_filters[f](torch.mul(x, Variable(self.mask[f].cuda()))) for f in range(self.n_cluster)]
        x_.append(self.conv_outer(x))
        x = torch.cat(x_, 1)
        return x

    def make_layer_fsplit(self, in_channel, out_channel, num_clusters):
        layers = []
        for i in range(num_clusters):
            conv2d = nn.Conv2d(in_channel, out_channel/num_clusters - self.global_num, 3, padding=1)
            layers += [conv2d]
        return nn.ModuleList(layers)

























