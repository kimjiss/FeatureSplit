import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from math import floor

class DropDepthOut(nn.Module):
    def __init__(self, in_channel, cfg):
        super(DropDepthOut, self).__init__()
        # self.training = True
        self.dropout_ratio = 0.5
        self.in_channel = in_channel
        self.out_channel = cfg[1]
        self.conv = nn.Conv2d(in_channel, self.out_channel, 3, padding=1)

    def forward(self, x):
        if self.training == True:
            mask_tensor =np.ones([x.size(0), x.size(1), 1, 1], dtype=np.float32)
            for i in range(x.size(0)):
                # print np.random.choice(x.size(1), int(floor(x.size(1) * self.dropout_ratio)), replace=False)
                mask_tensor[i, np.random.choice(x.size(1), int(floor(x.size(1) * self.dropout_ratio)), replace=False), 0, 0] = 0.0
            mask_tensor = torch.from_numpy(mask_tensor)
            # print mask_tensor.size()
            mask_tensor = Variable(mask_tensor.expand_as(x).cuda())
            # print mask_tensor.size(), x.size()
            x = torch.mul(x, mask_tensor)
            return self.conv(x)
        else:
            x = torch.mul(self.conv(x), self.dropout_ratio)
            return x