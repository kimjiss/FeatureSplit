import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import math
import time
import res


cfg = {
    'VGG':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'FSplit':[]
}

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class VGGNet16(nn.Module):
    def __init__(self, features):
        super(VGGNet16, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # print x.size()
        x = x.view(x.size(0), -1)
        # print x.size()
        x = self.classifier(x)
        return F.log_softmax(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
def vgg16(**kwargs):
    model = VGGNet16(make_layers(cfg['VGG']), **kwargs)
    return model

class FeatureSplitNet(nn.Module):
    def __init__(self, num_clusters):
        super(FeatureSplitNet, self).__init__()
        self.nClusters = num_clusters
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.SplitLayerConv1 = nn.Conv2d(32/self.nClusters, 32/self.nClusters, 5)
        self.fc1 = nn.Linear((32/self.nClusters * (self.nClusters + 1)) * 10 * 10, 1024)
        self.fc2 = nn.Linear(1024, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.SplitLayer(x, 32, self.SplitLayerConv1)
        x = x.view(-1, (32/self.nClusters * (self.nClusters + 1)) * 10 * 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x)

    def SplitLayer(self, x, in_channel, conv_func):
        x_ = torch.chunk(x, self.nClusters, 1)
        x = [conv_func(c) for c in x_]
        index = range(in_channel/self.nClusters)
        random.shuffle(index)
        index = torch.from_numpy(np.asarray(index[:len(index)/self.nClusters]))
        global_x = torch.cat([torch.index_select(c, 1, Variable(index.cuda())) for c in x_], 1)
        x.append(conv_func(global_x))
        x = torch.cat(x, 1)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def train(trainloader, model, epoch):

    acc = AverageMeter()
    losses = AverageMeter()
    for ep in range(epoch):
        running_loss = 0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = model(inputs)
            # print outputs, labels
            loss = F.nll_loss(outputs, labels)
            # print loss.data[0]

            acc_result = accuracy(outputs, labels)
            # print acc_result
            acc.update(acc_result, inputs.size(0))
            losses.update(loss.data[0], inputs.size(0))
            loss.backward()
            optimizer.step()
            if i%5 == 4:
                print '[{}, {}] loss: {:.4f}, training accuracy: {:.2f}%'.format(ep, i, losses.avg, acc.avg)

def accuracy(output, target):
    pred = output.data.max(1)[1]
    # print pred.size(), target.data.size()
    batch_size = pred.size()[0]
    correct = pred.eq(target.data).cpu().sum()
    return correct*(100.0/batch_size)



transform = transforms.Compose([#transforms.Scale(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
trainset = torchvision.datasets.CIFAR10(root='/home/jisu/Desktop/Data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/home/jisu/Desktop/Data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# net = vgg16()
cfg = {'num_classes':10}
net = res.resnet50(pretrained=False, **cfg)
net = net.cuda()

optimizer = optim.Adam(net.parameters(), lr=0.001)
epochCnt = 20
# criterion = F.nll_loss()
count = 0

train(trainloader, net, epochCnt)

# for epoch in range(epochCnt):
#     running_loss = 0
#     for i, (inputs, labels) in enumerate(trainloader):
#         count += 1
#         inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
#         optimizer.zero_grad()
#         outputs = net(inputs)
#
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.data[0]
#         if count%100 == 99:
#             # print '[%d, %d] loss: %.4f, '
#             print "succ"
#

















