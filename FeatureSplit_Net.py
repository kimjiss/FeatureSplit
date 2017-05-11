import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import time

def accuracy(output, label):
    pred = output.data.max(1)[1]
    batch_size = label.size(0)
    correct = pred.eq(label.data).cpu().sum()
    return correct*(100.0 / batch_size)

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
        return F.log_softmax(x)

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

if __name__ == "__main__":

    iterCnt = 2000
    epochCnt = 10
    num_steps = 20
    train_phase = 1
    NUM_clusters = 4

    trainset = torchvision.datasets.MNIST(root='/home/jisu/Desktop/Data', train=True, download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))
                                          ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testset = torchvision.datasets.MNIST(root='/home/jisu/Desktop/Data', train=False, download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))
                                         ]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    net = FeatureSplitNet(NUM_clusters).cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    count = 0
    for epoch in range(epochCnt):
        for i, (inputs, labels) in enumerate(trainloader):
            count += 1
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            output = net(inputs)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()

            if count%200 == 199:
                test_loss = 0
                correct = 0
                start_time = time.time()
                for data, target in testloader:
                    data, target = Variable(data.cuda()), Variable(target.cuda())
                    out = net(data)
                    test_loss += F.nll_loss(out, target).data[0]
                    a = accuracy(out, target)
                    print a
                    pred = out.data.max(1)[1]
                    correct += pred.eq(target.data).cpu().sum()
                    test_loss = test_loss
                    test_loss /= len(testloader)  # loss function already averages over batch size
                print('\nTest set: Average loss: {:.8f}, Accuracy: {}/{} ({:.0f}%) elapsed time: {:.4f}\n'.format(
                    test_loss, correct, len(testloader.dataset),
                    100. * correct / len(testloader.dataset), time.time() - start_time))
