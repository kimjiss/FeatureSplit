import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from FeatureSplit_Net import FeatureSplitNet
import numpy as np
import random
import time

iterCnt = 2000
epochCnt = 10
num_steps = 20
train_phase = 1
NUM_clusters = 8

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

def accuracy(output, label):
    pred = output.data.max(1)[1]
    print pred
    batch_size = label.size(0)
    correct = pred.eq(label.data).cpu().sum()
    return correct*(100/batch_size)


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

