import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import res
import time
from torchvision.models import resnet50
import jisu_util as utils
from FeatSplit_layer import featSplit

num_clusters = 4

# cfg = [['C', 40], ['C', 80], ['M'], ['C', 80], ['C', 160], ['M'], ['C', 160], ['C', 320], ['M'], ['C', 640], ['M']]
cfg = [['C', 40], ['F', 80, num_clusters], ['M'], ['F', 80, num_clusters], ['F', 160, num_clusters]
    , ['M'], ['F', 160, num_clusters], ['F', 320, num_clusters], ['M'], ['F', 640, num_clusters], ['M']]
# cfg = [32, 64, 128, 'M', 256, 256, 512, 'M']

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

class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()

    def forward(self, x):
        return x

class CNN(nn.Module):
    def __init__(self, cfg):
        super(CNN, self).__init__()
        self.features = self.make_layers(cfg, batch_norm=True)
        self.classifier = nn.Sequential(nn.Linear(6 * 6 * 640, 2048), nn.Linear(2048, 256), nn.Linear(256, 10))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x)

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v[0] == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v[0] == 'F':
                featSL_layer = featSplit(in_channels, v)
                if batch_norm:
                    layers += [featSL_layer, nn.BatchNorm2d(v[1]), nn.ReLU(inplace=True)]
                else:
                    layers += [featSL_layer, nn.ReLU(inplace=True)]
                in_channels = v[1]
            elif v[0] == 'C':
                conv2d = nn.Conv2d(in_channels, v[1], kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v[1]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v[1]
        return nn.Sequential(*layers)


def train(model, trainloader, testloader, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()


    model.train()
    end = time.time()
    for i, (input, label) in enumerate(trainloader):
        # Measure data loading time
        data_time.update(time.time() - end)

        input = Variable(input.cuda())
        label = Variable(label.cuda())

        # Compute output
        out = model(input)
        loss = F.nll_loss(out, label)

        # Update results
        accuracy = utils.accuracy(out, label)
        acc.update(accuracy)
        losses.update(loss.data[0])

        # Compute gradients and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                    epoch, i, len(trainloader), batch_time=batch_time,
                    data_time=data_time, loss=losses, acc=acc))

def validate(testloader, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    end = time.time()
    for i, (input, label) in enumerate(testloader):
        input = Variable(input.cuda(), volatile=True)
        label = Variable(label.cuda(), volatile=True)

        # compute output
        output = model(input)
        loss = F.nll_loss(output, label)

        # measure accuracy and record loss
        accuracy = utils.accuracy(output, label)
        losses.update(loss.data[0], input.size(0))
        acc.update(accuracy)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                i, len(testloader), batch_time=batch_time, loss=losses,
                acc=acc))

    print(' * Total Accuracy {acc.avg:.3f}'
          .format(acc=acc))

    return acc.avg


def main():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.STL10(root='/home/jisu/Desktop/Data', split='train', transform=transform, download=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)
    testset = torchvision.datasets.STL10(root='/home/jisu/Desktop/Data', split='test', transform=transform, download=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True, num_workers=2)

    # transform = transforms.Compose([transforms.ToTensor()])
    #                                 # , transforms.Scale((96, 96))])
    # trainset = torchvision.datasets.CIFAR10(root='/home/jisu/Desktop/Data', train=True, transform=transform,
    #                                       download=True)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    # testset = torchvision.datasets.CIFAR10(root='/home/jisu/Desktop/Data', train=False, transform=transform,
    #                                      download=True)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)

    # cfg = {'num_classes':10}
    # model = res.resnet50(pretrained=False, **cfg)

    model = CNN(cfg).cuda()
    print model

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    best_acc = 0

    for epoch in range(1, 20):
        train(model, trainloader, testloader, optimizer, epoch)
        valid_acc = validate(testloader, model)

        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'CNN',
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            }, is_best)

if __name__ == '__main__':
    main()