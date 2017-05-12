import argparse

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
import jisu_util
from FeatSplit_layer import featSplit, OnlySplit, OnlySplit_global, shuffleSplit, shuffleSplit_global
from dropdepthout import DropDepthOut

parser = argparse.ArgumentParser(description='CIFAR100 training', add_help=False)
parser.add_argument('--datadir', default='/home/david/FeatureSplit/data/')
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--print-freq', '-p', default=10, type=int)
parser.add_argument('--resume', default='', type=str, metavar='PATH')
parser.add_argument('--numcluster', '-nc', default=8, type=int)
parser.add_argument('--global_num', '-gc', default=2, type=int)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

class CNN(nn.Module):
    def __init__(self, cfg, arg):
        super(CNN, self).__init__()
        self.features = self.make_layers(cfg, arg, batch_norm=True)
        self.classifier = nn.Sequential(nn.Linear(8 * 8 * 320, 2048), nn.Linear(2048, 256), nn.Linear(256, 100))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x)

    def make_layers(self, cfg, arg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v[0] == 'ImageSize':
                imagesize = v[1]
            elif v[0] == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                imagesize /= 2
            elif v[0] == 'F':
                featSL_layer = featSplit(in_channels, v)
                if batch_norm:
                    layers += [featSL_layer, nn.BatchNorm2d(v[1]), nn.ReLU(inplace=True)]
                else:
                    layers += [featSL_layer, nn.ReLU(inplace=True)]
                in_channels = v[1]
            elif v[0] == 'D':
                dropDepth = DropDepthOut(in_channels, v)
                if batch_norm:
                    layers += [dropDepth, nn.BatchNorm2d(v[1]), nn.ReLU(inplace=True)]
                else:
                    layers += [dropDepth, nn.ReLU(inplace=True)]
                in_channels = v[1]
            elif v[0] == 'C':
                conv2d = nn.Conv2d(in_channels, v[1], kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v[1]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v[1]
            elif v[0] == 'O':
                splitOnly = OnlySplit(in_channels, v)
                if batch_norm:
                    layers += [splitOnly, nn.BatchNorm2d(v[1]), nn.ReLU(inplace=True)]
                else:
                    layers += [splitOnly, nn.ReLU(inplace=True)]
                in_channels = v[1]
            elif v[0] == 'S':
                shuffle_split = shuffleSplit(in_channels, v, arg.batchsize, imagesize)
                if batch_norm:
                    layers += [shuffle_split, nn.BatchNorm2d(v[1]), nn.ReLU(inplace=True)]
                else:
                    layers += [shuffle_split, nn.ReLU(inplace=True)]
                in_channels = v[1]
            elif v[0] == 'OG':
                splitonly_global = OnlySplit_global(in_channels, v, arg.batchsize,32)
                if batch_norm:
                    layers += [splitonly_global, nn.BatchNorm2d(v[1]), nn.ReLU(inplace=True)]
                else:
                    layers += [splitonly_global, nn.ReLU(inplace=True)]
                in_channels = v[1]
            elif v[0] == 'SG':
                shuffle_split_global = shuffleSplit_global(in_channels, v, arg.batchsize, imagesize)
                if batch_norm:
                    layers += [shuffle_split_global, nn.BatchNorm2d(v[1]), nn.ReLU(inplace=True)]
                else:
                    layers += [shuffle_split_global, nn.ReLU(inplace=True)]
                in_channels = v[1]
        return nn.Sequential(*layers)

def train(model, trainloader, testloader, optimizer, epoch):
    batch_time = jisu_util.AverageMeter()
    data_time = jisu_util.AverageMeter()
    losses = jisu_util.AverageMeter()
    acc = jisu_util.AverageMeter()


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
        # print loss

        # Update results
        accuracy = jisu_util.accuracy(out, label)
        acc.update(accuracy, input.size(0))
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
                  'Loss {loss.avg:.4f}\t'
                  'Accuracy {acc.avg:.3f}\t'.format(
                    epoch, i, len(trainloader), batch_time=batch_time,
                    data_time=data_time, loss=losses, acc=acc))

def validate(testloader, model):
    batch_time = jisu_util.AverageMeter()
    losses = jisu_util.AverageMeter()
    acc = jisu_util.AverageMeter()

    end = time.time()
    model.eval()
    for i, (input, label) in enumerate(testloader):
        input = Variable(input.cuda(), volatile=True)
        label = Variable(label.cuda(), volatile=True)

        # compute output
        output = model(input)
        loss = F.nll_loss(output, label)

        # measure accuracy and record loss
        accuracy = jisu_util.accuracy(output, label)
        losses.update(loss.data[0], input.size(0))
        acc.update(accuracy, input.size(0))

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
    args = parser.parse_args()

    #cfg = [['C', 40], ['C', 80], ['M'], ['C', 80], ['C', 160], ['M'], ['C', 160], ['C', 320], ['M'], ['C', 640], ['M']]
    # cfg = [['C', 40], ['F', 80, num_clusters], ['M'], ['F', 80, num_clusters], ['F', 160, num_clusters]
    #     , ['M'], ['F', 160, num_clusters], ['F', 320, num_clusters], ['M'], ['F', 640, num_clusters], ['M']]
    #cfg = [['ImageSize', 32], ['C', 40], ['C', 80, args.numcluster], ['M'], ['C', 80, args.numcluster], ['C', 160, args.numcluster]
    #     , ['C', 160, args.numcluster], ['C', 320, args.numcluster], ['M']]
    cfg = [['ImageSize', 32], ['C', 40], ['O', 80, args.numcluster], ['M'], ['O', 80, args.numcluster], ['O', 160, args.numcluster]
         , ['O', 160, args.numcluster], ['O', 320, args.numcluster], ['M']]
    # cfg = [['ImageSize', 32], ['C', 40], ['S', 80, args.numcluster], ['M'], ['S', 80, args.numcluster], ['S', 160, args.numcluster]
    #     , ['S', 160, args.numcluster], ['S', 320, args.numcluster], ['M']]
    #cfg = [['ImageSize', 32], ['C', 40], ['SG', 80, 2, args.numcluster], ['M'], ['SG', 80, 2, args.numcluster], ['SG', 160, 2, args.numcluster]
    #    , ['SG', 160, 2, args.numcluster], ['SG', 320, 2, args.numcluster], ['M']]
    # cfg = [['C', 40], ['D', 80], ['M'], ['D', 80], ['D', 160]
    #     , ['M'], ['D', 160], ['D', 320], ['M'], ['D', 640], ['M']]
    # cfg = [32, 64, 128, 'M', 256, 256, 512, 'M']

    trainset = torchvision.datasets.CIFAR100('/home/david/FeatureSplit/data/', train=True, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100('/home/david/FeatureSplit/data/', train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=True, num_workers=2)

    model = CNN(cfg, args).cuda()
    # print model

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    best_acc = 0

    for epoch in range(1, 100):
        train(model, trainloader, testloader, optimizer, epoch)
        valid_acc = validate(testloader, model)

        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        jisu_util.save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'CNN',
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
        }, is_best)
        print(' * Best Accuracy {acc:.3f}'
              .format(acc=best_acc))


if __name__ == '__main__':
    main()