import torch
import shutil
import pandas as pd

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / self.count

def accuracy(output, label):
    pred = output.data.max(1)[1]
    batch_size = label.size(0)
    correct = pred.eq(label.data).cpu().sum()
    return correct*(100.0 / batch_size)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

import torch.utils.data as data
import torchvision
from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(imagedir, labeldir):
    images = []
    label = pd.read_csv(labeldir, sep="\n", converters={"label":int})['label'].tolist()
    for root, _, fnames in sorted(os.walk(imagedir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = (path, label[int(fname[-13:-5])-1]-1)
                images.append(item)

    return images
# datadir = '/home/jisu/Desktop/Data/ILSVRC/CLS_LOC/ILSVRC2015/Data/CLS-LOC/val'
# labeldir = '/home/jisu/Desktop/Data/ILSVRC/ILSVRC2015_devkit/devkit/data/ILSVRC2015_clsloc_validation_ground_truth.txt'
# img = make_dataset(datadir, labeldir)
# print img[1]

def pil_loader(path):
    return Image.open(path).convert('RGB')

class IMAGENET_VAL(data.Dataset):

    def __init__(self, imgdir, labeldir, transform=None, target_transform=None,
                 loader=pil_loader):
        imgs = make_dataset(imgdir, labeldir)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + imgdir + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = imgdir
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)