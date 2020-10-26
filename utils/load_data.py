from __future__ import print_function

import numbers

import torch
import torch.utils.data as data_utils
import pickle
from scipy.io import loadmat

import numpy as np
import math

import os
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as vf
from torch.utils.data import ConcatDataset

from PIL import Image

import os
import os.path
from os.path import join
import sys
import tarfile


class ToTensorNoNorm():
    def __call__(self, X_i):
        return torch.from_numpy(np.array(X_i, copy=False)).permute(2, 0, 1)


class PadToMultiple(object):
    def __init__(self, multiple, fill=0, padding_mode='constant'):
        assert isinstance(multiple, numbers.Number)
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.multiple = multiple
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.
        Returns:
            PIL Image: Padded image.
        """
        w, h = img.size
        m = self.multiple
        nw = (w // m + int((w % m) != 0)) * m
        nh = (h // m + int((h % m) != 0)) * m
        padw = nw - w
        padh = nh - h

        out = vf.pad(img, (0, 0, padw, padh), self.fill, self.padding_mode)
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(multiple={0}, fill={1}, padding_mode={2})'.\
            format(self.mulitple, self.fill, self.padding_mode)


class CustomTensorDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        from PIL import Image

        X, y = self.tensors
        X_i, y_i, = X[index], y[index]

        if self.transform:
            X_i = self.transform(X_i)
            X_i = torch.from_numpy(np.array(X_i, copy=False))
            X_i = X_i.permute(2, 0, 1)

        return X_i, y_i

    def __len__(self):
        return self.tensors[0].size(0)


def load_cifar10(args, **kwargs):
    # set args
    args.input_size = [3, 32, 32]
    args.input_type = 'continuous'

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(int(math.ceil(32 * 0.04)), padding_mode='edge'),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
        transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip(),
        ToTensorNoNorm()
    ])

    test_transform = transforms.Compose([
        ToTensorNoNorm()
    ])

    data_train = torchvision.datasets.CIFAR10('./data', train=True, transform=train_transform,
                                              target_transform=None, download=True)

    train = torch.utils.data.Subset(data_train, torch.arange(0, 40000))

    data_val = torchvision.datasets.CIFAR10('./data', train=True,
                                              transform=test_transform,
                                              target_transform=None,
                                              download=False)

    val = torch.utils.data.Subset(data_val, torch.arange(40000, 50000))

    test = torchvision.datasets.CIFAR10('./data', train=False,
                                          transform=test_transform,
                                          target_transform=None,
                                          download=True)

    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    val_loader = data_utils.DataLoader(val, batch_size=args.batch_size, shuffle=False, **kwargs)

    test_loader = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, args


def extract_tar(tarpath):
    assert tarpath.endswith('.tar')

    startdir = tarpath[:-4] + '/'

    if os.path.exists(startdir):
        return startdir

    print('Extracting', tarpath)

    with tarfile.open(name=tarpath) as tar:
        t = 0
        done = False
        while not done:
            path = join(startdir, 'images{}'.format(t))
            os.makedirs(path, exist_ok=True)

            print(path)

            for i in range(50000):
                member = tar.next()

                if member is None:
                    done = True
                    break

                # Skip directories
                while member.isdir():
                    member = tar.next()
                    if member is None:
                        done = True
                        break

                member.name = member.name.split('/')[-1]

                tar.extract(member, path=path)

            t += 1

    return startdir


def load_imagenet(resolution, args, **kwargs):
    assert resolution == 32 or resolution == 64

    args.input_size = [3, resolution, resolution]

    trainpath = '../imagenet{res}/train_{res}x{res}.tar'.format(res=resolution)
    valpath = '../imagenet{res}/valid_{res}x{res}.tar'.format(res=resolution)

    trainpath = extract_tar(trainpath)
    valpath = extract_tar(valpath)

    data_transform = transforms.Compose([
        ToTensorNoNorm()
    ])

    print('Starting loading ImageNet')

    imagenet_data = torchvision.datasets.ImageFolder(
        trainpath,
        transform=data_transform)

    print('Number of data images', len(imagenet_data))

    val_idcs = np.random.choice(len(imagenet_data), size=20000, replace=False)
    train_idcs = np.setdiff1d(np.arange(len(imagenet_data)), val_idcs)

    train_dataset = torch.utils.data.dataset.Subset(
        imagenet_data, train_idcs)
    val_dataset = torch.utils.data.dataset.Subset(
        imagenet_data, val_idcs)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs)

    test_dataset = torchvision.datasets.ImageFolder(
        valpath,
        transform=data_transform)

    print('Number of val images:', len(test_dataset))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs)

    return train_loader, val_loader, test_loader, args


def load_svhn(args, **kwargs):
    # set args
    args.input_size = [3, 32, 32]

    train_transform = transforms.Compose([
        transforms.Pad(int(math.ceil(32 * 0.04)), padding_mode='edge'),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
        transforms.CenterCrop(32),
        ToTensorNoNorm()
    ])

    test_transform = transforms.Compose([
        ToTensorNoNorm()
    ])

    root = './data/svhn'
    train_data = torchvision.datasets.SVHN(
        root, split='train', transform=train_transform, target_transform=None,
        download=True)

    train_data = torch.utils.data.Subset(
        train_data, indices=np.arange(0, len(train_data) - 10000))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_data = torchvision.datasets.SVHN(
        root, split='train', transform=test_transform, target_transform=None,
        download=True)

    val_data = torch.utils.data.Subset(
        val_data, indices=np.arange(len(train_data) - 10000, len(train_data)))
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args.batch_size,
                                             shuffle=False, num_workers=4)

    test_data = torchvision.datasets.SVHN(
        root, split='test', transform=test_transform, target_transform=None,
        download=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size,
                                              shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, args


def load_dataset(args, **kwargs):

    if args.dataset == 'cifar10':
        train_loader, val_loader, test_loader, args = load_cifar10(args, **kwargs)
    elif args.dataset == 'svhn':
        train_loader, val_loader, test_loader, args = load_svhn(args, **kwargs)
    elif args.dataset == 'imagenet32':
        train_loader, val_loader, test_loader, args = load_imagenet(32, args, **kwargs)
    elif args.dataset == 'imagenet64':
        train_loader, val_loader, test_loader, args = load_imagenet(64, args, **kwargs)
    else:
        raise Exception('Wrong name of the dataset!')

    return train_loader, val_loader, test_loader, args


if __name__ == '__main__':
    pass
