from __future__ import print_function

import argparse
import os
import shutil
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models as models
import utils.tools as tools
import utils.function as func
from utils.logger import Logger

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--drop', '--dropout', default=0, type=float,
                                        metavar='Dropout', help='Dropout ratio')

parser.add_argument('--arch', '-a', metavar='ARCH')
parser.add_argument('--depth', type=int, default=20, help='Model depth.')
parser.add_argument('--widen-factor', type=int, default=10, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--expansion', type=int, default=6, help='expansion rate of shiftresnet')
parser.add_argument('--train-batch', type=int, default=128, help='expansion rate of shiftresnet')
parser.add_argument('--mult', type=float, default=1, help='multiplier of depth, width of shiftnet-A')
parser.add_argument('--verbose', action='store_true', help='print additional information')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--nest', action='store_true',
                    help='nesterov momentum')

parser.add_argument('--att-cfg' , metavar='DICT', default={}, type=tools.str2dict)

parser.add_argument('--num-block1', type=int, default = 0,
                    help='number of nl block')
parser.add_argument('--num-block2', type=int, default = 0,
                    help='number of nl block')
parser.add_argument('--num-block3', type=int, default = 0,
                    help='number of nl block')
parser.add_argument('--num-block4', type=int, default = 0,
                    help='number of nl block')


parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--gpu-id', default='0', type=str,
                                        help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()
print(args)
        
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    # Model
    print("==> creating model '{}'".format(args.arch))

    if args.dataset.startswith('cifar'):
        img_size = 32
        num_classes = 100
    elif args.dataset == 'imagenet':
        img_size = 224
        num_classes = 1000

    if args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                                    args = args,
                                    num_classes=num_classes,
                                    depth=args.depth,
                                    widen_factor=args.widen_factor,
                                    dropRate=args.drop,
                            )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                                    num_classes=num_classes,
                                    depth=args.depth,
                                    args = args
                            )
    else:
        model = models.__dict__[args.arch](args=args,num_classes=num_classes)

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    print(model)

    dummy_input = torch.randn(args.train_batch, 3, img_size,img_size, dtype=torch.float).cuda()
    dummy_target = torch.zeros(args.train_batch, dtype=torch.long).cuda(non_blocking=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov = args.nest)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    for _ in range(10):
        _ = model(dummy_input)

    for rep in range(repetitions):
        dummy_input, dummy_target = torch.autograd.Variable(dummy_input), torch.autograd.Variable(dummy_target)

        t0 = time.time()
        outputs = model(dummy_input)
        loss = criterion(outputs, dummy_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        timings[rep] = (t1 - t0)
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(f'mean: {mean_syn}, std: {std_syn}')


if __name__ == '__main__':
        main()
