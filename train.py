from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models as models
import ys_utils.tools as tools
import ys_utils.ys_function as ys_func
from ys_utils.logger import Logger
from tensorboard_logger import configure, log_value


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training')
# Datasets
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--datadir', default='./', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='nonlocal_',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=20, help='Model depth.')

parser.add_argument('--nest', action='store_true',
                    help='nesterov momentum')
parser.add_argument('--cos', action='store_true',
                    help='cosine anealing')
parser.add_argument('--plot', action='store_true',
                    help='plotting log file')
parser.add_argument('--log', default='none', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--freq', type=int, default=100, help='print frequency')
parser.add_argument('--logger', default='none', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')




parser.add_argument('--att-cfg' , metavar='DICT', default={}, type=tools.str2dict)

parser.add_argument('--num-block1', type=int, default = 0,
                    help='number of nl block')
parser.add_argument('--num-block2', type=int, default = 0,
                    help='number of nl block')
parser.add_argument('--num-block3', type=int, default = 0,
                    help='number of nl block')
parser.add_argument('--num-block4', type=int, default = 0,
                    help='number of nl block')

parser.add_argument('--verbose', action = 'store_true')
parser.add_argument('--save-pic', action='store_true')




parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=10, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--finetune', action='store_true')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(args)
# Validate dataset
if args.tensorboard:
  configure("runs/%s"%(args.logger))
    
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    if args.verbose:
      args.mean = 0
      args.variance = 0
    if not os.path.isdir(args.checkpoint):
        tools.mkdir_p(args.checkpoint)

    if args.plot:
        tools.LogtoMat(args.log, args.checkpoint)
        return

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    elif args.dataset == 'cifar100':
        dataloader = datasets.CIFAR100
        num_classes = 100
    elif args.dataset == 'imagenet':
        num_classes = 1000
    elif args.dataset == 'tiny-imagenet':
        num_classes = 200    
        
    if args.dataset.startswith('cifar'):
      transform_train = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
  
      transform_test = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
      
      trainset = dataloader(root='data', train=True, download=True, transform=transform_train)
      
      trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory =True)

      testset = dataloader(root='data', train=False, download=False, transform=transform_test)
      testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory =True)
    
    
    elif args.dataset == 'imagenet':
      
      traindir = os.path.join(args.datadir, 'train')
      valdir = os.path.join(args.datadir, 'val')
      
      
      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
      transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])   
      transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
            
            
      train_dataset = datasets.ImageFolder(
            traindir,
            transform_train)
      test_dataset = datasets.ImageFolder(valdir,               transform_test) 
      
      
      trainloader = torch.utils.data.DataLoader(
          train_dataset, batch_size=args.train_batch, shuffle=True,
          num_workers=args.workers, pin_memory=True)

      testloader = torch.utils.data.DataLoader(
          test_dataset, batch_size=args.test_batch, shuffle=False,
          num_workers=args.workers, pin_memory=True)                                      
    elif args.dataset == 'tiny-imagenet':
      
      traindir = os.path.join(args.datadir, 'train')
      valdir = os.path.join(args.datadir, 'val')
      
      
      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
      transform_train = transforms.Compose([
                transforms.RandomResizedCrop(56),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])   
      transform_test = transforms.Compose([
                transforms.CenterCrop(56),
                transforms.ToTensor(),
                normalize,
            ])
            
            
      train_dataset = datasets.ImageFolder(
            traindir,
            transform_train)
      test_dataset = datasets.ImageFolder(valdir,               transform_test) 
      
      
      trainloader = torch.utils.data.DataLoader(
          train_dataset, batch_size=args.train_batch, shuffle=True,
          num_workers=args.workers, pin_memory=True)

      testloader = torch.utils.data.DataLoader(
          test_dataset, batch_size=args.test_batch, shuffle=False,
          num_workers=args.workers, pin_memory=True)                                      

    # Model
    print("==> creating model '{}'".format(args.arch))
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
        model = models.__dict__[args.arch](num_classes=num_classes)

    print(model)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
    
    criterion = nn.CrossEntropyLoss(reduction = 'none')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov = args.nest)

    # Resume
    title = args.dataset + '/' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict = not args.finetune)
        
        if not args.finetune:
          best_acc = checkpoint['best_acc']
          start_epoch = checkpoint['epoch']
          print(best_acc, start_epoch)
          optimizer.load_state_dict(checkpoint['optimizer'])
        if args.finetune:
          logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
          logger.set_names(['Epoch','	Train_Loss','	Train_Acc','Test_loss','Test_Acc'])
          
          if os.path.isfile(args.logger + 'log.txt'):
            logger2 = Logger(args.logger + 'log.txt', title=title, resume = True)
          else:
            logger2 = Logger(args.logger + 'log.txt', title=title) 
            logger2.set_names(['Test Error'])
        
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch','	Train_Loss','	Train_Acc','Test_loss','Test_Acc'])
        if os.path.isfile(args.logger + 'log.txt'):
          logger2 = Logger(args.logger + 'log.txt', title=title, resume = True)
        else:
          logger2 = Logger(args.logger + 'log.txt', title=title) 
          logger2.set_names(['Test Error'])
        
    

    if args.evaluate:
        print('\nEvaluation only')
        ys_func.evaluate_test(testloader, model, criterion, start_epoch,use_cuda, args)
           
        return
    
    for epoch in range(start_epoch, args.epochs):
        if not args.cos: 
          adjust_learning_rate(optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        
        tr_loss,tr_top1,tr_top5 = ys_func.train(trainloader, model, criterion, optimizer, epoch, use_cuda,args, state)
        
        te_loss, te_top1, te_top5 = ys_func.evaluate_test(testloader, model, criterion, epoch, use_cuda, args)
        # append logger file
        logger.append([epoch, tr_loss, tr_top1, te_loss, te_top1])
        
        test_acc = te_top1
        if args.tensorboard:
          log_value('train_loss', tr_loss, epoch)
          log_value('train_acc', tr_top1, epoch)
          log_value('test_loss', te_loss, epoch)
          log_value('test_acc', te_top1, epoch)
    
        #save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        print(best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)
        

    
    print('Best acc:')
    print(100-best_acc)
    logger2.append([100-best_acc])
        

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
