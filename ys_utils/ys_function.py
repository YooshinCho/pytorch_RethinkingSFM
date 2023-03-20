import numpy as np
import torch
import os
import scipy.io as sio
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import time
import sys
import math
import torch.utils.data as data
from ys_utils.tools import *



    
def train(trainloader, model, criterion, optimizer, epoch, use_cuda, args, state):
    # switch to train mode
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    end = time.time()
    start_time = end
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        # Cosine Annealing
        if args.cos:
          state['lr'] = args.lr / 2.0 * (np.cos(np.pi*(epoch*len(trainloader)+batch_idx)/args.epochs/len(trainloader))+1.0)
          for param_group in optimizer.param_groups:
              param_group['lr'] = state['lr']
              
        if use_cuda:
          inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.mean().item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
                
        
        loss = loss.mean()
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % args.freq == 0:
          print('[%d/%d]%d/%d  train loss: %f, top1 acc: %f, top5 acc: %f, time: %f'%(epoch,args.epochs,batch_idx,len(trainloader),losses.avg, top1.avg, top5.avg, batch_time.avg))
          sys.stdout.flush()

    epoch_time = time.time() - start_time
    print('[%d/%d]epoch time: %f'%(epoch,args.epochs, epoch_time))
    sys.stdout.flush()

        
    return (losses.avg,top1.avg,top5.avg)


def evaluate_test(testloader, model, criterion, epoch, use_cuda, args):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    
    
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
          inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

          # compute output
          outputs = model(inputs)
          loss = criterion(outputs, targets)
          loss = loss.mean()
          
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
    print("test acc:%f, %f"%(top1.avg, top5.avg))
    print("test loss:%f"%losses.avg)
  
    print(100-top1.avg)
    
    return (losses.avg, top1.avg, top5.avg )
