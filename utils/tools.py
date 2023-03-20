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
import matplotlib as mpl
import matplotlib.pylab as plt
import torch.utils.data as data


def str2dict(s) -> dict:
  if s is None:
    return {}
  if not isinstance(s, str):
    return s
  s = s.split(',')
  d = {}
  for ss in s:
    if ss == '':
        continue
    ss = ss.split('=')
    assert len(ss) == 2
    key = ss[0].strip()
    value = str2num(ss[1])
    d[key] = value
  return d

def str2num(s: str):
    s.strip()
    try:
        value = int(s)
    except ValueError:
        try:
            value = float(s)
        except ValueError:
            if s == 'True':
                value = True
            elif s == 'False':
                value = False
            elif s == 'None':
                value = None
            else:
                value = s
    return value

def save_pic(tensor, savepath):
  tensor = tensor.cpu().numpy()
  for i in range(0, tensor.shape[0]):
    plt.figure(figsize=(18, 18))
    plt.title('attention_map')
    im = plt.imshow(tensor[i], cmap = 'hot', interpolation='nearest')
    plt.colorbar(im)
    plt.savefig(os.path.join(savepath,'att_%d.png'%(i)))
    
    plt.clf()

  
def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
            

        


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
