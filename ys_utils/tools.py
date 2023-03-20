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
            
def softmax(logits):
  batchsize = logits.size(0)
  logits = logits - logits.max(1)[0].view(batchsize,-1)
  predictions = logits.exp() / logits.exp().sum(1).view(batchsize,-1)
  return predictions


class YSNorm(nn.Module):
  def __init__(self, normalized_shape,mode = 'M', eps = 1e-6):
        super(YSNorm, self).__init__()
        self.n_shape = normalized_shape
        self.eps = eps
        self.mode = mode
  
  def forward(self, input):
        mean = input.mean(-1, keepdim=True)
        std = input.std(-1,unbiased=False, keepdim=True)
        if self.mode == 'M':
          return (input - mean)
        elif self.mode == 'S':
          return (input) / (std + self.eps)
        elif self.mode == 'A':
          return (input - mean) / (std + self.eps)  
        


class cross_entropy_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self, reduction=True, norm='softmax'):
        super(cross_entropy_loss, self).__init__()
        self.reduction = reduction
        self.norm = norm
        self.eps = 1e-8
    def forward(self, logits, targets):
        if self.norm =='softmax':
          pred = softmax(logits)
        elif self.norm == 'sum':
          pred = logits / logits.sum(1).view(logits.size(0),-1)
        elif self.norm == 'softmax_2':
          pred = softmax_2(logits)

        
        log_pred =  (self.eps + pred).log()
        targets_onehot = torch.zeros(logits.size()) - 1.0
        targets_onehot[torch.arange(logits.size(0)), targets] = 1.0
        targets_onehot = targets_onehot.cuda()
        if self.reduction:
          loss = -( targets * (self.eps + pred).log() ).sum(1).mean()
        else:
          loss = -( targets_onehot * log_pred ).sum(1)
        return loss
        

          
class kldiv_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self, reduction=True, norm='softmax'):
        super(kldiv_loss, self).__init__()
        self.eps = 1e-8
        self.reduction = reduction
        self.norm = norm
    def forward(self, logits1, logits2):
        if self.norm =='softmax':
          pred1 = softmax(logits1)
          pred2 = softmax(logits2)
        elif self.norm == 'sum':
          pred1 = logits1 / logits1.sum(1).view(logits1.size(0),-1)
          pred2 = logits2 / logits2.sum(1).view(logits2.size(0),-1)
        if self.reduction:
          loss =  (pred2 * (self.eps + pred2 / (pred1 + self.eps)).log()).sum(1).mean()
        else:
          loss =  (pred2 * (self.eps + pred2 / (pred1 + self.eps)).log()).sum(1)
        return loss

class ECEMeter(object):
    """Computes and stores the Expected calibration error
    """
    def __init__(self, num):
        self.cnt = np.zeros([num],dtype = float)
        self.acc = np.zeros([num],dtype = float)
        self.conf = np.zeros([num],dtype = float)
        self.bin = num
        self.mean = np.linspace(0,1,self.bin,endpoint = False) + 1/self.bin * 0.5
        self.ece = 0.0
        self.conf_mean = 0.0
    def update(self, outputs, targets):
        output = outputs - outputs.max(1)[0].view(-1,1)
        conf = output.exp().max(1)[0] / output.exp().sum(1) 
        pred = (outputs.max(1)[1] == targets) 
        for i in range(0,conf.size(0)):
          dist = abs( self.mean - conf[i] )
          j = dist.argmin()
          
          self.conf_mean += conf[i]
          self.cnt[j] += 1
          self.acc[j] += pred[i]
          self.conf[j] += conf[i]
        
    def calculate(self):
        self.ece = 0.0
        self.conf_mean = self.conf_mean / self.cnt.sum()
        for i in range(0, self.bin):
          if self.cnt[i] != 0:
            self.ece += abs(self.acc[i] / self.cnt[i] - self.conf[i] / self.cnt[i]) / self.cnt.sum() * self.cnt[i]
    
 

def LogtoMat(log_path, save_path):
  f = open(log_path,'r')
  f_r = f.readlines()
  num_f = len(f_r)-1
  print('num epoch : %d'%(num_f))
  names = f_r[0].split()
  print(names)
  matfiles = np.zeros([len(names),num_f], dtype =float)
  
  for i in range(1, num_f+1):
    for j in range(0,len(names)):
      matfiles[j,i-1] = f_r[i].split()[j]
  f.close()
  
  for i in range(0,len(names)):
    sio.savemat(os.path.join(save_path,names[i]),dict(data = matfiles[i,:]))
    
  return

  
class Graph2DMeter(object):
    """Computes and stores Distribution
    """
    def __init__(self, num, num_class):
        self.num = num
        self.graph = np.zeros([num,num_class],dtype = float) + 100
        self.n = 0
    def update(self, val):
        for i in range(0,val.size(0)):
          self.graph[self.n] = val[i]
          self.n += 1
  
class Graph1DMeter(object):
    """Computes and stores Distribution
    """
    def __init__(self, num):
        self.num = num
        self.graph = np.zeros([num],dtype = float) + 100
        self.n = 0
    def update(self, val):
        for i in range(0,val.size(0)):
          self.graph[self.n] = val[i]
          self.n += 1
    def sort(self):
        self.graph.sort()  

        
class HistoMeter(object):
  """Computes and stores Histogram
  """
  def __init__(self,bin_min,bin_max,bin_num):    
    self.histogram = np.zeros([bin_num+2])
    self.bin_sz = (bin_max - bin_min) / bin_num
    self.bin_num = bin_num
    self.bin_min = bin_min
    self.bin_max = bin_max
    self.bin_mean = np.linspace(bin_min,bin_max,bin_num,endpoint = False) + self.bin_sz * 0.5
    print("bin_sz : %f , bin_num : %f, bin_min : %f, bin_max : %f"%(self.bin_sz,self.bin_num, self.bin_min, self.bin_max))
    
  def update(self, val):
    for i in range(0,val.size(0)):
      if val[i] >= self.bin_max:
        self.histogram[-1] += 1
      elif val[i] < self.bin_min:
        self.histogram[0] += 1
      else:            
        v = np.abs(val[i].item() - self.bin_mean)
        idx = v.argmin()
        self.histogram[idx+1] += 1


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
