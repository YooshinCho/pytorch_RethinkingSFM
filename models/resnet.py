from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch.nn as nn
import math
from models.nlblock import *
import torch.nn.init as init
__all__ = ['resnet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,args, depth, num_classes=1000):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        self.args = args
        self.nlprint = self.args.verbose  
        
        if args.dataset.startswith('cifar'):
  
          block = Bottleneck if depth >=44 else BasicBlock
          if block == BasicBlock:
            n = (depth - 2) // 6
          elif block == Bottleneck:
            n = (depth -2) // 9
          self.inplanes = 16
          self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                                 bias=False)
          self.bn = nn.BatchNorm2d(16)
          self.relu = nn.ReLU(inplace=True)
          
          self.args.pos_x = 32
          self.layer1 = self._make_layer(block, 16, n, num_block = self.args.num_block1 )
          self.args.pos_x = 16
          self.layer2 = self._make_layer(block, 32, n, stride=2, num_block = self.args.num_block2)
          self.args.pos_x = 8
          self.layer3 = self._make_layer(block, 64, n, stride=2, num_block = self.args.num_block3)
          self.avgpool = nn.AvgPool2d(8)
          self.fc = nn.Linear(64 * block.expansion, num_classes)
          
  
        
              
        elif args.dataset == 'imagenet':
          blocks ={18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
          layers ={18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
          assert layers[depth], 'invalid detph for Pre-ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'

          self.inplanes = 64
          self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
          self.bn = nn.BatchNorm2d(64)
          self.relu = nn.ReLU(inplace=True)
          self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
          self.args.pos_x = 56
          self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0],num_block = self.args.num_block1)
          self.args.pos_x = 28
          self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2, num_block = self.args.num_block2)
          self.args.pos_x = 14
          self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2, num_block = self.args.num_block3)
          self.args.pos_x = 7
          self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2, num_block = self.args.num_block4)
          self.avgpool = nn.AvgPool2d(7, stride=1) 
          self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)

          
        elif args.dataset == 'tiny-imagenet':
          blocks ={18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
          layers ={18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
          assert layers[depth], 'invalid detph for Pre-ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'

          self.inplanes = 64
          self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
          self.bn = nn.BatchNorm2d(64)
          self.relu = nn.ReLU(inplace=True)
          self.args.pos_x = 56
          self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0], num_block = self.args.num_block1)
          self.args.pos_x = 28
          self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2, num_block = self.args.num_block2)
          self.args.pos_x = 14
          self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2, num_block = self.args.num_block3)
          self.args.pos_x = 7
          self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2, num_block = self.args.num_block4)
          self.avgpool = nn.AvgPool2d(7, stride=1) 
          self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)

        print('initializing resnet')    
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                  m.weight.data.fill_(1)
                  m.bias.data.zero_()
        
        for m in self.modules():
            if isinstance(m, nlblock):
              m.reset()
              
               
    def _make_layer(self, block, planes, blocks, stride=1, num_block=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = nn.Sequential()
        layers.add_module('0',block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        #if self.args.dataset.startswith('cifar'):
        for i in range(1, blocks):
          if i == blocks-1 and num_block != 0 :
            layers.add_module('nonlocal',nlblock(self.inplanes, args = self.args, bias_on = False,num_block = num_block, **self.args.att_cfg)) 
                            
          layers.add_module('%d'%(i),block(self.inplanes, planes))
        return layers
        
    def forward(self, x):
        if self.args.dataset.startswith('cifar'):
          x = self.conv1(x)
          x = self.bn(x)   
          x = self.relu(x)    # 32x32
  
          x = self.layer1(x)  # 32x32
          x = self.layer2(x)  # 16x16
          x = self.layer3(x)  # 8x8
  
          x = self.avgpool(x)
          x = x.view(x.size(0), -1)
          x = self.fc(x)
        elif self.args.dataset == 'imagenet':
          x = self.conv1(x)
          x = self.bn(x)
          x = self.relu(x)
          
          x = self.maxpool(x)

          x = self.layer1(x)
          x = self.layer2(x)
          x = self.layer3(x)
          x = self.layer4(x)

          x = self.avgpool(x)
          x = x.view(x.size(0), -1)
          x = self.fc(x)  
        elif self.args.dataset == 'tiny-imagenet':
          x = self.conv1(x)
          x = self.bn(x)
          x = self.relu(x)
          
          x = self.layer1(x)
          x = self.layer2(x)
          x = self.layer3(x)
          x = self.layer4(x)
          
          x = self.avgpool(x)
          x = x.view(x.size(0), -1)
          x = self.fc(x) 
          
        return x


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)
