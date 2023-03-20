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
__all__ = ['sepreresnet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, preact='no_preact', reduction = 1):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        self.preact = preact

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            if self.preact == 'preact':
                residual = self.downsample(out)
            else:
                residual = self.downsample(x)

        out = self.conv1(out)   

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)        
        out = self.se(out)
        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, preact='no_preact', reduction = 1):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride
        self.preact = preact

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            if self.preact == 'preact':
                residual = self.downsample(out)
            else:
                residual = self.downsample(x)

        out = self.conv1(out) 

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.se(out)
        out += residual

        return out

class PreResNet(nn.Module):

    def __init__(self,args, depth, num_classes=1000):
        super(PreResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        self.args = args
        self.nlprint = self.args.verbose  
        
        if args.dataset.startswith('cifar'):
          
          block = Bottleneck if depth >=44 else BasicBlock
          if block == BasicBlock:
            n = (depth - 2) // 6
          elif block == Bottleneck:
            n = (depth - 2) // 9
          self.inplanes = 16
          self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                                 bias=False)
          self.layer1 = self._make_layer(block, 16, n, num_block = self.args.num_block1 )
          self.layer2 = self._make_layer(block, 32, n, stride=2, num_block = self.args.num_block2)
          self.layer3 = self._make_layer(block, 64, n, stride=2, num_block = self.args.num_block3)
          self.bn = nn.BatchNorm2d(64 * block.expansion)
          self.relu = nn.ReLU(inplace=True)
          self.avgpool = nn.AvgPool2d(8)
          self.fc = nn.Linear(64 * block.expansion, num_classes)
          
  
        
              
        elif args.dataset == 'imagenet':
          blocks ={18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
          layers ={18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
          assert layers[depth], 'invalid detph for Pre-ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'

          self.inplanes = 64
          self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
          self.bn1 = nn.BatchNorm2d(64)
          self.relu = nn.ReLU(inplace=True)
          self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
          self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0], num_block = self.args.num_block1, preact = 'no_preact')
          self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2, num_block = self.args.num_block2)
          self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2, num_block = self.args.num_block3)
          self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2, num_block = self.args.num_block4)
          self.bn2 = nn.BatchNorm2d(512 * blocks[depth].expansion)
          self.avgpool = nn.AvgPool2d(7, stride=1) 
          self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)

          
        elif args.dataset == 'tiny-imagenet':
          blocks ={18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
          layers ={18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
          assert layers[depth], 'invalid detph for Pre-ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'

          self.inplanes = 64
          self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
          self.bn1 = nn.BatchNorm2d(64)
          self.relu = nn.ReLU(inplace=True)
          self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0], num_block = self.args.num_block1, preact = 'no_preact')
          self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2, num_block = self.args.num_block2)
          self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2, num_block = self.args.num_block3)
          self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2, num_block = self.args.num_block4)
          self.bn2 = nn.BatchNorm2d(512 * blocks[depth].expansion)
          self.avgpool = nn.AvgPool2d(7, stride=1) 
          self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        for m in self.layer1:
            if isinstance(m, nlblock):
              m.reset()
              
        for m in self.layer2:
            if isinstance(m, nlblock):
              m.reset()
            
        for m in self.layer3:
            if isinstance(m, nlblock):
              m.reset()
            
        if self.args.dataset == 'imagenet' or self.args.dataset == 'tiny_imagenet':
          for m in self.layer4:
              if isinstance(m, nlblock):
                m.reset()
                        
            
    def nl_print(self):
      for m in self.layer2:
        if isinstance(m, nlblock):
          for n in m.modules():
            if isinstance(n, nn.BatchNorm2d):
              print('bn')
              if n.weight is not None:
                print(n.weight.data.mean())
                print(n.bias.data.mean())
            if isinstance(n, nn.LayerNorm):  
              print('ln')
              if n.weight is not None:
                print(n.weight.data.mean())
                print(n.bias.data.mean())
          if self.args.bn =='z_false_aff_false_gamma':
            print(m.gamma.item())
          if self.args.input_norm.startswith('ChannelBias'):
            print(m.gamma.item())      

    def _make_layer(self, block, planes, blocks, stride=1, num_block=0, preact='preact'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, preact, reduction = self.args.reduction))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
          if i == blocks-self.args.pos_block and num_block != 0 :
            for j in range(0,num_block):
              print('appended')
              if self.args.pos_enc and j==0:
                pos_enc = True
              else:
                pos_enc = False
              layers.append(nlblock(self.inplanes, args = self.args, bias_on = False, pos_enc = pos_enc))
              
                              
          layers.append(block(self.inplanes, planes, reduction = self.args.reduction))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.nlprint:
          self.nl_print()
          self.nlprint = False
        if self.args.dataset.startswith('cifar'):
          x = self.conv1(x)
          x = self.layer1(x)  # 32x32
          x = self.layer2(x)  # 16x16
          x = self.layer3(x)  # 8x8
          x = self.bn(x)
          x = self.relu(x)
  
          x = self.avgpool(x)
          x = x.view(x.size(0), -1)
          x = self.fc(x)
        elif self.args.dataset == 'imagenet':
          x = self.conv1(x)
          x = self.bn1(x)
          x = self.relu(x)
          x = self.maxpool(x)

          x = self.layer1(x)
          x = self.layer2(x)
          x = self.layer3(x)
          x = self.layer4(x)

          x = self.bn2(x)
          x = self.relu(x)
          x = self.avgpool(x)
          x = x.view(x.size(0), -1)
          x = self.fc(x)  
        elif self.args.dataset == 'tiny-imagenet':
          x = self.conv1(x)

          x = self.layer1(x)
          x = self.layer2(x)
          x = self.layer3(x)
          x = self.layer4(x)
          
          x = self.bn2(x)
          x = self.relu(x)
          x = self.avgpool(x)
          x = x.view(x.size(0), -1)
          x = self.fc(x) 
          
        return x


def sepreresnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return PreResNet(**kwargs)
