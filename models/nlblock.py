from __future__ import absolute_import
import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib as mpl
import os
import matplotlib.pyplot as plt

__all__ = ['nlblock', 'get_sinusoid_encoding_table']
        
def save_pic(attention, path):
  batch_size = attention.size(0)
  tensor = attention[torch.randint(batch_size,(1,))].detach()
  tensor = tensor.cpu().numpy()
  plt.figure(figsize=(18, 18), frameon=False)
  plt.axis('off')
  im = plt.imshow(tensor, cmap = 'hot' )
  plt.savefig(os.path.join(path,'att.png'), bbox_inches='tight', pad_inches=0)
  plt.clf()    
  return

    
  
def get_sinusoid_encoding_table(n_posX, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''
  
    d_hid = d_hid // 2
    n_posY = n_posX
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_tableX = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_posX)])
    sinusoid_tableY = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_posY)])
    
    sinusoid_tableX[:, 0::2] = np.sin(sinusoid_tableX[:, 0::2])  # dim 2i
    sinusoid_tableX[:, 1::2] = np.cos(sinusoid_tableX[:, 1::2])  # dim 2i+1
    sinusoid_tableY[:, 0::2] = np.sin(sinusoid_tableY[:, 0::2])  # dim 2i
    sinusoid_tableY[:, 1::2] = np.cos(sinusoid_tableY[:, 1::2])  # dim 2i+1
    
    sinusoid_tableX = sinusoid_tableX.transpose()
    sinusoid_tableY = sinusoid_tableY.transpose()
    
    sinusoid_tableX = np.expand_dims(sinusoid_tableX, axis=2)
    sinusoid_tableY = np.expand_dims(sinusoid_tableY, axis=1)

    sinusoid_tableX = np.repeat(sinusoid_tableX, n_posY, axis=2)
    sinusoid_tableY = np.repeat(sinusoid_tableY, n_posX, axis=1)

    sinusoid_table = np.concatenate([sinusoid_tableX ,sinusoid_tableY], axis =0) 
    
    if padding_idx is not None:
        sinusoid_table[padding_idx, padding_idx] = 0.

    sinusoid_table = np.expand_dims(sinusoid_table, axis=0)
    
    return torch.FloatTensor(sinusoid_table)

    
class attention(nn.Module):
  def __init__(self,in_dim,args, num_c = -1,num_head = 1, act = 'softmax', normtype = 'base', pos_enc = 'add_init'):
        super(attention,self).__init__()
        self.args = args
        self.pos_enc = pos_enc
        self.theta_conv = nn.Conv2d(in_channels = in_dim , out_channels = num_c * num_head , kernel_size= 1, bias = False)     
        self.pi_conv = nn.Conv2d(in_channels = in_dim , out_channels = num_c * num_head , kernel_size= 1, bias = False)             
        self.g_conv = nn.Conv2d(in_channels = in_dim , out_channels = num_c * num_head , kernel_size= 1, bias = False)
        self.z_conv = nn.Conv2d(in_channels = num_c * num_head , out_channels = in_dim , kernel_size= 1, bias = False)
        
        self.z_bn = nn.BatchNorm2d(num_features=in_dim)
         
        self.num_c = num_c
        self.num_head = num_head
        self.normtype = normtype
        
        self.act = act
        if self.act == 'softmax':
          self.f_act  = nn.Softmax(dim=-1)      
          
    
  def forward(self,x):
      x_res = x
      x_theta = self.theta_conv(x)
      x_pi = self.pi_conv(x)
      x_g = self.g_conv(x) 
      
      size_b,c,h,w = x_theta.size()      
      x_theta = x_theta.view(size_b, self.num_head, self.num_c, h, w)
      x_pi = x_pi.view(size_b, self.num_head, self.num_c, h, w)
      x_g = x_g.view(size_b, self.num_head, self.num_c, h, w)
      
      
      x_theta = x_theta.view(-1, self.num_c,h,w)
      x_pi = x_pi.view(-1, self.num_c,h,w)
      x_g = x_g.view(-1, self.num_c,h,w)

      
      if self.normtype == 'mag':
        x_theta = x_theta.norm(dim=1, keepdim=True) 
        x_pi = x_pi.norm(dim=1, keepdim=True) 
        
      elif self.normtype == 'angle':
        x_theta = x_theta / x_theta.norm(dim=1, keepdim=True) 
        x_pi = x_pi / x_pi.norm(dim=1, keepdim=True) 

      
      x_theta = x_theta.view(size_b*self.num_head ,x_theta.size(1),-1)
      x_pi = x_pi.view(size_b*self.num_head,x_pi.size(1),-1)
      x_g = x_g.view(size_b*self.num_head,x_g.size(1),-1)
            
      x_theta = x_theta.permute(0,2,1).contiguous()
      x_g = x_g.permute(0,2,1).contiguous()
       
      attention = torch.bmm(x_theta,x_pi)
         
      if self.act != 'none':
        attention = self.f_act(attention)   
      else:
        attention = attention.div((h*w)**0.5)

      if self.args.save_pic:
        save_pic(attention, self.args.checkpoint)
        return
      
      x_g = torch.bmm(attention, x_g)  
      x_g = x_g.permute(0,2,1).contiguous()
      x_g = x_g.view(size_b,self.num_head, self.num_c,h,w)
      x_g = x_g.view(size_b,-1,h,w)
      x_z = self.z_conv(x_g)
      x_z = self.z_bn(x_z)
      x_z = x_res + x_z
      return x_z

class attention_T(nn.Module):
  def __init__(self,in_dim,args, var_scale = True, num_c = -1,num_head = 1, act = 'softmax', normtype = 'base'):
        super(attention_T,self).__init__()
        self.args = args
        self.num_c = num_c
        self.num_head = num_head
        self.act = act
        self.normtype = normtype
        self.var_scale = var_scale
        self.theta_conv = nn.Conv2d(in_channels = in_dim , out_channels = num_c * self.num_head , kernel_size= 1, bias = False)
        
        self.pi_conv = nn.Conv2d(in_channels = in_dim , out_channels = num_c * self.num_head , kernel_size= 1, bias = False)             
        self.g_conv = nn.Conv2d(in_channels = in_dim , out_channels = num_c * self.num_head , kernel_size= 1, bias = False)
        self.z_conv = nn.Conv2d(in_channels = num_c * num_head , out_channels = in_dim , kernel_size= 1, bias = False)
        
        self.z_bn = nn.BatchNorm2d(num_features=in_dim)
        if self.act == 'softmax':
          self.f_act  = nn.Softmax(dim=-1)
          
          

  def forward(self,x):
      x_res = x
      x_theta = self.theta_conv(x)
      x_pi = self.pi_conv(x)
      x_g = self.g_conv(x) 
      
      size_b,c,h,w = x_theta.size()
      
      x_theta = x_theta.view(size_b, self.num_head, self.num_c, h, w)
      x_pi = x_pi.view(size_b, self.num_head, self.num_c, h, w)
      x_g = x_g.view(size_b, self.num_head, self.num_c, h, w)
      
      x_theta = x_theta.view(size_b*self.num_head, self.num_c,-1)
      x_pi = x_pi.view(size_b*self.num_head, self.num_c,-1)
      x_g = x_g.view(size_b*self.num_head, self.num_c,-1)
      
      if self.normtype == 'mag':
        x_theta = x_theta.norm(dim=1, keepdim=True) 
        x_pi = x_pi.norm(dim=1, keepdim=True) 
        
      elif self.normtype == 'angle':
        x_theta = x_theta / x_theta.norm(dim=1, keepdim=True) 
        x_pi = x_pi / x_pi.norm(dim=1, keepdim=True) 

            
      attention = torch.bmm(x_g, x_pi.permute(0,2,1))
      
      if self.act != 'none':
        attention = self.f_act(attention)   
      else:
        if self.var_scale:
            attention = attention.div((h*w)**0.5)

      x = torch.bmm(attention, x_theta) 
      x = x.div(self.num_c**0.5)
        
      x = x.view(-1,self.num_c, h , w)
          
      x = x.view(size_b,self.num_head, self.num_c , h , w)
      x = x.view(size_b,-1,h,w)
      x = self.z_conv(x)
      x = self.z_bn(x)
      x = x + x_res
        
      return x
  
class nlblock(nn.Module):
    def __init__(self, in_dim, args, var_scale = True, num_c = -1, num_head = 1, pos_enc = 'add_init', cinit=0.01, act = 'softmax', headtype='base', normtype='both', num_block=1):
        super(nlblock,self).__init__()
        self.args = args
        self.pos_enc = pos_enc
        if self.pos_enc == 'add_init' :
          self.register_buffer('pos_vec', get_sinusoid_encoding_table(self.args.pos_x,in_dim, padding_idx=None))
        
        if num_c == -1:
          if num_head == 1:
            num_c = in_dim // 2
          else:
            num_c = in_dim // num_head

        self.num_block = num_block
        self.in_dim = in_dim 
        self.var_scale = var_scale
        self.num_c = num_c
        self.num_head = num_head
        self.cinit = cinit
        self.act = act
        self.headtype = headtype
        self.normtype = normtype
        self.cinit = cinit 
        head = []
        for i in range(0, self.num_block):
            assert headtype == 'base' or headtype == 'T'
            if headtype == 'base':
              head.append(attention(in_dim, args, num_c=num_c, num_head=num_head, act=act, normtype=normtype, pos_enc = self.pos_enc))
            elif headtype == 'T':
              head.append(attention_T(in_dim, args, num_c=num_c, num_head=num_head, act=act, normtype= normtype, var_scale=var_scale))
        self.head = nn.Sequential(*head)
        
          
    def reset(self):
      print('re-initializing nlblock')
      for m in self.modules():
        if isinstance(m, nn.Conv2d) and self.cinit == 0.01:
          m.weight.data.normal_(0, 0.01)
          print('Conv init with 0.01')
        elif isinstance(m, nn.Conv2d) and self.cinit == 0:
          m.weight.data.zero_()
          print('Conv init with 0.0')
        elif isinstance(m, nn.Conv2d) :
          print('Conv init with He')

        if isinstance(m, nn.BatchNorm2d) :
          if m.weight is not None:
            m.weight.data.zero_()
            m.bias.data.zero_()
            print('BN init with 0.0')
  
    def forward(self,x):
      if self.pos_enc == 'add_init':
        x = x + self.pos_vec.expand_as(x)
      x = self.head(x)  
      return x
    def extra_repr(self):
        return 'in_dim={in_dim}, num_block={num_block} num_channels={num_c}, num_heads={num_head}, headtype={headtype}, positional_enc={pos_enc}, init={cinit}, activation={act}, normtype={normtype}, variance scale={var_scale}'.format(**self.__dict__)
