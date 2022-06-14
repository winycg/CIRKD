
from torchvision import transforms
import numpy as np

import torch
from torch.optim.optimizer import Optimizer, required
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter


__all__ = ['Discriminator']
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, preprocess_GAN_mode, input_channel, distributed, conv_dim=64):
        super(Discriminator, self).__init__()
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        #layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(SpectralNorm(nn.Conv2d(input_channel, conv_dim, 4, 2, 1)))
        # layer1.append(nn.Conv2d(input_channel, conv_dim, 4, 2, 1))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim
        self.distributed = distributed
        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        # layer2.append(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        # layer3.append(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2


        layer4 = []
        layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        # layer4.append(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1))
        layer4.append(nn.LeakyReLU(0.1))
        self.l4 = nn.Sequential(*layer4)
        curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

        if preprocess_GAN_mode == 1: #'bn':
            if self.distributed:
                self.preprocess_additional = nn.SyncBatchNorm(input_channel)
            else:
                self.preprocess_additional = nn.BatchNorm2d(input_channel)
        elif preprocess_GAN_mode == 2: #'tanh':
            self.preprocess_additional = nn.Tanh()
        elif preprocess_GAN_mode == 3:
            self.preprocess_additional = lambda x: 2*(x/255 - 0.5)
        else:
            raise ValueError('preprocess_GAN_mode should be 1:bn or 2:tanh or 3:-1 - 1')

    def forward(self, x):
        #import pdb;pdb.set_trace()
        x = self.preprocess_additional(x)
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        #return [out.squeeze(), p1, p2]
        return [out, p1, p2]

if __name__ == '__main__':
    D_model = Discriminator(1, 128, 128, 256)
    img = torch.zeros((4, 128, 128, 256))
    out = D_model(img)
    print(out[0].shape)
