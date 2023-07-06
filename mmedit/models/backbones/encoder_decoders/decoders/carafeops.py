import torch
from torch import nn
from torch.nn import functional as F


class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''
    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation,
                 use_relu=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                c_in, c_out, kernel_size=kernel_size, stride=stride, 
                padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CARAFE(nn.Module):
    def __init__(self, channels, c_mid=64, scale_factor=2, up_kernel=5, k_enc=3,up_group=16):
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.
        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale_factor
        self.up_kernel=up_kernel
        self.group = up_group
        self.comp = ConvBNReLU(channels, c_mid, kernel_size=1, stride=1, 
                               padding=0, dilation=1)
        self.enc = ConvBNReLU(c_mid, (scale_factor*up_kernel)**2*self.group, kernel_size=k_enc, 
                              stride=1, padding=k_enc//2, dilation=1, 
                              use_relu=False)
        self.pix_shf = nn.PixelShuffle(scale_factor)

        self.upsmp = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=up_kernel, dilation=scale_factor, 
                                padding=up_kernel//2*scale_factor)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale
        
        W = self.comp(X)                                # b * m * h * w
        W = self.enc(W)                                 # b * 100 * h * w
        W = self.pix_shf(W)                             # b * 25 * h_ * w_
        W = F.softmax(W, dim=1)                         # b * 25 * h_ * w_

        X = self.upsmp(X)                               # b * c * h_ * w_
        X = self.unfold(X)                              # b * 25c * h_ * w_
        X = X.view(b,self.group,c//self.group, self.up_kernel**2, h_, w_)# b * 25 * c * h_ * w_
        W = W.view(b,self.group,self.up_kernel**2, h_, w_)
        X = torch.einsum('bgkhw,bgckhw->bgchw', [W, X]).reshape(b, c,h_, w_)   # b * c * h_ * w_
        return X
'''
from .vops import _involution_cuda 

class CARAFE(nn.Module):
    def __init__(self, channels, c_mid=64, scale_factor=2, up_kernel=5, k_enc=3,up_group=16):
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.
        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale_factor
        self.up_kernel=up_kernel
        self.group = up_group
        self.comp = ConvBNReLU(channels, c_mid, kernel_size=1, stride=1, 
                               padding=0, dilation=1)
        self.enc = ConvBNReLU(c_mid, (scale_factor*up_kernel)**2*self.group, kernel_size=k_enc, 
                              stride=1, padding=k_enc//2, dilation=1, 
                              use_relu=False)
        self.pix_shf = nn.PixelShuffle(scale_factor)

        self.upsmp = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=up_kernel, dilation=scale_factor, 
                                padding=up_kernel//2*scale_factor)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale
        
        W = self.comp(X)                                # b * m * h * w
        W = self.enc(W)                                 # b * 100 * h * w
        W = self.pix_shf(W)                             # b * 25 * h_ * w_
        weight = F.softmax(W, dim=1)                         # b * 25 * h_ * w_

        X = self.upsmp(X)                               # b * c * h_ * w_
        #X = self.unfold(X)                              # b * 25c * h_ * w_
        #X = X.view(b,self.group,c//self.group, self.up_kernel**2, h_, w_)# b * 25 * c * h_ * w_
        #W = W.view(b,self.group,self.up_kernel**2, h_, w_)
        #X = torch.einsum('bgkhw,bgckhw->bgchw', [W, X]).reshape(b, c,h_, w_)   # b * c * h_ * w_
        b, c, h, w = weight.shape
        weight = weight.view(b, self.group, self.up_kernel, self.up_kernel, h, w)
        print(X.shape,weight.shape)
        out = _involution_cuda(X, weight, stride=1, padding=(self.up_kernel-1)//2)
        return out
'''
