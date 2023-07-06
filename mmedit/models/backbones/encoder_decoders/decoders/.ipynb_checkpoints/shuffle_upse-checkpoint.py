import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
#from modelsummary import get_model_summary
from mmedit.models.registry import COMPONENTS
import numpy as np
def norm(dim, bn=False):
    if(bn is False):
        return nn.GroupNorm(32, dim)
    else:
        return nn.BatchNorm2d(dim)    
class Ada_GAP(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale=scale/32
    
    def forward(self,supp_feat_c):
        supp_feat = supp_feat_c[:,:-1,:,:]
        mask = supp_feat_c[:,-1:,:,:]
        supp_feat = supp_feat * mask
        feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
        area = F.adaptive_avg_pool2d(mask, (int(feat_h*self.scale), int(feat_w*self.scale))) * feat_h * feat_w + 0.000005
        supp_feat = F.adaptive_avg_pool2d(supp_feat, (int(feat_h*self.scale), int(feat_w*self.scale))) * feat_h * feat_w / area  
        return nn.functional.interpolate(supp_feat,(feat_h, feat_w),mode='bilinear', align_corners=False)
class tripool(nn.Module):

    def __init__(self,
                 channels,pool_size,
                 stride=1):
        super(tripool, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=pool_size, stride=stride,padding=(pool_size-1)//2,count_include_pad=False)
        self.conv = nn.Sequential(nn.Conv2d(channels, 256, kernel_size=1, bias=True),norm(256, True),
                nn.LeakyReLU(inplace=True))
    def forward(self, supp_feat_c):
        supp_feat = supp_feat_c[:,:-1,:,:]
        mask = supp_feat_c[:,-1:,:,:]
        supp_feat = self.conv(supp_feat)
        supp_feat_m = supp_feat * mask
        out = self.avgpool(supp_feat_m)/(self.avgpool(mask)+1e-6)
        return out


from .vops import _spdynamic_cuda 
##########################bl
class global_local_fusion(nn.Module):

    def __init__(self,
                 channels,inter_channels,out_channels,
                 kernel_size,upscale_factor,reduction_ratio = 4,
                 stride=1):
        super(global_local_fusion, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        #reduction_ratio = 4
        self.inter_channels = inter_channels
        self.group_channels = 16
        self.groups = self.inter_channels // self.group_channels
        self.reduce = nn.Conv2d(in_channels=channels,
            out_channels=inter_channels,
            kernel_size=1)
        self.out = nn.Sequential(nn.Conv2d(in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=1),nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=inter_channels,
            out_channels=inter_channels // reduction_ratio,
            kernel_size=1))
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=inter_channels // reduction_ratio,
            out_channels=kernel_size**2 * self.groups,
            kernel_size=3,padding=1)
        self.embconv = nn.Conv2d(in_channels=256,
            out_channels=inter_channels // reduction_ratio,
            kernel_size=1)
        self.upscale_factor = upscale_factor

    def forward(self, high_level,low_level,emb):
        cat = torch.cat([F.pixel_shuffle(high_level, upscale_factor=self.upscale_factor),low_level],dim=1)
        cat = self.reduce(cat)
        weight = self.conv2(self.relu(self.conv1(cat)+self.embconv(emb)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size, self.kernel_size, h, w)
        out = _spdynamic_cuda(cat, weight, stride=self.stride, padding=(self.kernel_size-1)//2)
        return self.out(out)



@COMPONENTS.register_module()
class TMFdecoder(nn.Module):
    def __init__(self,atrous_rates=None,batch_norm=True):
        super(TMFdecoder, self).__init__()
        pool_ksize = (31,17, 11, 5)
        self.batch_norm = batch_norm

        self.ppm = []
        for ksize in pool_ksize:
            self.ppm.append(tripool(2048,ksize))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_up1 = nn.Sequential(
            nn.Conv2d(2048 + len(self.ppm) * 256, 256,
                     kernel_size=3, padding=1, bias=True),

            norm(256, self.batch_norm),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            norm(256, self.batch_norm),
            nn.LeakyReLU(inplace=True)
        )
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        '''
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(256 + 256, 256,
                     kernel_size=3, padding=1, bias=True),
            norm(256, self.batch_norm),
            nn.LeakyReLU(inplace=True)
        )
        '''
        self.conv_up2 = global_local_fusion(channels=64 + 256, inter_channels=256,out_channels=256,kernel_size=3,upscale_factor=2)
        '''
        if(self.batch_norm):
            d_up3 = 128
        else:
            d_up3 = 64
        
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(256 + d_up3, 64,
                     kernel_size=3, padding=1, bias=True),
            norm(64, self.batch_norm),
            nn.LeakyReLU(inplace=True)
        )


        self.conv_up4 = nn.Sequential(
            nn.Conv2d(64 + 3 + 3 , 32,
                      kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 16,
                      kernel_size=3, padding=1, bias=True),

            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, padding=0, bias=True)
        )
        '''
        self.conv_up3 = global_local_fusion(channels=64 + 128, inter_channels=256,out_channels=64,kernel_size=3,upscale_factor=2)
        self.conv_up4_1 = global_local_fusion(channels=16 + 3 + 3, inter_channels=32,out_channels=32,kernel_size=3,upscale_factor=2)
        self.conv_up4_2 = nn.Sequential(nn.Conv2d(32, 16,
                      kernel_size=3, padding=1, bias=True),nn.LeakyReLU(inplace=True),nn.Conv2d(16, 1, kernel_size=1, padding=0, bias=True))
        
    def forward(self, conv_out):
        conv5 = conv_out[-1]
        mask = conv_out[0][:,-2:,:,:].sum(1).unsqueeze(1)
        input_size = conv5.size()
        mask = nn.functional.interpolate(mask,(input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)
        conv5_c = torch.cat([conv5,mask],dim=1)
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(pool_scale(conv5_c))
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_up1(ppm_out)
        emb = self.globalpool(x)#/(self.globalpool(mask)+1e-6)

        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        #x = F.interpolate(x, size=(conv_out[-4].size(2), conv_out[-4].size(3)), mode='bilinear', align_corners=True)
        #x = torch.cat((x, conv_out[-4]), 1)
        x = self.conv_up2(x,conv_out[-4],emb)

        #x = torch.cat((x, conv_out[-5]), 1)
        x = self.conv_up3(x, conv_out[-5],emb)

        #x = F.interpolate(x, size=(conv_out[-6].size(2), conv_out[-6].size(3)), mode='bilinear', align_corners=True)
        #x = torch.cat((x, conv_out[-6]), 1)
        
        x = self.conv_up4_1(x, conv_out[-6],emb)
        output = self.conv_up4_2(x)

        return output   
    
