import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .pixelwise_loss import l1_loss

_reduction_modes = ['none', 'mean', 'sum']


@LOSSES.register_module()
class GradientLoss(nn.Module):
    """Gradient loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(GradientLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        kx = torch.Tensor([[1, 0, -1], [2, 0, -2],
                           [1, 0, -1]]).view(1, 1, 3, 3).to(target)
        ky = torch.Tensor([[1, 2, 1], [0, 0, 0],
                           [-1, -2, -1]]).view(1, 1, 3, 3).to(target)

        pred_grad_x = F.conv2d(pred, kx, padding=1)
        pred_grad_y = F.conv2d(pred, ky, padding=1)
        target_grad_x = F.conv2d(target, kx, padding=1)
        target_grad_y = F.conv2d(target, ky, padding=1)

        loss = (
            l1_loss(
                pred_grad_x, target_grad_x, weight, reduction=self.reduction) +
            l1_loss(
                pred_grad_y, target_grad_y, weight, reduction=self.reduction))
        return loss * self.loss_weight

@LOSSES.register_module()
class GradientCompLoss(nn.Module):
    """Gradient loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(GradientCompLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')

    def forward(self, pred_alpha, fg, bg, ori_merged, weight=None):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        target = ori_merged
        kx = torch.Tensor([[1, 0, -1], [2, 0, -2],
                           [1, 0, -1]]).view(1, 1, 3, 3).repeat(3,1,1,1).to(target)
        ky = torch.Tensor([[1, 2, 1], [0, 0, 0],
                           [-1, -2, -1]]).view(1, 1, 3, 3).repeat(3,1,1,1).to(target)
        
        pred = pred_alpha * fg + (1. - pred_alpha) * bg
        
        pred_grad_x = F.conv2d(pred, kx, padding=1,groups=3)
        pred_grad_y = F.conv2d(pred, ky, padding=1,groups=3)
        target_grad_x = F.conv2d(target, kx, padding=1,groups=3)
        target_grad_y = F.conv2d(target, ky, padding=1,groups=3)

        loss = (
            l1_loss(
                pred_grad_x, target_grad_x, weight, reduction=self.reduction) +
            l1_loss(
                pred_grad_y, target_grad_y, weight, reduction=self.reduction))
        return loss * self.loss_weight


    
@LOSSES.register_module()
class LapLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(LapLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.gauss_filter = torch.tensor([[1., 4., 6., 4., 1.],
                                        [4., 16., 24., 16., 4.],
                                        [6., 24., 36., 24., 6.],
                                        [4., 16., 24., 16., 4.],
                                        [1., 4., 6., 4., 1.]]).cuda()
        self.gauss_filter /= 256.
        self.gauss_filter = self.gauss_filter.repeat(1, 1, 1, 1)
    def forward(self, pred, target, weight=None):
        return self.loss_weight*self.lap_loss(pred, target,self.gauss_filter,weight=weight)
    
    def lap_loss(self,logit, target, gauss_filter, weight=None):
            '''
            Based on FBA Matting implementation:
            https://gist.github.com/MarcoForte/a07c40a2b721739bb5c5987671aa5270
            '''
            def regression_loss(logit, target, weight=None):
                """
                Alpha reconstruction loss
                :param logit:
                :param target:
                :param loss_type: "l1" or "l2"
                :param weight: tensor with shape [N,1,H,W] weights for each pixel
                :return:
                """
                if weight is None:
                    return F.l1_loss(logit, target)
                else:
                    return F.l1_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
            def conv_gauss(x, kernel):
                x = F.pad(x, (2,2,2,2), mode='reflect')
                x = F.conv2d(x, kernel, groups=x.shape[1])
                return x

            def downsample(x):
                return x[:, :, ::2, ::2]

            def upsample(x, kernel):
                N, C, H, W = x.shape
                cc = torch.cat([x, torch.zeros(N,C,H,W).cuda()], dim = 3)
                cc = cc.view(N, C, H*2, W)
                cc = cc.permute(0,1,3,2)
                cc = torch.cat([cc, torch.zeros(N, C, W, H*2).cuda()], dim = 3)
                cc = cc.view(N, C, W*2, H*2)
                x_up = cc.permute(0,1,3,2)
                return conv_gauss(x_up, kernel=4*gauss_filter)
            def lap_pyramid(x, kernel, max_levels=3):
                current = x
                pyr = []
                for level in range(max_levels):
                    filtered = conv_gauss(current, kernel)
                    down = downsample(filtered)
                    up = upsample(down, kernel)
                    diff = current - up
                    pyr.append(diff)
                    current = down
                return pyr

            def weight_pyramid(x, max_levels=3):
                current = x
                pyr = []
                for level in range(max_levels):
                    down = downsample(current)
                    pyr.append(current)
                    current = down
                return pyr

            pyr_logit = lap_pyramid(x = logit, kernel = gauss_filter, max_levels = 5)
            pyr_target = lap_pyramid(x = target, kernel = gauss_filter, max_levels = 5)
            if weight is not None:
                pyr_weight = weight_pyramid(x = weight, max_levels = 5)
                return sum(regression_loss(A[0], A[1], weight=A[2]) * (2**i) for i, A in enumerate(zip(pyr_logit, pyr_target, pyr_weight)))
            else:
                return sum(regression_loss(A[0], A[1], weight=None) * (2**i) for i, A in enumerate(zip(pyr_logit, pyr_target)))