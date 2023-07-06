import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init
import torch
from mmedit.models.common import GCAModule
from mmedit.models.registry import COMPONENTS
from ..encoders.resnet_enc import BasicBlock


class BasicBlockDec(BasicBlock):
    """Basic residual block for decoder.

    For decoder, we use ConvTranspose2d with kernel_size 4 and padding 1 for
    conv1. And the output channel of conv1 is modified from `out_channels` to
    `in_channels`.
    """

    def build_conv1(self, in_channels, out_channels, kernel_size, stride,
                    conv_cfg, norm_cfg, act_cfg, with_spectral_norm):
        """Build conv1 of the block.

        Args:
            in_channels (int): The input channels of the ConvModule.
            out_channels (int): The output channels of the ConvModule.
            kernel_size (int): The kernel size of the ConvModule.
            stride (int): The stride of the ConvModule. If stride is set to 2,
                then ``conv_cfg`` will be overwritten as
                ``dict(type='Deconv')`` and ``kernel_size`` will be overwritten
                as 4.
            conv_cfg (dict): The conv config of the ConvModule.
            norm_cfg (dict): The norm config of the ConvModule.
            act_cfg (dict): The activation config of the ConvModule.
            with_spectral_norm (bool): Whether use spectral norm.

        Returns:
            nn.Module: The built ConvModule.
        """
        if stride == 2:
            conv_cfg = dict(type='Deconv')
            kernel_size = 4
            padding = 1
        else:
            padding = kernel_size // 2

        return ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            with_spectral_norm=with_spectral_norm)

    def build_conv2(self, in_channels, out_channels, kernel_size, conv_cfg,
                    norm_cfg, with_spectral_norm):
        """Build conv2 of the block.

        Args:
            in_channels (int): The input channels of the ConvModule.
            out_channels (int): The output channels of the ConvModule.
            kernel_size (int): The kernel size of the ConvModule.
            conv_cfg (dict): The conv config of the ConvModule.
            norm_cfg (dict): The norm config of the ConvModule.
            with_spectral_norm (bool): Whether use spectral norm.

        Returns:
            nn.Module: The built ConvModule.
        """
        return ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm)


@COMPONENTS.register_module()
class ResNetDec(nn.Module):
    """ResNet decoder for image matting.

    This class is adopted from https://github.com/Yaoyi-Li/GCA-Matting.

    Args:
        block (str): Type of residual block. Currently only `BasicBlockDec` is
            implemented.
        layers (list[int]): Number of layers in each block.
        in_channels (int): Channel num of input features.
        kernel_size (int): Kernel size of the conv layers in the decoder.
        conv_cfg (dict): dictionary to construct convolution layer. If it is
            None, 2d convolution will be applied. Default: None.
        norm_cfg (dict): Config dict for normalization layer. "BN" by default.
        act_cfg (dict): Config dict for activation layer, "ReLU" by default.
        with_spectral_norm (bool): Whether use spectral norm after conv.
            Default: False.
        late_downsample (bool): Whether to adopt late downsample strategy,
            Default: False.
    """

    def __init__(self,
                 block,
                 layers,
                 in_channels,
                 kernel_size=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(
                     type='LeakyReLU', negative_slope=0.2, inplace=True),
                 with_spectral_norm=False,
                 late_downsample=False):
        super(ResNetDec, self).__init__()
        if block == 'BasicBlockDec':
            block = BasicBlockDec
        else:
            raise NotImplementedError(f'{block} is not implemented.')

        self.kernel_size = kernel_size
        self.inplanes = in_channels
        self.midplanes = 64 if late_downsample else 32

        self.layer1 = self._make_layer(block, 256, layers[0], conv_cfg,
                                       norm_cfg, act_cfg, with_spectral_norm)
        self.layer2 = self._make_layer(block, 128, layers[1], conv_cfg,
                                       norm_cfg, act_cfg, with_spectral_norm)
        self.layer3 = self._make_layer(block, 64, layers[2], conv_cfg,
                                       norm_cfg, act_cfg, with_spectral_norm)
        self.layer4 = self._make_layer(block, self.midplanes, layers[3],
                                       conv_cfg, norm_cfg, act_cfg,
                                       with_spectral_norm)

        self.conv1 = ConvModule(
            self.midplanes,
            32,
            4,
            stride=2,
            padding=1,
            conv_cfg=dict(type='Deconv'),
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            with_spectral_norm=with_spectral_norm)

        self.conv2 = ConvModule(
            32,
            1,
            self.kernel_size,
            padding=self.kernel_size // 2,
            act_cfg=None)

    def init_weights(self):
        """Init weights for the module.
        """
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m.weight, 1)
                constant_init(m.bias, 0)

        # Zero-initialize the last BN in each residual branch, so that the
        # residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlockDec):
                constant_init(m.conv2.bn.weight, 0)

    def _make_layer(self, block, planes, num_blocks, conv_cfg, norm_cfg,
                    act_cfg, with_spectral_norm):
        upsample = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            ConvModule(
                self.inplanes,
                planes * block.expansion,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
                with_spectral_norm=with_spectral_norm))

        layers = [
            block(
                self.inplanes,
                planes,
                kernel_size=self.kernel_size,
                stride=2,
                interpolation=upsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                with_spectral_norm=with_spectral_norm)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    kernel_size=self.kernel_size,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    with_spectral_norm=with_spectral_norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            Tensor: Output tensor.
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv1(x)
        x = self.conv2(x)

        return x

@COMPONENTS.register_module()
class ClassiDec(nn.Module):
    def __init__(self,in_channels,numclass=1000):
        super(ClassiDec, self).__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),nn.Linear(in_features=in_channels, out_features=numclass, bias=True))
    def forward(self, x):
        return self.fc(x)
@COMPONENTS.register_module()
class ClassiDecSC(nn.Module):
    def __init__(self,in_channels,numclass=1000):
        super(ClassiDecSC, self).__init__()
        inc =  sum([32, 32, 64, 128, 256])+in_channels
        #print('wint',inc)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(nn.Linear(in_features=inc, out_features=numclass, bias=True))
    def forward(self, inputs):
        feat1 = inputs['feat1']
        feat2 = inputs['feat2']
        feat3 = inputs['feat3']
        feat4 = inputs['feat4']
        feat5 = inputs['feat5']
        feats = [feat1,feat2,feat3,feat4,feat5]
        #for i in feats:
        #    print(i.size())
        top = inputs['out']
        x = torch.cat([self.pool(i) for i in feats]+[self.pool(top)],dim=1).squeeze(-1).squeeze(-1)
        #print(x.size())
        return self.fc(x)
@COMPONENTS.register_module()
class ResShortcutDec(ResNetDec):
    """ResNet decoder for image matting with shortcut connection.

    ::

        feat1 --------------------------- conv2 --- out
                                       |
        feat2 ---------------------- conv1
                                  |
        feat3 ----------------- layer4
                             |
        feat4 ------------ layer3
                        |
        feat5 ------- layer2
                   |
        out ---  layer1

    Args:
        block (str): Type of residual block. Currently only `BasicBlockDec` is
            implemented.
        layers (list[int]): Number of layers in each block.
        in_channels (int): Channel number of input features.
        kernel_size (int): Kernel size of the conv layers in the decoder.
        conv_cfg (dict): Dictionary to construct convolution layer. If it is
            None, 2d convolution will be applied. Default: None.
        norm_cfg (dict): Config dict for normalization layer. "BN" by default.
        act_cfg (dict): Config dict for activation layer, "ReLU" by default.
        late_downsample (bool): Whether to adopt late downsample strategy,
            Default: False.
    """

    def forward(self, inputs):
        """Forward function of resnet shortcut decoder.

        Args:
            inputs (dict): Output dictionary of the ResNetEnc containing:

              - out (Tensor): Output of the ResNetEnc.
              - feat1 (Tensor): Shortcut connection from input image.
              - feat2 (Tensor): Shortcut connection from conv2 of ResNetEnc.
              - feat3 (Tensor): Shortcut connection from layer1 of ResNetEnc.
              - feat4 (Tensor): Shortcut connection from layer2 of ResNetEnc.
              - feat5 (Tensor): Shortcut connection from layer3 of ResNetEnc.

        Returns:
            Tensor: Output tensor.
        """
        feat1 = inputs['feat1']
        feat2 = inputs['feat2']
        feat3 = inputs['feat3']
        feat4 = inputs['feat4']
        feat5 = inputs['feat5']
        x = inputs['out']

        x = self.layer1(x) + feat5
        x = self.layer2(x) + feat4
        x = self.layer3(x) + feat3
        x = self.layer4(x) + feat2
        x = self.conv1(x) + feat1
        x = self.conv2(x)

        return x


@COMPONENTS.register_module()
class ResGCADecoder(ResShortcutDec):
    """ResNet decoder with shortcut connection and gca module.

    ::

        feat1 ---------------------------------------- conv2 --- out
                                                    |
        feat2 ----------------------------------- conv1
                                               |
        feat3 ------------------------------ layer4
                                          |
        feat4, img_feat -- gca_module - layer3
                        |
        feat5 ------- layer2
                   |
        out ---  layer1

    * gca module also requires unknown tensor generated by trimap which is \
    ignored in the above graph.

    Args:
        block (str): Type of residual block. Currently only `BasicBlockDec` is
            implemented.
        layers (list[int]): Number of layers in each block.
        in_channels (int): Channel number of input features.
        kernel_size (int): Kernel size of the conv layers in the decoder.
        conv_cfg (dict): Dictionary to construct convolution layer. If it is
            None, 2d convolution will be applied. Default: None.
        norm_cfg (dict): Config dict for normalization layer. "BN" by default.
        act_cfg (dict): Config dict for activation layer, "ReLU" by default.
        late_downsample (bool): Whether to adopt late downsample strategy,
            Default: False.
    """

    def __init__(self,
                 block,
                 layers,
                 in_channels,
                 kernel_size=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(
                     type='LeakyReLU', negative_slope=0.2, inplace=True),
                 with_spectral_norm=False,
                 late_downsample=False):
        super(ResGCADecoder,
              self).__init__(block, layers, in_channels, kernel_size, conv_cfg,
                             norm_cfg, act_cfg, with_spectral_norm,
                             late_downsample)
        self.gca = GCAModule(128, 128)

    def forward(self, inputs):
        """Forward function of resnet shortcut decoder.

        Args:
            inputs (dict): Output dictionary of the ResGCAEncoder containing:

              - out (Tensor): Output of the ResGCAEncoder.
              - feat1 (Tensor): Shortcut connection from input image.
              - feat2 (Tensor): Shortcut connection from conv2 of \
                    ResGCAEncoder.
              - feat3 (Tensor): Shortcut connection from layer1 of \
                    ResGCAEncoder.
              - feat4 (Tensor): Shortcut connection from layer2 of \
                    ResGCAEncoder.
              - feat5 (Tensor): Shortcut connection from layer3 of \
                    ResGCAEncoder.
              - img_feat (Tensor): Image feature extracted by guidance head.
              - unknown (Tensor): Unknown tensor generated by trimap.

        Returns:
            Tensor: Output tensor.
        """
        img_feat = inputs['img_feat']
        unknown = inputs['unknown']
        feat1 = inputs['feat1']
        feat2 = inputs['feat2']
        feat3 = inputs['feat3']
        feat4 = inputs['feat4']
        feat5 = inputs['feat5']
        x = inputs['out']

        x = self.layer1(x) + feat5
        x = self.layer2(x) + feat4
        x = self.gca(img_feat, x, unknown)
        x = self.layer3(x) + feat3
        x = self.layer4(x) + feat2
        x = self.conv1(x) + feat1
        x = self.conv2(x)

        return x
from ..hop import CONFIG
from .ops import GuidedCxtAttenEmbedding,LocalHOPBlock
@COMPONENTS.register_module()
class ResLocalHOP_PosEmb_Dec(ResShortcutDec):

    def __init__(self,
                 block,
                 layers,
                 in_channels,
                 kernel_size=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(
                     type='LeakyReLU', negative_slope=0.2, inplace=True),
                 with_spectral_norm=False,
                 late_downsample=False):
        super(ResLocalHOP_PosEmb_Dec,
              self).__init__(block, layers, in_channels, kernel_size, conv_cfg,
                             norm_cfg, act_cfg, with_spectral_norm,
                             late_downsample)
        self.gca = GuidedCxtAttenEmbedding(256 * 1,
                                           64,
                                           use_trimap_embed=True,
                                           rate=CONFIG.model.arch.global_hop_downsample,
                                           scale=CONFIG.model.arch.hop_softmax_scale,
                                           learnable_scale=CONFIG.model.arch.learnable_scale)
        self.localgca1 = LocalHOPBlock(128 * 1,
                                       64,
                                       ksize=CONFIG.model.arch.hop_ksize,
                                       use_pos_emb=True,
                                       scale=CONFIG.model.arch.hop_softmax_scale,
                                       learnable_scale=CONFIG.model.arch.learnable_scale)
        self.localgca2 = LocalHOPBlock(64 * 1,
                                       32,
                                       ksize=CONFIG.model.arch.hop_ksize,
                                       use_pos_emb=True,
                                       scale=CONFIG.model.arch.hop_softmax_scale,
                                       learnable_scale=CONFIG.model.arch.learnable_scale)
        self.localgca3 = LocalHOPBlock(32 * 1,
                                       16,
                                       ksize=CONFIG.model.arch.hop_ksize,
                                       use_pos_emb=True,
                                       scale=CONFIG.model.arch.hop_softmax_scale,
                                       learnable_scale=CONFIG.model.arch.learnable_scale)

    def forward(self, indict):#x, mid_fea):
        x = indict['out']
        mid_fea = indict['mid_fea']
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        im1, im2, im3, im4 = mid_fea['image_fea']
        x = self.layer1(x) + fea5 # N x 256 x 32 x 32
        x, offset = self.gca(im4, x, mid_fea['trimap'])
        x = self.layer2(x) + fea4 # N x 128 x 64 x 64
        x = self.localgca1(im3, x)
        x = self.layer3(x) + fea3 # N x 64 x 128 x 128
        x = self.localgca2(im2, x)
        x = self.layer4(x) + fea2 # N x 32 x 256 x 256
        x = self.localgca3(im1, x)
        x = self.conv1(x) + fea1
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x) + fea1
        '''
        alpha = self.conv2(x)

        #alpha = (self.tanh(x) + 1.0) / 2.0

        return alpha#, {'offset': offset}

import cv2   
Kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]
def get_unknown_tensor_from_pred(pred, rand_width=30, train_mode=True):
    ### pred: N, 1 ,H, W 
    N, C, H, W = pred.shape
    pred = F.interpolate(pred, size=(640,640), mode='nearest')
    pred = pred.data.cpu().numpy()
    uncertain_area = np.ones_like(pred, dtype=np.uint8)
    uncertain_area[pred<1.0/255.0] = 0
    uncertain_area[pred>1-1.0/255.0] = 0

    for n in range(N):
        uncertain_area_ = uncertain_area[n,0,:,:] # H, W
        if train_mode:
            width = np.random.randint(1, rand_width)
        else:
            width = rand_width // 2
        uncertain_area_ = cv2.dilate(uncertain_area_, Kernels[width])
        uncertain_area[n,0,:,:] = uncertain_area_

    weight = np.zeros_like(uncertain_area)
    weight[uncertain_area == 1] = 1
    weight = torch.from_numpy(weight).cuda()

    weight = F.interpolate(weight, size=(H,W), mode='nearest')

    return weight
class ResNet_D_Dec(nn.Module):

    def __init__(self, block, layers, norm_layer=None, large_kernel=False, late_downsample=False):
        super(ResNet_D_Dec, self).__init__()
        self.logger = logging.getLogger("Logger")
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.large_kernel = large_kernel
        self.kernel_size = 5 if self.large_kernel else 3

        self.inplanes = 512 if layers[0] > 0 else 256
        self.late_downsample = late_downsample
        self.midplanes = 64 if late_downsample else 32

        self.conv1 = SpectralNorm(nn.ConvTranspose2d(self.midplanes, 32, kernel_size=4, stride=2, padding=1, bias=False))
        self.bn1 = norm_layer(32)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.tanh = nn.Tanh()
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.midplanes, layers[3], stride=2)

        self.refine_OS1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False),
            norm_layer(32),
            self.leaky_relu,
            nn.Conv2d(32, 1, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),)
        
        self.refine_OS4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False),
            norm_layer(32),
            self.leaky_relu,
            nn.Conv2d(32, 1, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),)

        self.refine_OS8 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False),
            norm_layer(32),
            self.leaky_relu,
            nn.Conv2d(32, 1, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, "weight_bar"):
                    nn.init.xavier_uniform_(m.weight_bar)
                else:
                    nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

        self.logger.debug(self)

    def _make_layer(self, block, planes, blocks, stride=1):
        if blocks == 0:
            return nn.Sequential(nn.Identity())
        norm_layer = self._norm_layer
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                SpectralNorm(conv1x1(self.inplanes, planes * block.expansion)),
                norm_layer(planes * block.expansion),
            )
        elif self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                SpectralNorm(conv1x1(self.inplanes, planes * block.expansion)),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, upsample, norm_layer, self.large_kernel)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, large_kernel=self.large_kernel))

        return nn.Sequential(*layers)

    def forward(self, x, mid_fea):
        ret = {}

        x = self.layer1(x) # N x 256 x 32 x 32
        x = self.layer2(x) # N x 128 x 64 x 64
        x_os8 = self.refine_OS8(x)
        
        x = self.layer3(x) # N x 64 x 128 x 128
        x_os4 = self.refine_OS4(x)

        x = self.layer4(x) # N x 32 x 256 x 256
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x_os1 = self.refine_OS1(x)
        
        x_os4 = F.interpolate(x_os4, scale_factor=4.0, mode='bilinear', align_corners=False)
        x_os8 = F.interpolate(x_os8, scale_factor=8.0, mode='bilinear', align_corners=False)
        
        x_os1 = (torch.tanh(x_os1) + 1.0) / 2.0
        x_os4 = (torch.tanh(x_os4) + 1.0) / 2.0
        x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0

        ret['alpha_os1'] = x_os1
        ret['alpha_os4'] = x_os4
        ret['alpha_os8'] = x_os8

        return ret