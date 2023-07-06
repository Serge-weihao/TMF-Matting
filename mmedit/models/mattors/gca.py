import torch
from mmcv.runner import auto_fp16
import mmcv
from ..builder import build_loss
from ..registry import MODELS
from .base_mattor import BaseMattor
from .utils import get_unknown_tensor
import numpy as np

@MODELS.register_module()
class GCA(BaseMattor):
    """Guided Contextual Attention image matting model.

    https://arxiv.org/abs/2001.04069

    Args:
        backbone (dict): Config of backbone.
        train_cfg (dict): Config of training. In ``train_cfg``,
            ``train_backbone`` should be specified. If the model has a refiner,
            ``train_refiner`` should be specified.
        test_cfg (dict): Config of testing. In ``test_cfg``, If the model has a
            refiner, ``train_refiner`` should be specified.
        pretrained (str): Path of the pretrained model.
        loss_alpha (dict): Config of the alpha prediction loss. Default: None.
    """

    def __init__(self,
                 backbone,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_alpha=None):
        super(GCA, self).__init__(backbone, None, train_cfg, test_cfg,
                                  pretrained)
        self.loss_alpha = build_loss(loss_alpha)
        # support fp16
        self.fp16_enabled = False

    @auto_fp16(apply_to=('x', ))
    def _forward(self, x):
        raw_alpha = self.backbone(x)
        pred_alpha = (raw_alpha.tanh() + 1.0) / 2.0
        return pred_alpha

    def forward_dummy(self, inputs):
        return self._forward(inputs)

    def forward_train(self, merged, trimap, meta, alpha):
        """Forward function for training GCA model.

        Args:
            merged (Tensor): with shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            trimap (Tensor): with shape (N, C', H, W). Tensor of trimap. C'
                might be 1 or 3.
            meta (list[dict]): Meta data about the current data batch.
            alpha (Tensor): with shape (N, 1, H, W). Tensor of alpha.

        Returns:
            dict: Contains the loss items and batch infomation.
        """
        pred_alpha = self._forward(torch.cat((merged, trimap), 1))

        weight = get_unknown_tensor(trimap, meta)
        losses = {'loss': self.loss_alpha(pred_alpha, alpha, weight)}
        return {'losses': losses, 'num_samples': merged.size(0)}

    def forward_test(self,
                     merged,
                     trimap,
                     meta,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Defines the computation performed at every test call.

        Args:
            merged (Tensor): Image to predict alpha matte.
            trimap (Tensor): Trimap of the input image.
            meta (list[dict]): Meta data about the current data batch.
                Currently only batch_size 1 is supported. It may contain
                information needed to calculate metrics (``ori_alpha`` and
                ``ori_trimap``) or save predicted alpha matte
                (``merged_path``).
            save_image (bool, optional): Whether save predicted alpha matte.
                Defaults to False.
            save_path (str, optional): The directory to save predicted alpha
                matte. Defaults to None.
            iteration (int, optional): If given as None, the saved alpha matte
                will have the same file name with ``merged_path`` in meta dict.
                If given as an int, the saved alpha matte would named with
                postfix ``_{iteration}.png``. Defaults to None.

        Returns:
            dict: Contains the predicted alpha and evaluation result.
        """
        pred_alpha = self._forward(torch.cat((merged, trimap), 1))
        pred_alpha = pred_alpha.detach().cpu().numpy().squeeze()
        pred_alpha = self.restore_shape(pred_alpha, meta)
        eval_result = self.evaluate(pred_alpha, meta)

        if save_image:
            self.save_image(pred_alpha, meta, save_path, iteration)

        return {'pred_alpha': pred_alpha, 'eval_result': eval_result}
    
    def restore_shape(self, pred_alpha, meta):
        """Restore the predicted alpha to the original shape.

        The shape of the predicted alpha may not be the same as the shape of
        original input image. This function restores the shape of the predicted
        alpha.

        Args:
            pred_alpha (np.ndarray): The predicted alpha.
            meta (list[dict]): Meta data about the current data batch.
                Currently only batch_size 1 is supported.

        Returns:
            np.ndarray: The reshaped predicted alpha.
        """
        ori_trimap = meta[0]['ori_trimap'].squeeze()
        trimap = meta[0]['trimap'].squeeze()
        #print(trimap)
        ori_h, ori_w = meta[0]['merged_ori_shape'][:2]
        if 're_ori_shape' in meta[0]:
            rh,rw = meta[0]['re_ori_shape'][:2]
        else:
            rh,rw = ori_h, ori_w
        
        pred_alpha = np.clip(pred_alpha, 0, 1)
        pred_alpha[trimap == 0] = 0.
        pred_alpha[trimap == 255] = 1.
        if 'interpolation' in meta[0]:
            # images have been resized for inference, resize back
            pred_alpha = mmcv.imresize(
                pred_alpha, (ori_w, ori_h),
                interpolation=meta[0]['interpolation'])
        elif 'pad' in meta[0]:
            # images have been padded for inference, remove the padding
            pred_alpha = pred_alpha[:rh, :rw]

        

        # some methods do not have an activation layer after the last conv,
        # clip to make sure pred_alpha range from 0 to 1.
        pred_alpha = mmcv.imresize(
                pred_alpha, (ori_w, ori_h),
                interpolation='bicubic')
        pred_alpha[ori_trimap == 0] = 0.
        pred_alpha[ori_trimap == 255] = 1.
        assert pred_alpha.shape == (ori_h, ori_w)
        return pred_alpha
