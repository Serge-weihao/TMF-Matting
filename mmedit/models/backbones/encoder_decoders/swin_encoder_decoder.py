import torch.nn as nn

from mmedit.models.builder import build_component
from mmedit.models.registry import BACKBONES
from mmcv.runner import load_checkpoint
from mmedit.utils import get_root_logger

@BACKBONES.register_module()
class SwinEncoderDecoder(nn.Module):
    """Simple encoder-decoder model from matting.

    Args:
        encoder (dict): Config of the encoder.
        decoder (dict): Config of the decoder.
    """

    def __init__(self, encoder, decoder):
        super(SwinEncoderDecoder, self).__init__()

        self.encoder = build_component(encoder)
        self.decoder = build_component(decoder)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)


    def forward(self, *args, **kwargs):
        """Forward function.

        Returns:
            Tensor: The output tensor of the decoder.
        """
        out = self.encoder(*args, **kwargs)
        out = self.decoder(out)
        return out
