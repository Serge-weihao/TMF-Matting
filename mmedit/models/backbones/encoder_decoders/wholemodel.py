import torch.nn as nn

from mmedit.models.builder import build_component
from mmedit.models.registry import BACKBONES


@BACKBONES.register_module()
class Wholemodel(nn.Module):
    """Simple encoder-decoder model from matting.

    Args:
        encoder (dict): Config of the encoder.
        decoder (dict): Config of the decoder.
    """

    def __init__(self, encoder):
        super(Wholemodel, self).__init__()

        self.encoder = build_component(encoder)

    def init_weights(self, pretrained=None):
        return

    def forward(self, *args, **kwargs):
        """Forward function.

        Returns:
            Tensor: The output tensor of the decoder.
        """
        out = self.encoder(*args, **kwargs)
        return out
