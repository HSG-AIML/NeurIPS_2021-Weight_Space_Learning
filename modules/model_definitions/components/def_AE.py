# -*- coding: utf-8 -*-

import torch.nn as nn

from .def_encoder_attn import EncoderTransformer
from .def_decoder_attn import DecoderTransformer

from .def_encoder import Encoder
from .def_decoder import Decoder


###############################################################################
# define regular AE
# ##############################################################################


class AE(nn.Module):
    """
    tbd
    """

    def __init__(self, config):
        super(AE, self).__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x):
        z = self.forward_encoder(x)
        y = self.forward_decoder(z)
        return z, y

    def forward_encoder(self, x):
        z = self.encoder(x)
        return z

    def forward_decoder(self, z):
        y = self.decoder(z)
        return y


###############################################################################
# define Attention AE
# ##############################################################################


class AE_attn(nn.Module):
    """
    tbd
    """

    def __init__(self, config):
        super(AE_attn, self).__init__()

        self.encoder = EncoderTransformer(config)
        self.decoder = DecoderTransformer(config)

    def forward(self, x):
        z = self.forward_encoder(x)
        y = self.forward_decoder(z)
        return z, y

    def forward_encoder(self, x):
        z, _ = self.encoder(x)
        return z

    def forward_decoder(self, z):
        y, _ = self.decoder(z)
        return y

