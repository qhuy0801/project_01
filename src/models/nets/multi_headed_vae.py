from copy import deepcopy

import torch

from models.nets.vae import VAE


class MultiHeadedVAE(VAE):
    def __init__(self, input_size: int = None, dims: [int] = None, latent_dim: int = None, *args, **kwargs) -> None:
        super().__init__(input_size, dims, latent_dim, *args, **kwargs)
        # Make 2 more copy of the decoder
        self.decoder_1 = deepcopy(self.decoder)
        self.decoder_2 = deepcopy(self.decoder)

    def decode_1(self, x):
        """
        Decode from latent space into pixel space
        :param x:
        :return:
        """
        x = self.decoder_input(x)
        x = x.view(
            -1, self.dims[-1][-1], self.compressed_conv_size, self.compressed_conv_size
        )
        x = self.decoder_1(x)
        return torch.sigmoid(x)

    def decode_2(self, x):
        """
        Decode from latent space into pixel space
        :param x:
        :return:
        """
        x = self.decoder_input(x)
        x = x.view(
            -1, self.dims[-1][-1], self.compressed_conv_size, self.compressed_conv_size
        )
        x = self.decoder_2(x)
        return torch.sigmoid(x)
