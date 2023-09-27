import gc
import torch

from torch import nn

from models.nets.vae import VAE
from utils import get_conv_output_size


class VAE_v1(VAE):
    """
    A Variational Autoencoder (VAE) with a convolutional encoder and a decoder
    having a similar reversed architecture.

    The encoder is equipped with batch normalization and LeakyReLU activation
    functions. In the middle of the network, three linear layers serve the
    purpose of sampling. A separate sigmoid activation function is applied to
    the output of the encoder, which is not included in the model setup.
    """

    # Default setting
    input_size: int = 512
    dims: [int] = [3, 8, 16, 32, 64, 128]
    latent_dim: int = 1024
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1

    def __init__(
        self,
        input_size: int = None,
        dims: [int] = None,
        latent_dim: int = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # Define input size
        if input_size is not None:
            self.input_size = input_size

        # Define dimensions array
        if dims is not None:
            self.dims = dims

        # Define latent dim
        if latent_dim is not None:
            self.latent_dim = latent_dim

        # Convert dimensions array to tuples: [3, 4, 4, 4, 4] to [(3, 4), (4, 4), (4, 4), (4, 4)]
        self.dims = [
            (self.dims[i], self.dims[i + 1]) for i in range(len(self.dims) - 1)
        ]

        # Build the encoder
        layers = []
        for dim in self.dims:
            in_c, out_c = dim
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_c,
                        out_channels=out_c,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                    ),
                    nn.BatchNorm2d(out_c),
                    nn.LeakyReLU(),
                )
            )
        self.encoder = nn.Sequential(*layers)

        # Build the decoder
        layers = []
        for dim in reversed(self.dims):
            out_c, in_c = dim
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=in_c,
                        out_channels=out_c,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(out_c),
                    nn.LeakyReLU(),
                )
            )
        self.decoder = nn.Sequential(*layers)

        # Get compressed latent size (encoder's output and decoder's input)
        compressed_conv_size = self.input_size
        for _ in range(len(self.dims)):
            compressed_conv_size = get_conv_output_size(
                input_size=compressed_conv_size,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
        self.compressed_conv_size = compressed_conv_size
        self.compressed_size = (
            self.compressed_conv_size * self.compressed_conv_size * self.dims[-1][-1]
        )

        # Compressed fully-connected layers (mu and sigma)
        self.fc_mu = nn.Linear(self.compressed_size, self.latent_dim)
        self.fc_sigma = nn.Linear(self.compressed_size, self.latent_dim)

        # Decoder input
        self.decoder_input = nn.Linear(self.latent_dim, self.compressed_size)

        # Latent distribution (mu & sigma) and representation

        # Un-referent and clear unreferenced instance
        layers = None
        gc.collect()

    def decode(self, x):
        """
        Decode from latent space into pixel space
        In this VAE implementation, we add a sigmoid layer to this decode function
        as we didn't include sigmoid in to the architecture before
        :param x:
        :return:
        """
        x = self.decoder_input(x)
        x = x.view(
            -1, self.dims[-1][-1], self.compressed_conv_size, self.compressed_conv_size
        )
        x = self.decoder(x)
        return torch.sigmoid(x)
