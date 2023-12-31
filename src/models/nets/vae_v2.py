"""
Implementation tried for VAE architecture, however all VAE model has been discarded
due to low-performance
"""
import gc

from torch import nn

from models.nets.vae import VAE
from utils import get_conv_output_size


class VAE_v2(VAE):
    """
    This class represents another implementation of a Variational Autoencoder (VAE),
    where a sigmoid activation function is embedded inside the decoder architecture.

    The VAE consists of an encoder, a decoder, and a sampling layer in between.
    The encoder transforms the input data into a latent representation, from which
    samples are drawn via the reparameterisation trick. The decoder then reconstructs
    the original data from these samples, applying a sigmoid activation function
    as part of its architecture to ensure the output is in the appropriate range.
    """

    # Default setting
    intput_size: int = 512
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
            self.intput_size = input_size

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
        for dim in reversed(self.dims[1:]):
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
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=self.dims[0][1],
                    out_channels=self.dims[0][1],
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    output_padding=1,
                ),
                nn.BatchNorm2d(self.dims[0][1]),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=self.dims[0][1],
                    out_channels=self.dims[0][0],
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                ),
                nn.Sigmoid()
            )
        )
        self.decoder = nn.Sequential(*layers)

        # Get compressed latent size (encoder's output and decoder's input)
        compressed_conv_size = self.intput_size
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
