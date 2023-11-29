"""
Implementation tried for VAE architecture, however all VAE model has been discarded
due to low-performance
"""
import gc

from torch import nn

from models.nets.vae import VAE
from utils import get_conv_output_size


class VAE_v3(VAE):
    # Default setting
    input_size: int = 128
    encoder_dim: [int] = [3, 4]
    decoder_dim: [int] = [4, 4, 3]
    latent_dim = int = 512
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1

    def __init__(
        self,
        input_size: int = None,
        encoder_dim: [int] = None,
        decoder_dim: [int] = None,
        latent_dim: int = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # None-check all settings and grant
        if input_size is not None:
            self.input_size = input_size

        if encoder_dim is not None:
            self.encoder_dim = encoder_dim

        if decoder_dim is not None:
            self.decoder_dim = decoder_dim

        if latent_dim is not None:
            self.latent_dim = latent_dim

        # Convert array dimensions to tuples
        self.encoder_dim = [
            (self.encoder_dim[i], self.encoder_dim[i + 1])
            for i in range(len(self.encoder_dim) - 1)
        ]

        self.decoder_dim = [
            (self.decoder_dim[i], self.decoder_dim[i + 1])
            for i in range(len(self.decoder_dim) - 1)
        ]

        # Build encoder
        layers = []
        for dim in self.encoder_dim:
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

        # Build decoder
        layers = []
        for dim in self.decoder_dim[:-1]:
            in_c, out_c = dim
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=in_c,
                        out_channels=out_c,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(out_c),
                    nn.LeakyReLU(),
                )
            )
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=self.decoder_dim[-1][0],
                    out_channels=self.decoder_dim[-1][0],
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    output_padding=1,
                ),
                nn.BatchNorm2d(self.decoder_dim[-1][0]),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=self.decoder_dim[-1][0],
                    out_channels=self.decoder_dim[-1][1],
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                ),
                nn.Tanh(),
            )
        )
        self.decoder = nn.Sequential(*layers)

        # Get compressed latent size (encoder's output and decoder's input)
        compressed_conv_size = self.input_size
        for _ in range(len(self.encoder_dim)):
            compressed_conv_size = get_conv_output_size(
                input_size=compressed_conv_size,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
        self.compressed_conv_size = compressed_conv_size
        self.compressed_size = (
            self.compressed_conv_size
            * self.compressed_conv_size
            * self.encoder_dim[-1][-1]
        )

        # Compressed fully-connected layers (mu and sigma)
        self.fc_mu = nn.Linear(self.compressed_size, self.latent_dim)
        self.fc_sigma = nn.Linear(self.compressed_size, self.latent_dim)

        # Decoder input
        self.decoder_input = nn.Linear(self.latent_dim, self.compressed_size)

        # Un-referent and clear unreferenced instance
        layers = None
        gc.collect()

    def forward(self, x):
        """
        Modified forward function
        :param x:
        :return:
        """
        output, mu, sigma = super().forward(x)
        return (
            nn.functional.interpolate(
                output,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ),
            output,
            mu,
            sigma,
        )

    def decode(self, x):
        """
        Decode from latent space into pixel space
        :param x:
        :return:
        """
        x = self.decoder_input(x)
        x = x.view(
            -1,
            self.decoder_dim[0][0],
            self.compressed_conv_size,
            self.compressed_conv_size,
        )
        return self.decoder(x)
