"""
Implementation tried for VAE architecture, however all VAE model has been discarded
due to low-performance
"""
import gc

import torch
from torch import nn

from utils import arr_to_tuples


class VAE_v4(nn.Module):
    def __init__(self, input_size: int, fc_dims: int = 128, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_size = input_size

        self.encoder = Encoder(compressed_dims=[32, 32])
        self.decoder = Decoder(decompressed_dims=[32, 512])

        self.latent_size = (
            self.encoder.compressed_dims[-1][-1],
            input_size // 2 ** len(self.encoder.down_sampling_dims),
            input_size // 2 ** len(self.encoder.down_sampling_dims),
        )
        __flatten_size = self.latent_size[0] * self.latent_size[1] * self.latent_size[2]

        self.fc_mu = nn.Linear(in_features=__flatten_size, out_features=fc_dims)
        self.fc_sigma = nn.Linear(in_features=__flatten_size, out_features=fc_dims)
        self.decoder_input = nn.Linear(in_features=fc_dims, out_features=__flatten_size)

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc_mu(x), self.fc_sigma(x)

    def reparameterise(self, mu, sigma):
        std = torch.exp(sigma * .5)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, x):
        x = self.decoder_input(x)
        x = x.view(
            -1, self.latent_size[0], self.latent_size[1], self.latent_size[2]
        )
        return self.decoder(x)

    def forward(self, x):
        mu, sigma = self.encode(x)
        x = self.reparameterise(mu, sigma)
        return self.decode(x), mu, sigma


class Multi_headed_VAE_v1(VAE_v4):
    def __init__(self, input_size: int, fc_dims: int = 128, *args, **kwargs) -> None:
        super().__init__(input_size, fc_dims, *args, **kwargs)
        self.additional_decoder_1 = Decoder(
            decompressed_dims=[32, 512], output_activation="tanh"
        )

    def additional_decode(self, x):
        x = self.decoder_input(x)
        x = x.view(
            -1,
            self.latent_size[0],
            self.latent_size[1],
            self.latent_size[2],
        )
        return self.additional_decoder_1(x)

    def additional_forward(self, x):
        mu, sigma = self.encode(x)
        x = self.reparameterise(mu, sigma)
        return self.decode(x), self.additional_decode(x), mu, sigma

    def forward(self, x):
        mu, sigma = self.encode(x)
        x = self.reparameterise(mu, sigma)
        x = self.decode(x) + self.additional_decode(x)
        return x, mu, sigma


class Autoencoder_v1(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = Encoder(compressed_dims=[32, 32])
        self.decoder = Decoder(decompressed_dims=[32, 512])

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class Multi_headed_AE(Autoencoder_v1):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.additional_decoder = Decoder(decompressed_dims=[32, 512], output_activation="tanh")

    def additional_decode(self, x):
        return self.additional_decoder(x)

    def original_forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def additional_forward(self, x):
        x = self.encode(x)
        x = x.detach().clone()
        x = self.additional_decode(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.additional_decode(x) + self.decode(x)
        return torch.clamp(x, min=0, max=1)


class Decoder(nn.Module):
    # Default settings
    decompressed_dims: [int] = [4, 512]
    up_sampling_dims: [int] = [512, 256, 128]
    output_dim: int = 3

    def __init__(
        self,
        decompressed_dims: [int] = None,
        up_sampling_dims: [int] = None,
        output_dim: int = None,
        output_activation: str = "sigmoid",
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # None check
        if decompressed_dims is not None:
            self.decompressed_dims = decompressed_dims

        if up_sampling_dims is not None:
            self.up_sampling_dims = up_sampling_dims
        self.up_sampling_dims = [self.decompressed_dims[-1], *self.up_sampling_dims]

        if output_dim is not None:
            self.output_dim = output_dim
        self.output_dim = [self.up_sampling_dims[-1], self.output_dim]

        # Convert array to tuples
        self.decompressed_dims = arr_to_tuples(self.decompressed_dims)
        self.up_sampling_dims = arr_to_tuples(self.up_sampling_dims)
        self.output_dim = arr_to_tuples(self.output_dim)

        # Configure the layers
        decompress_layer = []
        for in_c, out_c in self.decompressed_dims:
            decompress_layer.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_c,
                        out_channels=out_c,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(out_c),
                    nn.LeakyReLU(),
                )
            )
        self.decompressor = nn.Sequential(*decompress_layer)

        up_sampling_layers = []
        for in_c, out_c in self.up_sampling_dims:
            up_sampling_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=in_c,
                        out_channels=out_c,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(out_c),
                    nn.LeakyReLU(),
                )
            )
        self.up_sampler = nn.Sequential(*up_sampling_layers)

        output_layers = []
        for in_c, out_c in self.output_dim:
            output_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_c,
                        out_channels=out_c,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(out_c),
                    nn.LeakyReLU(),
                )
            )
        output_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.output_dim[-1][-1],
                    out_channels=self.output_dim[-1][-1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                # nn.BatchNorm2d(self.output_dim[-1][-1]),
                nn.Tanh() if output_activation == "tanh" else nn.Sigmoid(),
            )
        )
        self.to_output = nn.Sequential(*output_layers)

        # Un-reference and collect
        decompress_layer = None
        up_sampling_dims = None
        output_layers = None
        gc.collect()

    def forward(self, x):
        x = self.decompressor(x)
        x = self.up_sampler(x)
        x = self.to_output(x)
        return x


class Encoder(nn.Module):
    # Default settings
    input_dim: int = 3
    feature_extract_dims: [int] = [128]
    down_sampling_dims: [int] = [128, 256, 512]
    compressed_dims: [int] = [8, 4]

    def __init__(
        self,
        input_dim: [int] = None,
        feature_extract_dims: [int] = None,
        down_sampling_dims: [int] = None,
        compressed_dims: [int] = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # None check
        if input_dim is not None:
            self.input_dim = input_dim

        if feature_extract_dims is not None:
            self.feature_extract_dims = feature_extract_dims
        self.feature_extract_dims = [self.input_dim, *self.feature_extract_dims]

        if down_sampling_dims is not None:
            self.down_sampling_dims = down_sampling_dims
        self.down_sampling_dims = [
            self.feature_extract_dims[-1],
            *self.down_sampling_dims,
        ]

        if compressed_dims is not None:
            self.compressed_dims = compressed_dims
        self.compressed_dims = [self.down_sampling_dims[-1], *self.compressed_dims]

        self.feature_extract_dims = arr_to_tuples(self.feature_extract_dims)
        self.down_sampling_dims = arr_to_tuples(self.down_sampling_dims)
        self.compressed_dims = arr_to_tuples(self.compressed_dims)

        # Configure feature extract layers
        feature_extract_layers = []
        for in_c, out_c in self.feature_extract_dims:
            feature_extract_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_c,
                        out_channels=out_c,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(out_c),
                    nn.LeakyReLU(),
                )
            )
        self.feature_extractor = nn.Sequential(*feature_extract_layers)

        # Configure down-sampling layers
        down_sampling_layers = []
        for in_c, out_c in self.down_sampling_dims:
            down_sampling_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_c,
                        out_channels=out_c,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(out_c),
                    nn.LeakyReLU(),
                )
            )
        self.down_sampler = nn.Sequential(*down_sampling_layers)

        # Compressed layers
        compressed_layers = []
        for in_c, out_c in self.compressed_dims:
            compressed_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_c,
                        out_channels=out_c,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(out_c),
                    nn.LeakyReLU(),
                )
            )
        self.compressor = nn.Sequential(*compressed_layers)

        # Un-reference and collect
        feature_extract_layers = None
        down_sampling_layers = None
        compressed_layers = None
        gc.collect()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.down_sampler(x)
        x = self.compressor(x)
        return x
