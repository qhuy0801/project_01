import gc

from torch import nn

from utils import arr_to_tuples


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
        self.down_sampling_dims = [self.feature_extract_dims[-1], *self.down_sampling_dims]

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
                    nn.SiLU()
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
                    nn.SiLU()
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
                    nn.SiLU()
                )
            )
        self.compressor = nn.Sequential(*compressed_layers)

        # Un-reference and collect
        feature_extract_layers = None
        down_sampling_layers = None
        compressed_layers = None
        gc.collect()

    def forward(self,x):
        x = self.feature_extractor(x)
        x = self.down_sampler(x)
        x = self.compressor(x)
        return x
