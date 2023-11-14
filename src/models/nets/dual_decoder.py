from torch import nn

from models.nets.unet_components import DoubleConvolution


class DualDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        middle_channels: int = 256,
        kernel_size: int = 3,
        pooling_kernel_size: int = 2,
        middle_activation: str = "ReLU",
        output_activation: str = "Sigmoid",
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.middle_channels = middle_channels
        self.kernel_size = kernel_size
        self.middle_activation = middle_activation
        self.output_activation = output_activation
        self.pooling_kernel_size = pooling_kernel_size

        self.feature_extractor = DoubleConvolution(
            in_channels=self.in_channels,
            out_channels=256,
            bias=True,
        )

        self.decoder_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            DoubleConvolution(
                in_channels=256,
                out_channels=128,
                bias=True,
            ),
        )

        self.decoder_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            DoubleConvolution(
                in_channels=128,
                out_channels=64,
                bias=True,
            ),
        )

        self.to_output = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.to_output(x)
        return x
