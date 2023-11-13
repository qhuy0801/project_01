from torch import nn
from torchinfo import summary

from models.nets.unet_components import DoubleConvolution


class SuperResAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        middle_channels: int = 256,
        kernel_size: int = 3,
        pooling_kernel_size: int = 2,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.middle_channels = middle_channels
        self.kernel_size = kernel_size
        self.pooling_kernel_size = pooling_kernel_size

        self.feature_extractor = DoubleConvolution(
            in_channels=self.in_channels,
            out_channels=64,
            bias=True,
        )

        self.encoder_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=self.pooling_kernel_size),
            DoubleConvolution(
                in_channels=64,
                out_channels=128,
                bias=True,
            ),
        )

        self.encoder_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=self.pooling_kernel_size),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=self.kernel_size,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.decoder_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            DoubleConvolution(
                in_channels=256,
                out_channels=128,
                bias=True,
            ),
        )

        self.decoder_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            DoubleConvolution(
                in_channels=128,
                out_channels=64,
                bias=True,
            ),
        )

        self.to_output = nn.Sequential(
            DoubleConvolution(
                in_channels=64,
                out_channels=self.out_channels,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.feature_extractor(x)

        x_e1 = self.encoder_1(x)
        x_e2 = self.encoder_2(x_e1)

        x_d2 = self.decoder_2(x_e2)
        x_d1 = self.decoder_1(x_d2 + x_e1)
        x_out = self.to_output(x_d1 + x)
        return x_out


if __name__ == "__main__":
    model_stats = str(summary(SuperResAE(), (1, 3, 256, 256)))
    print(model_stats)
