from torch import nn
from torchinfo import summary

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

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=256,
                kernel_size=self.kernel_size,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=self.kernel_size,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.decoder_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=self.kernel_size,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=self.kernel_size,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.decoder_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=self.kernel_size,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=self.kernel_size,
                padding=1,
            ),
            nn.ReLU(),
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


if __name__ == "__main__":
    model_stats = str(summary(DualDecoder(), (1, 3, 64, 64)))
    print(model_stats)
