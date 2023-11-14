from torch import nn
from torchinfo import summary

from models.nets.unet_components import DoubleConvolution
from utils import get_activation


class DualDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
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
        self.kernel_size = kernel_size
        self.middle_activation = middle_activation
        self.output_activation = output_activation
        self.pooling_kernel_size = pooling_kernel_size

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=256,
                kernel_size=self.kernel_size,
                stride=1,
                padding=1,
            )
        )

        self.up_sampler = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=self.kernel_size,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            get_activation(self.middle_activation),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=self.kernel_size,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            get_activation(self.middle_activation),
            nn.BatchNorm2d(64),
        )

        self.to_output = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=1,
            ),
            nn.Tanh() if self.output_activation == "Tanh" else nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.up_sampler(x)
        x = self.to_output(x)
        return x


if __name__ == "__main__":
    model_stats = str(summary(DualDecoder(output_activation="Sigmoid"), (1, 3, 64, 64)))
    print(model_stats)
