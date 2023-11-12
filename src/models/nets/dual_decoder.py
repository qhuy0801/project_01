from torch import nn

from utils import get_activation


class DualDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        middle_channels: int = 256,
        kernel_size: int = 3,
        middle_activation: str = "LeakyReLU",
        output_activation: str = "Tanh",
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

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.middle_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=1,
            )
        )

        self.up_sampler = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.middle_channels,
                out_channels=self.middle_channels,
                kernel_size=self.kernel_size,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            get_activation(self.middle_activation),
            nn.BatchNorm2d(self.middle_channels),
            nn.ConvTranspose2d(
                in_channels=self.middle_channels,
                out_channels=self.middle_channels,
                kernel_size=self.kernel_size,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            get_activation(self.middle_activation),
            nn.BatchNorm2d(self.middle_channels),
        )

        self.to_output = nn.Sequential(
            nn.Conv2d(
                in_channels=self.middle_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=1,
            ),
            nn.Tanh() if self.output_activation == "Tanh" else nn.Sigmoid(),
        )

    def forward(self,x):
        x = self.feature_extraction(x)
        x = self.up_sampler(x)
        x = self.to_output(x)
        return x
