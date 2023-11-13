from torch import nn

from utils import get_activation


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

        self.encoder_3 = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=64,
                    kernel_size=self.kernel_size,
                    padding=1,
                ),
                get_activation(self.middle_activation),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=self.kernel_size,
                    padding=1,
                ),
                get_activation(self.middle_activation),
                nn.MaxPool2d(
                    kernel_size=self.pooling_kernel_size,
                )
            )
        )

        self.encoder_3 = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=64,
                    kernel_size=self.kernel_size,
                    padding=1,
                ),
                get_activation(self.middle_activation),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=self.kernel_size,
                    padding=1,
                ),
                get_activation(self.middle_activation),
                nn.MaxPool2d(
                    kernel_size=self.pooling_kernel_size,
                )
            )
        )



    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.up_sampler(x)
        x = self.to_output(x)
        return x
