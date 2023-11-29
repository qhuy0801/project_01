"""
In this directory we tried a version of Super-Resolution GAN
Which followed the original implementation of this article: https://arxiv.org/abs/1609.04802
We discarded the model because the training procedure took a lot of resources and not straight-forward
"""
import math

from torch import nn
from torchinfo import summary
from torchvision.models import vgg19


class VGGExtractor(nn.Module):
    """
    Feature extractor with 18 first layer of VGG
    Discarded the last classification layer
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Constructor of the class
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

        # Gather the model
        model = vgg19(pretrained=True)

        # Collected layers
        # In this case, we skip the last layer as we need the features only, not classification results
        layers = [*list(model.features.children())[:18]]

        self.extractor = nn.Sequential(*layers)

    def forward(self, img):
        """
        Torch module forward function
        :param img:
        :return:
        """
        return self.extractor(img)


class ResBlock(nn.Module):
    """
    Residual block implemenation
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        *args,
        **kwargs
    ) -> None:
        """
        Constructor
        :param channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(channels, 0.8),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(channels, 0.8),
        )

    def forward(self, x):
        """
        Classic residual module architecture with sum of input and output
        :param x:
        :return:
        """
        return x + self.block(x)


class Generator(nn.Module):
    """
    SRGan's inspired architecture Generator
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_res_block: int = 16,
        scaling_factor: int = 4,
        *args,
        **kwargs
    ) -> None:
        """
        Constructor
        :param in_channels:
        :param out_channels:
        :param num_res_block:
        :param scaling_factor:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

        # Base layer, with large kernel_size to extract high-level features
        # `out_channel = 64` similar to original paper
        self.base_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=9,
                stride=1,
                padding=4,
            ),
            nn.PReLU(),
        )

        # Residual layers, with multiple residual blocks
        res_layers = []
        for _ in range(num_res_block):
            res_layers.append(ResBlock(channels=64))
        self.res_layers = nn.Sequential(*res_layers)

        # Post-residual layers
        self.post_res_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64, 0.8),
        )

        # Up-sampling layers
        num_scaling_block = int(math.log2(scaling_factor))
        up_sampling_layers = []
        for _ in range(num_scaling_block):
            up_sampling_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=64 * scaling_factor,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(256),
                    nn.PixelShuffle(upscale_factor=2),
                    nn.PReLU(),
                )
            )
        self.up_sampling_layers = nn.Sequential(*up_sampling_layers)

        # To output layers
        self.to_output = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=out_channels,
                kernel_size=9,
                stride=1,
                padding=4,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        Forward function
        :param x:
        :return:
        """
        x_1 = self.base_layers(x)
        x = self.res_layers(x_1)
        x_2 = self.post_res_layers(x)
        x = self.up_sampling_layers(x_1 + x_2)
        x = self.to_output(x)
        return x


class DiscriminatorBlock(nn.Module):
    """
    Feature extractor block inside the Discriminator
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        initial_block: bool = False,
        *args,
        **kwargs
    ) -> None:
        """
        Construction function
        :param in_channels:
        :param out_channels:
        :param initial_block:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        ]
        if not initial_block:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward function
        :param x:
        :return:
        """
        return self.block(x)


class Discriminator(nn.Module):
    """
    SRGan's inspired architecture Discriminator
    """
    def __init__(
        self,
        in_channels: int = 3,
        discriminator_channels: [int] = [64, 128, 256, 512],
        avg_pool_size: (int, int) = (6, 6),
        fc_size: int = 1024,
        *args,
        **kwargs
    ) -> None:
        """
        Constructor
        :param in_channels:
        :param discriminator_channels:
        :param avg_pool_size:
        :param fc_size:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

        conv_layers = []

        for i, out_channels in enumerate(discriminator_channels):
            conv_layers.append(
                DiscriminatorBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    initial_block=(i == 0),
                )
            )
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        self.avg_pool = nn.AdaptiveAvgPool2d(avg_pool_size)
        self.fc_1 = nn.Linear(
            discriminator_channels[-1] * avg_pool_size[0] * avg_pool_size[1], fc_size
        )
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fc_2 = nn.Linear(fc_size, 1)

    def forward(self, x):
        """
        Forward function
        :param x:
        :return:
        """
        batch_size = x.size(0)
        x = self.conv_layers(x)
        x = self.avg_pool(x)
        x = self.fc_1(x.view(batch_size, -1))
        x = self.leaky_relu(x)
        return self.fc_2(x)


if __name__ == "__main__":
    model_stats = str(summary(Discriminator(), (1, 3, 256, 256), depth=5))
    print(model_stats)
