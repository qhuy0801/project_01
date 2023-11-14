import torch
from torch import nn
from torch.nn import functional

from utils import get_activation


class UpBlock(nn.Module):
    """
    Up sampling component includes:
    Up sample layer (default scale factor here is "bi-linear")
    Double convolutional component
    """

    def __init__(
        self, in_channels, out_channels, embedded_dim=256, *args, **kwargs
    ) -> None:
        """
        Constructor
        :param in_channels:
        :param out_channels:
        :param embedded_dim:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedded_dim = embedded_dim
        self.up_conv = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.double_conv = nn.Sequential(
            DoubleConvolution(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                residual=True,
            ),
            DoubleConvolution(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                additional_channels=(self.in_channels // 2),
            ),
        )

        self.embedded_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=embedded_dim, out_features=out_channels),
        )

    def forward(self, x, skip_x, embeddings):
        """
        Forward function
        :param x:
        :param skip_x: skip connection from equivalent down block
        :param embeddings:
        :return:
        """
        x = self.up_conv(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.double_conv(x)
        embedded = self.embedded_layer(embeddings)[:, :, None, None].repeat(
            1, 1, x.shape[-2], x.shape[-1]
        )
        return x + embedded


class DownBlock(nn.Module):
    """
    Down sampling component includes:
    Max-pooling layer (Down sample)
    Double convolutional component5
    """

    def __init__(
        self, in_channels, out_channels, embedded_dim=256, *args, **kwargs
    ) -> None:
        """
        Constructor
        :param in_channels:
        :param out_channels:
        :param embedded_dim:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedded_dim = embedded_dim
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConvolution(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                residual=True,
            ),
            DoubleConvolution(
                in_channels=self.in_channels, out_channels=self.out_channels
            ),
        )

        self.embedded_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=self.embedded_dim, out_features=self.out_channels),
        )

    def forward(self, x, embeddings):
        """
        Forward function
        :param x:
        :param embeddings:
        :return:
        """
        x = self.down_conv(x)
        embeddings = self.embedded_layer(embeddings)
        embeddings = embeddings[:, :, None, None]
        embeddings = embeddings.repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + embeddings


class DoubleConvolution(nn.Module):
    """
    This a combination of 2 convolutional layer created for convenience of constructing the U-Net.
    The component can be modified by changing `kernel_size`.
    In this implementation, we will utilise same setting as original UNet, with a kernel size of 3.
    GeLU activation function as possesses a smoother and more continuous form compared to the
    ReLU function, potentially enhancing its ability to discern intricate patterns in the data.
    """

    # Default setting: kernel_size of convolutional layers
    kernel_size: int = 3
    padding: int = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        additional_channels: int = None,
        residual: bool = False,
        bias: bool = False,
        middle_activation: str = "GELU",
        *args,
        **kwargs
    ) -> None:
        """
        Constructor
        :param in_channels: number of input convolutional chanel
        :param out_channels: number of output convolutional chanel
        :param additional_channels: (optional) number of convolutional chanel in the middle layer
        :param residual: whether if we want to use residual mode (similar to ResNet backbone)
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.residual = residual
        if not additional_channels:
            additional_channels = out_channels
        self.double_convolution = nn.Sequential(
            nn.Conv2d(
                in_channels,
                additional_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=bias,
            ),
            nn.GroupNorm(1, additional_channels),
            get_activation(middle_activation),
            nn.Conv2d(
                additional_channels,
                out_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=bias,
            ),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        """
        Forward function
        :param x:
        :return:
        """
        if self.residual:
            return functional.gelu(x + self.double_convolution(x))
        return self.double_convolution(x)
