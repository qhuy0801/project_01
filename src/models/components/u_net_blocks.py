import torch
from torch import nn

from models import DoubleConvolution


class DownBlock(nn.Module):
    """
    Down sampling component includes:
    Max-pooling layer (Down sample)
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
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConvolution(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                residual=True,
            ),
            DoubleConvolution(
                in_channels=self.out_channels, out_channels=self.out_channels
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
        embeddings = self.embedded_layer(embeddings)[:, :, None, None].repeat(
            1, 1, x.shape[-2], x.shape[-1]
        )
        return x + embeddings


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
