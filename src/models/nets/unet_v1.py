import torch
from torch import nn
from torch.nn import functional

from models.nets.attention import SelfAttention


class UNet(nn.Module):
    """
    This class is an implementation of UNet which originally designed to do Biomedical segmentation.
    Based on the mechanism of encoder-decoder, UNet is utilised in diffusion models.
    Original paper of UNet:
    https://arxiv.org/abs/1505.04597
    In this implementation, the UNet designed with 3 channels input and output which represent
    3 color channel of RGB image.
    We also implement a time (step) embedding attribute in this class in order to further control
    the training process
    """

    def __init__(
        self, in_channels=3, out_channels=3, embedded_dim=256, *args, **kwargs
    ) -> None:
        """
        Constructor
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedded_dim = embedded_dim

        self.in_conv = DoubleConvolution(self.in_channels, 64)
        self.down_sampling_1 = DownBlock(64, 128)
        self.self_attention_1 = SelfAttention(128)
        self.down_sampling_2 = DownBlock(128, 256)
        self.self_attention_2 = SelfAttention(256)
        self.down_sampling_3 = DownBlock(256, 256)
        self.self_attention_3 = SelfAttention(256)

        self.mid_conv = DoubleConvolution(256, 256)
        self.mid_conv = DoubleConvolution(256, 256)

        self.up_sampling_1 = UpBlock(256, 128)
        self.self_attention_1 = SelfAttention(128)
        self.up_sampling_2 = UpBlock(256, 64)
        self.self_attention_5 = SelfAttention(64)
        self.up_sampling_3 = UpBlock(128, 64)
        self.self_attention_6 = SelfAttention(64)
        self.out_conv = nn.Conv2d(64, self.out_channels, kernel_size=1)

    def forward(self, x, embeddings):
        """
        Forward function
        :param x:
        :param embeddings:
        :return:
        """
        _x1 = self.in_conv(x)
        _x2 = self.down_sampling_1(_x1, embeddings)
        _x2 = self.self_attention_1(_x2)
        _x3 = self.down_sampling_2(_x2, embeddings)
        _x3 = self.self_attention_2(_x3)
        _x4 = self.down_sampling_3(_x3, embeddings)
        _x4 = self.self_attention_3(_x4)

        _x4 = self.mid_conv(_x4)
        _x4 = self.mid_conv(_x4)

        x = self.up_sampling_1(_x4, _x3, embeddings)
        x = self.self_attention_1(x)
        x = self.up_sampling_2(x, _x2, embeddings)
        x = self.self_attention_5(x)
        x = self.up_sampling_3(x, _x1, embeddings)
        x = self.self_attention_6(x)
        return self.out_conv(x)


class UNet_Light(nn.Module):
    """
    A lighter version of U-Net with fewer layers
    """

    def __init__(
        self, in_channels=4, out_channels=4, embedded_dim=256, *args, **kwargs
    ) -> None:
        """
        Constructor
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedded_dim = embedded_dim

        self.in_conv = DoubleConvolution(self.in_channels, 64)
        self.down_sampling_1 = DownBlock(64, 128)
        self.self_attention_1 = SelfAttention(128)

        self.mid_conv = DoubleConvolution(128, 128)
        self.mid_conv = DoubleConvolution(128, 128)

        self.up_sampling_1 = UpBlock(128, 64)
        self.self_attention_1 = SelfAttention(64)

        self.out_conv = nn.Conv2d(64, self.out_channels, kernel_size=1)

    def forward(self, x, embeddings):
        """
        Forward function
        :param x:
        :param embeddings:
        :return:
        """
        _x1 = self.in_conv(x)
        _x2 = self.down_sampling_1(_x1, embeddings)
        _x2 = self.self_attention_1(_x2)
        _x3 = self.down_sampling_2(_x2, embeddings)
        _x3 = self.self_attention_2(_x3)
        _x4 = self.down_sampling_3(_x3, embeddings)
        _x4 = self.self_attention_3(_x4)

        _x4 = self.mid_conv(_x4)
        _x4 = self.mid_conv(_x4)

        x = self.up_sampling_1(_x4, _x3, embeddings)
        x = self.self_attention_1(x)
        x = self.up_sampling_2(x, _x2, embeddings)
        x = self.self_attention_5(x)
        x = self.up_sampling_3(x, _x1, embeddings)
        x = self.self_attention_6(x)
        return self.out_conv(x)


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
        a = x + embeddings
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
    kernel_size = 3

    def __init__(
        self,
        in_channels,
        out_channels,
        additional_channels=None,
        residual=False,
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
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, additional_channels),
            nn.GELU(),
            nn.Conv2d(
                additional_channels,
                out_channels,
                kernel_size=self.kernel_size,
                padding=1,
                bias=False,
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
