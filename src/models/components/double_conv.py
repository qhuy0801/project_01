from torch import nn
from torch.nn import functional


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
