import torch
from torch import nn
import torch.nn.functional as functional


class SelfAttentionLayer(nn.Module):
    """
    Self-Attention considers the interrelationships between elements within the same matrix.
    This module build a self-attention layers based on the design of its publication.
    https://arxiv.org/abs/1706.03762
    In the self attention layer, we implemented multi-headed attention.
    We set the default head number of multi-headed attention component is 4
    """

    # Depth of embedded layer
    channels: int

    # Number of heads in multi-headed attention component
    head_num = 4

    def __init__(self, _channels, *args, **kwargs) -> None:
        """
        Constructor
        :param _channels:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.channels = _channels
        self.multi_head_at = nn.MultiheadAttention(
            _channels, self.head_num, batch_first=True
        )
        self.layer_norm = nn.LayerNorm([_channels])
        self.self_forward = nn.Sequential(
            nn.LayerNorm([_channels]),
            nn.Linear(_channels, _channels),
            nn.GELU(),
            nn.Linear(_channels, _channels),
        )

    def forward(self, _x):
        """
        Forward function
        :param _x:
        :return:
        """
        depth = _x.shape[-1]
        _x = _x.view(-1, self.channels, depth * depth).swapaxes(1, 2)
        x_ln = self.ln(_x)
        at_value, _ = self.multi_head_at(x_ln, x_ln, x_ln)
        at_value = at_value + _x
        at_value = self.self_forward(at_value) + at_value
        return at_value.swapaxes(2, 1).view(-1, self.channels, depth, depth)


class DoubleConvolutionComponent(nn.Module):
    """
    This a combination of 2 convolutional layer created for convenience of constructing the U-Net.
    The component can be modified by changing `kernel_size`.
    GeLU activation function as possesses a smoother and more continuous form compared to the
    ReLU function, potentially enhancing its ability to discern intricate patterns in the data.
    """

    # Default setting: kernel_size of convolutional layers
    kernel_size = 3

    def __init__(
        self,
        _in_channels,
        _out_channels,
        _middle_channels=None,
        _residual=False,
        *args,
        **kwargs
    ) -> None:
        """
        Constructor
        :param _in_channels:
        :param _out_channels:
        :param _middle_channels:
        :param _residual:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.residual = _residual
        if not _middle_channels:
            _middle_channels = _out_channels
        self.double_convolution = nn.Sequential(
            nn.Conv2d(
                _in_channels,
                _middle_channels,
                kernel_size=self.kernel_size,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, _middle_channels),
            nn.GELU(),
            nn.Conv2d(
                _middle_channels,
                _out_channels,
                kernel_size=self.kernel_size,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, _out_channels),
        )

    def forward(self, _x):
        """
        Forward function
        :param _x:
        :return:
        """
        if self.residual:
            return functional.gelu(_x + self.double_convolution(_x))
        return self.double_convolution(_x)


class DownSamplingComponent(nn.Module):
    """
    Down sampling component includes:
    Max-pooling layer (Down sample)
    Double convolutional component
    """

    def __init__(
        self, _in_channels, _out_channels, _embedding_dimensions=256, *args, **kwargs
    ) -> None:
        """
        Constructor
        :param _in_channels:
        :param _out_channels:
        :param _embedding_dimensions:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.down_sampling_convolution = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvolutionComponent(_in_channels, _in_channels, residual=True),
            DoubleConvolutionComponent(_in_channels, _out_channels),
        )

        self.embedding_layer = nn.Sequential(
            nn.SiLU(), nn.Linear(_embedding_dimensions, _out_channels)
        )

    def forward(self, _x, _encoding):
        """
        Forward function
        :param _x:
        :param _encoding:
        :return:
        """
        _x = self.down_sampling_convolution(_x)
        embedding = self.embedding_layer(_encoding)[:, :, None, None].repeat(
            1, 1, _x.shape[-2], _x.shape[-1]
        )
        return _x + embedding


class UpSamplingComponent(nn.Module):
    """
    Up sampling component includes:
    Up sample layer (default scale factor here is "bi-linear")
    Double convolutional component
    """

    def __init__(
        self, _in_channels, _out_channels, _embedding_dimensions=256, *args, **kwargs
    ) -> None:
        """
        Constructor
        :param _in_channels:
        :param _out_channels:
        :param _embedding_dimensions:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.up_sampling = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.double_convolution = nn.Sequential(
            DoubleConvolutionComponent(_in_channels, _in_channels, residual=True),
            DoubleConvolutionComponent(_in_channels, _out_channels, _in_channels // 2),
        )

        self.embedding_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(_embedding_dimensions, _out_channels),
        )

    def forward(self, _x, _skip_x, _encoding):
        """
        Forward function
        :param _x:
        :param _skip_x:
        :param _encoding:
        :return:
        """
        _x = self.up_sampling(_x)
        _x = torch.cat([_skip_x, _x], dim=1)
        _x = self.double_convolution(_x)
        embedding = self.embedding_layer(_encoding)[:, :, None, None].repeat(
            1, 1, _x.shape[-2], _x.shape[-1]
        )
        return _x + embedding
