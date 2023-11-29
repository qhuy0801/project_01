"""
This implementation of U-Net is discarded due to faulty code
"""
from torch import nn

from models.nets.attention import SelfAttention
from models.nets.unet_components import DoubleConvolution, DownBlock, UpBlock


class UNet_v1(nn.Module):
    """
    This class is an implementation of UNet which originally designed to do Biomedical segmentation.
    Based on the mechanism of encoder-decoder, UNet is utilised in diffusion models.
    Original paper of UNet:
    https://arxiv.org/abs/1505.04597
    In this implementation, the UNet designed with 3 channels input and output which represent
    3 color channel of RGB image.
    We also implement a time (step) embedding attribute in this class in order to further control
    the training process.
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
