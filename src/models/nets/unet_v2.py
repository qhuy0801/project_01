from torch import nn

from models.nets.attention import SelfAttention
from models.nets.unet_components import DoubleConvolution, DownBlock, UpBlock


class UNet_v2(nn.Module):
    """
    This class represents an reconstructed version of the U-Net architecture, building upon the original
    U-Net model which is renowned for its effectiveness.
    The original U-Net paper provides the foundational methodology for this architecture and can be found at the
    specified reference link: https://arxiv.org/abs/1505.04597
    The self-attention mechanism has been upgraded with a flexible configuration of the number of
    attention heads. By utilizing `nn.MultiheadAttention`, the model can now adapt its focus on
    various features within the input data, enhancing its ability to manage and interpret
    interdependencies.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        attn_heads: int = 1,
        embedded_dim: int = 256,
        *args,
        **kwargs
    ) -> None:
        """

        :param in_channels:
        :param out_channels:
        :param attn_heads:
        :param embedded_dim:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedded_dim = embedded_dim
        self.attn_heads = attn_heads

        # Input convolution
        self.in_conv = DoubleConvolution(in_channels=self.in_channels, out_channels=64)

        # Down-sampling process
        self.down_1 = DownBlock(
            in_channels=64, out_channels=128, embedded_dim=self.embedded_dim
        )
        self.sa_1 = SelfAttention(input_channels=128, head_num=self.attn_heads)
        self.down_2 = DownBlock(
            in_channels=128, out_channels=256, embedded_dim=self.embedded_dim
        )
        self.sa_2 = SelfAttention(input_channels=256, head_num=self.attn_heads)
        self.down_3 = DownBlock(
            in_channels=256, out_channels=512, embedded_dim=self.embedded_dim
        )
        self.sa_3 = SelfAttention(input_channels=512, head_num=self.attn_heads)
        self.down_4 = DownBlock(
            in_channels=512, out_channels=512, embedded_dim=self.embedded_dim
        )
        self.sa_4 = SelfAttention(input_channels=512, head_num=self.attn_heads)

        # Middle convolution
        self.mid_conv0 = DoubleConvolution(in_channels=512, out_channels=1024)
        self.mid_conv1 = DoubleConvolution(in_channels=1024, out_channels=1024)
        self.mid_conv2 = DoubleConvolution(in_channels=1024, out_channels=512)

        # Up-sampling process
        self.up_3 = UpBlock(
            in_channels=1024, out_channels=256, embedded_dim=self.embedded_dim
        )
        self.sa_up_3 = SelfAttention(input_channels=256, head_num=self.attn_heads)
        self.up_2 = UpBlock(
            in_channels=512, out_channels=128, embedded_dim=self.embedded_dim
        )
        self.sa_up_2 = SelfAttention(input_channels=128, head_num=self.attn_heads)
        self.up_1 = UpBlock(
            in_channels=256, out_channels=64, embedded_dim=self.embedded_dim
        )
        self.sa_up_1 = SelfAttention(input_channels=64, head_num=self.attn_heads)
        self.up_0 = UpBlock(
            in_channels=128, out_channels=64, embedded_dim=self.embedded_dim
        )
        self.sa_up_0 = SelfAttention(input_channels=64, head_num=self.attn_heads)

        self.out_conv = nn.Conv2d(
            in_channels=64, out_channels=self.out_channels, kernel_size=1
        )

    def forward(self, x, embeddings):
        """
        Forward function
        :param x:
        :param embeddings:
        :return:
        """
        x0 = self.in_conv(x)

        x1 = self.down_1(x0, embeddings)
        x1 = self.sa_1(x1)
        x2 = self.down_2(x1, embeddings)
        x2 = self.sa_2(x2)
        x3 = self.down_3(x2, embeddings)
        x3 = self.sa_3(x3)
        x4 = self.down_4(x3, embeddings)
        x4 = self.sa_4(x4)

        x4 = self.mid_conv0(x4)
        x4 = self.mid_conv1(x4)
        x4 = self.mid_conv2(x4)

        x = self.up_3(x4, x3, embeddings)
        x = self.sa_up_3(x)
        x = self.up_2(x, x2, embeddings)
        x = self.sa_up_2(x)
        x = self.up_1(x, x1, embeddings)
        x = self.sa_up_1(x)
        x = self.up_0(x, x0, embeddings)
        x = self.sa_up_0(x)

        x = self.out_conv(x)
        return x
