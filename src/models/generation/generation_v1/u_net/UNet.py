import torch
from torch import nn

from models.generation.generation_v1.u_net.u_net_components import (
    DoubleConvolution,
    DownBlock,
    SelfAttentionLayer,
    UpBlock,
)


class UNet(nn.Module):
    """
    This class is an implementation of UNet which originally designed to do Biomedical segmentation.
    Based on the mechanism of encoder-decoder, UNet is utilised in diffusion models.
    Original paper of UNet:
    https://arxiv.org/abs/1505.04597
    In this implementation, the UNet designed with 3 channels input and output which represent
    3 color channel of RGB image.
    We also implement a time (step) encoding attribute in this class in order to further control
    the training process
    """

    # Number of channels for the input and output
    in_channels = 3
    out_channels = 3

    # Time encoding
    time_channel = 256

    def __init__(self, *args, **kwargs) -> None:
        """
        Constructor
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.input_layer = DoubleConvolution(self.in_channels, 64)
        self.down_sampling_1 = DownBlock(64, 128)
        self.self_attention_1 = SelfAttentionLayer(128)
        self.down_sampling_1 = DownBlock(128, 256)
        self.self_attention_2 = SelfAttentionLayer(256)
        self.down_sampling_3 = DownBlock(256, 256)
        self.self_attention_3 = SelfAttentionLayer(256)

        self.convolution_1 = DoubleConvolution(256, 256)
        self.convolution_2 = DoubleConvolution(256, 256)

        self.up_sampling_1 = UpBlock(512, 128)
        self.self_attention_1 = SelfAttentionLayer(128)
        self.up_sampling_2 = UpBlock(256, 64)
        self.self_attention_5 = SelfAttentionLayer(64)
        self.up_sampling_3 = UpBlock(128, 64)
        self.self_attention_6 = SelfAttentionLayer(64)
        self.output_layer = nn.Conv2d(64, self.out_channels, kernel_size=1)

    def position_embedding(self, _encoding, _target_channel):
        """
        Position (step) embedding to allow network know which noise step it is at
        :param _encoding:
        :param _target_channel:
        :return:
        """
        normalising_factor = 1.0 / (
            10000
            ** (
                torch.arange(
                    0, _target_channel, 2, device=next(iter(self.parameters())).device
                ).float()
                / _target_channel
            )
        )
        position_a = torch.sin(
            _encoding.repeat(1, _target_channel // 2) * normalising_factor
        )
        position_b = torch.cos(
            _encoding.repeat(1, _target_channel // 2) * normalising_factor
        )
        return torch.cat([position_a, position_b], dim=-1)

    def unet_forward(self, _x, _embedding):
        """
        Forward function
        :param _x:
        :param _embedding:
        :return:
        """
        _x1 = self.input_layer(_x)
        _x2 = self.down_sampling_1(_x1, _embedding)
        _x2 = self.self_attention_1(_x2)
        _x3 = self.down_sampling_1(_x2, _embedding)
        _x3 = self.self_attention_2(_x3)
        _x4 = self.down_sampling_3(_x3, _embedding)
        _x4 = self.self_attention_3(_x4)

        _x4 = self.convolution_1(_x4)
        _x4 = self.convolution_2(_x4)

        _x = self.up_sampling_1(_x4, _x3, _embedding)
        _x = self.self_attention_1(_x)
        _x = self.up_sampling_2(_x, _x2, _embedding)
        _x = self.self_attention_5(_x)
        _x = self.up_sampling_3(_x, _x1, _embedding)
        _x = self.self_attention_6(_x)
        output = self.output_layer(_x)
        return output

    def forward(self, _x, _embedding):
        """
        Forward with encodings
        :param _x:
        :param _embedding:
        :return:
        """
        encoding = _embedding.unsqueeze(-1)
        encoding = self.position_embedding(encoding, self.time_channel)
        return self.unet_forward(_x, encoding)


class UNet_Conditional(UNet):
    class_count = None

    def __init__(self, _class_count=class_count, *args, **kwargs) -> None:
        """
        Constructor, here, we add number of class for further turn it into
        semantic embedding (encoding)
        Further works: In this implementation, we only do simple class embedding
        In the case that we need more specified encoding for labels (or semantics)
        we can do separated dedicated label and timestep encoding and embed in tho the network
        :param _class_count:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        if _class_count is not None:
            self.label_embedding = nn.Embedding(_class_count, self.time_channel)

    def forward(self, _x, _embedding, _y=None):
        """
        Forward function with encoding
        :param _x:
        :param _embedding:
        :param _y:
        :return:
        """
        encoding = _embedding.unsqueeze(-1)
        encoding = self.position_embedding(encoding, self.time_dim)
        if _y is not None:
            encoding += self.label_embedding(_y)
        return self.unet_forwad(_x, encoding)
