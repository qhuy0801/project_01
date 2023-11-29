from torch import nn


class SelfAttention(nn.Module):
    """
    Self-Attention considers the interrelationships between elements within the same matrix.
    This module build a self-attention layers based on the design of its publication.
    https://arxiv.org/abs/1706.03762
    In the self attention layer, we implemented multi-headed attention.
    We set the default head number of multi-headed attention component is 1
    """

    def __init__(self, input_channels, head_num=1, *args, **kwargs) -> None:
        """
        Constructor
        :param input_channels: dimension of input (which equal to convolution size)
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.channels = input_channels
        self.head_num = head_num
        self.multi_head_at = nn.MultiheadAttention(
            input_channels, self.head_num, batch_first=True
        )
        self.layer_norm = nn.LayerNorm([input_channels])
        self.self_forward = nn.Sequential(
            nn.LayerNorm([input_channels]),
            nn.Linear(input_channels, input_channels),
            nn.GELU(),
            nn.Linear(input_channels, input_channels),
        )

    def forward(self, x):
        """
        Forward function of attention module
        :param x:
        :return:
        """
        depth = x.shape[-1]
        x = x.view(-1, self.channels, depth * depth).swapaxes(1, 2)
        x_normalised = self.layer_norm(x)
        at_value, _ = self.multi_head_at(x_normalised, x_normalised, x_normalised)
        at_value = at_value + x
        at_value = self.self_forward(at_value) + at_value
        return at_value.swapaxes(2, 1).view(-1, self.channels, depth, depth)
