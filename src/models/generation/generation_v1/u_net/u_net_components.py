from torch import nn


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
        self.multi_head_at = nn.MultiheadAttention(_channels, self.head_num, batch_first=True)
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
