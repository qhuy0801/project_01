from torch import nn


class SelfAttention(nn.Module):
    """
    Self-Attention considers the interrelationships between elements within the same matrix.
    This module build a self-attention layers based on the design of its publication.
    https://arxiv.org/abs/1706.03762
    In the self attention layer, we implemented multi-headed attention.
    We set the default head number of multi-headed attention component is 1
    """

    def __init__(self, embedded_dim, head_num=1, *args, **kwargs) -> None:
        """
        Constructor
        :param embedded_dim: dimension of input (which equal to convolution size)
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.channels = embedded_dim
        self.head_num = head_num
        self.multi_head_at = nn.MultiheadAttention(
            embedded_dim, self.head_num, batch_first=True
        )
        self.layer_norm = nn.LayerNorm([embedded_dim])
        self.self_forward = nn.Sequential(
            nn.LayerNorm([embedded_dim]),
            nn.Linear(embedded_dim, embedded_dim),
            nn.GELU(),
            nn.Linear(embedded_dim, embedded_dim),
        )

    def forward(self, x):
        """
        Forward function of attention module
        :param x:
        :return:
        """
        depth = x.shape[-1]
        x = x.view(-1, self.channels, depth * depth).swapaxes(1, 2)
        x_ln = self.layer_norm(x)
        at_value, _ = self.multi_head_at(x_ln, x_ln, x_ln)
        at_value = at_value + x
        at_value = self.self_forward(at_value) + at_value
        return at_value.swapaxes(2, 1).view(-1, self.channels, depth, depth)
