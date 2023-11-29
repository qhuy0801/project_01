import torch
from torch import nn
from torchinfo import summary


class UpScaler(nn.Module):
    """
    The implementation of image up-scaler module
    This module is used to enhance the image resolution
    The model followed the inspiration from: https://arxiv.org/abs/1609.05158
    However, we used same model constructing approach but everything is re-designed
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_channels: int = 64,
        scale_factor: int = 4,
        middle_activation: str = "ReLU",
        output_module: str = "sub-pix",
        *args,
        **kwargs
    ) -> None:
        """
        Construction class
        :param in_channels:
        :param out_channels:
        :param hidden_channels:
        :param scale_factor:
        :param middle_activation:
        :param output_module:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU() if middle_activation == "ReLU" else nn.Tanh(),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU() if middle_activation == "ReLU" else nn.Tanh(),
        )

        # To output mode
        if output_module == "sub-pix":
            sub_pix_channels = int(out_channels * (scale_factor**2))
            self.to_output = nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_channels // 2,
                    out_channels=sub_pix_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.PixelShuffle(scale_factor),
            )
        elif output_module == "conv":
            self.to_output = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=hidden_channels // 2,
                    out_channels=hidden_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.ConvTranspose2d(
                    in_channels=hidden_channels // 2,
                    out_channels=hidden_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.ReLU() if middle_activation == "ReLU" else nn.Tanh(),
                nn.Conv2d(
                    in_channels=hidden_channels // 2,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU() if middle_activation == "ReLU" else nn.Tanh(),
            )
        else:
            self.to_output = nn.Sequential(
                nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
                nn.ReLU() if middle_activation == "ReLU" else nn.Tanh(),
                nn.Conv2d(
                    in_channels=hidden_channels // 2,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU() if middle_activation == "ReLU" else nn.Tanh(),
            )

    def forward(self, x):
        """
        Forwarding fucntion
        :param x:
        :return:
        """
        x = self.feature_extractor(x)
        x = self.to_output(x)
        x = torch.clamp(x, 0, 1)
        return x


if __name__ == "__main__":
    model_stats = str(summary(UpScaler(hidden_channels=512, output_module="sub-pix"), (1, 3, 64, 64)))
    print(model_stats)
