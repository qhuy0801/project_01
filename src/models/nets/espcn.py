import torch
from torch import nn
from torchinfo import summary


class ESPCN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_channels: int = 64,
        scale_factor: int = 4,
        *args,
        **kwargs
    ) -> None:
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
            nn.Tanh(),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Tanh(),
        )

        # Convolution and sub-pixel
        sub_pix_channels = int(out_channels * (scale_factor**2))
        self.sub_pixel = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels // 2,
                out_channels=sub_pix_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.PixelShuffle(scale_factor),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.sub_pixel(x)
        x = torch.clamp(x, 0, 1)
        return x

    
if __name__ == "__main__":
    model_stats = str(summary(ESPCN(), (1, 3, 64, 64)))
    print(model_stats)
