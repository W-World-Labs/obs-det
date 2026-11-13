import torch
from torch import nn


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True) if activation else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AntiAliasStem(nn.Module):
    def __init__(self, out_channels: int = 64) -> None:
        super().__init__()
        mid_channels = max(out_channels // 2, 8)
        self.net = nn.Sequential(
            ConvBNAct(3, mid_channels, kernel_size=3, stride=2),
            ConvBNAct(mid_channels, out_channels, kernel_size=3, stride=2),
            ConvBNAct(out_channels, out_channels, kernel_size=3, groups=out_channels),
            ConvBNAct(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedConvBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.BatchNorm2d(channels)
        self.proj = nn.Conv2d(channels, hidden * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.out = nn.Conv2d(hidden, channels, kernel_size=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        value, gate = self.proj(self.norm(x)).chunk(2, dim=1)
        value = self.dwconv(value)
        return residual + self.out(self.act(value) * torch.sigmoid(gate))
