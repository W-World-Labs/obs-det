import torch

from omnidet.models.blocks import AntiAliasStem, ConvBNAct, GatedConvBlock


def test_conv_bn_act_changes_channels() -> None:
    layer = ConvBNAct(3, 16, kernel_size=3, stride=1)
    output = layer(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 16, 64, 64)


def test_stem_downsamples_to_quarter_resolution() -> None:
    stem = AntiAliasStem(out_channels=64)
    output = stem(torch.randn(2, 3, 960, 960))
    assert output.shape == (2, 64, 240, 240)


def test_gated_block_preserves_shape() -> None:
    block = GatedConvBlock(channels=64)
    output = block(torch.randn(2, 64, 40, 48))
    assert output.shape == (2, 64, 40, 48)
