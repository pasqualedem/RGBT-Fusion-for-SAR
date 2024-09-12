import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv


def tensor_is_padding(tensor):
    """
    Check if a tensor is a padding tensor.
    Padding tensors are tensors filled with a single value.
    """
    return tensor is None or tensor.unique().numel() == 1


class OptionalConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        use_embedding=True,
    ):
        super(OptionalConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        if use_embedding:
            self.embedding = nn.Embedding(1, out_channels)
        else:
            self.embedding = None

    def forward(self, x, shape):
        if x is None:
            device = self.conv.weight.device
            B, C, H, W = shape
            # Conv reduction formula
            H = (
                H
                + 2 * self.conv.padding[0]
                - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
                - 1
            ) // self.conv.stride[0] + 1
            W = (
                W
                + 2 * self.conv.padding[1]
                - self.conv.dilation[1] * (self.conv.kernel_size[1] - 1)
                - 1
            ) // self.conv.stride[1] + 1
            if self.embedding is None:
                return torch.zeros(B, self.conv.out_channels, H, W, device=device)
            return (
                self.embedding(torch.tensor([0], device=device))
                .view(1, -1, 1, 1)
                .expand(B, -1, H, W)
            )
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class FusionConv(nn.Module):
    IGNORE_VALUE = 114

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=True,
        use_embedding=True,
    ):
        super(FusionConv, self).__init__()
        self.optional_rgb = OptionalConv(
            3,
            out_channels,
            kernel_size,
            stride=1,
            padding=1,
            dilation=dilation,
            groups=groups,
            bias=bias,
            use_embedding=use_embedding,
        )
        self.optional_ir = OptionalConv(
            1,
            out_channels,
            kernel_size,
            stride=1,
            padding=1,
            dilation=dilation,
            groups=groups,
            bias=bias,
            use_embedding=use_embedding,
        )
        assert in_channels == 4, "in_channels must be 4"

        self.conv = Conv(
            out_channels * 2,
            out_channels,
            kernel_size,
            s=stride,
            p=padding,
            d=dilation,
            g=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def detect_channels(x):
        channels = []
        for img in x:
            if img.shape[0] == 3:
                channels.append((img, None))
            elif img.shape[0] == 1:
                channels.append((None, img))
            elif img.shape[0] == 4:
                if img[:3, :, :].eq(FusionConv.IGNORE_VALUE).all():
                    channels.append((None, img[3:, :, :]))
                elif img[3:, :, :].eq(FusionConv.IGNORE_VALUE).all():
                    channels.append((img[:3, :, :], None))
                else:
                    channels.append((img[:3, :, :], img[3:, :, :]))
            else:
                raise ValueError("Unsupported number of channels")
        return channels

    def forward(self, x):
        channels = FusionConv.detect_channels(x)
        rgb, ir = zip(*channels)
        rgb = [
            item.unsqueeze(0) if not (tensor_is_padding(item)) else None for item in rgb
        ]
        ir = [
            item.unsqueeze(0) if not (tensor_is_padding(item)) else None for item in ir
        ]

        if all(item is not None for item in rgb):
            rgb = torch.cat(rgb)
            rgb = self.optional_rgb(rgb, x.shape)
        else:
            rgb = torch.cat(
                [self.optional_rgb(item, (1,) + x.shape[1:]) for item in rgb]
            )
        if all(item is not None for item in ir):
            ir = torch.cat(ir)
            ir = self.optional_ir(ir, x.shape)
        else:
            ir = torch.cat([self.optional_ir(item, (1,) + x.shape[1:]) for item in ir])
        x = torch.cat([rgb, ir], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x