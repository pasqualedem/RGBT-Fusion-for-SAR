import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv


class OptionalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(OptionalConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.embedding = nn.Embedding(1, out_channels)

    def forward(self, x, shape):
        if x is None:
            device = self.embedding.weight.device
            B, C, H, W = shape
            # Conv reduction formula
            H = (H + 2 * self.conv.padding[0] - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1) // self.conv.stride[0] + 1
            W = (W + 2 * self.conv.padding[1] - self.conv.dilation[1] * (self.conv.kernel_size[1] - 1) - 1) // self.conv.stride[1] + 1
            return self.embedding(torch.tensor([0], device=device)).view(1, -1, 1, 1).expand(B, -1, H, W)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
    
class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(FusionConv, self).__init__()
        self.optional_rgb = OptionalConv(3, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.optional_ir = OptionalConv(1, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        assert in_channels == 4, "in_channels must be 4"
        
        self.conv = Conv(out_channels * 2, out_channels, kernel_size, s=stride, p=padding, d=dilation, g=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        
    def forward(self, x):
        C = x.shape[1]
        if C == 4:
            rgb = x[:, :3, :, :]
            ir = x[:, 3:, :, :]
        elif C == 3:
            rgb = x
            ir = None
        elif C == 1:
            rgb = None
            ir = x
        else:
            raise ValueError("in_channels must be 1, 3, or 4")
        
        rgb = self.optional_rgb(rgb, shape=x.shape)
        ir = self.optional_ir(ir, shape=x.shape)
        x = torch.cat([rgb, ir], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x