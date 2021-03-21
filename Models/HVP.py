import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.PReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)

        return x


class gycblock(nn.Module):
    def __init__(self,channels_in,channels_out):
        super(gycblock, self).__init__()
        self.recp7 = nn.Sequential(
            BasicConv(channels_in*1, channels_in, kernel_size=7, dilation=1, padding=3, groups=channels_in, bias=False),
            BasicConv(channels_in, channels_in, kernel_size=1, dilation=1, bias=False),

            BasicConv(channels_in, channels_in, kernel_size=3, dilation=7, padding=7, groups=channels_in, bias=False,
                      relu=False),
            BasicConv(channels_in, channels_in, kernel_size=1, dilation=1, bias=False)
        )

        self.recp5 = nn.Sequential(
            BasicConv(channels_in*2, channels_in, kernel_size=5, dilation=1, padding=2, groups=channels_in, bias=False),
            BasicConv(channels_in, channels_in, kernel_size=1, dilation=1, bias=False),

            BasicConv(channels_in, channels_in, kernel_size=3, dilation=5, padding=5, groups=channels_in, bias=False,
                      relu=False),
            BasicConv(channels_in, channels_in, kernel_size=1, dilation=1, bias=False)
        )

        self.recp3 = nn.Sequential(
            BasicConv(channels_in*3, channels_in, kernel_size=3, dilation=1, padding=1, groups=channels_in, bias=False),
            BasicConv(channels_in, channels_in, kernel_size=1, dilation=1, bias=False),

            BasicConv(channels_in, channels_in, kernel_size=3, dilation=3, padding=3, groups=channels_in, bias=False,
                      relu=False),
            BasicConv(channels_in, channels_in, kernel_size=1, dilation=1, bias=False)
        )

        self.recp1 = nn.Sequential(
            BasicConv(channels_in*4, channels_out, kernel_size=1, dilation=1, bias=False,relu=True)
        )

    def forward(self, x):
        x0 = self.recp7(x)
        x1 = self.recp5(torch.cat([x,x0],dim=1))
        x2 = self.recp3(torch.cat([x,x0,x1],dim=1))
        out = self.recp1(torch.cat([x,x0,x1,x2],dim=1))

        return out


class Channelatt(nn.Module):
    def __init__(self, channel, reduction=4):
        super(Channelatt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


class Spatialatt(nn.Module):
    def __init__(self,channels_in):
        super(Spatialatt, self).__init__()
        kernel_size = 3
        self.spatial = BasicConv(channels_in, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_out = self.spatial(x)
        scale = torch.sigmoid(x_out) # broadcasting

        return x * scale


class residual_att(nn.Module):
    def __init__(self,channels_in, reduction=4):
        super(residual_att, self).__init__()
        self.channel_att=Channelatt(channels_in, reduction=reduction)
        self.spatialatt=Spatialatt(channels_in)

    def forward(self, x):
        return x + self.spatialatt(self.channel_att(x))
