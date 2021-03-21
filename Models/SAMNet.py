import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSal(nn.Module):
    def __init__(self, pretrained=None):
        super(FastSal, self).__init__()
        self.context_path = VAMM_backbone(pretrained)
        self.pyramid_pooling = PyramidPooling(128, 128)
        self.prepare = nn.ModuleList([
                convbnrelu(128, 128, k=1, s=1, p=0, relu=False),
                convbnrelu(96, 96, k=1, s=1, p=0, relu=False),
                convbnrelu(64, 64, k=1, s=1, p=0, relu=False),
                convbnrelu(32, 32, k=1, s=1, p=0, relu=False),
                convbnrelu(16, 16, k=1, s=1, p=0, relu=False)
                ])
        self.fuse = nn.ModuleList([
                DSConv3x3(128, 96, dilation=1),
                DSConv3x3(96, 64, dilation=2),
                DSConv5x5(64, 32, dilation=2),
                DSConv5x5(32, 16, dilation=2),
                DSConv5x5(16, 16, dilation=2)
                ])
        self.heads = nn.ModuleList([
                SalHead(in_channel=96),
                SalHead(in_channel=64),
                SalHead(in_channel=32),
                SalHead(in_channel=16),
                SalHead(in_channel=16)
                ])

    def forward(self, x): # (3, 1)
        ct_stage1, ct_stage2, ct_stage3, ct_stage4, ct_stage5 = self.context_path(x)
        # (16, 1/2) (32, 1/4) (64, 1/8)  (96, 1/16) (128, 1/32)
        ct_stage6 = self.pyramid_pooling(ct_stage5)                          # (128, 1/32)

        fused_stage1 = self.fuse[0](self.prepare[0](ct_stage5) + ct_stage6)  # (96, 1/32)
        refined1 = interpolate(fused_stage1, ct_stage4.size()[2:])           # (96, 1/16)

        fused_stage2 = self.fuse[1](self.prepare[1](ct_stage4) + refined1)   # (64, 1/16)
        refined2 = interpolate(fused_stage2, ct_stage3.size()[2:])           # (64, 1/8)

        fused_stage3 = self.fuse[2](self.prepare[2](ct_stage3) + refined2)   # (32, 1/8)
        refined3 = interpolate(fused_stage3, ct_stage2.size()[2:]) 		     # (32, 1/4)

        fused_stage4 = self.fuse[3](self.prepare[3](ct_stage2) + refined3)   # (16, 1/4)
        refined4 = interpolate(fused_stage4, ct_stage1.size()[2:])		     # (16, 1/2)

        fused_stage5 = self.fuse[4](self.prepare[4](ct_stage1) + refined4)   # (16, 1/2)

        output_side1 = interpolate(self.heads[0](fused_stage1), x.size()[2:])
        output_side2 = interpolate(self.heads[1](fused_stage2), x.size()[2:])
        output_side3 = interpolate(self.heads[2](fused_stage3), x.size()[2:])
        output_side4 = interpolate(self.heads[3](fused_stage4), x.size()[2:])
        output_main  = interpolate(self.heads[4](fused_stage5), x.size()[2:])

        return torch.cat([output_main, output_side1, output_side2, output_side3, output_side4], dim=1)


interpolate = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)


class DSConv5x5(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv5x5, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=5, s=stride, p=2*dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)


class SalHead(nn.Module):
    def __init__(self, in_channel):
        super(SalHead, self).__init__()
        self.conv = nn.Sequential(
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channel, 1, 1, stride=1, padding=0),
                nn.Sigmoid()
                )

    def forward(self, x):
        return self.conv(x)


class VAMM_backbone(nn.Module):
    def __init__(self, pretrained=None):
        super(VAMM_backbone, self).__init__()
        self.layer1 = nn.Sequential(
                convbnrelu(3, 16, k=3, s=2, p=1),
                VAMM(16, dilation_level=[1,2,3])
                )
        self.layer2 = nn.Sequential(
                DSConv3x3(16, 32, stride=2),
                VAMM(32, dilation_level=[1,2,3])
                )
        self.layer3 = nn.Sequential(
                DSConv3x3(32, 64, stride=2),
                VAMM(64, dilation_level=[1,2,3]),
                VAMM(64, dilation_level=[1,2,3]),
                VAMM(64, dilation_level=[1,2,3])
                )
        self.layer4 = nn.Sequential(
                DSConv3x3(64, 96, stride=2),
                VAMM(96, dilation_level=[1,2,3]),
                VAMM(96, dilation_level=[1,2,3]),
                VAMM(96, dilation_level=[1,2,3]),
                VAMM(96, dilation_level=[1,2,3]),
                VAMM(96, dilation_level=[1,2,3]),
                VAMM(96, dilation_level=[1,2,3])
                )
        self.layer5 = nn.Sequential(
                DSConv3x3(96, 128, stride=2),
                VAMM(128, dilation_level=[1,2]),
                VAMM(128, dilation_level=[1,2]),
                VAMM(128, dilation_level=[1,2])
                )

        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained))
            print('Pretrained model loaded!')

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out1, out2, out3, out4, out5


class VAMM(nn.Module):
    def __init__(self, channel, dilation_level=[1,2,4,8], reduce_factor=4):
        super(VAMM, self).__init__()
        self.planes = channel
        self.dilation_level = dilation_level
        self.conv = DSConv3x3(channel, channel, stride=1)
        self.branches = nn.ModuleList([
                DSConv3x3(channel, channel, stride=1, dilation=d) for d in dilation_level
                ])
        ### ChannelGate
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = convbnrelu(channel, channel, 1, 1, 0, bn=True, relu=True)
        self.fc2 = nn.Conv2d(channel, (len(self.dilation_level) + 1) * channel, 1, 1, 0, bias=False)
        self.fuse = convbnrelu(channel, channel, k=1, s=1, p=0, relu=False)
        ### SpatialGate
        self.convs = nn.Sequential(
                convbnrelu(channel, channel // reduce_factor, 1, 1, 0, bn=True, relu=True),
                DSConv3x3(channel // reduce_factor, channel // reduce_factor, stride=1, dilation=2),
                DSConv3x3(channel // reduce_factor, channel // reduce_factor, stride=1, dilation=4),
                nn.Conv2d(channel // reduce_factor, 1, 1, 1, 0, bias=False)
                )

    def forward(self, x):
        conv = self.conv(x)
        brs = [branch(conv) for branch in self.branches]
        brs.append(conv)
        gather = sum(brs)

        ### ChannelGate
        d = self.gap(gather)
        d = self.fc2(self.fc1(d))
        d = torch.unsqueeze(d, dim=1).view(-1, len(self.dilation_level) + 1, self.planes, 1, 1)

        ### SpatialGate
        s = self.convs(gather).unsqueeze(1)

        ### Fuse two gates
        f = d * s
        f = F.softmax(f, dim=1)

        return self.fuse(sum([brs[i] * f[:, i, ...] for i in range(len(self.dilation_level) + 1)]))	+ x


class PyramidPooling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PyramidPooling, self).__init__()
        hidden_channel = int(in_channel / 4)
        self.conv1 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv2 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv3 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv4 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.out = convbnrelu(in_channel*2, out_channel, k=1, s=1, p=0)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size)
        feat2 = interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size)
        feat3 = interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size)
        feat4 = interpolate(self.conv4(F.adaptive_avg_pool2d(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)

        return x
