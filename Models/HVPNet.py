import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Models.HVP import gycblock, residual_att


class ConvBlock(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)
        self.add = add

    def forward(self, input):
        output = self.conv(input)
        if self.add:
            output = input + output

        output = self.bn(output)
        output = self.act(output)

        return output


class DilatedParallelConvBlockD2(nn.Module):
    def __init__(self, nIn, nOut, add=False):
        super(DilatedParallelConvBlockD2, self).__init__()
        n = int(np.ceil(nOut / 2.))
        n2 = nOut - n

        self.conv0 = nn.Conv2d(nIn, nOut, 1, stride=1, padding=0, dilation=1, bias=False)
        self.conv1 = nn.Conv2d(n, n, 3, stride=1, padding=1, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(n2, n2, 3, stride=1, padding=2, dilation=2, bias=False)

        self.bn = nn.BatchNorm2d(nOut)
        #self.act = nn.PReLU(nOut)
        self.add = add

    def forward(self, input):
        in0 = self.conv0(input)
        in1, in2 = torch.chunk(in0, 2, dim=1)
        b1 = self.conv1(in1)
        b2 = self.conv2(in2)
        output = torch.cat([b1, b2], dim=1)

        if self.add:
            output = input + output
        output = self.bn(output)
        #output = self.act(output)

        return output


class DownsamplerBlockDepthwiseConv(nn.Module):
    def __init__(self, nIn, nOut):
        super(DownsamplerBlockDepthwiseConv, self).__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            self.conv0 = nn.Conv2d(nIn, nOut-nIn, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
            self.conv1 = nn.Conv2d(nOut-nIn, nOut-nIn, 5, stride=2, padding=2, dilation=1, groups=nOut-nIn, bias=False)
            #self.pool = nn.MaxPool2d(2, stride=2)
            self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        else:
            self.conv0 = nn.Conv2d(nIn, nOut, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
            self.conv1 = nn.Conv2d(nOut, nOut, 5, stride=2, padding=2, dilation=1, groups=nOut, bias=False)

        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        if self.nIn < self.nOut:
            output = torch.cat([self.conv1(self.conv0(input)), self.pool(input)], 1)
        else:
            output = self.conv1(self.conv0(input))

        output = self.bn(output)
        output = self.act(output)

        return output


class DownsamplerBlockConv(nn.Module):
    def __init__(self, nIn, nOut):
        super(DownsamplerBlockConv, self).__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            self.conv = nn.Conv2d(nIn, nOut-nIn, 3, stride=2, padding=1, bias=False)
            #self.pool = nn.MaxPool2d(2, stride=2)
            self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        else:
            self.conv = nn.Conv2d(nIn, nOut, 3, stride=2, padding=1, bias=False)

        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        if self.nIn < self.nOut:
            output = torch.cat([self.conv(input), self.pool(input)], 1)
        else:
            output = self.conv(input)

        output = self.bn(output)
        output = self.act(output)

        return output


class Backbone(nn.Module):
    def __init__(self, P1=1, P2=1, P3=3, P4=5, reduction=4, pretrained=None):
        super(Backbone, self).__init__()
        self.level1_0 = DownsamplerBlockConv(3, 16)
        self.level1 = nn.ModuleList()
        for i in range(0, P1):
            self.level1.append(ConvBlock(16, 16))
        self.level1.append(residual_att(16, reduction=reduction))
        self.branch1 = nn.Conv2d(16, 16, 1, stride=1, padding=0,bias=False)
        self.br1 = nn.Sequential(nn.BatchNorm2d(16), nn.PReLU(16))

        self.level2_0 = DownsamplerBlockDepthwiseConv(16, 32)
        self.level2 = nn.ModuleList()
        for i in range(0, P2):
            self.level2.append(nn.Dropout2d(0.1, True))
            self.level2.append(gycblock(32, 32))
            self.level2.append(residual_att(32, reduction=reduction))
        self.branch2 = nn.Conv2d(32, 32, 1, stride=1, padding=0,bias=False)
        self.br2 = nn.Sequential(nn.BatchNorm2d(32), nn.PReLU(32))

        self.level3_0 = DownsamplerBlockDepthwiseConv(32, 64)
        self.level3 = nn.ModuleList()
        for i in range(0, P3):
            self.level3.append(nn.Dropout2d(0.1, True))
            self.level3.append(gycblock(64, 64))
            self.level3.append(residual_att(64, reduction=reduction))
        self.branch3 = nn.Conv2d(64, 64, 1, stride=1, padding=0,bias=False)
        self.br3 = nn.Sequential(nn.BatchNorm2d(64), nn.PReLU(64))

        self.level4_0 = DownsamplerBlockDepthwiseConv(64, 128)
        self.level4 = nn.ModuleList()
        for i in range(0, P4):
            self.level4.append(nn.Dropout2d(0.1, True))
            self.level4.append(gycblock(128, 128))
            self.level4.append(residual_att(128, reduction=reduction))
        self.branch4 = nn.Conv2d(128, 128, 1, stride=1, padding=0,bias=False)
        self.br4 = nn.Sequential(nn.BatchNorm2d(128), nn.PReLU(128))

        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained))
            print('Pretrained Model Loaded!')

    def forward(self, input):
        output1_0 = self.level1_0(input)

        output1 = output1_0
        for layer in self.level1:
            output1 = layer(output1)
        output1 = self.br1(self.branch1(output1_0) + output1)

        output2_0 = self.level2_0(output1)
        output2 = output2_0
        for layer in self.level2:
            output2 = layer(output2)
        output2 = self.br2(self.branch2(output2_0) + output2)

        output3_0 = self.level3_0(output2)
        output3 = output3_0
        for layer in self.level3:
            output3 = layer(output3)
        output3 = self.br3(self.branch3(output3_0) + output3)

        output4_0 = self.level4_0(output3)
        output4 = output4_0
        for layer in self.level4:
            output4 = layer(output4)
        output4 = self.br4(self.branch4(output4_0) + output4)

        return output1, output2, output3, output4


class FastSal(nn.Module):
    '''
    This class defines the MiniNetV2 network
    '''
    def __init__(self, P1=1, P2=1, P3=3, P4=5, reduction=4, pretrained=None):
        super(FastSal, self).__init__()
        self.backbone = Backbone(P1, P2, P3, P4, reduction, pretrained)

        self.up3_conv4 = DilatedParallelConvBlockD2(128, 64)
        self.up3_conv3 = nn.Conv2d(64, 64, 1, stride=1, padding=0,bias=False)
        self.up3_bn3 = nn.BatchNorm2d(64)
        self.up3_act = nn.PReLU(64)

        self.up2_conv3 = DilatedParallelConvBlockD2(64, 32)
        self.up2_conv2 = nn.Conv2d(32, 32, 1, stride=1, padding=0,bias=False)
        self.up2_bn2 = nn.BatchNorm2d(32)
        self.up2_act = nn.PReLU(32)

        self.up1_conv2 = DilatedParallelConvBlockD2(32, 16)
        self.up1_conv1 = nn.Conv2d(16, 16, 1, stride=1, padding=0,bias=False)
        self.up1_bn1 = nn.BatchNorm2d(16)
        self.up1_act = nn.PReLU(16)

        self.classifier4 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(128, 1, 1, stride=1, padding=0, bias=False))
        self.classifier3 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, 1, 1, stride=1, padding=0, bias=False))
        self.classifier2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(32, 1, 1, stride=1, padding=0, bias=False))
        self.classifier1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(16, 1, 1, stride=1, padding=0, bias=False))

    def forward(self, input):
        output1, output2, output3, output4 = self.backbone(input)

        up4 = F.interpolate(output4, output3.size()[2:], mode='bilinear', align_corners=False)
        up3_conv4 = self.up3_conv4(up4)
        up3_conv3 = self.up3_bn3(self.up3_conv3(output3))
        up3 = self.up3_act(up3_conv4 + up3_conv3)

        up3 = F.interpolate(up3, output2.size()[2:], mode='bilinear', align_corners=False)
        up2_conv3 = self.up2_conv3(up3)
        up2_conv2 = self.up2_bn2(self.up2_conv2(output2))
        up2 = self.up2_act(up2_conv3 + up2_conv2)

        up2 = F.interpolate(up2, output1.size()[2:], mode='bilinear', align_corners=False)
        up1_conv2 = self.up1_conv2(up2)
        up1_conv1 = self.up1_bn1(self.up1_conv1(output1))
        up1 = self.up1_act(up1_conv2 + up1_conv1)

        classifier4 = torch.sigmoid(self.classifier4(up4))
        classifier3 = torch.sigmoid(self.classifier3(up3))
        classifier2 = torch.sigmoid(self.classifier2(up2))
        classifier1 = torch.sigmoid(self.classifier1(up1))
        classifier4 = F.interpolate(classifier4, input.size()[2:], mode='bilinear', align_corners=False)
        classifier3 = F.interpolate(classifier3, input.size()[2:], mode='bilinear', align_corners=False)
        classifier2 = F.interpolate(classifier2, input.size()[2:], mode='bilinear', align_corners=False)
        classifier1 = F.interpolate(classifier1, input.size()[2:], mode='bilinear', align_corners=False)

        return torch.cat([classifier1, classifier2, classifier3, classifier4], dim=1)
