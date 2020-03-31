import torch
from torch import nn


__all__ = ['MiniXception']


class MiniXception(nn.Module):
    """MiniXception classification model.
        Constructing MiniXception network supporting number of classes and number of input channels.
        """
    def __init__(self, emotion_map, in_channels=3):
        """:param emotion_map: emotion type list
           :param in_channels [optional]: number of input channels"""
        super(MiniXception, self).__init__()
        num_classes = len(emotion_map)
        self.emotion_map = emotion_map

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=8)
        self.act2 = nn.ReLU()

        self.blocks = self._make_xception_blocks(in_channels=8, n=3)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

    @staticmethod
    def _make_xception_blocks(in_channels, n):
        cur_channels = in_channels
        blocks = list()
        for i in range(n):
            blocks.append(MiniXceptionBlock(cur_channels))
            cur_channels *= 2

        return nn.Sequential(*blocks)

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'act1'], ['conv2', 'bn2', 'act2']], inplace=True)
        for mxblock in self.blocks.children():
            torch.quantization.fuse_modules(mxblock, [['res_conv.0', 'res_conv.1']], inplace=True)
            torch.quantization.fuse_modules(mxblock, [['block.0.pointwise', 'block.1', 'block.2'], ['block.3.pointwise', 'block.4']], inplace=True)


class MiniXceptionBlock(nn.Module):
    def __init__(self, in_channels):
        super(MiniXceptionBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 2 * self.in_channels

        self.res_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(self.out_channels)
        )
        self.block = nn.Sequential(
            DepthwiseSeparableConv(self.in_channels, self.out_channels),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            DepthwiseSeparableConv(self.out_channels, self.out_channels),
            nn.BatchNorm2d(self.out_channels),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.skip_add = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.skip_add.add(self.res_conv(x), self.block(x))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
