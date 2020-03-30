import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ConvBlock, self).__init__()

        if out_channels is None:
            out_channels = in_channels * 2

        layers = list()
        layers.append(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        )
        layers.append(nn.BatchNorm2d(num_features=out_channels))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class ConvNet(nn.Module):
    """ResNet-based classification model.
        """
    def __init__(self, emotion_map):
        """:param: emotion_map: emotions list"""
        super(ConvNet, self).__init__()

        self.emotion_map = emotion_map

        self.layer1 = ConvBlock(in_channels=3, out_channels=16)
        self.layer2 = ConvBlock(in_channels=16)
        self.layer3 = ConvBlock(in_channels=32)
        self.layer4 = ConvBlock(in_channels=64)
        self.layer5 = ConvBlock(in_channels=128)
        self.layer6 = ConvBlock(in_channels=256)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=len(emotion_map))

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.quantization.fuse_modules(m.layer, [['0', '1', '2']], inplace=True)


class PretrConvNet(nn.Module):
    def __init__(self, emotion_map, freeze=True):
        import pretrainedmodels
        super(PretrConvNet, self).__init__()

        self.emotion_map = emotion_map
        self.backbone = pretrainedmodels.resnet34(pretrained='imagenet')

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.last_linear = torch.nn.Linear(self.backbone.last_linear.in_features, len(emotion_map))

    def forward(self, x):
        x = self.backbone(x)
        return x
