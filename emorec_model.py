import torch
from torch import nn


class ConvNet(nn.Module):
    def __init__(self, emotion_map):
        super(ConvNet, self).__init__()

        self.emotion_map = emotion_map

        self.layer1 = self._make_conv_block(in_channels=3, out_channels=16)
        self.layer2 = self._make_conv_block(in_channels=16)
        self.layer3 = self._make_conv_block(in_channels=32)
        self.layer4 = self._make_conv_block(in_channels=64)
        self.layer5 = self._make_conv_block(in_channels=128)
        self.layer6 = self._make_conv_block(in_channels=256)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=len(emotion_map))

    @staticmethod
    def _make_conv_block(in_channels, out_channels=None):

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
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
