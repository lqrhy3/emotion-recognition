from torch import nn
from utils.summary import summary


class FacedModel(nn.Module):
    def __init__(self, grid_size, num_bboxes, n_classes=1):
        super(FacedModel, self).__init__()
        self.S = grid_size
        self.B = num_bboxes
        self.C = n_classes

        self.layer1 = self._make_conv_block(in_channels=3, out_channels=8)
        self.layer2 = self._make_conv_block(in_channels=8)
        self.layer3 = self._make_conv_block(in_channels=16)
        self.layer4 = self._make_conv_block(in_channels=32)
        self.layer5 = self._make_conv_block(in_channels=64)
        self.layer6 = self._make_conv_block(in_channels=128)
        self.layer7 = self._make_conv_block(in_channels=256, out_channels=320, n_convs=4, max_pool=False)
        self.layer8 = nn.Conv2d(in_channels=320,
                                out_channels=self.B * 5 + self.C,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)

    @staticmethod
    def _make_conv_block(in_channels, out_channels=None, n_convs=2, max_pool=True):
        if not out_channels:
            out_channels = in_channels * 2

        layers = list()
        for _ in range(n_convs):
            layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False)
            )
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            in_channels = out_channels

        if max_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x
