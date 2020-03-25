import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, n_convs=2, max_pool=True):
        """Conv block builder.
                        :param in_channels: number of input channels
                        :param out_channels: number of output channels. Doubles in_channel if None
                        :param n_convs: number of sequential convolutions
                        :param max_pool: boolean flag whether to put max pooling layer or not
                        """
        super(ConvBlock, self).__init__()
        self.n_convs = n_convs
        self.max_pool = max_pool

        if not out_channels:
            out_channels = in_channels * 2

        layers = list()
        for _ in range(self.n_convs):
            layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False)
            )
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=False))
            # layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            in_channels = out_channels

        if self.max_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class FacedModel(nn.Module):
    """Convolutional network for face detection task. """
    def __init__(self, grid_size, num_bboxes, n_classes=1):
        super(FacedModel, self).__init__()
        self.S = grid_size  # Number of grid cells
        self.B = num_bboxes  # Number of bounding boxes
        self.C = n_classes  # Number of classes (for loss.py compatibility)

        self.layer1 = ConvBlock(in_channels=3, out_channels=8)
        self.layer2 = ConvBlock(in_channels=8)
        self.layer3 = ConvBlock(in_channels=16)
        self.layer4 = ConvBlock(in_channels=32)
        self.layer5 = ConvBlock(in_channels=64)
        self.layer6 = ConvBlock(in_channels=128)
        self.layer7 = ConvBlock(in_channels=256, out_channels=320, n_convs=4, max_pool=False)
        # self.layer7 = self._make_conv_block(in_channels=128, out_channels=192, n_convs=4, max_pool=False)
        self.layer8 = nn.Conv2d(in_channels=320,
                                out_channels=self.B * 5 + self.C,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # Forward pass of the model

        x = x.float()
        x = self.quant(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        # Output tensor has shape [5 * number of bboxes, number of grid cells, number of grid cells]
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for m in self.children():
            if type(m) == ConvBlock:
                for conv_m in m.children():
                    if type(conv_m) == nn.Sequential:
                        for i in range(0, m.n_convs):
                            torch.quantization.fuse_modules(conv_m,
                                                            [str(3 * i), str(3 * i + 1), str(3 * i + 2)],
                                                            inplace=True)
