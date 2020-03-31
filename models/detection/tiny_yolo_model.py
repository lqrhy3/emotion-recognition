import torch
from torch import nn


class ConvBlock(nn.Module):
    """Making standard convolutional block.
            Convolution layer, next Batch normalization layer, next Max pooling layer, next Activation function.
            """
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
        # layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class TinyYolo(nn.Module):
    """Tiny YOLOv1 detection_model.
    Constructing TinyYOLO network supporting grid cell size, number of bboxes per cell and number of classes modification.
    """
    def __init__(self, grid_size, num_bboxes, n_classes=1):
        super(TinyYolo, self).__init__()
        self.S = grid_size   # grid size
        self.B = num_bboxes   # number of bbox
        self.C = n_classes   # number of classes

        self.layer1 = ConvBlock(in_channels=3, out_channels=16)
        self.layer2 = ConvBlock(in_channels=16)
        self.layer3 = ConvBlock(in_channels=32)
        self.layer4 = ConvBlock(in_channels=64)
        self.layer5 = ConvBlock(in_channels=128)
        self.layer6 = ConvBlock(in_channels=256)

        self.conv1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(in_features=256*self.S*self.S, out_features=self.S*self.S*(5*self.B + self.C))
        self.sigmoid = nn.Sigmoid()

        self.quant = torch.quantizatiuon.QuantStub()
        self.dequant = torch.quantizatiuon.DeQuantStub()

    def forward(self, x):
        # Declaring forward pass of detection_model
        x = x.float()
        x = self.quant(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view((x.size(0), -1))
        x = self.fc(x)
        x = self.sigmoid(x)

        # Transfroming to YOLO output tensor
        x = x.view(-1, 5 * self.B + self.C, self.S, self.S)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.quantization.fuse_modules(m.layer, [['0', '1', '2']], inplace=True)
