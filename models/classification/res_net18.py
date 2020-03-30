from torch import nn
import torchvision


class ResNet18(nn.Module):
    """Network based on ResNet18 with a pretrained backbone on the ImageNet"""
    def __init__(self, emotion_map):
        super(ResNet18, self).__init__()
        self.backbone = torchvision.models.resnet18()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, len(emotion_map))

    def forward(self, x):
        x = self.backbone(x)
        return x
