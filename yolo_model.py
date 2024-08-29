from torch import nn
from torchvision.models import resnet34, resnet50
from config_parser import config

S = config.S
B = config.B
C = config.C
IMAGE_SIZE = config.IMAGE_SIZE
VOC_DETECTION_CATEGORIES = config.VOC_DETECTION_CATEGORIES


class Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0,
    ):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class DetectionHead(nn.Module):
    """The layers added on for detection as described in the paper."""

    def __init__(self, in_channels):
        super().__init__()

        inner_channels = 512
        self.depth = 5 * B + C
        self.model = nn.Sequential(
            Block(in_channels, inner_channels, 1),
            Block(inner_channels, inner_channels * 2, 3, 2, 1),
            Block(inner_channels * 2, inner_channels, 1),
            Block(inner_channels, inner_channels * 2, 3, 1, 1),
            Block(inner_channels * 2, inner_channels, 1),
            nn.Conv2d(inner_channels, self.depth, 1),
        )

    def forward(self, x):
        return self.model(x).view(-1, S, S, self.depth)


class YOLOv1ResNet(nn.Module):
    def __init__(self, backbone="resnet34", mode="detection"):
        super().__init__()
        self.mode = mode
        if backbone == "resnet34":
            self.resnet = resnet34(weights="DEFAULT")
        elif backbone == "resnet50":
            self.resnet = resnet50(weights="DEFAULT")

        in_features = self.resnet.fc.in_features
        if mode == "detection":
            self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])
            self.detection_head = DetectionHead(in_features)

        self.backbone.requires_grad_(False)

    def forward(self, x):
        if self.mode == "detection":
            x = self.backbone(x)
            x = self.detection_head(x)
        elif self.mode == "classification":
            x = self.resnet(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, downsample=False
    ):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        if downsample:
            self.layer.add_module("MaxPool", nn.MaxPool2d(2, 2))

    def forward(self, x):
        return self.layer(x)


class YOLOV1_tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 16, 3, 1, 1, downsample=True),
            ConvBlock(16, 32, 3, 1, 1, downsample=True),
            ConvBlock(32, 64, 3, 1, 1, downsample=True),
            ConvBlock(64, 128, 3, 1, 1, downsample=True),
            ConvBlock(128, 256, 3, 1, 1, downsample=True),
            ConvBlock(256, 512, 3, 1, 1, downsample=True),
        )
        self.detection_head = nn.Sequential(
            ConvBlock(512, 1024, 3, 1, 1),
            ConvBlock(1024, 1024, 3, 1, 1),
            ConvBlock(1024, 1024, 3, 1, 1),
            nn.Flatten(),
            nn.Linear(1024 * S * S, 256),
            nn.Linear(256, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, S * S * (B * 5 + C)),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.detection_head(x)
        x = x.view(-1, S, S, B * 5 + C)
        return x
