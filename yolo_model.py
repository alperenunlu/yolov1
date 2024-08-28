from torch import nn
from torchvision.models import resnet50
from config_parser import config

S = config.S
B = config.B
C = config.C
IMAGE_SIZE = config.IMAGE_SIZE
VOC_DETECTION_CATEGORIES = config.VOC_DETECTION_CATEGORIES


class DetectionHead(nn.Module):
    """The layers added on for detection as described in the paper."""

    def __init__(self, in_channels):
        super().__init__()

        inner_channels = 1024
        self.depth = 5 * B + C
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inner_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(
                inner_channels, inner_channels, kernel_size=3, stride=2, padding=1
            ),  # (Ch, 14, 14) -> (Ch, 7, 7)
            nn.BatchNorm2d(inner_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inner_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inner_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Flatten(),
            nn.Linear(7 * 7 * inner_channels, 4096),
            # nn.Dropout(),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(4096, S * S * self.depth),
        )

    def forward(self, x):
        return self.model(x).view(-1, S, S, self.depth)


class YOLOv1ResNet(nn.Module):
    def __init__(self, mode="detection"):
        super().__init__()
        self.mode = mode
        self.resnet = resnet50(weights="DEFAULT")
        if mode == "detection":
            self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])
            self.detection_head = DetectionHead(2048)

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
