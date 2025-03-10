from torch import nn
from timm import create_model

from torch import Tensor

from config_parser import YOLOConfig


class YOLO_V1(nn.Module):
    def __init__(self, config: YOLOConfig):
        super().__init__()
        self.S = config.S
        self.B = config.B
        self.C = config.C

        self.backbone = create_model("resnet18.tv_in1k", pretrained=True, num_classes=0)

        self.head = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, config.S * config.S * (config.B * 5 + config.C)),
        )

        self.init_weights()

    def init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone.forward_features(x)
        x = self.head(x)
        return x.reshape(x.size(0), self.S, self.S, self.B * 5 + self.C)


if __name__ == "__main__":
    import torch
    from config_parser import load_config

    config = load_config("yolo_config.yaml")

    model = YOLO_V1(config)
    print(model)
    print(model(torch.empty(1, 3, 448, 448)).shape)
