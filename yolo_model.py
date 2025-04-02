from config_parser import YOLOConfig
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import Tensor, nn
from torchvision.ops import Conv2dNormActivation


class YOLO_V1(nn.Module):
    def __init__(self, config: YOLOConfig):
        super().__init__()
        self.S = config.S
        self.B = config.B
        self.C = config.C

        self.backbone = create_yolov1_backbone(cfg, pretrained=True)

        self.head = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
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
        x = self.backbone(x)
        x = self.head(x)
        return x.reshape(x.size(0), self.S, self.S, self.B * 5 + self.C)


cfg = [
    ("conv", 64, 7, 2, 1),
    ("maxpool", 2, 2),
    ("conv", 192, 3, 1, 1),
    ("maxpool", 2, 2),
    ("conv", 128, 1, 1, 1),
    ("conv", 256, 3, 1, 1),
    ("conv", 256, 1, 1, 1),
    ("conv", 512, 3, 1, 1),
    ("maxpool", 2, 2),
    ("conv", 256, 1, 1, 1),
    ("conv", 512, 3, 1, 1),
    ("conv", 256, 1, 1, 1),
    ("conv", 512, 3, 1, 1),
    ("conv", 256, 1, 1, 1),
    ("conv", 512, 3, 1, 1),
    ("conv", 256, 1, 1, 1),
    ("conv", 512, 3, 1, 1),
    ("conv", 512, 1, 1, 1),
    ("conv", 1024, 3, 1, 1),
    ("maxpool", 2, 2),
    ("conv", 512, 1, 1, 1),
    ("conv", 1024, 3, 1, 1),
    ("conv", 512, 1, 1, 1),
    ("conv", 1024, 3, 1, 1),
]


def create_yolov1_backbone(cfg, pretrained=True):
    layers = []
    in_channels = 3
    for layer in cfg:
        if layer[0] == "conv":
            filters, size, stride, pad = layer[1:]
            pad = (size - 1) // 2 if pad == 1 else 0
            conv_layer = Conv2dNormActivation(in_channels, filters, size, stride, pad)
            layers.append(conv_layer)
            in_channels = filters
        elif layer[0] == "maxpool":
            size, stride = layer[1:]
            maxpool_layer = nn.MaxPool2d(size, stride)
            layers.append(maxpool_layer)

    model = nn.Sequential(*layers)

    if pretrained:
        model_file = hf_hub_download(
            repo_id="alperenunlu/yolov1-backbone",
            filename="yolov1_backbone.safetensors",
        )

        state_dict = load_file(model_file)
        model.load_state_dict(state_dict)

    return model


if __name__ == "__main__":
    import torch
    from config_parser import load_config

    config = load_config("yolo_config.yaml")

    model = YOLO_V1(config)
    print(model)
    print(model(torch.empty(1, 3, 448, 448)).shape)
