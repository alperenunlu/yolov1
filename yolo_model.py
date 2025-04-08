from functools import partial

from huggingface_hub import hf_hub_download
from safetensors.torch import load_model
from timm import create_model
from torch import Tensor, nn
from torchvision.ops import Conv2dNormActivation
from yolo_config import YOLOConfig


class YOLOv1(nn.Module):
    def __init__(self, config: YOLOConfig):
        super().__init__()
        self.S = config.S
        self.B = config.B
        self.C = config.C

        # self.backbone = self.create_yolov1_backbone(pretrained=True)
        self.backbone = create_model(
            "resnet34.tv_in1k",
            pretrained=True,
            num_classes=0,
            global_pool="",
            drop_path_rate=0.2,
            drop_block_rate=0.2,
        )

        activation = partial(nn.LeakyReLU, negative_slope=0.1)
        self.head = nn.Sequential(
            Conv2dNormActivation(
                512, 1024, 3, padding=1, stride=1, activation_layer=activation
            ),
            Conv2dNormActivation(
                1024, 1024, 3, padding=1, stride=2, activation_layer=activation
            ),
            Conv2dNormActivation(
                1024, 1024, 3, padding=1, stride=1, activation_layer=activation
            ),
            Conv2dNormActivation(
                1024, 1024, 3, padding=1, stride=1, activation_layer=activation
            ),
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
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.kaiming_normal_(self.head[-4].weight, a=0.1, nonlinearity="leaky_relu")
        nn.init.constant_(self.head[-4].bias, 0)
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.head[-1].bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x.view(x.size(0), self.S, self.S, self.B * 5 + self.C)

    @staticmethod
    def create_yolov1_backbone(pretrained=True):
        activation = partial(nn.LeakyReLU, negative_slope=0.1)
        Conv2d = partial(
            Conv2dNormActivation,
            activation_layer=activation,
        )

        backbone = nn.Sequential(
            Conv2d(3, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(2, 2),
            Conv2d(64, 192, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            Conv2d(192, 128, 1, stride=1, padding=0),
            Conv2d(128, 256, 3, stride=1, padding=1),
            Conv2d(256, 256, 1, stride=1, padding=0),
            Conv2d(256, 512, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            Conv2d(512, 256, 1, stride=1, padding=0),
            Conv2d(256, 512, 3, stride=1, padding=1),
            Conv2d(512, 256, 1, stride=1, padding=0),
            Conv2d(256, 512, 3, stride=1, padding=1),
            Conv2d(512, 256, 1, stride=1, padding=0),
            Conv2d(256, 512, 3, stride=1, padding=1),
            Conv2d(512, 256, 1, stride=1, padding=0),
            Conv2d(256, 512, 3, stride=1, padding=1),
            Conv2d(512, 512, 1, stride=1, padding=0),
            Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            Conv2d(1024, 512, 1, stride=1, padding=0),
            Conv2d(512, 1024, 3, stride=1, padding=1),
            Conv2d(1024, 512, 1, stride=1, padding=0),
            Conv2d(512, 1024, 3, stride=1, padding=1),
        )

        if pretrained:
            model_file = hf_hub_download(
                repo_id="alperenunlu/yolov1",
                filename="yolov1_backbone.darknet.safetensors",
            )
            load_model(backbone, model_file)

        return backbone


if __name__ == "__main__":
    import torch
    from yolo_config import load_config

    config = load_config("yolo_config.toml")

    model = YOLOv1(config)
    print(model)
    print(model(torch.empty(1, 3, 448, 448)).shape)
