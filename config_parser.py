import yaml
from dataclasses import dataclass

config = yaml.safe_load(open("yolo_config.yaml"))

@dataclass
class YOLOCONFIG:
    S: int
    B: int
    C: int
    IMAGE_SIZE: int
    VOC_DETECTION_CATEGORIES: list

config = YOLOCONFIG(**config)