from typing import List, Tuple

import yaml
from dataclasses import dataclass, asdict


@dataclass
class YOLOConfig:
    S: int
    B: int
    C: int
    IMAGE_SIZE: Tuple[int, int]
    VOC_DETECTION_CATEGORIES: List[str]
    L_coord: float
    L_obj: float
    L_noobj: float
    L_class: float

    Rescore: bool
    Sqrt: bool

    BATCH_SIZE: int
    NUM_EPOCHS: int

    Jitter: float
    Brightness: float
    Saturation: float
    Hue: float

    LEARNING_RATE: float
    WEIGHT_DECAY: float

    def asdict(self):
        d = asdict(self)
        return d


def load_config(path: str) -> YOLOConfig:
    config = yaml.safe_load(open(path, "r"))
    return YOLOConfig(**config)


config = load_config("yolo_config.yaml")

if __name__ == "__main__":
    from pprint import pprint

    pprint(config)
