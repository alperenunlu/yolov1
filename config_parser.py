from dataclasses import asdict, dataclass

import yaml


@dataclass
class YOLOConfig:
    S: int
    B: int
    C: int
    IMAGE_SIZE: tuple[int, int]
    VOC_DETECTION_CATEGORIES: list[str]
    L_coord: float
    L_obj: float
    L_noobj: float
    L_class: float

    Rescore: bool
    Sqrt: bool

    BATCH_SIZE: int
    NUM_EPOCHS: int

    Jitter: float

    LR: float
    WEIGHT_DECAY: float

    def asdict(self):
        return asdict(self)


def load_config(path: str) -> YOLOConfig:
    config = yaml.safe_load(open(path, "r"))
    return YOLOConfig(**config)


config = load_config("yolo_config.yaml")

if __name__ == "__main__":
    from pprint import pprint

    pprint(config)
