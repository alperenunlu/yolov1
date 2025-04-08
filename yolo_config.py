from dataclasses import asdict, dataclass

from tomllib import load


@dataclass
class YOLOConfig:
    S: int
    B: int
    C: int
    IMAGE_SIZE: tuple[int, int]
    VOC_DETECTION_CATEGORIES: tuple[str]
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

    def __post_init__(self):
        self.IMAGE_SIZE = tuple(self.IMAGE_SIZE)
        self.VOC_DETECTION_CATEGORIES = tuple(self.VOC_DETECTION_CATEGORIES)

    def asdict(self):
        return asdict(self)


def load_config(path: str = "yolo_config.toml") -> YOLOConfig:
    with open(path, "rb") as f:
        config = load(f)
    return YOLOConfig(**config)


if __name__ == "__main__":
    from pprint import pprint

    cfg = load_config()
    pprint(cfg.IMAGE_SIZE)
