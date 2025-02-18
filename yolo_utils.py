import torch

from torchvision.ops import box_convert, nms

from typing import Tuple, List, Dict, Union
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat
from torch import Tensor

from config_parser import YOLOConfig


def xyxy_to_yolo_target(boxes: BoundingBoxes, labels: Tensor, config: YOLOConfig) -> Tensor:
    S = config.S
    C = config.C
    CANVAS_SIZE = config.IMAGE_SIZE

    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)

    assert boxes.format == BoundingBoxFormat.XYXY

    # Since the boxes could be in the same cell we shuffle them to avoid bias
    indices = torch.randperm(len(boxes))
    boxes = boxes[indices]
    labels = labels[indices]

    cx, cy, w, h = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh").unbind(-1)
    cell_w = CANVAS_SIZE[0] / S
    cell_h = CANVAS_SIZE[1] / S

    i = (cx // cell_w).long().clamp(0, S - 1)
    j = (cy // cell_h).long().clamp(0, S - 1)

    x = (cx - i * cell_w) / cell_w
    y = (cy - j * cell_h) / cell_h

    w = w / CANVAS_SIZE[0]
    h = h / CANVAS_SIZE[1]

    target = torch.zeros((S, S, 5 + C))
    target[i, j, labels - 1] = 1
    target[i, j, C] = 1
    target[i, j, C + 1 : C + 5] = torch.stack((x, y, w, h), -1)

    return target


@torch.no_grad()
def yolo_target_to_xyxy(target: Tensor, config: YOLOConfig) -> Tuple[Tensor, Tensor]:
    S = config.S
    C = config.C
    CANVAS_SIZE = config.IMAGE_SIZE
    device = target.device

    cell_w = CANVAS_SIZE[0] / S
    cell_h = CANVAS_SIZE[1] / S

    c, x, y, w, h = target[..., C:].unbind(-1)

    x_grid, y_grid = torch.meshgrid(
        torch.arange(S, device=device) * cell_w,
        torch.arange(S, device=device) * cell_h,
        indexing="ij",
    )

    x_grid = x_grid * c
    y_grid = y_grid * c

    x = x * cell_w + x_grid
    y = y * cell_h + y_grid
    w = w * CANVAS_SIZE[0]
    h = h * CANVAS_SIZE[1]

    coords = box_convert(
        torch.stack((x, y, w, h), -1),
        in_fmt="cxcywh",
        out_fmt="xyxy",
    )

    boxes = torch.cat((c[..., None], coords), -1)

    return boxes, target[..., :C]


@torch.no_grad()
def yolo_pred_to_xyxy(pred: Tensor, config: YOLOConfig) -> Tuple[Tensor, Tensor]:
    S = config.S
    C = config.C
    B = config.B
    CANVAS_SIZE = config.IMAGE_SIZE
    device = pred.device

    cell_w = CANVAS_SIZE[0] / S
    cell_h = CANVAS_SIZE[1] / S

    boxes = pred[..., C:].view(-1, S, S, B, 5)

    c, x, y, w, h = boxes.unbind(-1)

    x_grid, y_grid = torch.meshgrid(
        torch.arange(S, device=device) * cell_w,
        torch.arange(S, device=device) * cell_h,
        indexing="ij",
    )

    x_grid = x_grid[..., None] * c
    y_grid = y_grid[..., None] * c

    x = x * cell_w + x_grid
    y = y * cell_h + y_grid
    w = torch.abs(w) * CANVAS_SIZE[0]
    h = torch.abs(h) * CANVAS_SIZE[1]

    coords = box_convert(
        torch.stack((x, y, w, h), -1),
        in_fmt="cxcywh",
        out_fmt="xyxy",
    )

    boxes = torch.cat((c[..., None], coords), -1)

    return boxes, pred[..., :C]


@torch.no_grad()
def yolo_pred_to_dict(pred: Tensor, config: YOLOConfig) -> List[Dict[str, Union[BoundingBoxes, Tensor]]]:
    boxes, classes = yolo_pred_to_xyxy(pred, config)
    box_mask = boxes[..., 0].argmax(dim=-1)
    selected_boxes = boxes.gather(
        dim=-2, index=box_mask[..., None, None].expand(-1, -1, -1, -1, 5)
    ).squeeze(-2)
    conf_mask = selected_boxes[..., 0] > 0.5
    selected_boxes = selected_boxes[conf_mask]
    selected_classes = classes[conf_mask].argmax(dim=-1) + 1
    count = conf_mask.sum((1, 2)).tolist()

    pred_dict = [
        dict(
            boxes=BoundingBoxes(
                boxes[..., 1:], format="xyxy", canvas_size=config.IMAGE_SIZE
            ),
            scores=boxes[..., 0],
            labels=labels,
        )
        for boxes, labels in zip(
            selected_boxes.split(count), selected_classes.split(count)
        )
    ]
    return pred_dict