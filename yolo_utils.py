from typing import Dict, List, Tuple, Union

import torch
from config_parser import YOLOConfig
from torch import Tensor
from torchvision.ops import box_convert, clip_boxes_to_image, nms
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat


def xyxy_to_yolo_target(
    boxes: BoundingBoxes, labels: Tensor, config: YOLOConfig
) -> Tensor:
    """Converts :class:`BoundingBoxes` boxes and :class:`Tensor` labels to yolo style cxcywh format target.

    Args:
        boxes (BoundingBoxes): Target boxes in xyxy format.
        labels (Tensor): Target labels.
        config (YOLOConfig): Configuration object.

    Returns:
        Tensor: Converted boxes in yolo format.
    """
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
    """Converts :class:`torch.Tensor` target from yolo style cxcywh format to xyxy format.

    Args:
        boxes (Tensor[Batch, S, S, C + 5]): Target boxes in yolo format.
        config (YOLOConfig): Configuration object.

    Returns:
        Tuple[Tensor[Batch, S, S, 5], Tensor[Batch, S, S, C]]: Converted boxes in xyxy format and classes
    """
    box_convert
    S = config.S
    C = config.C
    CANVAS_SIZE = config.IMAGE_SIZE
    device = target.device

    cell_w = CANVAS_SIZE[0] / S
    cell_h = CANVAS_SIZE[1] / S

    is_obj, x, y, w, h = target[..., C:].unbind(-1)

    x_grid, y_grid = torch.meshgrid(
        torch.arange(S, device=device) * cell_w,
        torch.arange(S, device=device) * cell_h,
        indexing="ij",
    )

    x_grid = x_grid * is_obj
    y_grid = y_grid * is_obj

    x = x * cell_w + x_grid
    y = y * cell_h + y_grid
    w = w * CANVAS_SIZE[0]
    h = h * CANVAS_SIZE[1]

    coords = box_convert(
        torch.stack((x, y, w, h), -1),
        in_fmt="cxcywh",
        out_fmt="xyxy",
    )

    boxes = torch.cat((is_obj[..., None], coords), -1)

    return boxes, target[..., :C]


@torch.no_grad()
def yolo_pred_to_xyxy(pred: Tensor, config: YOLOConfig) -> Tuple[Tensor, Tensor]:
    """Converts :class:`torch.Tensor` prediction from yolo style cxcywh format to xyxy format.

    Args:
        boxes (Tensor[Batch, S, S, C + B * 5]): Prediction boxes in yolo format.
        config (YOLOConfig): Configuration object.

    Returns:
        Tuple[Tensor[Batch, S, S, B, 5], Tensor[Batch, S, S, C]]: Converted boxes in xyxy format and classes
    """
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

    x_grid = x_grid[..., None]
    y_grid = y_grid[..., None]

    x = x * cell_w + x_grid
    y = y * cell_h + y_grid
    if config.Sqrt:
        w = w.pow(2)
        h = h.pow(2)
    w = w * CANVAS_SIZE[0]
    h = h * CANVAS_SIZE[1]

    coords = box_convert(
        torch.stack((x, y, w, h), -1),
        in_fmt="cxcywh",
        out_fmt="xyxy",
    )

    boxes = torch.cat((c[..., None], coords), -1)

    return boxes, pred[..., :C]


@torch.no_grad()
def yolo_pred_to_dict(
    pred: Tensor, config: YOLOConfig
) -> List[Dict[str, Union[BoundingBoxes, Tensor]]]:
    """Converts :class:`torch.Tensor` prediction to a list of :class:`BoundingBoxes`, :class:`Tensor` labels and :class:`Tensor` scores.

    List of dictionaries containing `boxes`, `labels` and `scores`.

    Args:
        pred (Tensor[Batch, S, S, C + B * 5]): Prediction boxes in yolo format.
        config (YOLOConfig): Configuration

    Returns:
        List[Dict[str, Union[BoundingBoxes, Tensor]]]: List of dictionaries containing boxes, labels and scores.
    """
    boxes, classes = yolo_pred_to_xyxy(pred, config)
    conf_mask = boxes[..., 0] > 0.05
    max_class_score, max_class_arg = classes.max(dim=-1, keepdim=True)
    scores, boxes = boxes.split([1, 4], dim=-1)
    scores = scores[..., 0] * max_class_score
    selected_scores = scores[conf_mask]
    selected_boxes = boxes[conf_mask]

    selected_classes = max_class_arg.repeat_interleave(config.B, dim=-1)[conf_mask] + 1
    count = conf_mask.sum((1, 2, 3)).tolist()

    selected_boxes = clip_boxes_to_image(selected_boxes, size=config.IMAGE_SIZE)

    index_list = [
        nms(
            boxes=boxes,
            scores=scores,
            iou_threshold=0.5,
        )
        for boxes, scores in zip(
            selected_boxes.split(count),
            selected_scores.split(count),
        )
    ]

    pred_dict = [
        {
            "boxes": BoundingBoxes(
                boxes[index], format="xyxy", canvas_size=config.IMAGE_SIZE
            ),
            "labels": labels[index],
            "scores": scores[index],
        }
        for boxes, labels, scores, index in zip(
            selected_boxes.split(count),
            selected_classes.split(count),
            selected_scores.split(count),
            index_list,
        )
    ]

    return pred_dict


def box_iou(pred: Tensor, target: Tensor, S: int = 1) -> Tensor:
    """Calculates Intersection over Union (IoU) between predicted and target boxes. For each cell in the grid."""
    l1 = pred[..., 1:3] / S - pred[..., 3:] / 2
    l2 = target[..., 1:3] / S - target[..., 3:] / 2
    left = torch.max(l1, l2[..., None, :])
    r1 = pred[..., 1:3] / S + pred[..., 3:] / 2
    r2 = target[..., 1:3] / S + target[..., 3:] / 2
    right = torch.min(r1, r2[..., None, :])
    i = torch.relu(right - left).prod(-1)
    u = pred[..., 3:].prod(-1) + target[..., None, 3:].prod(-1) - i
    return i / (u + 1e-6)


def box_rmse(pred: Tensor, target: Tensor, S: int = 1) -> Tensor:
    """Calculates Root Mean Squared Error (RMSE) between predicted and target boxes. For each cell in the grid"""
    coord_diff = (pred[..., 1:3] / S - target[..., None, 1:3] / S).pow(2).sum(-1)
    dim_diff = (pred[..., 3:] - target[..., None, 3:]).pow(2).sum(-1)

    return (coord_diff + dim_diff).sqrt()
