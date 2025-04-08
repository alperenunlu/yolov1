import torch
from torch import Tensor
from torchvision.ops import batched_nms, box_convert, clip_boxes_to_image
from torchvision.tv_tensors import BoundingBoxes
from yolo_config import YOLOConfig


def xyxy_to_yolo_target(
    boxes: BoundingBoxes | Tensor, labels: Tensor, config: YOLOConfig
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
    IMAGE_SIZE = config.IMAGE_SIZE

    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)

    # Since the boxes could be in the same cell we shuffle them to avoid bias
    indices = torch.randperm(len(boxes))
    boxes = boxes[indices]
    labels = labels[indices]

    cx, cy, w, h = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh").unbind(-1)
    cell_w = IMAGE_SIZE[1] / S
    cell_h = IMAGE_SIZE[0] / S

    j = (cx // cell_w).long().clamp(0, S - 1)
    i = (cy // cell_h).long().clamp(0, S - 1)

    x = (cx % cell_w) / cell_w
    y = (cy % cell_h) / cell_h

    w = w / IMAGE_SIZE[1]
    h = h / IMAGE_SIZE[0]

    target = torch.zeros((S, S, 5 + C))
    target[i, j, labels - 1] = 1
    target[i, j, C] = 1
    target[i, j, C + 1 : C + 5] = torch.stack((x, y, w, h), -1)

    return target


@torch.no_grad()
def yolo_target_to_xyxy(
    target: Tensor, config: YOLOConfig
) -> tuple[Tensor, Tensor, Tensor]:
    """Converts :class:`torch.Tensor` target from yolo style cxcywh format to xyxy format.

    Args:
        boxes (Tensor[Batch, S, S, C + 5]): Target boxes in yolo format.
        config (YOLOConfig): Configuration object.

    Returns:
        Tuple[Tensor[Batch, S, S], Tensor[Batch, S, S, 4], Tensor[Batch, S, S, C]]: Confidences, boxes in xyxy format and classes
    """
    S = config.S
    C = config.C
    IMAGE_SIZE = config.IMAGE_SIZE
    device = target.device

    cell_w = IMAGE_SIZE[1] / S
    cell_h = IMAGE_SIZE[0] / S

    classes, boxes = target.split([C, 5], dim=-1)
    boxes = boxes.view(-1, S, S, 5)
    classes = classes.view(-1, S, S, C)

    is_obj, x, y, w, h = boxes.unbind(-1)

    x_grid, y_grid = torch.meshgrid(
        torch.arange(S, device=device),
        torch.arange(S, device=device),
        indexing="xy",
    )

    x_grid = x_grid * is_obj
    y_grid = y_grid * is_obj

    x = cell_w * (x + x_grid)
    y = cell_h * (y + y_grid)
    w = w * IMAGE_SIZE[1]
    h = h * IMAGE_SIZE[0]

    coords = box_convert(
        torch.stack((x, y, w, h), -1),
        in_fmt="cxcywh",
        out_fmt="xyxy",
    ).floor()

    return is_obj, coords, classes


@torch.no_grad()
def yolo_pred_to_xyxy(
    pred: Tensor, config: YOLOConfig
) -> tuple[Tensor, Tensor, Tensor]:
    """Converts :class:`torch.Tensor` prediction from yolo style cxcywh format to xyxy format.

    Args:
        boxes (Tensor[Batch, S, S, C + B * 5]): Prediction boxes in yolo format.
        config (YOLOConfig): Configuration object.

    Returns:
        Tuple[Tensor[Batch, S, S], Tensor[Batch, S, S, 4], Tensor[Batch, S, S, C]]: Confidences, boxes in xyxy format and classes
    """
    S = config.S
    C = config.C
    B = config.B
    CANVAS_SIZE = config.IMAGE_SIZE
    device = pred.device

    cell_w = CANVAS_SIZE[1] / S
    cell_h = CANVAS_SIZE[0] / S

    classes, boxes = pred.split([C, B * 5], dim=-1)
    boxes = boxes.view(-1, S, S, B, 5)
    classes = classes.view(-1, S, S, C)

    c, x, y, w, h = boxes.unbind(-1)

    x_grid, y_grid = torch.meshgrid(
        torch.arange(S, device=device),
        torch.arange(S, device=device),
        indexing="xy",
    )

    x_grid = x_grid[..., None]
    y_grid = y_grid[..., None]

    x = cell_w * (x + x_grid)
    y = cell_h * (y + y_grid)
    if config.Sqrt:
        w = w.pow(2)
        h = h.pow(2)
    w = w * CANVAS_SIZE[1]
    h = h * CANVAS_SIZE[0]

    coords = box_convert(
        torch.stack((x, y, w, h), -1),
        in_fmt="cxcywh",
        out_fmt="xyxy",
    ).floor()

    return c, coords, classes


@torch.no_grad()
def yolo_pred_to_dict(
    pred: Tensor, config: YOLOConfig, thresh: float = 0.01
) -> list[dict[str, BoundingBoxes | Tensor]]:
    """Converts :class:`torch.Tensor` prediction to a list of :class:`BoundingBoxes`, :class:`Tensor` labels and :class:`Tensor` scores.

    List of dictionaries containing `boxes`, `labels` and `scores`.

    Args:
        pred (Tensor[Batch, S, S, C + B * 5]): Prediction boxes in yolo format.
        config (YOLOConfig): Configuration

    Returns:
        List[Dict[str, Union[BoundingBoxes, Tensor]]]: List of dictionaries containing boxes, labels and scores.
    """
    conf, coords, classes = yolo_pred_to_xyxy(pred, config)
    coords = coords.unsqueeze(4).expand(-1, -1, -1, -1, config.C, -1)
    score = classes.unsqueeze(-2) * conf.unsqueeze(-1)
    label = torch.arange(config.C, device=pred.device).expand_as(score) + 1

    mask = score > thresh
    count = mask.sum(dim=[1, 2, 3, 4]).tolist()

    scores_list = score[mask].split(count)
    bboxes_list = clip_boxes_to_image(coords[mask], size=config.IMAGE_SIZE).split(count)
    label_list = label[mask].split(count)

    index_list = [
        batched_nms(boxes=bboxes, scores=scores, idxs=idxs, iou_threshold=0.5)
        for bboxes, scores, idxs in zip(bboxes_list, scores_list, label_list)
    ]

    pred_dict = [
        dict(
            boxes=BoundingBoxes(
                bboxes[index], format="xyxy", canvas_size=config.IMAGE_SIZE
            ),
            labels=labels[index],
            scores=scores[index],
        )
        for index, bboxes, labels, scores in zip(
            index_list, bboxes_list, label_list, scores_list
        )
    ]

    return pred_dict


def box_iou(pred: Tensor, target: Tensor) -> Tensor:
    """Calculates Intersection over Union (IoU) between predicted and target boxes. For each cell in the grid."""
    l1 = pred[..., 1:3] - pred[..., 3:] / 2
    l2 = target[..., 1:3] - target[..., 3:] / 2
    left = torch.max(l1, l2[..., None, :])
    r1 = pred[..., 1:3] + pred[..., 3:] / 2
    r2 = target[..., 1:3] + target[..., 3:] / 2
    right = torch.min(r1, r2[..., None, :])
    i = torch.relu(right - left).prod(-1)
    u = pred[..., 3:].prod(-1) + target[..., None, 3:].prod(-1) - i
    return i / (u + 1e-6)


def box_rmse(pred: Tensor, target: Tensor) -> Tensor:
    """Calculates Root Mean Squared Error (RMSE) between predicted and target boxes. For each cell in the grid"""
    return (pred[..., 1:] - target[..., 1:].unsqueeze(-2)).pow(2).sum(-1).sqrt()
