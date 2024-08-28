import torch
import torch.nn.functional as F
from torchvision.ops import box_convert, box_iou
from torchvision import tv_tensors
import torchvision.transforms.v2 as T
from torchvision.utils import draw_bounding_boxes

from config_parser import config


def xyxy_to_yolo_target(boxes, labels, config=config):
    S = config.S
    C = config.C
    x, y, w, h = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh").unbind(-1)
    cell_w = boxes.canvas_size[0] / S
    cell_h = boxes.canvas_size[1] / S

    center_row = (x / cell_w).floor().long().clamp(0, S - 1)
    center_col = (y / cell_h).floor().long().clamp(0, S - 1)

    norm_center_x = (x % cell_w) / cell_w
    norm_center_y = (y % cell_h) / cell_h
    norm_bndbox_w = w / boxes.canvas_size[0]
    norm_bndbox_h = h / boxes.canvas_size[1]

    target = torch.zeros(S, S, 5 + C)
    target[center_row, center_col, 0:C] = F.one_hot(labels - 1, C).float()
    target[center_row, center_col, C] = 1
    target[center_row, center_col, C + 1] = norm_center_x
    target[center_row, center_col, C + 2] = norm_center_y
    target[center_row, center_col, C + 3] = norm_bndbox_w
    target[center_row, center_col, C + 4] = norm_bndbox_h

    return target


def yolo_target_to_xyxy(target, threshold=0.5, config=config):
    S = config.S
    C = config.C
    canvas_size = config.IMAGE_SIZE

    center_row, center_col = torch.where(target[..., C] > threshold)

    confidences = target[center_row, center_col, C]
    norm_center_x = target[center_row, center_col, C + 1]
    norm_center_y = target[center_row, center_col, C + 2]
    norm_bndbox_w = target[center_row, center_col, C + 3]
    norm_bndbox_h = target[center_row, center_col, C + 4]

    cell_w = canvas_size[0] / S
    cell_h = canvas_size[1] / S

    x = (center_row + norm_center_x) * cell_w
    y = (center_col + norm_center_y) * cell_h
    w = norm_bndbox_w * canvas_size[0]
    h = norm_bndbox_h * canvas_size[1]

    boxes = box_convert(
        torch.stack([x, y, w, h], dim=-1), in_fmt="cxcywh", out_fmt="xyxy"
    )

    labels = torch.argmax(target[center_row, center_col, :C], dim=-1) + 1

    boxes = tv_tensors.BoundingBoxes(
        boxes.ceil(),
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=canvas_size,
    )

    return boxes, labels, confidences


def yolo_output_to_xyxy(output, threshold=0.005, config=config):
    S = config.S
    C = config.C
    B = config.B

    classes = output[..., :C]
    boxes = output[..., C:].reshape(S, S, B, 5)
    box_conf, box_max_indices = boxes[..., 0].max(dim=-1)
    best_boxes = boxes.gather(
        -2, box_max_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 5)
    ).squeeze()
    boxes, labels, confidences = yolo_target_to_xyxy(
        torch.cat([classes, best_boxes], dim=-1), threshold, config
    )
    return boxes, labels, confidences


def draw_yolo_target(image, target, threshold=0.5, config=config, mode="output"):
    if mode == "output":
        boxes, labels, _ = yolo_output_to_xyxy(target, threshold, config)
    else:
        boxes, labels, _ = yolo_target_to_xyxy(target, threshold, config)
    print(boxes, labels)
    return T.ToPILImage()(
        draw_bounding_boxes(
            image,
            boxes,
            labels=[config.VOC_DETECTION_CATEGORIES[i].upper() for i in labels],
            width=3,
            font="FiraCodeNerdFont-Bold.ttf",
            font_size=25,
        )
    )


def yolo_multi_bbox_to_xyxy(bbox, config=config):
    """
    bbox: Tensor of shape (N, S, S, B, 4)

    x and y are the center of the box relative to the cell
    w and h are the width and height of the box relative to the image size

    vectorized version
    """
    S = config.S
    canvas_size = config.IMAGE_SIZE

    batch_idx, i_idx, j_idx, b_idx = torch.meshgrid(
        torch.arange(bbox.size(0)),
        torch.arange(bbox.size(1)),
        torch.arange(bbox.size(2)),
        torch.arange(bbox.size(3)),
        indexing="ij",
    )

    batch_idx = batch_idx.to(bbox.device)
    i_idx = i_idx.to(bbox.device)
    j_idx = j_idx.to(bbox.device)
    b_idx = b_idx.to(bbox.device)

    cell_w = canvas_size[0] / S
    cell_h = canvas_size[1] / S

    bbox[batch_idx, i_idx, j_idx, b_idx, 0] = (
        i_idx + bbox[batch_idx, i_idx, j_idx, b_idx, 0]
    ) * cell_w
    bbox[batch_idx, i_idx, j_idx, b_idx, 1] = (
        j_idx + bbox[batch_idx, i_idx, j_idx, b_idx, 1]
    ) * cell_h
    bbox[batch_idx, i_idx, j_idx, b_idx, 2] = (
        bbox[batch_idx, i_idx, j_idx, b_idx, 2] * cell_w
    )
    bbox[batch_idx, i_idx, j_idx, b_idx, 3] = (
        bbox[batch_idx, i_idx, j_idx, b_idx, 3] * cell_h
    )

    return bbox


def yolo_resp_bbox(output, target, config=config):
    S = config.S
    B = config.B
    batch_size = output.size(0)

    output_coords = yolo_multi_bbox_to_xyxy(output[..., 1:], config)
    target_coords = yolo_multi_bbox_to_xyxy(
        target[..., 1:].unsqueeze(-2), config
    ).squeeze()

    ious = (
        box_iou(output_coords.view(-1, 4), target_coords.view(-1, 4))
        .view((S * S * batch_size), B, (S * S * batch_size))
        .permute(1, 0, 2)
    )
    ious = ious.diagonal(dim1=1, dim2=2).permute(1, 0).view(batch_size, S, S, B)
    ious, best_bbox = ious.max(dim=-1)

    # if ious is 0 then responsible box is the one with the lowest rmse
    # zero_batch, zero_i, zero_j = torch.where(ious == 0)
    # if zero_batch.size(0) > 0:
    #     zero_output = output[zero_batch, zero_i, zero_j]
    #     zero_target = target[zero_batch, zero_i, zero_j].unsqueeze(1)
    #     rmse = F.mse_loss(
    #         zero_output[..., 1:],
    #         zero_target[..., 1:],
    #         reduction="none",
    #     ).sqrt().sum(-1)

    #     _, best_bbox[zero_batch, zero_i, zero_j] = rmse.min(dim=-1)
    #     print(best_bbox[zero_batch, zero_i, zero_j])

    resp_boxes = output.gather(
        -2,
        best_bbox.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, output.size(-1)),
    ).squeeze()


    return resp_boxes


# train_transforms = T.Compose(
#     [
#         T.ToImage(),
#         T.RandomAffine(degrees=(-30, 30), scale=(0.8, 1.2), translate=(0.2, 0.2)),
#         T.ColorJitter(brightness=0.5, saturation=0.5, contrast=0.5),
#         T.RandomHorizontalFlip(),
#         T.Resize(config.IMAGE_SIZE),
#         T.ToDtype(torch.float32, scale=True),
#     ]
# )

train_transforms = T.Compose([
        T.ToImage(),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=(-15, 15)),
        T.RandomPerspective(distortion_scale=0.2, p=0.5),
        T.Resize(config.IMAGE_SIZE),
        T.ToDtype(torch.float32, scale=True),
        T.RandomAffine(
            degrees=10,
            scale=(0.8, 1.2),
            translate=(0.2, 0.2),
            shear=10,
        ),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transforms = T.Compose(
    [
        T.ToImage(),
        T.Resize(config.IMAGE_SIZE),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def train_transforms_func(*args):
    transformed = train_transforms(*args)
    transformed[1]["target"] = xyxy_to_yolo_target(
        transformed[1]["boxes"], transformed[1]["labels"]
    )
    return transformed


def test_transforms_func(*args):
    transformed = test_transforms(*args)
    transformed[1]["target"] = xyxy_to_yolo_target(
        transformed[1]["boxes"], transformed[1]["labels"]
    )
    return transformed


def collate_fn(batch):
    images, annotations = zip(*batch)
    images = torch.stack(images)
    targets = torch.stack([annotation["target"] for annotation in annotations])
    return images, targets
