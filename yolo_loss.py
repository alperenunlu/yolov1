import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap

from torchvision.ops import complete_box_iou

from torch import Tensor

from yolo_utils import yolo_pred_to_xyxy, yolo_target_to_xyxy
from config_parser import YOLOConfig

batched_box_iou = vmap(vmap(vmap(complete_box_iou)))


class YOLOLoss(nn.Module):
    def __init__(self, config: YOLOConfig):
        super().__init__()
        self.config = config
        self.S = config.S
        self.B = config.B
        self.C = config.C
        self.L_coord = config.L_coord
        self.L_obj = config.L_obj
        self.L_noobj = config.L_noobj
        self.L_class = config.L_class

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_xyxy, _ = yolo_pred_to_xyxy(pred, self.config)
        target_xyxy, _ = yolo_target_to_xyxy(target, self.config)

        ious = batched_box_iou(
            pred_xyxy[..., 1:], target_xyxy[..., 1:].unsqueeze(3)
        ).squeeze(-1)
        ious, best_bbox = ious.max(-1, keepdim=True)
        idx = self.C + (best_bbox * 5)
        idx = torch.arange(5, device=idx.device) + idx
        pred_boxes = pred.gather(-1, idx)
        target_boxes = target[..., self.C :]

        obj_mask = target_xyxy[..., 0].bool()
        noobj_mask = ~obj_mask

        xy_loss = masked_mse(pred_boxes[..., 1:3], target_boxes[..., 1:3], obj_mask)
        wh_loss = masked_mse(
            pred_boxes[..., 3:5].abs().sqrt(), target_boxes[..., 3:5].sqrt(), obj_mask
        )
        coord_loss = self.L_coord * (xy_loss + wh_loss)

        conf_loss = self.L_obj * masked_mse(
            pred_boxes[..., 0], target_boxes[..., 0], obj_mask
        )
        noobj_loss = self.L_noobj * masked_mse(
            pred_boxes[..., 0], target_boxes[..., 0], noobj_mask
        )

        class_loss = self.L_class * masked_mse(
            pred[..., : self.C], target[..., : self.C], obj_mask
        )

        total_loss = coord_loss + conf_loss + noobj_loss + class_loss

        return total_loss / pred.size(0)


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    return F.mse_loss(pred[mask], target[mask], reduction="sum")


if __name__ == "__main__":

    def random_pred_and_target(BATCH_SIZE=16, S=7, B=2, C=20):
        torch.manual_seed(0)
        classes = F.one_hot(
            torch.randint(0, C, (S, S)),
            num_classes=C,
        )
        coords = torch.rand(S, S, B * 5)
        pred = torch.cat((classes, coords), dim=-1)
        pred.unsqueeze_(0)
        pred = torch.cat([pred] * BATCH_SIZE, dim=0)

        target_classes = F.one_hot(
            torch.randint(0, C, (S, S)),
            num_classes=C,
        )
        target_coords = torch.cat(
            (torch.randint(0, 2, (S, S, 1)), torch.rand(S, S, 4)), dim=-1
        )
        target = torch.cat((target_classes, target_coords), dim=2)
        target.unsqueeze_(0)
        target = torch.cat([target] * BATCH_SIZE, dim=0)

        return pred, target

    from config_parser import load_config

    config = load_config("yolo_config.yaml")

    loss = YOLOLoss(config)
    pred, target = random_pred_and_target()
    print(loss(pred, target))
