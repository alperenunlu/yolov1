import torch
import torch.nn as nn
import torch.nn.functional as F

from yolo_utils import box_iou, box_rmse

from torch import Tensor
from config_parser import YOLOConfig


class YOLOLoss(nn.Module):
    def __init__(self, config: YOLOConfig):
        super().__init__()
        self.S = config.S
        self.B = config.B
        self.C = config.C
        self.L_coord = config.L_coord
        self.L_obj = config.L_obj
        self.L_noobj = config.L_noobj
        self.L_class = config.L_class

        self.rescore = config.Rescore
        self.sqrt = config.Sqrt

    @staticmethod
    def _masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        return F.mse_loss(pred * mask, target.expand_as(pred) * mask, reduction="sum")

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        BATCH = pred.size(0)
        with torch.no_grad():
            obj_i = target[..., self.C, None] > 0

            pred_boxes = pred[..., self.C :].clone().view(-1, self.S, self.S, self.B, 5)
            target_boxes = target[..., self.C :].clone().view(-1, self.S, self.S, 5)
            if self.sqrt:
                pred_boxes[..., 3:] **= 2

            ious = box_iou(pred_boxes, target_boxes)
            ious, best_bbox = ious.max(-1)
            rmse_idx = box_rmse(pred_boxes, target_boxes).argmin(-1)
            best_bbox[torch.where(ious == 0)] = rmse_idx[torch.where(ious == 0)]
            resp_box = (
                torch.zeros(BATCH, self.S, self.S, self.B, device=pred.device)
                .scatter(-1, best_bbox[..., None], 1)
                .bool()
            )

            obj_ij = obj_i * resp_box
            noobj_ij = ~obj_ij

            if self.rescore:
                target[..., self.C] = ious
            if self.sqrt:
                target[..., self.C + 3 :].sqrt_()

        x_loss = self._masked_mse(
            pred[..., self.C + 1 :: 5], target[..., self.C + 1 :: 5], obj_ij
        )
        y_loss = self._masked_mse(
            pred[..., self.C + 2 :: 5], target[..., self.C + 2 :: 5], obj_ij
        )
        w_loss = self._masked_mse(
            pred[..., self.C + 3 :: 5], target[..., self.C + 3 :: 5], obj_ij
        )
        h_loss = self._masked_mse(
            pred[..., self.C + 4 :: 5], target[..., self.C + 4 :: 5], obj_ij
        )

        coord_loss = self.L_coord * (x_loss + y_loss + w_loss + h_loss)

        conf_loss = self.L_obj * self._masked_mse(
            pred[..., self.C :: 5], target[..., self.C :: 5], obj_ij
        )
        noobj_loss = self.L_noobj * self._masked_mse(
            pred[..., self.C :: 5], torch.zeros_like(target[..., self.C :: 5]), noobj_ij
        )
        class_loss = self.L_class * self._masked_mse(
            pred[..., : self.C], target[..., : self.C], obj_i
        )

        total_loss = coord_loss + conf_loss + noobj_loss + class_loss

        return total_loss / BATCH


if __name__ == "__main__":

    def random_pred_and_target(config: YOLOConfig, BATCH_SIZE: int = 1):
        S = config.S
        C = config.C
        B = config.B
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

    loss_fn = YOLOLoss(config)
    pred, target = random_pred_and_target(config=config)
    pred.requires_grad_(True)

    loss = loss_fn(pred, target)
    loss.backward()
    print(loss)
    print(pred.grad[0, 0, 0])
