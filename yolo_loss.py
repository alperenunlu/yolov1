import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.ops import box_convert, box_iou

from torch import Tensor

from config_parser import config

from yolo_utils import yolo_resp_bbox

S = config.S
B = config.B
C = config.C


class YOLOLoss(nn.Module):
    def __init__(self, S=S, B=B, C=C, L_coord=5, L_noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.L_coord = L_coord
        self.L_noobj = L_noobj

    def forward(self, output: Tensor, target: Tensor):
        """
        output: (batch_size, S, S, B * 5 + C)
        target: (batch_size, S, S, 5 + C)

        output format: [c1, c2, ..., C, B1[conf, cx, cy, w, h], B2[conf, cx, cy, w, h], B]
        target format: [c1, c2, ..., C, conf, cx, cy, w, h]

        loss =
            L_coord * obj * [(x - x_hat)^2 + (y - y_hat)^2]
            L_coord * obj * [(sqrt(w) - sqrt(w_hat))^2 + (sqrt(h) - sqrt(h_hat))^2]
            obj * [(conf - conf_hat)^2]
            L_noobj * noobj * [(conf - conf_hat)^2]
            obj * [(c - c_hat)^2]

        where:
            obj = 1 if object exists in cell
            noobj = 1 if no object exists in cell
        """
        S = self.S
        B = self.B
        C = self.C
        canvas_size = config.IMAGE_SIZE

        obj_mask = target[..., C + 0] == 1
        noobj_mask = target[..., C + 0] == 0

        # Which boxes in each cell is responsible for the prediction
        output_boxes = output[..., C:].contiguous().view(-1, S, S, B, 5)
        target_boxes = target[..., C:].contiguous().view(-1, S, S, 5)

        ## calculating iou hardest part

        resp_boxes = yolo_resp_bbox(output_boxes, target_boxes, config)
        resp_coords = resp_boxes[..., 1:]
        target_coords = target_boxes[..., 1:]

        # Box Loss

        center_loss = F.mse_loss(
            target_coords[obj_mask][..., :2],
            resp_coords[obj_mask][..., :2],
            reduction="sum",
        )

        wh_loss = F.mse_loss(
            target_coords[obj_mask][..., 2:],
            resp_coords[obj_mask][..., 2:],
            reduction="sum",
        ).sqrt()

        box_loss = self.L_coord * (center_loss + wh_loss)

        # Object Loss

        conf_loss = F.mse_loss(
            resp_boxes[obj_mask][..., 0],
            target_boxes[obj_mask][..., 0],
            reduction="sum",
        )

        # No Object Loss
        noobj_loss = self.L_noobj * F.mse_loss(
            resp_boxes[noobj_mask][..., 0],
            target_boxes[noobj_mask][..., 0],
            reduction="sum",
        )

        # Class Loss

        class_loss = F.mse_loss(
            output[..., :C][obj_mask],
            target[..., :C][obj_mask],
            reduction="sum",
        )

        loss = box_loss + conf_loss + noobj_loss + class_loss
        loss = loss / output.size(0)

        return loss
