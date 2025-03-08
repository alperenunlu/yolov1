import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap

# from torchvision.ops import box_iou
from torch import Tensor
from yolo_utils import yolo_pred_to_xyxy, yolo_target_to_xyxy, box_iou, box_rmse
from config_parser import YOLOConfig

# batched_box_iou = vmap(vmap(vmap(box_iou)))


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

        self.rescore = config.Rescore
        self.sqrt = config.Sqrt

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        with torch.no_grad():
            obj_mask = target[..., self.C, None] > 0
            noobj_mask = ~obj_mask

            pred_boxes = pred[..., self.C :].view(-1, self.S, self.S, self.B, 5)
            target_boxes = target[..., self.C :].view(-1, self.S, self.S, 5)

            ious = box_iou(
                pred_boxes,
                target_boxes,
            )
            ious, best_bbox = ious.max(-1)
            rmse_idx = box_rmse(pred_boxes, target_boxes).argmin(-1)
            best_bbox[torch.where(ious == 0)] = rmse_idx[torch.where(ious == 0)]
            idx = self.C + (best_bbox * 5)
            idx = torch.arange(5, device=idx.device) + idx[..., None]
            pred_boxes = pred.gather(-1, idx)

            # Trick for expanding the best box gradient to the other boxes
            pred.data[..., self.C :] = pred_boxes.tile(self.B).detach()
            target_boxes = target[..., self.C :]
            if self.config.Rescore:
                target_boxes[..., 0] = ious
            target = torch.cat(
                (target[..., : self.C], target_boxes.tile(self.B)), dim=-1
            )
            if self.sqrt:
                target[..., self.C + 3 :: 5].sqrt_()
                target[..., self.C + 4 :: 5].sqrt_()

        mse_loss = F.mse_loss(pred, target, reduction="none")
        obj_mse_loss = mse_loss * obj_mask

        x_loss = obj_mse_loss[..., self.C + 1 :: 5]
        y_loss = obj_mse_loss[..., self.C + 2 :: 5]
        w_loss = obj_mse_loss[..., self.C + 3 :: 5]
        h_loss = obj_mse_loss[..., self.C + 4 :: 5]
        coord_loss = self.L_coord * (x_loss + y_loss + w_loss + h_loss).sum()

        conf_loss = self.L_obj * obj_mse_loss[..., self.C :: 5].sum()
        noobj_loss = self.L_noobj * (mse_loss[..., self.C :: 5] * noobj_mask).sum()

        class_loss = self.L_class * obj_mse_loss[..., : self.C].sum()

        # print(
        #     "Coord Loss: ",
        #     coord_loss.item(),
        #     "Conf Loss: ",
        #     conf_loss.item(),
        #     "Noobj Loss: ",
        #     noobj_loss.item(),
        #     "Class Loss: ",
        #     class_loss.item(),
        # )
        total_loss = coord_loss + conf_loss + noobj_loss + class_loss

        return total_loss / pred.size(0)


if __name__ == "__main__":

    def random_pred_and_target(BATCH_SIZE=16, config=YOLOConfig):
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
    # print(pred.grad[0, 0, 0])
    # pred.requires_grad_(False)

    # from darknet_detection import DarknetDetection

    # darknet = DarknetDetection(config)
    # delta = darknet(pred, target)
    # print(delta[0, 0, 0])
