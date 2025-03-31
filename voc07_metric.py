from collections import defaultdict
from typing import Dict, List, Union

import torch
from torch import Tensor
from torchvision.ops import box_iou


class VOC07_mAP:
    def __init__(self, iou_thresh: float = 0.5) -> None:
        self.iou_thresh = iou_thresh
        self.reset()

    def reset(self) -> None:
        """Reset buffers for predictions and targets."""
        self.pred_buff: Dict[int, List[Dict[str, Tensor]]] = defaultdict(list)
        self.target_buff: Dict[int, List[Dict[str, Tensor]]] = defaultdict(list)

    def update(
        self, preds: List[Dict[str, Tensor]], targets: List[Dict[str, Tensor]]
    ) -> None:
        """Update buffers with sorted predictions and targets."""
        for pred, target in zip(preds, targets):
            # Get unique labels from both predictions and targets.
            all_labels = torch.unique(torch.cat([pred["labels"], target["labels"]]))
            for label in all_labels:
                self._update_label_buffers(label.item(), pred, target)

    def _update_label_buffers(
        self, label: int, pred: Dict[str, Tensor], target: Dict[str, Tensor]
    ) -> None:
        # Process predictions: filter by label and sort descending by score.
        pred_mask = pred["labels"] == label
        scores = pred["scores"][pred_mask]
        sorted_idx = scores.argsort(descending=True)
        self.pred_buff[label].append(
            {
                "boxes": pred["boxes"][pred_mask][sorted_idx],
                "scores": scores[sorted_idx],
            }
        )
        # Process targets: filter by label.
        target_mask = target["labels"] == label
        self.target_buff[label].append({"boxes": target["boxes"][target_mask]})

    def compute(self) -> float:
        """Compute mean Average Precision (mAP) across all classes."""
        average_precisions = [
            self.compute_ap_for_label(label) for label in self.target_buff.keys()
        ]
        return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0

    def compute_ap_for_label(self, label: int) -> float:
        """Compute AP for a given label using VOC07 11-point interpolation."""
        all_detections, gt_counts, total_gt = self._prepare_detections_and_gt_counts(label)
        if not all_detections or total_gt == 0:
            return 0.0

        tp, fp = self._process_detections(all_detections, label)
        recalls, precisions = self._compute_precision_recall(tp, fp, total_gt)
        return self._compute_voc07_ap(recalls, precisions)

    def _prepare_detections_and_gt_counts(self, label: int) -> Union[List, List, int]:
        all_detections = []
        gt_counts = []
        num_images = len(self.pred_buff[label])
        for img_idx in range(num_images):
            img_dets = self.pred_buff[label][img_idx]
            for score, box in zip(img_dets["scores"], img_dets["boxes"]):
                all_detections.append((img_idx, score.item(), box))
            gt_counts.append(len(self.target_buff[label][img_idx]["boxes"]))
        # Sort all detections by score in descending order.
        all_detections.sort(key=lambda x: x[1], reverse=True)
        return all_detections, gt_counts, sum(gt_counts)

    def _process_detections(
        self, all_detections: List, label: int
    ) -> Union[Tensor, Tensor]:
        tp = []
        fp = []
        # Use a dictionary to track which GT boxes have been matched for each image.
        matched = {}
        for img_idx, score, box in all_detections:
            gt_boxes = self.target_buff[label][img_idx]["boxes"]
            # If there are no GT boxes for this image, count as false positive.
            if len(gt_boxes) == 0:
                fp.append(1)
                tp.append(0)
                continue

            # Initialize matched for this image if not already done.
            if img_idx not in matched:
                matched[img_idx] = torch.zeros(len(gt_boxes), dtype=torch.bool)

            # Compute IoU between the predicted box and all GT boxes.
            ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
            max_iou, gt_idx = ious.max(dim=0)
            # Check if the detection is a true positive and hasn't been matched yet.
            if max_iou >= self.iou_thresh and not matched[img_idx][gt_idx]:
                tp.append(1)
                fp.append(0)
                matched[img_idx][gt_idx] = True
            else:
                fp.append(1)
                tp.append(0)

        return torch.tensor(tp, dtype=torch.float32), torch.tensor(fp, dtype=torch.float32)

    @staticmethod
    def _compute_precision_recall(
        tp: Tensor, fp: Tensor, total_gt: int
    ) -> Union[Tensor, Tensor]:
        tp_cumsum = tp.cumsum(0)
        fp_cumsum = fp.cumsum(0)
        recalls = tp_cumsum / total_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        return recalls, precisions

    @staticmethod
    def _compute_voc07_ap(recalls: Tensor, precisions: Tensor) -> float:
        ap = 0.0
        # VOC07 uses an 11-point interpolation method.
        for t in torch.arange(0, 1.1, 0.1):
            mask = recalls >= t
            ap += precisions[mask].max().item() if mask.any() else 0.0
        return ap / 11.0


if __name__ == "__main__":
    # Test Case 1: Single image, one correct detection.
    preds = [{
        "boxes": torch.tensor([[10, 10, 20, 20]], dtype=torch.float32),
        "scores": torch.tensor([0.9], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
    }]
    targets = [{
        "boxes": torch.tensor([[10, 10, 20, 20]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
    }]
    metric = VOC07_mAP(iou_thresh=0.5)
    metric.update(preds, targets)
    print("Test Case 1 mAP (expect 1.0):", metric.compute())
    metric.reset()

    # Test Case 2: Single image, one false detection.
    preds = [{
        "boxes": torch.tensor([[30, 30, 40, 40]], dtype=torch.float32),
        "scores": torch.tensor([0.9], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
    }]
    targets = [{
        "boxes": torch.tensor([[10, 10, 20, 20]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
    }]
    metric.update(preds, targets)
    print("Test Case 2 mAP (expect 0.0):", metric.compute())
    metric.reset()

    # Test Case 3: Two images, multiple labels.
    preds = [
        {
            "boxes": torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=torch.float32),
            "scores": torch.tensor([0.9, 0.8], dtype=torch.float32),
            "labels": torch.tensor([1, 2], dtype=torch.int64),
        },
        {
            "boxes": torch.tensor([[15, 15, 25, 25]], dtype=torch.float32),
            "scores": torch.tensor([0.85], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=torch.float32),
            "labels": torch.tensor([1, 2], dtype=torch.int64),
        },
        {
            "boxes": torch.tensor([[15, 15, 25, 25]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
    ]
    metric.update(preds, targets)
    print("Test Case 3 mAP (expect 1.0):", metric.compute())
    metric.reset()

    # Test Case 4: Single image, multiple detections for one GT.
    preds = [{
        "boxes": torch.tensor([[10, 10, 20, 20], [10, 10, 20, 20]], dtype=torch.float32),
        "scores": torch.tensor([0.9, 0.8], dtype=torch.float32),
        "labels": torch.tensor([1, 1], dtype=torch.int64),
    }]
    targets = [{
        "boxes": torch.tensor([[10, 10, 20, 20]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
    }]
    metric.update(preds, targets)
    print("Test Case 4 mAP (expect 1.0):", metric.compute())
    metric.reset()
