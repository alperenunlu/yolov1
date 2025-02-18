import torch
import torch.nn as nn
from collections import defaultdict
from typing import List, Dict
from torchvision.ops import box_iou

class MAP50Metric(nn.Module):
    def __init__(self):
        super().__init__()
        self.reset()
    
    def reset(self):
        """Reset buffers for new evaluation."""
        self.detections = defaultdict(list)
        self.ground_truths = defaultdict(list)
    
    def update(self, pred_dict_list: List[Dict[str, torch.Tensor]], labels_dict_list: List[Dict[str, torch.Tensor]]):
        """Accumulate detections and ground truths.
        
        Args:
            pred_dict_list: List of dicts with keys 'boxes', 'labels', 'scores' (predictions), each dict corresponds to an image.
            labels_dict_list: List of dicts with keys 'boxes', 'labels' (ground truth), each dict corresponds to an image.
        """
        for preds, gts in zip(pred_dict_list, labels_dict_list):
            pred_boxes = preds['boxes']  # Tensor[N, 4]
            pred_labels = preds['labels']  # Tensor[N]
            pred_scores = preds['scores']  # Tensor[N]
            gt_boxes = gts['boxes']  # Tensor[M, 4]
            gt_labels = gts['labels']  # Tensor[M]
            
            for label in torch.unique(torch.cat([pred_labels, gt_labels])):
                pred_mask = pred_labels == label
                gt_mask = gt_labels == label
                
                self.detections[label.item()].extend(
                    zip(pred_scores[pred_mask].tolist(), pred_boxes[pred_mask].tolist())
                )
                self.ground_truths[label.item()].extend(gt_boxes[gt_mask].tolist())
    
    def compute(self):
        """Compute the mAP@0.5 based on accumulated predictions and ground truths."""
        aps = []
        
        for label, detections in self.detections.items():
            gt_boxes = torch.tensor(self.ground_truths[label])  # All GT boxes for this label
            if gt_boxes.numel() == 0:
                continue
            
            detections.sort(reverse=True, key=lambda x: x[0])  # Sort by confidence score
            pred_boxes = torch.tensor([d[1] for d in detections])
            pred_scores = torch.tensor([d[0] for d in detections])
            
            ious = box_iou(pred_boxes, gt_boxes)  # IoU matrix
            detected = torch.zeros(len(gt_boxes))  # Track matched GT boxes
            tp = torch.zeros(len(pred_boxes))
            fp = torch.zeros(len(pred_boxes))
            
            for i, row in enumerate(ious):
                max_iou, max_idx = row.max(0)
                if max_iou >= 0.5 and detected[max_idx] == 0:
                    tp[i] = 1
                    detected[max_idx] = 1
                else:
                    fp[i] = 1
            
            tp_cumsum = torch.cumsum(tp, dim=0)
            fp_cumsum = torch.cumsum(fp, dim=0)
            recalls = tp_cumsum / len(gt_boxes)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            
            ap = self.compute_ap(precisions, recalls)
            aps.append(ap)
        
        return sum(aps) / len(aps) if aps else 0.0
    
    @staticmethod
    def compute_ap(precisions, recalls):
        """Compute the AP using 11-point interpolation (YOLOv1 style)."""
        ap = 0.0
        for t in torch.linspace(0, 1, 11):
            p = precisions[recalls >= t].max() if (recalls >= t).any() else 0
            ap += p / 11
        return ap
