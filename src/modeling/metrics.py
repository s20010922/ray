"""Detection metrics: greedy IoU-matching -> precision / recall / F1 at a
single IoU threshold. Not a full mAP but enough for "is this getting better".
"""

from typing import Dict, List

import torch
from torchvision.ops import box_iou

from src.data.targets import xywhn_to_xyxy_px


@torch.no_grad()
def match_and_score(dets_per_image: List[torch.Tensor],
                    labels_pad: torch.Tensor,
                    boxes_pad_xywhn: torch.Tensor,
                    match_iou: float = 0.5) -> Dict[str, int]:
    """Greedy class-aware matching, IoU>=match_iou counts as TP.

    Args:
        dets_per_image: NMS output, list of (M, 6) [x1, y1, x2, y2, conf, cls]
        labels_pad: (N, MAX_BOXES) int64, PAD_LABEL for unused slots
        boxes_pad_xywhn: (N, MAX_BOXES, 4) float32 normalised xywh

    Returns:
        {"n_pred": int, "n_gt": int, "n_match": int}
    """
    n_pred = n_gt = n_match = 0
    device = labels_pad.device

    for i, det in enumerate(dets_per_image):
        gt_mask = labels_pad[i] >= 0
        gt_lab  = labels_pad[i][gt_mask]
        gt_xyxy = xywhn_to_xyxy_px(boxes_pad_xywhn[i][gt_mask])

        n_gt += len(gt_lab)
        n_pred += len(det)
        if len(det) == 0 or len(gt_lab) == 0:
            continue

        ious = box_iou(det[:, :4], gt_xyxy)
        pred_cls = det[:, 5].long()
        matched = torch.zeros(len(gt_lab), dtype=torch.bool, device=device)
        for j in range(len(det)):
            same = (gt_lab == pred_cls[j]) & ~matched
            if not same.any():
                continue
            row = ious[j].clone()
            row[~same] = 0
            best, idx = row.max(0)
            if best >= match_iou:
                n_match += 1
                matched[idx] = True

    return {"n_pred": n_pred, "n_gt": n_gt, "n_match": n_match}


def precision_recall_f1(n_pred: int, n_gt: int, n_match: int):
    p = n_match / max(n_pred, 1)
    r = n_match / max(n_gt, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return p, r, f1
