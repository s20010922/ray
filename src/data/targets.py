"""Target padding / unpacking helpers.

Ray Data Arrow batches require every column to share the same first
dim, but YOLO targets are inherently ragged (varying boxes per image).
We pad to `MAX_BOXES` with `PAD_LABEL` in preprocessing, then unmask
back to flat form when feeding `v8DetectionLoss` in the train loop.
"""

from typing import Tuple

import torch

from src.config import IMG_SIZE, MAX_BOXES, PAD_LABEL


def pad_to_max(boxes_xywhn, labels):
    """numpy in, numpy out. Used in the Ray Data preprocess UDF."""
    import numpy as np
    k = min(len(labels), MAX_BOXES)
    b_pad = np.zeros((MAX_BOXES, 4), dtype=np.float32)
    l_pad = np.full((MAX_BOXES,), PAD_LABEL, dtype=np.int64)
    b_pad[:k] = boxes_xywhn[:k]
    l_pad[:k] = labels[:k]
    return b_pad, l_pad


def unpack_to_v8_targets(batch, device) -> Tuple[torch.Tensor, torch.Tensor,
                                                  torch.Tensor, dict]:
    """(N, MAX_BOXES, *) padded -> the flat dict v8DetectionLoss wants.

    Returns (labels_pad, boxes_pad, mask, targets_dict). The first three are
    kept for evaluation (which still needs per-image structure for IoU
    matching against predictions).
    """
    labels_pad = torch.as_tensor(batch["labels"], device=device)
    boxes_pad  = torch.as_tensor(batch["boxes_xywhn"], device=device)
    mask = labels_pad >= 0
    n = labels_pad.shape[0]
    bidx_full = torch.arange(n, device=device).unsqueeze(1) \
                     .expand_as(labels_pad)
    targets = {
        "cls":       labels_pad[mask].float().unsqueeze(1),
        "bboxes":    boxes_pad[mask].float(),
        "batch_idx": bidx_full[mask].float(),
    }
    return labels_pad, boxes_pad, mask, targets


def xywhn_to_xyxy_px(boxes_xywhn: torch.Tensor) -> torch.Tensor:
    """Convert per-image (k, 4) xywh normalised -> (k, 4) xyxy in IMG_SIZE px."""
    gx = boxes_xywhn[:, 0] * IMG_SIZE
    gy = boxes_xywhn[:, 1] * IMG_SIZE
    gw = boxes_xywhn[:, 2] * IMG_SIZE
    gh = boxes_xywhn[:, 3] * IMG_SIZE
    return torch.stack(
        [gx - gw / 2, gy - gh / 2, gx + gw / 2, gy + gh / 2], dim=1)
