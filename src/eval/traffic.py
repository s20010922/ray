"""車流偵測評估：自寫 mAP@0.5（單類 Vehicle）。

ultralytics 的 val 需要 YOLO 資料夾格式；這裡直接吃記憶體 records + 推論結果，
不落地。單類所以 mAP = AP(Vehicle)。

AP 算法（VOC2010 後的 all-point 積分）：
  1. 每張圖內，pred 依信心由高到低，greedy 配對 GT（IoU≥0.5 且 GT 未被佔用 → TP）
  2. 跨全資料集把所有 pred 依信心排序，累積 TP/FP → precision-recall 曲線
  3. AP = PR 曲線下面積（precision 取右側最大值的單調包絡後積分）
"""

from typing import List, Tuple

import numpy as np


def _iou_matrix(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """(P,4) 與 (G,4) 的 IoU 矩陣，xyxy 像素。回傳 (P,G)。"""
    if len(preds) == 0 or len(gts) == 0:
        return np.zeros((len(preds), len(gts)), np.float32)
    p = preds[:, None, :]          # (P,1,4)
    g = gts[None, :, :]            # (1,G,4)
    x1 = np.maximum(p[..., 0], g[..., 0])
    y1 = np.maximum(p[..., 1], g[..., 1])
    x2 = np.minimum(p[..., 2], g[..., 2])
    y2 = np.minimum(p[..., 3], g[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_p = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])
    area_g = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    union = area_p + area_g - inter
    return inter / np.clip(union, 1e-9, None)


def compute_map50(preds_per_img: List[Tuple[np.ndarray, np.ndarray]],
                  gts_per_img: List[np.ndarray],
                  iou_thr: float = 0.5) -> dict:
    """計算 mAP@0.5。

    Args:
        preds_per_img: 每張圖 (boxes_xyxy (P,4), scores (P,))
        gts_per_img:   每張圖 GT boxes_xyxy (G,4)
    Returns:
        {"map50", "precision", "recall", "n_gt", "n_pred"}（precision/recall 為
        信心門檻掃到底時的整體值）
    """
    scores_all: List[float] = []
    tp_all: List[int] = []
    n_gt_total = 0

    for (boxes, scores), gts in zip(preds_per_img, gts_per_img):
        n_gt_total += len(gts)
        if len(boxes) == 0:
            continue
        order = np.argsort(-scores)
        boxes, scores = boxes[order], scores[order]
        ious = _iou_matrix(boxes, gts)
        matched = np.zeros(len(gts), dtype=bool)
        for i in range(len(boxes)):
            j = int(np.argmax(ious[i])) if len(gts) else -1
            if j >= 0 and ious[i, j] >= iou_thr and not matched[j]:
                matched[j] = True
                tp_all.append(1)
            else:
                tp_all.append(0)
            scores_all.append(float(scores[i]))

    n_pred = len(scores_all)
    if n_pred == 0 or n_gt_total == 0:
        return {"map50": 0.0, "precision": 0.0, "recall": 0.0,
                "n_gt": n_gt_total, "n_pred": n_pred}

    order = np.argsort(-np.asarray(scores_all))
    tp = np.asarray(tp_all)[order]
    fp = 1 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt_total
    precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-9, None)

    # all-point 積分：precision 取單調包絡（右側最大）後對 recall 積分
    mrec = np.concatenate(([0.0], recall, [recall[-1]]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    return {"map50": ap,
            "precision": float(precision[-1]),
            "recall": float(recall[-1]),
            "n_gt": n_gt_total, "n_pred": n_pred}
