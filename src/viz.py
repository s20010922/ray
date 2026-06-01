"""cv2-based bbox drawing helpers. No Ray, no torch -- usable from any
inference path (offline batch demo OR Serve client-side rendering).
"""

import cv2
import numpy as np

from src.config import COCO_NAMES, IMG_SIZE

GREEN = (0, 255, 0)
BLUE = (255, 120, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def draw_detections(img: np.ndarray, dets, gt_boxes_xyxy=None,
                    gt_labels=None, conf_thres: float = 0.25) -> tuple:
    """In-place annotation. dets: (M,6) numpy or torch [x1,y1,x2,y2,conf,cls]
    in IMG_SIZE coords (will be rescaled to img's native size).
    Returns (n_pred_drawn, n_gt_drawn).
    """
    h0, w0 = img.shape[:2]
    sx, sy = w0 / IMG_SIZE, h0 / IMG_SIZE
    n_pred = 0

    if dets is not None and len(dets) > 0:
        arr = dets.cpu().numpy() if hasattr(dets, "cpu") else np.asarray(dets)
        for x1, y1, x2, y2, conf, cls in arr:
            if conf < conf_thres:
                continue
            x1, x2 = int(x1 * sx), int(x2 * sx)
            y1, y2 = int(y1 * sy), int(y2 * sy)
            cls = int(cls)
            cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)
            label = f"{COCO_NAMES.get(cls, str(cls))} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 4)),
                          (x1 + tw, y1), GREEN, -1)
            cv2.putText(img, label, (x1, max(th, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1)
            n_pred += 1

    n_gt = 0
    if gt_boxes_xyxy is not None:
        for box in gt_boxes_xyxy:
            x1, y1, x2, y2 = np.asarray(box).astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), BLUE, 1)
            n_gt += 1

    cv2.rectangle(img, (0, 0), (w0, 28), BLACK, -1)
    cv2.putText(img, f"GT={n_gt}  pred={n_pred}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
    cv2.putText(img, "GREEN=pred  BLUE=GT",
                (w0 - 230, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    return n_pred, n_gt
