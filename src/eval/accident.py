"""車禍分類評估指標（test set 用）。

compute_cls_metrics() 接收 y_true / y_pred（整數 label），
回傳 accuracy、per-class precision/recall/F1、macro F1、confusion matrix。

label 對應：0=accident, 1=non-accident（對齊 CLASSES 順序）。
"""

from typing import Dict, List

import numpy as np

from src.modeling.accident import CLASSES


def compute_cls_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """計算分類評估指標。

    Args:
        y_true: 真實 label 列表（0=accident, 1=non-accident）
        y_pred: 預測 label 列表

    Returns:
        {
          "accuracy": float,
          "macro_f1": float,
          "per_class": {cls_name: {"precision", "recall", "f1", "support"}},
          "confusion_matrix": np.ndarray (n_cls × n_cls),
              rows=真實, cols=預測
        }
    """
    n = len(CLASSES)
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)

    accuracy = float((yt == yp).mean())

    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(yt, yp):
        if 0 <= t < n and 0 <= p < n:
            cm[t, p] += 1

    per_class = {}
    f1s = []
    for i, cls_name in enumerate(CLASSES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        support = int(cm[i, :].sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[cls_name] = {
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "support":   support,
        }
        f1s.append(f1)

    return {
        "accuracy":        round(accuracy, 4),
        "macro_f1":        round(float(np.mean(f1s)), 4),
        "per_class":       per_class,
        "confusion_matrix": cm,
    }
