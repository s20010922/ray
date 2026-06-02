"""Ray Data 串流預處理：records → 訓練 batch。

build_ray_dataset() 把 sources.py 的 records 包成 Ray Dataset，在多 CPU 上
串流處理：解碼 →（train）劣化增強 + 水平翻轉 → resize 640 → 正規化 → pad。

輸出每筆：
  image:       (3, 640, 640) float32, RGB, 0~1
  boxes_xywhn: (MAX_BOXES, 4) float32, 正規化中心點+寬高（補零）
  labels:      (MAX_BOXES,)  int64, padding 補 -1（表示無效框）

劣化增強只在 train split（augment=True）；val 保持原樣以反映真實評估。
"""

from typing import Any, Dict, List

import cv2
import numpy as np
import ray
from ray.data import Dataset

from src.data.augment import degrade
from src.modeling.traffic import IMG_SIZE

cv2.setNumThreads(1)  # Ray 已在多 worker 平行，單 worker 內不再多開執行緒

# 單幀最多保留的框數（UA-DETRAC 密集場景可能數十台車）。超過則截斷。
MAX_BOXES = 100


def _pad(xywhn: np.ndarray, labels: np.ndarray) -> tuple:
    """把變長的 (N,4)/(N,) pad 到固定 MAX_BOXES。padding：box=0, label=-1。"""
    n = min(len(labels), MAX_BOXES)
    boxes = np.zeros((MAX_BOXES, 4), dtype=np.float32)
    labs = np.full(MAX_BOXES, -1, dtype=np.int64)
    if n:
        boxes[:n] = xywhn[:n]
        labs[:n] = labels[:n]
    return boxes, labs


def _preprocess(batch: Dict[str, np.ndarray],
                augment: bool) -> Dict[str, np.ndarray]:
    """對一個 batch 的 records 做解碼 + 增強 + resize + 正規化 + pad。"""
    rng = np.random.default_rng() if augment else None
    paths = batch["image_path"]
    boxes_in = batch["boxes_xyxy"]
    labels_in = batch["labels"]

    imgs: List[np.ndarray] = []
    boxes_pad: List[np.ndarray] = []
    labels_pad: List[np.ndarray] = []

    for i, path in enumerate(paths):
        img = cv2.imread(str(path))
        if img is None:
            continue
        boxes = np.asarray(boxes_in[i], dtype=np.float32).copy()  # (N,4) xyxy 像素
        labels = np.asarray(labels_in[i], dtype=np.int64)

        if augment and rng is not None:
            img = degrade(img, rng)                  # 劣化（bbox 不變）
            if rng.random() < 0.5:                   # 水平翻轉（同步翻 bbox）
                w = img.shape[1]
                img = img[:, ::-1, :].copy()
                if boxes.size:
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

        h0, w0 = img.shape[:2]
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        imgs.append(np.transpose(img, (2, 0, 1)))    # CHW

        # xyxy 像素 → 正規化 xywh（正規化與 resize 後尺寸無關，用原圖 w0/h0）
        xc = (boxes[:, 0] + boxes[:, 2]) * 0.5 / w0
        yc = (boxes[:, 1] + boxes[:, 3]) * 0.5 / h0
        bw = (boxes[:, 2] - boxes[:, 0]) / w0
        bh = (boxes[:, 3] - boxes[:, 1]) / h0
        xywhn = np.stack([xc, yc, bw, bh], axis=1).astype(np.float32)

        b_pad, l_pad = _pad(xywhn, labels)
        boxes_pad.append(b_pad)
        labels_pad.append(l_pad)

    if not imgs:
        return {
            "image": np.zeros((0, 3, IMG_SIZE, IMG_SIZE), np.float32),
            "boxes_xywhn": np.zeros((0, MAX_BOXES, 4), np.float32),
            "labels": np.zeros((0, MAX_BOXES), np.int64),
        }
    return {
        "image": np.stack(imgs).astype(np.float32),
        "boxes_xywhn": np.stack(boxes_pad),
        "labels": np.stack(labels_pad),
    }


def build_ray_dataset(records: List[Dict[str, Any]],
                      augment: bool = False,
                      cpu_per_task: int = 1,
                      batch_size: int = 16) -> Dataset:
    """records → Ray Dataset（串流預處理）。train split 傳 augment=True。"""
    ds = ray.data.from_items(records)
    return ds.map_batches(
        _preprocess,
        fn_kwargs={"augment": augment},
        batch_size=batch_size,
        num_cpus=cpu_per_task,
        batch_format="numpy",
    )
