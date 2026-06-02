"""Ray Data 串流預處理（分類）：records → 訓練 batch。

build_ray_dataset_cls() 把 accident sources 的 records 包成 Ray Dataset，
串流：解碼 →（train）劣化增強 + 水平翻轉 → resize 224 → 正規化。

輸出每筆：
  image: (3, 224, 224) float32, RGB, 0~1
  label: int64

比 traffic 簡單：分類沒有 bbox，翻轉不需變換座標、也不用 pad。
劣化增強與 traffic 共用 src/data/augment.py（同樣適應高公局低畫質）。
"""

from typing import Dict, List

import cv2
import numpy as np
import ray
from ray.data import Dataset

from src.data.augment import degrade
from src.modeling.accident import IMG_SIZE

cv2.setNumThreads(1)


def _preprocess(batch: Dict[str, np.ndarray],
                augment: bool) -> Dict[str, np.ndarray]:
    """解碼 +（train）劣化增強 + 翻轉 → resize → 正規化。"""
    rng = np.random.default_rng() if augment else None
    paths = batch["image_path"]
    labels = batch["label"]

    imgs: List[np.ndarray] = []
    labs: List[int] = []
    for i, path in enumerate(paths):
        img = cv2.imread(str(path))
        if img is None:
            continue
        if augment and rng is not None:
            img = degrade(img, rng)              # 劣化（共用 augment）
            if rng.random() < 0.5:               # 水平翻轉（分類 label 不變）
                img = img[:, ::-1, :].copy()
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        imgs.append(np.transpose(img, (2, 0, 1)))
        labs.append(int(labels[i]))

    if not imgs:
        return {
            "image": np.zeros((0, 3, IMG_SIZE, IMG_SIZE), np.float32),
            "label": np.zeros((0,), np.int64),
        }
    return {
        "image": np.stack(imgs).astype(np.float32),
        "label": np.asarray(labs, dtype=np.int64),
    }


def build_ray_dataset_cls(records: List[Dict],
                          augment: bool = False,
                          cpu_per_task: int = 1,
                          batch_size: int = 32) -> Dataset:
    """records → Ray Dataset（分類串流預處理）。train split 傳 augment=True。"""
    ds = ray.data.from_items(records)
    return ds.map_batches(
        _preprocess,
        fn_kwargs={"augment": augment},
        batch_size=batch_size,
        num_cpus=cpu_per_task,
        batch_format="numpy",
    )
