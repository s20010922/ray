"""Ray Data: streaming preprocessing pipeline.

`build_ray_dataset()` wraps a list of records (from `src.data.sources`) into
a Ray Dataset that decodes + resizes + pads on the fly across CPU workers.
"""

from typing import Any, Dict, List

import cv2
import numpy as np
import ray
from ray.data import Dataset

from src.config import CLS_IMG_SIZE, IMG_SIZE, MAX_BOXES
from src.data.targets import pad_to_max

cv2.setNumThreads(1)


def _augment_det(img: np.ndarray, boxes_xyxy: np.ndarray,
                 rng: np.random.Generator):
    """Augmentations for detection. Applied to the raw image (HWC BGR uint8)
    BEFORE resize so the bbox coordinate transform stays simple. Closes the
    biggest gap from the v1 traffic model: no night, no color/lighting
    variation, no flips => brittle on freeway live cameras after dusk."""
    h, w = img.shape[:2]

    # 1. Horizontal flip (50%) — also flip box x-coords.
    if rng.random() < 0.5:
        img = img[:, ::-1, :].copy()
        if boxes_xyxy.size:
            boxes_xyxy = boxes_xyxy.copy()
            boxes_xyxy[:, [0, 2]] = w - boxes_xyxy[:, [2, 0]]

    # 2. HSV jitter (hue ±10, sat ±0.4x, val ±0.4x) — color robustness.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + int(rng.integers(-10, 11))) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(rng.uniform(0.6, 1.4)), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * float(rng.uniform(0.6, 1.4)), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 3. Random global darkening (30%) — crude proxy for "night camera".
    # Helps the model see vehicles in low-light freeway footage.
    if rng.random() < 0.30:
        img = (img.astype(np.float32) * float(rng.uniform(0.25, 0.55))) \
              .clip(0, 255).astype(np.uint8)

    return img, boxes_xyxy


def preprocess_batch(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return _preprocess_det_impl(batch, augment=False)


def preprocess_batch_aug(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return _preprocess_det_impl(batch, augment=True)


def _preprocess_det_impl(batch: Dict[str, np.ndarray],
                         augment: bool) -> Dict[str, np.ndarray]:
    """Decode image -> [optional augment] -> resize 640 -> CHW float -> pad."""
    imgs:       List[np.ndarray] = []
    boxes_pad:  List[np.ndarray] = []
    labels_pad: List[np.ndarray] = []
    rng = np.random.default_rng() if augment else None

    paths     = batch["image_path"]
    boxes_in  = batch["boxes_xyxy"]
    labels_in = batch["labels"]

    for i, path in enumerate(paths):
        img = cv2.imread(str(path))
        if img is None:
            continue
        boxes = np.asarray(boxes_in[i], dtype=np.float32).copy()
        if augment and rng is not None:
            img, boxes = _augment_det(img, boxes, rng)
        h0, w0 = img.shape[:2]
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        imgs.append(np.transpose(img, (2, 0, 1)))

        labels = np.asarray(labels_in[i], dtype=np.int64)
        sx, sy = IMG_SIZE / w0, IMG_SIZE / h0
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        xc = (boxes[:, 0] + boxes[:, 2]) * 0.5 / IMG_SIZE
        yc = (boxes[:, 1] + boxes[:, 3]) * 0.5 / IMG_SIZE
        bw = (boxes[:, 2] - boxes[:, 0]) / IMG_SIZE
        bh = (boxes[:, 3] - boxes[:, 1]) / IMG_SIZE
        xywhn = np.stack([xc, yc, bw, bh], axis=1).astype(np.float32)

        b_pad, l_pad = pad_to_max(xywhn, labels)
        boxes_pad.append(b_pad)
        labels_pad.append(l_pad)

    if not imgs:
        return {
            "image":       np.zeros((0, 3, IMG_SIZE, IMG_SIZE), np.float32),
            "boxes_xywhn": np.zeros((0, MAX_BOXES, 4),          np.float32),
            "labels":      np.zeros((0, MAX_BOXES),             np.int64),
        }
    return {
        "image":       np.stack(imgs).astype(np.float32),
        "boxes_xywhn": np.stack(boxes_pad),
        "labels":      np.stack(labels_pad),
    }


def build_ray_dataset(records: List[Dict[str, Any]],
                      cpu_per_task: int = 1,
                      augment: bool = False) -> Dataset:
    """from_items -> map_batches. Pass augment=True for the train split."""
    fn = preprocess_batch_aug if augment else preprocess_batch
    ds = ray.data.from_items(records)
    return ds.map_batches(
        fn,
        batch_size=16,
        num_cpus=cpu_per_task,
        batch_format="numpy",
        zero_copy_batch=True,
    )


def _augment_cls(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stronger augmentation for accident classification (HWC BGR uint8
    in/out). v2 was already much better than v1 thanks to light aug; this
    pushes harder to close the val->test generalization gap (0.86 -> 0.72)
    and to make the model less reliant on UCF's visual style."""
    h, w = img.shape[:2]

    # 1. Horizontal flip (50%).
    if rng.random() < 0.5:
        img = img[:, ::-1, :]

    # 2. Random crop-and-resize: 60-100% area (was 80-100%). Wider range
    #    encourages partial-scene robustness.
    scale = float(rng.uniform(0.6, 1.0))
    ch, cw = int(h * scale), int(w * scale)
    y0 = int(rng.integers(0, h - ch + 1))
    x0 = int(rng.integers(0, w - cw + 1))
    img = img[y0:y0 + ch, x0:x0 + cw]

    # 3. HSV jitter (hue/sat/val): UCF clips are colour-graded; without this
    #    the model latches onto that specific palette.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + int(rng.integers(-10, 11))) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(rng.uniform(0.7, 1.3)), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * float(rng.uniform(0.7, 1.3)), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 4. Brightness/contrast (was alpha 0.85-1.15, beta ±20).
    alpha = float(rng.uniform(0.7, 1.3))
    beta  = float(rng.uniform(-40, 40))
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # 5. Random darken (25%) — simulate night/poor camera.
    if rng.random() < 0.25:
        img = (img.astype(np.float32) * float(rng.uniform(0.3, 0.6))) \
              .clip(0, 255).astype(np.uint8)

    return img


def preprocess_batch_cls(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return _preprocess_cls_impl(batch, augment=False)


def preprocess_batch_cls_aug(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return _preprocess_cls_impl(batch, augment=True)


def _preprocess_cls_impl(batch: Dict[str, np.ndarray],
                         augment: bool) -> Dict[str, np.ndarray]:
    """Decode image -> [optional augment] -> resize CLS_IMG_SIZE -> CHW float."""
    imgs:   List[np.ndarray] = []
    labels: List[int]        = []
    rng = np.random.default_rng()   # per-call seed = thread-safe-enough
    for path, lbl in zip(batch["image_path"], batch["label"]):
        img = cv2.imread(str(path))
        if img is None:
            continue
        if augment:
            img = _augment_cls(img, rng)
        img = cv2.resize(img, (CLS_IMG_SIZE, CLS_IMG_SIZE),
                         interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        imgs.append(np.transpose(img, (2, 0, 1)))
        labels.append(int(lbl))

    if not imgs:
        return {
            "image": np.zeros((0, 3, CLS_IMG_SIZE, CLS_IMG_SIZE), np.float32),
            "label": np.zeros((0,), np.int64),
        }
    return {
        "image": np.stack(imgs).astype(np.float32),
        "label": np.asarray(labels, dtype=np.int64),
    }


def build_ray_dataset_cls(records: List[Dict[str, Any]],
                          cpu_per_task: int = 1,
                          augment: bool = False) -> Dataset:
    """Classification variant. Pass augment=True for the train split."""
    fn = preprocess_batch_cls_aug if augment else preprocess_batch_cls
    ds = ray.data.from_items(records)
    return ds.map_batches(
        fn,
        batch_size=32,
        num_cpus=cpu_per_task,
        batch_format="numpy",
        zero_copy_batch=True,
    )
