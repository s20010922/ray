"""車禍分類推論工具。

載入 Ray Train 存下的 checkpoint（state_dict 格式），重建模型後推論。

checkpoint 結構：
  {'model': OrderedDict (state_dict), 'classes': ['accident', 'non-accident']}
"""

import glob
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.modeling.accident import CLASSES, IMG_SIZE, load_accident_model


def _reshape_head(model: nn.Module, num_classes: int) -> nn.Module:
    head = model.model[-1]
    if hasattr(head, "linear") and head.linear.out_features != num_classes:
        head.linear = nn.Linear(head.linear.in_features, num_classes)
    return model


def load_classifier(checkpoint_path: str,
                    device: str = "cpu") -> Tuple[nn.Module, torch.device]:
    """載入 Ray Train checkpoint，回傳 (model, device)。"""
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu"
                       else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=dev)
    yolo = load_accident_model("yolo11n-cls.pt")
    model = _reshape_head(yolo.model, len(CLASSES))
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()
    return model, dev


def find_best_accident_checkpoint(results_root: str = "/workspace/ray_results",
                                  experiment: str = "accident") -> str:
    """自動找 val_acc 最高的 checkpoint。

    Ray Train 依 val_acc 命名 checkpoint 資料夾（checkpoint_000NNN），
    最後一個數字最大的通常是最佳（搭配 CheckpointConfig num_to_keep=2）。
    """
    pattern = os.path.join(results_root, experiment, "**/model.pt")
    candidates = sorted(glob.glob(pattern, recursive=True))
    if not candidates:
        raise FileNotFoundError(
            f"找不到 accident checkpoint，搜尋路徑：{pattern}")
    return candidates[-1]


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    """BGR ndarray → (1, 3, 224, 224) float32 tensor（ImageNet normalize）。"""
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE),
                     interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(img).unsqueeze(0)


def classify(model: nn.Module, img_bgr: np.ndarray,
             device: torch.device) -> Tuple[int, float]:
    """推論單張圖，回傳 (predicted_label, confidence)。"""
    x = preprocess(img_bgr).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(probs.argmax())
    return pred, float(probs[pred])
