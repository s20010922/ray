"""車流偵測推論：載入自訓 checkpoint，對單張影像出 bbox。

訓練存的 checkpoint 是我們自己的格式（model.pt 內含 state_dict + classes + nc），
不是 ultralytics 的 YOLO 封裝，所以這裡：
  1. 用 yolo11n.yaml(nc) 重建 DetectionModel，載入 state_dict
  2. 預處理「完全照訓練 pipeline」：直接 resize 640、BGR→RGB、/255（非 letterbox）
  3. eval forward → non_max_suppression → 框還原回原圖像素座標

座標還原：訓練是直接拉伸 resize（非等比），所以 x 乘 w0/640、y 乘 h0/640。
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

from src.modeling.traffic import CLASSES, IMG_SIZE


def load_detector(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """從自訓 checkpoint（model.pt）重建可推論的偵測模型。"""
    from ultralytics.nn.tasks import DetectionModel

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    nc = ckpt.get("nc", len(CLASSES))

    model = DetectionModel("yolo11n.yaml", nc=nc, verbose=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.nc = nc
    model.names = {i: c for i, c in enumerate(ckpt.get("classes", CLASSES))}
    model.eval().to(device)
    return model


def _preprocess(img_bgr: np.ndarray, device: str) -> torch.Tensor:
    """BGR 原圖 → (1,3,640,640) tensor，與訓練 pipeline 一致。"""
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0).to(device)


@torch.no_grad()
def detect(model: torch.nn.Module,
           img_bgr: np.ndarray,
           conf: float = 0.25,
           iou: float = 0.45,
           device: str = "cuda") -> Tuple[np.ndarray, np.ndarray]:
    """對單張 BGR 影像偵測車輛。

    Returns:
        boxes_xyxy: (N,4) float32，原圖像素座標
        scores:     (N,)  float32
    """
    from ultralytics.utils.ops import non_max_suppression

    h0, w0 = img_bgr.shape[:2]
    x = _preprocess(img_bgr, device)
    preds = model(x)                       # eval：回傳 (B, 4+nc, 8400) 或 tuple
    dets = non_max_suppression(preds, conf_thres=conf, iou_thres=iou)[0]

    if dets is None or not len(dets):
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

    dets = dets.cpu().numpy()
    boxes = dets[:, :4].copy()
    boxes[:, [0, 2]] *= w0 / IMG_SIZE      # 還原 x（直接拉伸的反向）
    boxes[:, [1, 3]] *= h0 / IMG_SIZE      # 還原 y
    return boxes.astype(np.float32), dets[:, 4].astype(np.float32)


def draw(img_bgr: np.ndarray, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """在影像上畫出偵測框與信心值（回傳新圖）。"""
    out = img_bgr.copy()
    for (x1, y1, x2, y2), s in zip(boxes.astype(int), scores):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{s:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(out, f"vehicles: {len(boxes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return out


def find_best_checkpoint(results_dir: str = "/workspace/ray_results/traffic") -> str:
    """找最近一次 traffic 訓練、編號最大的 checkpoint 的 model.pt。"""
    root = Path(results_dir)
    trials = sorted(root.glob("TorchTrainer_*"), key=lambda p: p.stat().st_mtime)
    if not trials:
        raise FileNotFoundError(f"找不到任何訓練結果於 {results_dir}")
    ckpts = sorted((trials[-1]).glob("checkpoint_*"))
    if not ckpts:
        raise FileNotFoundError(f"{trials[-1]} 下沒有 checkpoint")
    return str(ckpts[-1] / "model.pt")
