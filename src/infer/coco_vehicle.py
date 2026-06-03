"""用官方 YOLO11 COCO 預訓練模型自動標「車輛」。

自訓 base（UA-DETRAC 單類 + 640 變形）在高公局 domain gap 大、夜間/小目標漏一堆，
不適合拿來自動標。官方 COCO 模型見過更多場景，且 ultralytics predict 內建 letterbox，
小目標表現好很多——拿來預標品質高，人工只需抽查刪錯框。

把 COCO 的 car/motorcycle/bus/truck 全部映射成單類 Vehicle(class 0)，
介面與 src.infer.traffic.detect 一致（回傳 boxes_xyxy 像素 + scores），
可直接餵給 prelabel。
"""

from typing import Tuple

import numpy as np

# COCO 類別 id：2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_COCO_IDS = [2, 3, 5, 7]


def load_coco_vehicle_model(size: str = "x"):
    """載入官方 YOLO11 COCO 模型（yolo11{n,s,m,l,x}.pt，首次會自動下載）。"""
    from ultralytics import YOLO
    return YOLO(f"yolo11{size}.pt")


def detect(model, img_bgr: np.ndarray, conf: float = 0.25,
           iou: float = 0.45, device: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """偵測車輛，所有車種映射成 Vehicle。

    ultralytics predict 直接吃 BGR ndarray、內建 letterbox，回傳的 xyxy
    已是原圖像素座標，不需自己還原。

    Returns:
        boxes_xyxy: (N,4) float32 原圖像素；scores: (N,) float32
    """
    r = model.predict(img_bgr, conf=conf, iou=iou, classes=VEHICLE_COCO_IDS,
                      device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
    scores = r.boxes.conf.cpu().numpy().astype(np.float32)
    return boxes, scores
