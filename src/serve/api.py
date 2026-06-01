"""Ray Serve: two HTTP deployments sharing one GPU.

  TrafficDeployment   - YOLOv8n detection,  fine-tuned on UA-DETRAC
  AccidentDeployment  - YOLOv8n-cls binary, fine-tuned on UCF Crime

Each takes ~600MB VRAM at inference, so num_gpus=0.5 lets both coexist on the
3060 Ti. They're served as two separate apps (different route prefixes) so the
client can hit one without the other being up.
"""

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from ray import serve
from ultralytics.utils.ops import non_max_suppression

from src.config import (
    CLS_IMG_SIZE,
    COCO_NAMES,
    IMG_SIZE,
    TRAFFIC_COUNT_THRESHOLDS,
    TRAFFIC_DENSITY_THRESHOLDS,
    classify_level,
)
from src.modeling.yolo import (
    load_yolo_cls_for_inference,
    load_yolo_for_inference,
)


# ---------------------------- traffic ----------------------------------

traffic_api = FastAPI(title="Ray Serve - traffic detection")


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.5, "num_cpus": 2},
)
@serve.ingress(traffic_api)
class TrafficDeployment:
    def __init__(self,
                 checkpoint_path: str,
                 model_weights: str = "yolov8n.pt",
                 conf_thres: float = 0.25,
                 nms_iou: float = 0.45) -> None:
        self.conf_thres = conf_thres
        self.nms_iou = nms_iou
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, ckpt = load_yolo_for_inference(
            model_weights, Path(checkpoint_path), self.device)
        self.ckpt_meta = {
            "epoch": ckpt.get("epoch"),
            "val_f1@0.5": ckpt.get("val_f1@0.5"),
        }

    @traffic_api.get("/health")
    async def health(self) -> Dict[str, Any]:
        return {
            "status":     "ok",
            "model":      "traffic-yolov8n-detect",
            "device":     str(self.device),
            "checkpoint": self.ckpt_meta,
            "thresholds": {
                "count":   list(TRAFFIC_COUNT_THRESHOLDS),
                "density": list(TRAFFIC_DENSITY_THRESHOLDS),
            },
        }

    @traffic_api.post("/detect")
    async def detect(self, file: UploadFile = File(...)) -> Dict[str, Any]:
        raw = await file.read()
        nparr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "could not decode image", "detections": []}

        h0, w0 = img.shape[:2]
        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)) \
                      .unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)
            nms_input = out[0] if isinstance(out, tuple) else out
            dets = non_max_suppression(nms_input,
                                       self.conf_thres, self.nms_iou)[0]

        sx, sy = w0 / IMG_SIZE, h0 / IMG_SIZE
        detections: List[Dict[str, Any]] = []
        total_area = 0.0
        for det in dets.cpu().numpy():
            x1, y1, x2, y2, conf, cls = det.tolist()
            bx1, by1 = round(x1 * sx, 1), round(y1 * sy, 1)
            bx2, by2 = round(x2 * sx, 1), round(y2 * sy, 1)
            total_area += max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            detections.append({
                "box":      [bx1, by1, bx2, by2],
                "class":    COCO_NAMES.get(int(cls), str(int(cls))),
                "class_id": int(cls),
                "conf":     round(float(conf), 4),
            })

        density = total_area / max(w0 * h0, 1)
        n = len(detections)
        return {
            "image_size":     [w0, h0],
            "num_detections": n,
            "density":        round(density, 4),
            "count_level":    classify_level(n, TRAFFIC_COUNT_THRESHOLDS),
            "density_level":  classify_level(density, TRAFFIC_DENSITY_THRESHOLDS),
            "detections":     detections,
        }


def build_traffic_app(checkpoint_path: str, **kwargs):
    """`serve.run(build_traffic_app(...), route_prefix='/traffic')`"""
    return TrafficDeployment.bind(checkpoint_path=checkpoint_path, **kwargs)


# ---------------------------- accident ---------------------------------

accident_api = FastAPI(title="Ray Serve - accident classification")


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.5, "num_cpus": 2},
)
@serve.ingress(accident_api)
class AccidentDeployment:
    def __init__(self,
                 checkpoint_path: str,
                 model_weights: str = "yolov8n-cls.pt",
                 threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, ckpt = load_yolo_cls_for_inference(
            model_weights, Path(checkpoint_path), self.device)
        self.ckpt_meta = {
            "epoch":        ckpt.get("epoch"),
            "val_f1":       ckpt.get("val_f1"),
            "val_accuracy": ckpt.get("val_accuracy"),
        }

    @accident_api.get("/health")
    async def health(self) -> Dict[str, Any]:
        return {
            "status":     "ok",
            "model":      "accident-yolov8n-cls",
            "device":     str(self.device),
            "checkpoint": self.ckpt_meta,
            "classes":    ["normal", "accident"],
            "threshold":  self.threshold,
        }

    @accident_api.post("/detect")
    async def detect(self, file: UploadFile = File(...)) -> Dict[str, Any]:
        raw = await file.read()
        nparr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "could not decode image"}

        resized = cv2.resize(img, (CLS_IMG_SIZE, CLS_IMG_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)) \
                      .unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().tolist()

        # class 1 = accident (see ACCIDENT_LABEL_ACCIDENT in src/data/sources.py)
        accident_conf = float(probs[1])
        return {
            "is_accident": accident_conf >= self.threshold,
            "confidence":  round(accident_conf, 4),
            "threshold":   self.threshold,
            "probs":       {"normal":   round(probs[0], 4),
                            "accident": round(probs[1], 4)},
        }


def build_accident_app(checkpoint_path: str, **kwargs):
    """`serve.run(build_accident_app(...), route_prefix='/accident')`"""
    return AccidentDeployment.bind(checkpoint_path=checkpoint_path, **kwargs)
