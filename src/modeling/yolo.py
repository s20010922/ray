"""YOLO model loading. Centralised so training, eval, inference, and serve
all use the same 4-fix routine for ultralytics' yolov8n.pt.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.utils import IterableSimpleNamespace

from src.config import CLS_IMG_SIZE, CLS_NUM_CLASSES, IMG_SIZE


def _normalise_args(model: torch.nn.Module) -> None:
    """Fixes the four things ultralytics' loader leaves in a bad state for
    custom training loops:

    1. `model.args` is a dict; v8DetectionLoss does attribute access on it.
    2. The dict misses box/cls/dfl loss weights.
    3. The loaded weights are the EMA copy with `requires_grad=False`.
    4. (Not done here, handled by caller) strides/anchors need a warm-up fwd.
    """
    raw_args = dict(model.args) if isinstance(model.args, dict) \
        else dict(vars(model.args))
    for k, default in (("box", 7.5), ("cls", 0.5), ("dfl", 1.5)):
        raw_args.setdefault(k, default)
    model.args = IterableSimpleNamespace(**raw_args)


def load_yolo_for_training(weights: str, device: torch.device) -> torch.nn.Module:
    """Returns a fresh-from-pretrained model ready for training:
    train mode, on device, requires_grad enabled, strides primed.
    """
    yolo = YOLO(weights)
    model = yolo.model
    _normalise_args(model)
    model.to(device).train()
    for p in model.parameters():
        p.requires_grad_(True)
    with torch.no_grad():
        model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device))
    return model


def load_yolo_for_inference(weights: str, checkpoint: Optional[Path],
                            device: torch.device) -> Tuple[torch.nn.Module, dict]:
    """Build architecture from `weights`, optionally splice in fine-tuned
    state_dict from `checkpoint`. Returns (model_in_eval_mode, checkpoint_dict).
    """
    yolo = YOLO(weights)
    model = yolo.model
    _normalise_args(model)
    ckpt: dict = {}
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def _swap_classify_head(model: torch.nn.Module, num_classes: int) -> None:
    """YOLOv8-cls last module is a `Classify` block with .linear (Linear).
    Replace it so the 1000-class ImageNet head becomes our binary head."""
    head = model.model[-1]
    if not hasattr(head, "linear") or not isinstance(head.linear, nn.Linear):
        raise RuntimeError(
            f"expected Classify head with .linear, got {type(head).__name__}")
    in_features = head.linear.in_features
    head.linear = nn.Linear(in_features, num_classes)


def load_yolo_cls_for_training(weights: str, device: torch.device,
                               num_classes: int = CLS_NUM_CLASSES
                               ) -> torch.nn.Module:
    """YOLOv8n-cls fine-tuning setup: load ImageNet-pretrained backbone,
    replace the 1000-class head with `num_classes`, enable all grads,
    warm up with a zeros forward pass to prime any lazy buffers."""
    yolo = YOLO(weights)
    model = yolo.model
    _swap_classify_head(model, num_classes)
    model.to(device).train()
    for p in model.parameters():
        p.requires_grad_(True)
    with torch.no_grad():
        model(torch.zeros(1, 3, CLS_IMG_SIZE, CLS_IMG_SIZE, device=device))
    return model


def load_yolo_cls_for_inference(weights: str, checkpoint: Optional[Path],
                                device: torch.device,
                                num_classes: int = CLS_NUM_CLASSES
                                ) -> Tuple[torch.nn.Module, dict]:
    """Mirror of load_yolo_for_inference, but for classification."""
    yolo = YOLO(weights)
    model = yolo.model
    _swap_classify_head(model, num_classes)
    ckpt: dict = {}
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt
