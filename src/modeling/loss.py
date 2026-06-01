"""v8DetectionLoss wrapper. Trivial today but isolates ultralytics import so
swapping loss impl (e.g. for a custom focal variant) touches one file.
"""

import torch
from ultralytics.utils.loss import v8DetectionLoss


def build_loss(model: torch.nn.Module) -> v8DetectionLoss:
    return v8DetectionLoss(model)
