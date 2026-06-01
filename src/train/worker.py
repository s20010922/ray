"""Ray Train user-defined training loop.

Runs *inside* each Ray Train worker. Reads `config` from `train_loop_config`
(also overridable by Ray Tune via param_space), pulls shards via
`train.get_dataset_shard`, reports per-epoch metrics + checkpoint via
`train.report`.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from ray import train
from ray.train import Checkpoint
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from ultralytics.utils.ops import non_max_suppression

from src.data.targets import unpack_to_v8_targets
from src.modeling.loss import build_loss
from src.modeling.metrics import match_and_score, precision_recall_f1
from src.modeling.yolo import load_yolo_cls_for_training, load_yolo_for_training


def _evaluate(model, loss_fn, val_shard, device, batch_size: int,
              conf_thres: float = 0.25, nms_iou: float = 0.45):
    """Val loss + greedy P/R/F1 @ IoU>=0.5. Swaps to eval() and back."""
    model.eval()
    running_loss, n_batches = 0.0, 0
    n_pred = n_gt = n_match = 0
    inner = model.module if hasattr(model, "module") else model
    try:
        with torch.no_grad():
            for batch in val_shard.iter_batches(
                    batch_size=batch_size, batch_format="numpy"):
                if batch["image"].shape[0] == 0:
                    continue
                imgs = torch.as_tensor(batch["image"], device=device)
                labels_pad, boxes_pad, _, targets = unpack_to_v8_targets(
                    batch, device)

                out = inner(imgs)
                if isinstance(out, tuple):
                    nms_input, raw_for_loss = out[0], out[1]
                else:
                    nms_input, raw_for_loss = out, out

                loss, _ = loss_fn(raw_for_loss, targets)
                running_loss += float(loss.detach())
                n_batches += 1

                dets = non_max_suppression(nms_input, conf_thres, nms_iou)
                stats = match_and_score(dets, labels_pad, boxes_pad)
                n_pred += stats["n_pred"]
                n_gt   += stats["n_gt"]
                n_match += stats["n_match"]
    finally:
        model.train()

    p, r, f1 = precision_recall_f1(n_pred, n_gt, n_match)
    return {
        "val_loss":    running_loss / max(n_batches, 1),
        "val_p@0.5":   p,
        "val_r@0.5":   r,
        "val_f1@0.5":  f1,
        "val_n_pred":  n_pred,
        "val_n_gt":    n_gt,
        "val_n_match": n_match,
    }


def train_loop_per_worker(config: Dict[str, Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_yolo_for_training(config["model_weights"], device)
    loss_fn = build_loss(model)             # before DDP wrapping (needs raw .args)
    model = train.torch.prepare_model(model)
    optimizer = AdamW(model.parameters(), lr=config["lr"])
    scheduler = (CosineAnnealingLR(optimizer, T_max=config["epochs"])
                 if config.get("use_cosine_lr", False) else None)

    train_shard = train.get_dataset_shard("train")
    val_shard   = train.get_dataset_shard("val")

    for epoch in range(config["epochs"]):
        model.train()
        running, n_batches = 0.0, 0
        for batch in train_shard.iter_batches(
                batch_size=config["batch_size"], batch_format="numpy"):
            if batch["image"].shape[0] == 0:
                continue
            imgs = torch.as_tensor(batch["image"], device=device)
            _, _, _, targets = unpack_to_v8_targets(batch, device)
            preds = model(imgs)
            loss, _ = loss_fn(preds, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        metrics = {"epoch": epoch,
                   "train_loss": running / max(n_batches, 1),
                   "train_batches": n_batches,
                   "lr": optimizer.param_groups[0]["lr"]}

        if val_shard is not None:
            metrics.update(_evaluate(
                model, loss_fn, val_shard, device, config["batch_size"]))

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "model.pt"
            inner = model.module if hasattr(model, "module") else model
            torch.save({"model_state_dict": inner.state_dict(),
                        "epoch": epoch, **metrics}, ckpt_path)
            train.report(metrics, checkpoint=Checkpoint.from_directory(tmp))


# ----------------------------- classification ------------------------------

def _evaluate_cls(model, val_shard, device, batch_size: int):
    """Accuracy + precision/recall/F1 for binary accident detection
    (positive class = label 1 = accident)."""
    model.eval()
    running_loss, n_batches = 0.0, 0
    n_correct = n_total = 0
    tp = fp = fn = 0
    inner = model.module if hasattr(model, "module") else model
    try:
        with torch.no_grad():
            for batch in val_shard.iter_batches(
                    batch_size=batch_size, batch_format="numpy"):
                if batch["image"].shape[0] == 0:
                    continue
                imgs   = torch.as_tensor(batch["image"], device=device)
                labels = torch.as_tensor(batch["label"], device=device)
                logits = inner(imgs)
                loss   = F.cross_entropy(logits, labels)
                running_loss += float(loss.detach())
                n_batches += 1

                preds = logits.argmax(dim=1)
                n_correct += int((preds == labels).sum())
                n_total   += int(labels.numel())
                tp += int(((preds == 1) & (labels == 1)).sum())
                fp += int(((preds == 1) & (labels == 0)).sum())
                fn += int(((preds == 0) & (labels == 1)).sum())
    finally:
        model.train()

    acc = n_correct / max(n_total, 1)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {
        "val_loss":     running_loss / max(n_batches, 1),
        "val_accuracy": acc,
        "val_p":        p,
        "val_r":        r,
        "val_f1":       f1,
        "val_tp":       tp,
        "val_fp":       fp,
        "val_fn":       fn,
    }


def train_loop_per_worker_cls(config: Dict[str, Any]) -> None:
    """Binary accident classifier. Same Ray Train scaffolding as the detection
    loop, but uses YOLOv8n-cls + CrossEntropy + accuracy/F1 reporting.

    v4 regularization: mixup (alpha>0) blends pairs of samples in the batch
    to fight memorisation of the 17 train videos. Cosine LR schedule (when
    use_cosine_lr) anneals over `epochs` to find flatter minima.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_yolo_cls_for_training(config["model_weights"], device)
    model = train.torch.prepare_model(model)
    optimizer = AdamW(model.parameters(),
                      lr=config["lr"],
                      weight_decay=config.get("weight_decay", 0.0))
    scheduler = (CosineAnnealingLR(optimizer, T_max=config["epochs"])
                 if config.get("use_cosine_lr", False) else None)
    mixup_alpha = float(config.get("mixup_alpha", 0.0))
    rng = np.random.default_rng()

    train_shard = train.get_dataset_shard("train")
    val_shard   = train.get_dataset_shard("val")

    for epoch in range(config["epochs"]):
        model.train()
        running, n_batches = 0.0, 0
        for batch in train_shard.iter_batches(
                batch_size=config["batch_size"], batch_format="numpy"):
            if batch["image"].shape[0] == 0:
                continue
            imgs   = torch.as_tensor(batch["image"], device=device)
            labels = torch.as_tensor(batch["label"], device=device)
            if mixup_alpha > 0:
                lam = float(rng.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(imgs.size(0), device=device)
                imgs = lam * imgs + (1 - lam) * imgs[perm]
                logits = model(imgs)
                loss = (lam * F.cross_entropy(logits, labels)
                        + (1 - lam) * F.cross_entropy(logits, labels[perm]))
            else:
                logits = model(imgs)
                loss   = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        metrics = {"epoch": epoch,
                   "train_loss": running / max(n_batches, 1),
                   "train_batches": n_batches,
                   "lr": optimizer.param_groups[0]["lr"]}

        if val_shard is not None:
            metrics.update(_evaluate_cls(
                model, val_shard, device, config["batch_size"]))

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "model.pt"
            inner = model.module if hasattr(model, "module") else model
            torch.save({"model_state_dict": inner.state_dict(),
                        "epoch": epoch, **metrics}, ckpt_path)
            train.report(metrics, checkpoint=Checkpoint.from_directory(tmp))
