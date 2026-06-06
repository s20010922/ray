"""車禍分類訓練迴圈（在 Ray Train worker 上跑）。

train_loop_per_worker 是 Ray Train 在每個 worker 執行的函式：
  接 Ray Data 分片 → YOLO11n-cls 前向 → CrossEntropy → backward → report+checkpoint

這是「先打通 Ray Train 骨架」的最單純案例（分類）。骨架驗證後，traffic
偵測會套同一套結構，只是把 loss 換成 v8DetectionLoss。
"""

import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

import ray.train
from ray.train import Checkpoint

from src.modeling.accident import CLASSES, load_accident_model


def _reshape_head(model: nn.Module, num_classes: int) -> nn.Module:
    """yolo11n-cls.pt 預訓練是 ImageNet 1000 類，把分類頭換成本任務的類別數。"""
    head = model.model[-1]                       # ultralytics Classify head
    if hasattr(head, "linear") and head.linear.out_features != num_classes:
        head.linear = nn.Linear(head.linear.in_features, num_classes)
    return model


def train_loop_per_worker(config: dict) -> None:
    epochs = config["epochs"]
    lr = config["lr"]
    batch_size = config["batch_size"]

    device = ray.train.torch.get_device()

    # 模型：YOLO11n-cls 的底層 nn.Module，分類頭改成 2 類
    yolo = load_accident_model(config.get("weights", "yolo11n-cls.pt"))
    model = _reshape_head(yolo.model, len(CLASSES))
    model = ray.train.torch.prepare_model(model)   # 搬 GPU（+ DDP wrap）

    # weight_decay 預設 0；Ray Tune 可經 config 搜尋（見 scripts/tune_accident.py）
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=config.get("weight_decay", 0.0))

    train_shard = ray.train.get_dataset_shard("train")
    val_shard = ray.train.get_dataset_shard("val")

    for epoch in range(epochs):
        # ---- train ----
        model.train()
        running = 0.0
        n_batch = 0
        for batch in train_shard.iter_torch_batches(batch_size=batch_size,
                                                    device=device):
            images = batch["image"]                 # (B,3,224,224) float32
            labels = batch["label"].long()          # (B,)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batch += 1
        train_loss = running / max(1, n_batch)

        # ---- val ----
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_shard.iter_torch_batches(batch_size=batch_size,
                                                      device=device):
                logits = model(batch["image"])
                pred = logits.argmax(dim=1)
                labels = batch["label"].long()
                correct += (pred == labels).sum().item()
                total += labels.numel()
        val_acc = correct / max(1, total)

        # ---- report + checkpoint ----
        metrics = {"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc}
        state = (model.module if hasattr(model, "module") else model).state_dict()
        with tempfile.TemporaryDirectory() as td:
            torch.save({"model": state, "classes": CLASSES},
                       os.path.join(td, "model.pt"))
            ray.train.report(metrics, checkpoint=Checkpoint.from_directory(td))
