"""車流偵測訓練迴圈（在 Ray Train worker 上跑）。

套用 accident 驗證過的同一套 Ray Train 骨架，但偵測比分類硬兩塊：
  1. v8DetectionLoss：要從 YOLO11 偵測模型取出 loss，接我們的 batch
  2. batch 格式轉換：pipeline 的 pad 過的 (B,MAX,4)/(B,MAX) → ultralytics 要的
     攤平格式 batch_idx / cls / bboxes（去 padding、標記每個 box 屬於哪張圖）

單類別（Vehicle）：yolo11n.pt 是 COCO 80 類，用 yolo11n.yaml(nc=1) 重建模型、
載入相容的預訓練權重（backbone/neck），偵測頭隨機初始化重學。
"""

import os
import tempfile

import torch

import ray.train
from ray.train import Checkpoint

from src.modeling.traffic import CLASSES


def _build_detection_model(nc: int, weights: str) -> torch.nn.Module:
    """以 nc 類重建 YOLO11n 偵測模型，載入相容的預訓練權重。"""
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import DEFAULT_CFG

    model = DetectionModel("yolo11n.yaml", nc=nc, verbose=False)

    # 載入預訓練：只取 shape 相符的層（backbone/neck），偵測頭 cls 分支跳過
    pretrained = YOLO(weights).model.float().state_dict()
    own = model.state_dict()
    compat = {k: v for k, v in pretrained.items()
              if k in own and own[k].shape == v.shape}
    model.load_state_dict(compat, strict=False)

    model.nc = nc
    model.args = DEFAULT_CFG          # v8DetectionLoss 需要 box/cls/dfl gains
    return model


def _to_ultralytics_batch(images: torch.Tensor,
                          boxes_xywhn: torch.Tensor,
                          labels: torch.Tensor) -> dict:
    """pipeline 輸出 → ultralytics 偵測 batch 格式。

    pipeline: boxes_xywhn (B,MAX,4)、labels (B,MAX)，padding 為 box=0/label=-1。
    ultralytics: 攤平成 (N,*)，batch_idx 標記每個 box 屬於哪張圖。
    """
    device = images.device
    idx, cls, box = [], [], []
    for b in range(images.shape[0]):
        valid = labels[b] >= 0                       # 去掉 padding(-1)
        n = int(valid.sum())
        if n == 0:
            continue
        idx.append(torch.full((n,), b, device=device, dtype=torch.float32))
        cls.append(labels[b][valid].float().unsqueeze(1))   # (n,1)
        box.append(boxes_xywhn[b][valid])                   # (n,4) xywh normalized
    if box:
        return {"img": images,
                "batch_idx": torch.cat(idx),
                "cls": torch.cat(cls),
                "bboxes": torch.cat(box)}
    return {"img": images,
            "batch_idx": torch.zeros(0, device=device),
            "cls": torch.zeros(0, 1, device=device),
            "bboxes": torch.zeros(0, 4, device=device)}


def _run_epoch(model, shard, batch_size, device, optimizer=None):
    """跑一個 epoch。optimizer=None 表示 val（不更新）。回傳平均 total loss。"""
    train_mode = optimizer is not None
    # 注意：偵測的 loss 需要 training 模式的 forward（raw feature maps）；
    # eval 模式 forward 會回傳 NMS 後的推論結果，算不了 loss。val 也保持 train()。
    model.train()
    running, n = 0.0, 0
    torch.set_grad_enabled(train_mode)
    for batch in shard.iter_torch_batches(batch_size=batch_size, device=device):
        b = _to_ultralytics_batch(batch["image"], batch["boxes_xywhn"],
                                  batch["labels"])
        loss, _items = model.loss(b)
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        running += float(loss.detach())
        n += 1
    torch.set_grad_enabled(True)
    return running / max(1, n)


def train_loop_per_worker(config: dict) -> None:
    epochs = config["epochs"]
    lr = config["lr"]
    batch_size = config["batch_size"]

    device = ray.train.torch.get_device()

    model = _build_detection_model(len(CLASSES), config.get("weights", "yolo11n.pt"))
    # 單 worker：不需要 DDP，只搬到 device（DDP 會擋住 model.loss 自訂方法）
    model = ray.train.torch.prepare_model(model, parallel_strategy=None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_shard = ray.train.get_dataset_shard("train")
    val_shard = ray.train.get_dataset_shard("val")

    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_shard, batch_size, device, optimizer)
        val_loss = _run_epoch(model, val_shard, batch_size, device, None)

        metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        state = (model.module if hasattr(model, "module") else model).state_dict()
        with tempfile.TemporaryDirectory() as td:
            torch.save({"model": state, "classes": CLASSES, "nc": len(CLASSES)},
                       os.path.join(td, "model.pt"))
            ray.train.report(metrics, checkpoint=Checkpoint.from_directory(td))
