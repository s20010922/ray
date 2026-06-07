"""逐幀事故二分類 CNN：torchvision backbone(ImageNet 預訓練)→ 單 logit。

容器無網路 → backbone 一律 weights=None，從掛載的本地權重 load_state_dict。
輸出單一 logit，配 BCEWithLogitsLoss(pos_weight) 處理類別不平衡(事故少、正常多)。
"""

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tvm

WEIGHTS_DIR = "/workspace/datasets/weights"
LOCAL_WEIGHTS = {
    "mobilenet_v2": "mobilenet_v2-b0353104.pth",
    "resnet18": "resnet18-f37072fd.pth",
}


def _load_pretrained(model, name):
    """從掛載目錄載入 ImageNet 權重(離線)。檔案不在就略過(隨機初始化)。"""
    p = Path(WEIGHTS_DIR) / LOCAL_WEIGHTS[name]
    if p.exists():
        model.load_state_dict(torch.load(p, map_location="cpu"))
    return model


def build_model(config):
    """依 config 建模。config: backbone, freeze, dropout。回傳 nn.Module(輸出 logit B,)。"""
    name = config.get("backbone", "mobilenet_v2")
    freeze = bool(config.get("freeze", True))
    dropout = float(config.get("dropout", 0.2))

    if name == "mobilenet_v2":
        net = tvm.mobilenet_v2(weights=None)
        _load_pretrained(net, name)
        in_feat = net.classifier[-1].in_features
        backbone = net.features
        pool = nn.AdaptiveAvgPool2d(1)
    elif name == "resnet18":
        net = tvm.resnet18(weights=None)
        _load_pretrained(net, name)
        in_feat = net.fc.in_features
        net.fc = nn.Identity()
        backbone = net
        pool = None
    else:
        raise ValueError(f"未知 backbone: {name}")

    if freeze:
        for prm in backbone.parameters():
            prm.requires_grad = False

    return _Classifier(backbone, pool, in_feat, dropout, name)


class _Classifier(nn.Module):
    def __init__(self, backbone, pool, in_feat, dropout, name):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.name = name
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feat, 1),
        )

    def forward(self, x):
        f = self.backbone(x)
        if self.pool is not None:          # mobilenet：features 出 4D，需池化
            f = self.pool(f).flatten(1)
        return self.head(f).squeeze(-1)    # (B,)
