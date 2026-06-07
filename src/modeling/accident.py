"""事故時序模型：吃軌跡運動特徵序列 (T, F)，輸出事故機率。

不吃像素，只吃抽象運動數字(速度/加速度/航向/車距…)，故與偵測器/domain 解耦：
UCF/AccidentBench 訓練、高公局部署共用同一個模型。輕量(LSTM/GRU/1D-CNN)，
小到 CPU 就能快速訓練 → Ray Tune 可在三節點 CPU 平行搜參。
"""

import torch
import torch.nn as nn


class TemporalAccidentNet(nn.Module):
    """二元時序分類器。kind: 'lstm' | 'gru' | 'cnn'。forward 回傳 logits (B,)。"""

    def __init__(self, in_features: int = 10, kind: str = "lstm",
                 hidden: int = 64, layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.kind = kind
        if kind in ("lstm", "gru"):
            rnn = nn.LSTM if kind == "lstm" else nn.GRU
            self.encoder = rnn(in_features, hidden, layers, batch_first=True,
                               dropout=dropout if layers > 1 else 0.0)
            feat = hidden
        elif kind == "cnn":
            self.encoder = nn.Sequential(
                nn.Conv1d(in_features, hidden, 3, padding=1), nn.ReLU(),
                nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(),
                nn.AdaptiveMaxPool1d(1))
            feat = hidden
        else:
            raise ValueError(f"未知模型類型: {kind}")
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat, 1))

    def forward(self, x):                      # x: (B, T, F)
        if self.kind in ("lstm", "gru"):
            out, _ = self.encoder(x)
            h = out[:, -1]                     # 取最後一步
        else:
            h = self.encoder(x.transpose(1, 2)).squeeze(-1)   # (B, hidden)
        return self.head(h).squeeze(-1)        # logits (B,)


def build_model(config: dict, in_features: int = 10) -> TemporalAccidentNet:
    """從超參 dict 建模型(供 Tune/Train 共用)。"""
    return TemporalAccidentNet(
        in_features=in_features,
        kind=config.get("kind", "lstm"),
        hidden=int(config.get("hidden", 64)),
        layers=int(config.get("layers", 1)),
        dropout=float(config.get("dropout", 0.2)))
