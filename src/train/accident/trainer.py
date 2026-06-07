"""Stage 2/3 — 事故時序模型的訓練核心(Ray Tune 與 Ray Train 共用)。

資料是高度不平衡的(事故正樣本稀少)→ 用 BCEWithLogitsLoss 的 pos_weight 補償，
模型挑選看驗證集的 Average Precision(AP，對不平衡比 accuracy 有意義)。

模型小,Tune 走 CPU、三節點(head+2 worker)平行搜參;最終 Train 用 TorchTrainer。
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.modeling.accident import build_model

DATA_DIR = "/workspace/datasets/accident_seq"


# ---------- 資料 ----------
def _load(data_dir, split, scaler):
    d = np.load(Path(data_dir) / f"{split}.npz")
    X = d["X"].astype(np.float32)
    if len(X):
        X = (X - scaler["mean"]) / scaler["std"]
    return (torch.from_numpy(X),
            torch.from_numpy(d["y"].astype(np.float32)))


def make_loaders(data_dir, batch):
    sc = np.load(Path(data_dir) / "scaler.npz")
    scaler = {"mean": sc["mean"], "std": sc["std"]}
    Xtr, ytr = _load(data_dir, "train", scaler)
    Xva, yva = _load(data_dir, "val", scaler)
    pos = float(ytr.sum())
    pos_weight = (len(ytr) - pos) / max(pos, 1.0)      # neg/pos
    tl = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True)
    vl = DataLoader(TensorDataset(Xva, yva), batch_size=512)
    return tl, vl, pos_weight, Xtr.shape[-1], scaler


# ---------- 指標 ----------
def average_precision(y_true, y_score):
    """AP（不依賴 sklearn）。y_* 為 1D numpy。"""
    if y_true.sum() == 0:
        return 0.0
    order = np.argsort(-y_score)
    yt = y_true[order]
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    recall = tp / yt.sum()
    precision = tp / (tp + fp)
    # 以 recall 增量積分
    ap, prev_r = 0.0, 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_r)
        prev_r = r
    return float(ap)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    scores, trues, loss_sum, n = [], [], 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logit = model(xb)
        loss_sum += loss_fn(logit, yb).item() * len(yb)
        n += len(yb)
        scores.append(torch.sigmoid(logit).cpu().numpy())
        trues.append(yb.cpu().numpy())
    s = np.concatenate(scores) if scores else np.zeros(0)
    t = np.concatenate(trues) if trues else np.zeros(0)
    pred = (s >= 0.5).astype(np.float32)
    tp = float(((pred == 1) & (t == 1)).sum())
    recall = tp / max(t.sum(), 1.0)
    precision = tp / max(pred.sum(), 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"val_loss": loss_sum / max(n, 1), "ap": average_precision(t, s),
            "recall": recall, "precision": precision, "f1": f1}


# ---------- 共用訓練迴圈 ----------
def fit(config, device, report, save_path=None):
    """跑完整訓練;每 epoch 呼叫 report(metrics)。回傳最佳 AP。"""
    tl, vl, pos_weight, in_feat, scaler = make_loaders(
        config["data_dir"], int(config.get("batch", 256)))
    model = build_model(config, in_features=in_feat).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]),
                            weight_decay=float(config.get("weight_decay", 1e-4)))
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device))

    best_ap, best_state = 0.0, None
    for ep in range(int(config["epochs"])):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        m = evaluate(model, vl, device, loss_fn)
        if m["ap"] >= best_ap:
            best_ap = m["ap"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        report(m)

    if save_path and best_state is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "config": dict(config),
                    "in_features": in_feat,
                    "scaler_mean": scaler["mean"], "scaler_std": scaler["std"]},
                   save_path)
    return best_ap
