"""Stage 2/3 — 逐幀事故二分類的訓練核心(Ray Tune 與 Ray Train 共用)。

類別不平衡(事故 6k : 正常 15k)→ BCEWithLogitsLoss(pos_weight)；
選優看 val F1 / PR-AUC(對齊部署要的精準-召回平衡)，不看 accuracy。
影像存 uint8，訓練時用 ImageNet mean/std 標準化(backbone 為 ImageNet 預訓練)。
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.modeling.accident_cnn import build_model

DATA_DIR = "/workspace/datasets/accident_cnn_seq"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class _FrameDS(Dataset):
    """uint8 NHWC → 正規化 CHW float；train 時做翻轉+亮度抖動。"""

    def __init__(self, X, y, train=False, aug=0.2):
        self.X, self.y, self.train, self.aug = X, y, train, aug

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = torch.from_numpy(self.X[i]).float().permute(2, 0, 1) / 255.0
        if self.train:
            if torch.rand(1).item() < 0.5:                 # 水平翻轉
                img = torch.flip(img, dims=[2])
            if self.aug > 0:                               # 亮度抖動
                f = 1.0 + (torch.rand(1).item() * 2 - 1) * self.aug
                img = (img * f).clamp(0, 1)
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return img, float(self.y[i])


def load_arrays(data_dir):
    """driver 端載一次 train/val 原始陣列（給 ray.put / with_parameters 共享）。"""
    tr = np.load(Path(data_dir) / "train.npz")
    va = np.load(Path(data_dir) / "val.npz")
    return {"Xtr": tr["X"], "ytr": tr["y"], "Xva": va["X"], "yva": va["y"]}


def make_loaders_from_arrays(arr, batch, aug=0.2):
    """用已在記憶體/物件存儲的陣列建 loader（零拷貝共享，多 trial 不複製）。"""
    ytr = arr["ytr"]
    pos = float(ytr.sum())
    pos_weight = (len(ytr) - pos) / max(pos, 1.0)          # neg/pos
    # num_workers=0：影像已在 RAM、轉換很輕，不 fork 子行程避免複製大陣列爆記憶體
    tl = DataLoader(_FrameDS(arr["Xtr"], ytr, train=True, aug=aug),
                    batch_size=batch, shuffle=True, num_workers=0,
                    pin_memory=True, drop_last=True)
    vl = DataLoader(_FrameDS(arr["Xva"], arr["yva"]), batch_size=256,
                    num_workers=0, pin_memory=True)
    return tl, vl, pos_weight


def make_loaders(data_dir, batch, aug=0.2):
    return make_loaders_from_arrays(load_arrays(data_dir), batch, aug)


def average_precision(y_true, y_score):
    if y_true.sum() == 0:
        return 0.0
    order = np.argsort(-y_score)
    yt = y_true[order]
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    recall = tp / yt.sum()
    precision = tp / (tp + fp)
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
    acc = float((pred == t).mean()) if len(t) else 0.0
    return {"val_loss": loss_sum / max(n, 1), "ap": average_precision(t, s),
            "recall": recall, "precision": precision, "f1": f1, "acc": acc}


def fit(config, device, report, save_path=None, arrays=None):
    """完整訓練；每 epoch 呼叫 report(metrics)。回傳最佳 val F1。

    arrays 不為 None 時，直接用共享的物件存儲陣列建 loader（Tune 多 trial 零拷貝）；
    否則退回從 config['data_dir'] 的 npz 載入（單次 Train 用）。
    """
    batch = int(config.get("batch", 64))
    aug = float(config.get("aug", 0.2))
    if arrays is not None:
        tl, vl, pos_weight = make_loaders_from_arrays(arrays, batch, aug)
    else:
        tl, vl, pos_weight = make_loaders(config["data_dir"], batch, aug)
    model = build_model(config).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(config["lr"]),
                            weight_decay=float(config.get("weight_decay", 1e-4)))
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device))

    best_f1, best_state = -1.0, None
    for ep in range(int(config["epochs"])):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        m = evaluate(model, vl, device, loss_fn)
        if m["f1"] >= best_f1:                  # 用 F1 挑最佳(不看 accuracy)
            best_f1 = m["f1"]
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        report(m)

    if save_path and best_state is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "config": dict(config)},
                   save_path)
    return best_f1
