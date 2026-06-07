"""逐幀事故二分類 — test 集評估(PR-AUC / F1 / 混淆矩陣 + 門檻掃描 + 多數基線)。

  docker compose exec ray-head python scripts/eval_accident_cnn.py
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.modeling.accident_cnn import build_model
from src.train.accident_cnn.trainer import (IMAGENET_MEAN, IMAGENET_STD,
                                            average_precision)

SAVE_PATH = "/workspace/ray_results/accident_cnn_final/accident_cnn.pt"


def _scores(model, X, device, batch=256):
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float().permute(0, 3, 1, 2) / 255.0
        xb = (xb - IMAGENET_MEAN) / IMAGENET_STD
        with torch.no_grad():
            out.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SAVE_PATH)
    ap.add_argument("--data-dir", default="/workspace/datasets/accident_cnn_seq")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    model = build_model(ck["config"]).to(device).eval()
    model.load_state_dict(ck["state_dict"])

    d = np.load(Path(args.data_dir) / "test.npz")
    X, y = d["X"], d["y"].astype(np.int64)
    s = _scores(model, X, device)

    print(f"\n=== test 集逐幀事故二分類({len(y)} 張，事故 {int(y.sum())})===\n")
    print(f"[PR-AUC] {average_precision(y, s):.4f}"
          f"  (隨機基線 = 正樣本率 {y.mean():.4f})")

    # 門檻掃描，挑最佳 F1
    best = (0, 0.5, 0, 0)
    for thr in np.linspace(0.1, 0.9, 33):
        pred = (s >= thr).astype(np.int64)
        tp = int(((pred == 1) & (y == 1)).sum())
        rec = tp / max(y.sum(), 1)
        prec = tp / max(pred.sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best[0]:
            best = (f1, thr, prec, rec)
    f1, thr, prec, rec = best
    print(f"[最佳門檻] thr={thr:.2f}  F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}")

    pred = (s >= thr).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    print(f"[混淆矩陣 @thr] TP={tp} FP={fp} FN={fn} TN={tn}"
          f"  accuracy={(tp + tn) / len(y):.3f}")

    # 多數基線(全猜「正常」)
    maj = 1 - int(y.mean() >= 0.5)
    print(f"[多數基線] 全猜「{'事故' if maj else '正常'}」 "
          f"accuracy={max(y.mean(), 1 - y.mean()):.3f}  F1(事故)=0.000")


if __name__ == "__main__":
    main()
