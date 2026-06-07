"""TAD 車禍偵測評估 — 幀級 + **影片級**(誠實主指標)。

幀級：每幀判事故(部署時即時用)。
影片級：把每支影片的幀分數聚合(取較高分位數)→ 判該片是否車禍。這對齊
        TAD 影片級標籤、也最能回答「能不能抓到事故影片」，是該放簡報的主數字。
        對照 notebook 的幀級隨機切分(洩漏)版 99%。

  docker compose exec ray-head python scripts/eval_accident_tad.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.modeling.accident_cnn import build_model
from src.train.accident_cnn.trainer import (IMAGENET_MEAN, IMAGENET_STD,
                                            average_precision)

SAVE_PATH = "/workspace/ray_results/accident_tad_final/accident_tad.pt"
DATA_DIR = "/workspace/datasets/accident_tad_seq"


def _scores(model, X, device, batch=256):
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float().permute(0, 3, 1, 2) / 255.0
        xb = (xb - IMAGENET_MEAN) / IMAGENET_STD
        with torch.no_grad():
            out.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(out)


def _best_f1(y, s):
    best = (0, 0.5, 0, 0)
    for thr in np.linspace(0.05, 0.95, 37):
        pred = (s >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        rec = tp / max(y.sum(), 1)
        prec = tp / max(pred.sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best[0]:
            best = (f1, thr, prec, rec)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SAVE_PATH)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--agg-pct", type=float, default=90,
                    help="影片級聚合用的分位數(取每片幀分數的高分位)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    model = build_model(ck["config"]).to(device).eval()
    model.load_state_dict(ck["state_dict"])

    d = np.load(Path(args.data_dir) / "test.npz")
    X, y, vid = d["X"], d["y"].astype(int), d["vid"]
    meta = json.loads((Path(args.data_dir) / "videos.json").read_text(encoding="utf-8"))
    s = _scores(model, X, device)

    # ── 幀級 ──
    print(f"\n=== TAD test 幀級({len(y)} 幀，事故 {int(y.sum())})===")
    print(f"[PR-AUC] {average_precision(y, s):.4f}  (隨機={y.mean():.4f})")
    f1, thr, prec, rec = _best_f1(y, s)
    print(f"[最佳F1] {f1:.3f} @thr{thr:.2f}  P={prec:.3f} R={rec:.3f}")

    # ── 影片級(主指標)──
    vids = sorted(set(vid.tolist()))
    vy = np.array([meta[str(v)]["label"] for v in vids])
    vs = np.array([np.percentile(s[vid == v], args.agg_pct) for v in vids])
    print(f"\n=== TAD test 影片級({len(vids)} 片，事故 {int(vy.sum())}，"
          f"聚合=每片第{args.agg_pct:.0f}百分位)===")
    print(f"[ROC-AUC] {_roc_auc(vy, vs):.4f}   [PR-AUC] {average_precision(vy, vs):.4f}")
    vf1, vthr, vp, vr = _best_f1(vy, vs)
    pred = (vs >= vthr).astype(int)
    tp = int(((pred == 1) & (vy == 1)).sum()); fp = int(((pred == 1) & (vy == 0)).sum())
    fn = int(((pred == 0) & (vy == 1)).sum()); tn = int(((pred == 0) & (vy == 0)).sum())
    print(f"[最佳F1] {vf1:.3f} @thr{vthr:.2f}  P={vp:.3f} R={vr:.3f}")
    print(f"[混淆矩陣] TP={tp} FP={fp} FN={fn} TN={tn}  acc={(tp + tn) / len(vids):.3f}")


def _roc_auc(y, s):
    if y.sum() == 0 or y.sum() == len(y):
        return 0.0
    order = np.argsort(-s)
    y = y[order]
    tps = np.cumsum(y); fps = np.cumsum(1 - y)
    tpr = tps / y.sum(); fpr = fps / (len(y) - y.sum())
    return float(np.trapz(tpr, fpr))


if __name__ == "__main__":
    main()
