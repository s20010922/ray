"""事故時序模型評估：窗級 AP + 事件級(每支影片)偵測率/誤報率。

窗級 AP 在極不平衡下數字本來就低；對「事故偵測」真正有意義的是**事件級**：
  - 影片偵測率(recall)：肇事片中，肇事區段最高分 ≥ 門檻 → 視為偵測到。
  - 背景誤報率(FPR)：所有非事故窗中，分數 ≥ 門檻的比例(越低越好)。
掃不同門檻，給出可讀的權衡;並報正/負樣本平均分數的分離度。

  docker compose exec ray-head python scripts/eval_accident.py
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.modeling.accident import build_model
from src.train.accident.trainer import average_precision

CKPT = "/workspace/ray_results/accident_final/accident_seq.pt"
DATA = "/workspace/datasets/accident_seq"


def main():
    ap = argparse.ArgumentParser(description="事故模型評估(窗級+事件級)")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mean, std = ck["scaler_mean"], ck["scaler_std"]
    model = build_model(ck["config"], in_features=ck["in_features"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    d = np.load(Path(args.data_dir) / f"{args.split}.npz")
    X = (d["X"].astype(np.float32) - mean) / std
    y = d["y"].astype(np.int64)
    clip = d["clip"]
    with torch.no_grad():
        s = torch.sigmoid(model(torch.from_numpy(X))).numpy()

    base = y.mean()
    win_ap = average_precision(y.astype(np.float32), s)
    print(f"=== {args.split} 集 ===")
    print(f"樣本 {len(y)} | 正樣本 {int(y.sum())} ({100*base:.2f}%)")
    print(f"\n[窗級] AP = {win_ap:.4f}  (隨機 {base:.4f} 的 {win_ap/max(base,1e-9):.1f} 倍)")
    print(f"[分離度] 正樣本平均分 {s[y==1].mean():.3f} vs 負樣本 {s[y==0].mean():.3f}")

    # 事件級：以 clip 為單位
    acc_clips = sorted({int(c) for c in clip[y == 1]})      # 肇事片
    print(f"\n[事件級] 測試集肇事片 {len(acc_clips)} 支")
    print(f"{'門檻':>6} {'影片偵測率':>10} {'背景誤報率':>10} {'平均誤報/片':>10}")
    neg_mask = y == 0
    for th in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        detected = 0
        for c in acc_clips:
            m = (clip == c) & (y == 1)
            if s[m].max() >= th:
                detected += 1
        recall = detected / max(len(acc_clips), 1)
        bg_fpr = (s[neg_mask] >= th).mean()
        # 每支片平均有幾個背景窗誤觸發
        fp_per_clip = (s[neg_mask] >= th).sum() / max(len(set(clip)), 1)
        print(f"{th:>6} {recall:>9.1%} {bg_fpr:>10.3f} {fp_per_clip:>10.1f}")


if __name__ == "__main__":
    main()
