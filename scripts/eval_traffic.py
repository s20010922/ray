"""車流偵測評估：Traffic base 在 DETRAC held-out test 序列上跑 mAP@0.5。

Traffic base（Ray Train 自訓 checkpoint）依「序列」切 train/val/test，
本腳本在訓練全程未見過的 test 序列上評估，作為可信的 held-out 指標。

  # 自動找最新 traffic checkpoint，跑 DETRAC test
  docker compose exec ray-head python scripts/eval_traffic.py

  # 指定 checkpoint
  docker compose exec ray-head python scripts/eval_traffic.py \\
      --checkpoint /workspace/ray_results/traffic/TorchTrainer_xxx/checkpoint_000029/model.pt

  # 同時輸出畫框圖（--vis N 張）
  docker compose exec ray-head python scripts/eval_traffic.py --vis 8
"""

import argparse
import os
from pathlib import Path

import cv2

from src.data.traffic.split import list_detrac_splits
from src.eval.traffic import compute_map50
from src.infer.traffic import detect, draw, find_best_checkpoint, load_detector


def main():
    ap = argparse.ArgumentParser(description="車流偵測評估（DETRAC test 序列）")
    ap.add_argument("--checkpoint", default=None,
                    help="model.pt 路徑；省略則自動找最新 traffic 訓練的 checkpoint")
    ap.add_argument("--detrac-root", default="/data/detrac",
                    help="DETRAC 掛載根")
    ap.add_argument("--frame-stride", type=int, default=10,
                    help="抽幀間隔（需與訓練一致，確保 test 序列定義相同）")
    ap.add_argument("--vis", type=int, default=0, help="輸出畫框圖張數（0=不輸出）")
    ap.add_argument("--conf-map", type=float, default=0.001,
                    help="算 mAP 用的低門檻（掃完整 PR 曲線）")
    ap.add_argument("--conf-vis", type=float, default=0.4,
                    help="畫框用的門檻")
    ap.add_argument("--out", default="/workspace/ray_results/traffic_eval")
    args = ap.parse_args()

    ckpt = args.checkpoint or find_best_checkpoint()
    print(f"[載入] checkpoint: {ckpt}")
    model = load_detector(ckpt)

    splits = list_detrac_splits(detrac_root=args.detrac_root,
                                frame_stride=args.frame_stride)
    records = splits["test"]
    if not records:
        print(f"[錯誤] DETRAC test 序列是空的：{args.detrac_root}")
        return
    print(f"[資料] test set: {len(records)} 幀（held-out 序列）")

    preds_per_img, gts_per_img = [], []
    for rec in records:
        img = cv2.imread(rec["image_path"])
        if img is None:
            continue
        boxes, scores = detect(model, img, conf=args.conf_map)
        preds_per_img.append((boxes, scores))
        gts_per_img.append(rec["boxes_xyxy"])

    m = compute_map50(preds_per_img, gts_per_img)
    print("\n=== 車流偵測評估（DETRAC Test 序列）===")
    print(f"  mAP@0.5   : {m['map50']:.4f}  ({m['map50']*100:.1f}%)")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  GT 框數   : {m['n_gt']}   預測框數: {m['n_pred']}")

    if args.vis > 0:
        os.makedirs(args.out, exist_ok=True)
        step = max(1, len(records) // args.vis)
        saved = 0
        for rec in records[::step][:args.vis]:
            img = cv2.imread(rec["image_path"])
            if img is None:
                continue
            boxes, scores = detect(model, img, conf=args.conf_vis)
            vis = draw(img, boxes, scores)
            name = Path(rec["image_path"]).name
            cv2.imwrite(os.path.join(args.out, name), vis)
            saved += 1
        print(f"\n  畫框圖 → {args.out}（{saved} 張）")


if __name__ == "__main__":
    main()
