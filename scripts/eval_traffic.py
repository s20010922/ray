"""車流偵測評估與可視化（用自訓 checkpoint）。

  # A 肉眼看框 + B 算 mAP@0.5（全 val）
  docker compose exec ray-head python scripts/eval_traffic.py

  # 只抽 8 張畫框、val 只取前 20 序列加速
  docker compose exec ray-head python scripts/eval_traffic.py --vis 8 --limit 20

輸出：
  - 畫框圖存到 /workspace/ray_results/traffic_eval/*.jpg
  - mAP@0.5 印在終端
"""

import argparse
import os
from pathlib import Path

import cv2

from src.data.traffic.sources import list_detrac_records
from src.eval.traffic import compute_map50
from src.infer.traffic import detect, draw, find_best_checkpoint, load_detector


def main():
    ap = argparse.ArgumentParser(description="車流偵測評估（mAP@0.5 + 可視化）")
    ap.add_argument("--checkpoint", default=None,
                    help="model.pt 路徑；省略則自動找最新訓練的最後 checkpoint")
    ap.add_argument("--detrac-root", default="/data/detrac")
    ap.add_argument("--frame-stride", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None,
                    help="val 只取前 N 序列（加速）")
    ap.add_argument("--vis", type=int, default=8, help="畫框輸出張數")
    ap.add_argument("--conf-map", type=float, default=0.001,
                    help="算 mAP 用的低門檻（掃完整 PR 曲線）")
    ap.add_argument("--conf-vis", type=float, default=0.4,
                    help="畫框用的部署門檻")
    ap.add_argument("--out", default="/workspace/ray_results/traffic_eval")
    args = ap.parse_args()

    ckpt = args.checkpoint or find_best_checkpoint()
    print(f"[載入] checkpoint: {ckpt}")
    model = load_detector(ckpt)

    _, val_records = list_detrac_records(
        detrac_root=args.detrac_root, frame_stride=args.frame_stride,
        limit_sequences=args.limit)
    print(f"[資料] val 幀數: {len(val_records)}")

    # ---- B. mAP@0.5（全 val，低 conf 掃 PR 曲線）----
    preds_per_img, gts_per_img = [], []
    for rec in val_records:
        img = cv2.imread(rec["image_path"])
        if img is None:
            continue
        boxes, scores = detect(model, img, conf=args.conf_map)
        preds_per_img.append((boxes, scores))
        gts_per_img.append(rec["boxes_xyxy"])

    m = compute_map50(preds_per_img, gts_per_img)
    print("\n=== mAP@0.5 評估 ===")
    print(f"  mAP@0.5   : {m['map50']:.4f}")
    print(f"  precision : {m['precision']:.4f}")
    print(f"  recall    : {m['recall']:.4f}")
    print(f"  GT 框數   : {m['n_gt']}   預測框數: {m['n_pred']}")

    # ---- A. 可視化（部署門檻 0.4，肉眼看框）----
    os.makedirs(args.out, exist_ok=True)
    step = max(1, len(val_records) // max(1, args.vis))
    saved = 0
    for rec in val_records[::step][:args.vis]:
        img = cv2.imread(rec["image_path"])
        if img is None:
            continue
        boxes, scores = detect(model, img, conf=args.conf_vis)
        vis = draw(img, boxes, scores)
        name = f"{Path(rec['image_path']).parent.name}_{Path(rec['image_path']).name}"
        cv2.imwrite(os.path.join(args.out, name), vis)
        saved += 1
    print(f"\n=== 可視化 ===\n  已存 {saved} 張畫框圖到 {args.out}")


if __name__ == "__main__":
    main()
