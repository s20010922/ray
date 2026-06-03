"""高公局車輛偵測 fine-tune（ultralytics 原生 train）。

知識蒸餾流程：COCO yolo11x 自動標 → 從官方 yolo11n.pt fine-tune 單類 Vehicle。
ultralytics train 內建 letterbox + imgsz 960（治小目標）、mAP、early-stopping、
資料增強、checkpoint，補齊自刻 Ray loop 缺的功能。

  # 先用現有 200 張驗證 pipeline
  docker compose exec ray-head python scripts/finetune_freeway.py --epochs 100

  # 之後擴大資料：先 prelabel --coco --sample 0 全量，再重跑本腳本
"""

import argparse


def main():
    ap = argparse.ArgumentParser(description="高公局車輛偵測 fine-tune")
    ap.add_argument("--src", default="/workspace/datasets/freeway_yolo",
                    help="COCO 自動標的來源（images+labels 平鋪）")
    ap.add_argument("--split-out", default="/workspace/datasets/freeway_det",
                    help="切好的 ultralytics 結構輸出處")
    ap.add_argument("--weights", default="yolo11n.pt",
                    help="起點權重（官方 COCO 預訓練）")
    ap.add_argument("--imgsz", type=int, default=960, help="輸入尺寸（治小目標）")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=20, help="early-stop 容忍 epoch")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    args = ap.parse_args()

    from ultralytics import YOLO

    from src.data.freeway.split import make_det_split

    yaml_path, n_tr, n_va = make_det_split(args.src, args.split_out, args.val_ratio)
    print(f"[切分] train {n_tr} / val {n_va} → {yaml_path}")

    model = YOLO(args.weights)
    model.train(
        data=yaml_path,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        single_cls=True,                 # 強制單類 Vehicle
        project="/workspace/ray_results",
        name="freeway_finetune",
        exist_ok=True,
    )
    print("=== fine-tune 完成 ===")
    print("best.pt 在 /workspace/ray_results/freeway_finetune/weights/best.pt")


if __name__ == "__main__":
    main()
