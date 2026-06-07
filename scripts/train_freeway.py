"""Stage 3 — Ray Train：freeway 正式訓練（yolo11s，進入點）。

用 Ray Train（TorchTrainer）編排 ultralytics 訓練；資料吃 Stage 1 產出的
freeway_prepared/dataset.yaml。預設超參可由 Stage 2（tune_freeway）最佳值覆寫。

  docker compose exec ray-head python scripts/train_freeway.py --epochs 100
"""

import argparse

from src.core.cluster import init_ray
from src.train.freeway.trainer import run_train


def main():
    ap = argparse.ArgumentParser(description="Stage 3 Ray Train（yolo11s）")
    ap.add_argument("--data",
                    default="/workspace/datasets/freeway_prepared/dataset.yaml")
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--name", default="freeway_final")
    # ↓ 預設保守值；跑完 tune_freeway.py 後以最佳超參覆寫
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--lrf", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--hsv-v", type=float, default=0.03)
    ap.add_argument("--scale", type=float, default=0.1)
    ap.add_argument("--fliplr", type=float, default=0.5)
    ap.add_argument("--mosaic", type=float, default=0.1)
    args = ap.parse_args()

    init_ray()
    hp = {
        "lr0": args.lr0, "lrf": args.lrf, "weight_decay": args.weight_decay,
        "hsv_v": args.hsv_v, "scale": args.scale,
        "fliplr": args.fliplr, "mosaic": args.mosaic,
    }
    run_train(args.data, weights=args.weights, imgsz=args.imgsz,
              epochs=args.epochs, batch=args.batch, patience=args.patience,
              hp=hp, name=args.name)


if __name__ == "__main__":
    main()
