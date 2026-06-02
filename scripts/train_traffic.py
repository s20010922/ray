"""車流偵測訓練進入點。

  # 先驗證骨架（少序列、小 epoch）
  docker compose exec ray-head python scripts/train_traffic.py --limit 5 --epochs 2

  # 正式訓練（全 100 序列、抽幀 10 ≈ 1.4 萬張）
  docker compose exec ray-head python scripts/train_traffic.py --epochs 30
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse

from src.core.cluster import init_ray
from src.train.traffic.trainer import build_trainer


def main():
    ap = argparse.ArgumentParser(description="車流偵測訓練（Ray Train）")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--frame-stride", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None,
                    help="只用前 N 序列（先驗證用）")
    args = ap.parse_args()

    init_ray()
    trainer = build_trainer(epochs=args.epochs, lr=args.lr,
                            batch_size=args.batch_size,
                            frame_stride=args.frame_stride,
                            limit_sequences=args.limit)
    result = trainer.fit()

    print("=== 訓練完成 ===")
    print("metrics:", result.metrics)
    print("checkpoint:", result.checkpoint)


if __name__ == "__main__":
    main()
