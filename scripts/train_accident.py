"""車禍分類訓練進入點。

  docker compose exec ray-head python scripts/train_accident.py --epochs 20

骨架驗證可先用小 epoch：--epochs 2
"""

# BLAS 執行緒上限要在 import numpy/torch/cv2 前設好，避免多 worker 超訂 CPU。
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse

from src.core.cluster import init_ray
from src.train.accident.trainer import build_trainer


def main():
    ap = argparse.ArgumentParser(description="車禍分類訓練（Ray Train）")
    ap.add_argument("--epochs", type=int, default=20)
    # 預設為 tune_accident 搜出的最佳超參（val_acc 0.790）
    ap.add_argument("--lr", type=float, default=3.33e-4,
                    help="Ray Tune 最佳值；lr≥1e-3 會讓加入合成台灣資料後崩潰")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--weight-decay", type=float, default=3.8e-4,
                    help="Ray Tune 最佳值")
    args = ap.parse_args()

    init_ray()
    trainer = build_trainer(epochs=args.epochs, lr=args.lr,
                            batch_size=args.batch_size,
                            weight_decay=args.weight_decay)
    result = trainer.fit()

    print("=== 訓練完成 ===")
    print("metrics:", result.metrics)
    print("checkpoint:", result.checkpoint)


if __name__ == "__main__":
    main()
