"""車禍分類超參搜尋（Ray Tune + Ray Train）。

動機：Accident 對超參極敏感——實測 lr=1e-3 會讓加入合成台灣資料後的訓練崩潰
（全判非事故、train_loss 衝到 ~3），lr=1e-4 才穩定收斂。與其手動試 lr，讓 ASHA
在多組 lr / batch_size / weight_decay 間分時搜尋、提早砍掉崩潰 trial，自動挑最佳。

與 freeway 的差異：freeway 用 ultralytics 內建 model.tune(use_ray=True)；accident
是自刻的 Ray Train TorchTrainer，故用原生 Ray Tune——Tuner 包 TorchTrainer，
param_space 覆寫 train_loop_config，沿用 trainer 內建的 RunConfig/CheckpointConfig。

  # 先小規模驗證（2 trial、各 5 epoch）
  docker compose exec ray-head python scripts/tune_accident.py --iterations 2 --epochs 5
  # 正式搜尋
  docker compose exec ray-head python scripts/tune_accident.py --iterations 12 --epochs 30

註：in-loop 指標是 val_acc（土耳其 held-out）。它足以讓 ASHA 避開崩潰區
（崩潰模型 val_acc≈0.5），但台灣域鑑別力仍須事後用注入驗證／diag 腳本確認。
單 GPU 下各 trial 依序跑（ScalingConfig 每 trial 佔 1 GPU）。
"""

# BLAS 執行緒上限要在 import numpy/torch/cv2 前設好。
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse

from src.core.cluster import init_ray
from src.train.accident.trainer import build_trainer


def main():
    ap = argparse.ArgumentParser(description="車禍分類超參搜尋（Ray Tune）")
    ap.add_argument("--epochs", type=int, default=30, help="每個 trial 的 epoch 上限")
    ap.add_argument("--iterations", type=int, default=12, help="trial 數（超參組合）")
    ap.add_argument("--grace-period", type=int, default=8,
                    help="ASHA 最少跑幾 epoch 才可被砍（須 ≤ epochs）")
    args = ap.parse_args()

    from ray import tune
    from ray.tune import TuneConfig, Tuner
    from ray.tune.schedulers import ASHAScheduler

    init_ray()

    # 沿用 accident 的 Ray Train 骨架，輸出到獨立實驗 accident_tune（不蓋正式模型）
    trainer = build_trainer(epochs=args.epochs, experiment_name="accident_tune")

    # 搜尋空間：lr 跨數量級（涵蓋崩潰的 1e-3 與穩定的 1e-4）→ log 尺度
    space = {"train_loop_config": {
        "epochs": args.epochs,
        "lr": tune.loguniform(1e-5, 3e-3),
        "batch_size": tune.choice([16, 32, 64]),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
    }}

    grace = min(args.grace_period, args.epochs)   # ASHA 要求 grace_period ≤ max_t
    tuner = Tuner(
        trainer,
        param_space=space,
        tune_config=TuneConfig(
            scheduler=ASHAScheduler(max_t=args.epochs, grace_period=grace,
                                    reduction_factor=2),
            num_samples=args.iterations,
            metric="val_acc",
            mode="max",
        ),
    )
    results = tuner.fit()

    best = results.get_best_result(metric="val_acc", mode="max")
    print("=== Ray Tune（accident）完成 ===")
    print("最佳超參：", best.config.get("train_loop_config"))
    print("最佳 val_acc：", best.metrics.get("val_acc"))
    print("結果在 /workspace/ray_results/accident_tune（各 trial + 最佳）")


if __name__ == "__main__":
    main()
