"""Stage 2 入口：逐幀事故二分類 Ray Tune(ASHA)。

CNN 吃 GPU，單卡用 fractional GPU 共享同時跑 2 組試驗(num_gpus=0.5)。
選優指標 = val F1(類別不平衡，不看 accuracy)。

  docker compose exec ray-head python scripts/tune_accident_cnn.py --samples 12
"""

import argparse

import ray
import torch
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

from src.train.accident_cnn.trainer import DATA_DIR, fit, load_arrays


def trainable(config, data=None):
    # data 由 tune.with_parameters 注入：driver 端 ray.put 一次、各 trial 零拷貝共享
    fit(config, torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        report=lambda m: train.report(m), arrays=data)


def main():
    ap = argparse.ArgumentParser(description="逐幀事故二分類 Ray Tune")
    ap.add_argument("--data-dir",
                    default="/workspace/datasets/accident_tad_seq")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--gpu-per-trial", type=float, default=0.5,
                    help="0.5=單卡跑2個trial(資料已物件存儲共享，記憶體安全)")
    ap.add_argument("--run-name", default="accident_cnn_tune")
    args = ap.parse_args()

    ray.init(address="auto", ignore_reinit_error=True)
    # driver 端載一次資料 → ray.put 進 object store；with_parameters 讓各 trial 零拷貝共享
    arrays = load_arrays(args.data_dir)
    trainable_shared = tune.with_parameters(trainable, data=arrays)
    space = {
        "data_dir": args.data_dir,
        "epochs": args.epochs,
        "backbone": tune.choice(["mobilenet_v2", "resnet18"]),
        "freeze": tune.choice([True, False]),
        "dropout": tune.uniform(0.0, 0.4),
        "aug": tune.uniform(0.1, 0.3),
        "lr": tune.loguniform(1e-4, 3e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "batch": tune.choice([16, 32]),     # 限 ≤32：batch 64 會吃滿 8G VRAM 卡桌面
    }
    scheduler = ASHAScheduler(metric="f1", mode="max", max_t=args.epochs,
                              grace_period=min(3, args.epochs))
    tuner = tune.Tuner(
        tune.with_resources(trainable_shared, {"gpu": args.gpu_per_trial}),
        param_space=space,
        tune_config=tune.TuneConfig(num_samples=args.samples,
                                    scheduler=scheduler),
        run_config=train.RunConfig(name=args.run_name,
                                   storage_path="/workspace/ray_results"),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="f1", mode="max")
    print("\n=== 最佳超參 ===")
    print(best.config)
    print("=== 最佳驗證指標 ===")
    print({k: round(v, 4) for k, v in best.metrics.items()
           if k in ("ap", "recall", "precision", "f1", "acc", "val_loss")})


if __name__ == "__main__":
    main()
