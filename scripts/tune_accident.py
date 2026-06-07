"""Stage 2 入口：事故時序模型 Ray Tune(ASHA)— CPU 三節點平行搜參。

模型小,每組試驗只吃 CPU → head + 2 worker 同時跑多組,worker 終於派上用場。

  docker compose exec ray-head python scripts/tune_accident.py --samples 24
"""

import argparse

import ray
import torch
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

from src.train.accident.trainer import DATA_DIR, fit


def trainable(config):
    fit(config, torch.device("cpu"), report=lambda m: train.report(m))


def main():
    ap = argparse.ArgumentParser(description="事故時序模型 Ray Tune (CPU 平行)")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--samples", type=int, default=24, help="超參組合數")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--cpus-per-trial", type=int, default=2)
    args = ap.parse_args()

    ray.init(address="auto", ignore_reinit_error=True)
    space = {
        "data_dir": args.data_dir,
        "epochs": args.epochs,
        "kind": tune.choice(["lstm", "gru", "cnn"]),
        "hidden": tune.choice([32, 64, 128]),
        "layers": tune.choice([1, 2]),
        "dropout": tune.uniform(0.0, 0.4),
        "lr": tune.loguniform(1e-4, 5e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "batch": tune.choice([128, 256, 512]),
    }
    scheduler = ASHAScheduler(metric="ap", mode="max", max_t=args.epochs,
                              grace_period=min(5, args.epochs))
    tuner = tune.Tuner(
        tune.with_resources(trainable, {"cpu": args.cpus_per_trial}),
        param_space=space,
        tune_config=tune.TuneConfig(num_samples=args.samples,
                                    scheduler=scheduler),
        run_config=train.RunConfig(name="accident_tune",
                                   storage_path="/workspace/ray_results"),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="ap", mode="max")
    print("\n=== 最佳超參 ===")
    print(best.config)
    print("=== 最佳驗證指標 ===")
    print({k: round(v, 4) for k, v in best.metrics.items()
           if k in ("ap", "recall", "precision", "f1", "val_loss")})


if __name__ == "__main__":
    main()
