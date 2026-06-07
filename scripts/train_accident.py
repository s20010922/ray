"""Stage 3 入口：事故時序模型 Ray Train(TorchTrainer)正式訓練 + 存檔。

超參預設可被 Stage 2 搜得的最佳值覆寫(CLI)。輸出自帶 scaler 的 checkpoint，
部署端載入即可用。

  docker compose exec ray-head python scripts/train_accident.py \
      --kind lstm --hidden 64 --layers 1 --lr 1e-3 --epochs 60
"""

import argparse

import ray
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from src.train.accident.trainer import DATA_DIR, fit

SAVE_PATH = "/workspace/ray_results/accident_final/accident_seq.pt"


def _train_loop(config):
    import ray.train
    import torch
    device = (ray.train.torch.get_device() if config["use_gpu"]
              else torch.device("cpu"))
    best_ap = fit(config, device, report=lambda m: ray.train.report(m),
                  save_path=config["save_path"])
    print(f"[Ray Train] 最佳 val AP = {best_ap:.4f} → {config['save_path']}")


def main():
    ap = argparse.ArgumentParser(description="事故時序模型 Ray Train")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--kind", default="lstm", choices=["lstm", "gru", "cnn"])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--save-path", default=SAVE_PATH)
    ap.add_argument("--cpu", action="store_true", help="強制用 CPU 訓練")
    args = ap.parse_args()

    ray.init(address="auto", ignore_reinit_error=True)
    cfg = {
        "data_dir": args.data_dir, "kind": args.kind, "hidden": args.hidden,
        "layers": args.layers, "dropout": args.dropout, "lr": args.lr,
        "weight_decay": args.weight_decay, "batch": args.batch,
        "epochs": args.epochs, "save_path": args.save_path,
        "use_gpu": not args.cpu,
    }
    trainer = TorchTrainer(
        _train_loop, train_loop_config=cfg,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=not args.cpu),
        run_config=RunConfig(name="accident_final_raytrain",
                             storage_path="/workspace/ray_results"),
    )
    result = trainer.fit()
    print(f"[Ray Train] 完成 | metrics={result.metrics} | 模型={args.save_path}")


if __name__ == "__main__":
    main()
