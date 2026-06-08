"""Stage 3 入口：逐幀事故二分類 Ray Train(TorchTrainer)正式訓練 + 存檔。

  docker compose exec ray-head python scripts/train_accident_cnn.py \
      --backbone mobilenet_v2 --lr 1e-3 --epochs 15
"""

import argparse

import ray
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from src.train.accident_cnn.trainer import DATA_DIR, fit

SAVE_PATH = "/workspace/ray_results/accident_tad_final/accident_tad.pt"  # 對齊 eval/serve


def _train_loop(config):
    import ray.train
    import torch
    device = (ray.train.torch.get_device() if config["use_gpu"]
              else torch.device("cpu"))
    best_f1 = fit(config, device, report=lambda m: ray.train.report(m),
                  save_path=config["save_path"])
    print(f"[Ray Train] 最佳 val F1 = {best_f1:.4f} → {config['save_path']}")


def main():
    ap = argparse.ArgumentParser(description="逐幀事故二分類 Ray Train")
    ap.add_argument("--data-dir",
                    default="/workspace/datasets/accident_tad_seq")
    ap.add_argument("--backbone", default="mobilenet_v2",
                    choices=["mobilenet_v2", "resnet18"])
    ap.add_argument("--freeze", action="store_true", default=False,
                    help="凍結 backbone 只訓練分類頭；預設解凍全網微調（0.92 設定）")
    ap.add_argument("--no-freeze", dest="freeze", action="store_false")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--aug", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-4)   # 解凍微調用小 lr（0.92 設定）
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)   # 限 32：batch 64 會吃滿 8G VRAM 卡桌面
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--save-path", default=SAVE_PATH)
    ap.add_argument("--run-name", default="accident_tad_final_raytrain",
                    help="Ray Train run 目錄名（MONITOR Train % 依此名抓進度）")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    ray.init(address="auto", ignore_reinit_error=True)
    cfg = {
        "data_dir": args.data_dir, "backbone": args.backbone,
        "freeze": args.freeze, "dropout": args.dropout, "aug": args.aug,
        "lr": args.lr, "weight_decay": args.weight_decay, "batch": args.batch,
        "epochs": args.epochs, "save_path": args.save_path,
        "use_gpu": not args.cpu,
    }
    trainer = TorchTrainer(
        _train_loop, train_loop_config=cfg,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=not args.cpu),
        run_config=RunConfig(name=args.run_name,
                             storage_path="/workspace/ray_results"),
    )
    result = trainer.fit()
    print(f"[Ray Train] 完成 | metrics={result.metrics} | 模型={args.save_path}")


if __name__ == "__main__":
    main()
