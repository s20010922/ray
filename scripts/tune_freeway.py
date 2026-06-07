"""freeway 車輛偵測 — Ray Tune (ASHA) 超參搜尋。

student = yolo11s（蒸餾自 yolo11x 自動標）。用 ultralytics 內建 Ray Tune
（use_ray=True，內含 ASHA 排程器），跨叢集搜 lr/增強，依 val mAP50-95 挑最佳。
搜完用 scripts/train_freeway.py 帶最佳超參跑正式訓練。

  docker compose exec ray-head python scripts/tune_freeway.py --iterations 12 --epochs 30
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="freeway Ray Tune (ASHA)")
    ap.add_argument("--data",
                    default="/workspace/datasets/freeway_prepared/dataset.yaml",
                    help="Stage 1（prepare_freeway）產出的資料集")
    ap.add_argument("--weights", default="yolo11s.pt", help="student 起點權重")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=30, help="每組超參訓練 epoch")
    ap.add_argument("--iterations", type=int, default=12, help="搜尋組數")
    ap.add_argument("--gpu-per-trial", type=float, default=1.0)
    args = ap.parse_args()

    from ray import tune
    from ultralytics import YOLO

    yaml_path = args.data
    if not Path(yaml_path).exists():
        raise SystemExit(f"找不到 {yaml_path}，請先跑 scripts/prepare_freeway.py")

    # use_ray=True 時 space 必須用 Ray Tune 取樣器（非 tuple）。
    # 高公局固定機位 → mosaic 宜低；聚焦 lr/wd 與輕度增強
    space = {
        "lr0": tune.loguniform(1e-4, 1e-2),
        "lrf": tune.uniform(0.01, 0.5),
        "weight_decay": tune.uniform(0.0, 1e-3),
        "hsv_v": tune.uniform(0.0, 0.1),       # 日夜亮度
        "scale": tune.uniform(0.0, 0.3),       # 縮放
        "fliplr": tune.uniform(0.0, 0.5),      # 雙向車流
        "mosaic": tune.uniform(0.0, 0.3),      # 高公局宜低
    }

    model = YOLO(args.weights)
    model.tune(
        data=yaml_path,
        space=space,
        epochs=args.epochs,
        iterations=args.iterations,
        imgsz=args.imgsz,
        single_cls=True,
        optimizer="AdamW",       # 固定 optimizer，否則 auto 會覆蓋搜到的 lr0
        use_ray=True,
        gpu_per_trial=args.gpu_per_trial,
        project="/workspace/ray_results",
        name="freeway_tune",
    )
    print("=== Ray Tune 完成 ===")
    print("最佳超參見 /workspace/ray_results/freeway_tune/，"
          "填入 scripts/train_freeway.py 跑正式訓練")


if __name__ == "__main__":
    main()
