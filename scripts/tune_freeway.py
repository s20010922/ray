"""高公局 fine-tune 超參搜尋（Ray Tune）。

用 ultralytics 內建的 Ray Tune 整合（model.tune(use_ray=True)）：底層是
Ray Tune + ASHA 排程器，會在多組超參（trial）間分時跑，提早砍掉沒前途的，
最後挑 mAP 最佳的組合。

搜尋空間聚焦高公局痛點：
  lr0/lrf  收斂與精度
  hsv_v    亮度抖動 → 日夜光線差異
  scale    縮放增強 → 小目標（遠車）
  fliplr/mosaic  小資料正則化

  # 先小規模驗證能跑（2 trial、各 5 epoch）
  docker compose exec ray-head python scripts/tune_freeway.py --iterations 2 --epochs 5
  # 正式搜尋
  docker compose exec ray-head python scripts/tune_freeway.py --iterations 12 --epochs 30
"""

import argparse


def main():
    ap = argparse.ArgumentParser(description="高公局 fine-tune 超參搜尋（Ray Tune）")
    ap.add_argument("--src", default="/workspace/datasets/freeway_yolo")
    ap.add_argument("--split-out", default="/workspace/datasets/freeway_det")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=30, help="每個 trial 訓練 epoch")
    ap.add_argument("--iterations", type=int, default=12, help="trial 數（超參組合數）")
    ap.add_argument("--gpu-per-trial", type=float, default=1.0)
    ap.add_argument("--grace-period", type=int, default=10,
                    help="ASHA 最少跑幾 epoch 才可被砍（須 ≤ epochs）")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--test-ratio", type=float, default=0.1,
                    help="隔離的 test 鏡頭比例（與正式訓練一致，搜參時不可見）")
    args = ap.parse_args()

    from ray import tune
    from ultralytics import YOLO

    from src.data.freeway.split import make_det_split

    yaml_path, n_tr, n_va, n_te = make_det_split(
        args.src, args.split_out, args.val_ratio, args.test_ratio)
    print(f"[切分] train {n_tr} / val {n_va} / test {n_te}（隔離）→ {yaml_path}")

    # 搜尋空間（use_ray=True 須用 Ray Tune 分布物件，不是 tuple）
    space = {
        "lr0": tune.loguniform(1e-5, 1e-1),   # 跨數量級 → log 尺度
        "lrf": tune.uniform(0.01, 1.0),
        "hsv_v": tune.uniform(0.0, 0.5),      # 亮度抖動 → 日夜
        "scale": tune.uniform(0.0, 0.5),      # 縮放 → 小目標
        "fliplr": tune.uniform(0.0, 0.5),
        "mosaic": tune.uniform(0.0, 1.0),
    }

    grace = min(args.grace_period, args.epochs)   # ASHA 要求 grace_period ≤ max_t(epochs)

    model = YOLO(args.weights)
    result_grid = model.tune(
        data=yaml_path,
        space=space,
        epochs=args.epochs,
        iterations=args.iterations,
        grace_period=grace,
        imgsz=args.imgsz,
        single_cls=True,
        use_ray=True,                 # ← 走 Ray Tune（ASHA）
        gpu_per_trial=args.gpu_per_trial,
        project="/workspace/ray_results",
        name="freeway_tune",
    )

    print("=== Ray Tune 完成 ===")
    print("結果在 /workspace/ray_results/freeway_tune（各 trial + 最佳超參）")


if __name__ == "__main__":
    main()
