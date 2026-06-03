"""高公局 Freeway fine-tune 模型在 held-out test 鏡頭上評估 mAP@0.5。

Freeway 模型由 ultralytics 原生 train 微調（best.pt 為 ultralytics 格式），
所以這裡用 ultralytics 原生 val 評估，在訓練全程未見過的 test 鏡頭上跑。

make_det_split（finetune 時呼叫）已把整個 test 鏡頭隔離到 images/test、
labels/test；本腳本動態補一份含 test: 的 data.yaml 再評估。

  # 自動找 freeway_final/weights/best.pt，跑 test 鏡頭
  docker compose exec ray-head python scripts/eval_freeway.py

  # 指定權重
  docker compose exec ray-head python scripts/eval_freeway.py \\
      --weights /workspace/ray_results/freeway_final/weights/best.pt
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Freeway fine-tune 評估（test 鏡頭）")
    ap.add_argument("--weights",
                    default="/workspace/ray_results/freeway_final/weights/best.pt",
                    help="fine-tune 後的 best.pt")
    ap.add_argument("--split-root", default="/workspace/datasets/freeway_det",
                    help="make_det_split 輸出根（含 images/test、labels/test）")
    ap.add_argument("--imgsz", type=int, default=960, help="需與訓練一致")
    args = ap.parse_args()

    split_root = Path(args.split_root)
    test_img_dir = split_root / "images" / "test"
    if not test_img_dir.exists() or not any(test_img_dir.glob("*.jpg")):
        print(f"[錯誤] 找不到 test 鏡頭：{test_img_dir}")
        print("       請先用 scripts/finetune_freeway.py --test-ratio 0.1 切出 test set")
        return

    n_test = len(list(test_img_dir.glob("*.jpg")))
    print(f"[資料] test set: {n_test} 張（held-out 鏡頭）")

    # 動態補一份含 test: 的 data.yaml（原 data.yaml 只有 train/val）
    eval_yaml = split_root / "data_eval.yaml"
    eval_yaml.write_text(
        f"path: {split_root}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: 1\nnames:\n  0: Vehicle\n")

    from ultralytics import YOLO

    print(f"[載入] weights: {args.weights}")
    model = YOLO(args.weights)
    metrics = model.val(data=str(eval_yaml), split="test",
                        imgsz=args.imgsz, single_cls=True, verbose=False)

    print("\n=== Freeway 偵測評估（Test 鏡頭）===")
    print(f"  mAP@0.5     : {metrics.box.map50:.4f}  ({metrics.box.map50*100:.1f}%)")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
