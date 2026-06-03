"""車禍分類評估（held-out test set）。

載入 Ray Train checkpoint，在完全隔離的 test set 上跑推論，
印出 accuracy / precision / recall / F1 / confusion matrix。

  # 自動找最新 checkpoint，跑 test set
  docker compose exec ray-head python scripts/eval_accident.py

  # 指定 checkpoint
  docker compose exec ray-head python scripts/eval_accident.py \\
      --checkpoint /workspace/ray_results/accident/TorchTrainer_xxx/checkpoint_000018/model.pt
"""

import argparse

import cv2

from src.data.accident.sources import list_accident_records
from src.eval.accident import compute_cls_metrics
from src.infer.accident import classify, find_best_accident_checkpoint, load_classifier
from src.modeling.accident import CLASSES


def main():
    ap = argparse.ArgumentParser(description="車禍分類評估（test set）")
    ap.add_argument("--checkpoint", default=None,
                    help="model.pt 路徑；省略則自動找最新 checkpoint")
    ap.add_argument("--data-root", default="/workspace/datasets/accident",
                    help="accident 資料根（含 test/ 子目錄）")
    ap.add_argument("--device", default="cuda", help="cuda 或 cpu")
    args = ap.parse_args()

    ckpt = args.checkpoint or find_best_accident_checkpoint()
    print(f"[載入] checkpoint: {ckpt}")
    model, device = load_classifier(ckpt, device=args.device)

    test_records = list_accident_records(args.data_root, split="test")
    if not test_records:
        print("[錯誤] test set 是空的。請先執行 src/data/accident/split.py 切出 test set。")
        return
    print(f"[資料] test set: {len(test_records)} 張")

    y_true, y_pred = [], []
    for rec in test_records:
        img = cv2.imread(rec["image_path"])
        if img is None:
            continue
        pred, _ = classify(model, img, device)
        y_true.append(rec["label"])
        y_pred.append(pred)

    m = compute_cls_metrics(y_true, y_pred)

    print("\n=== Accident 分類評估（Test Set）===")
    print(f"  樣本數   : {len(y_true)}")
    print(f"  Accuracy : {m['accuracy']:.4f}  ({m['accuracy']*100:.1f}%)")
    print(f"  Macro F1 : {m['macro_f1']:.4f}")
    print()
    print(f"  {'類別':<16} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print(f"  {'-'*52}")
    for cls_name, s in m["per_class"].items():
        print(f"  {cls_name:<16} {s['precision']:>10.4f} {s['recall']:>8.4f} "
              f"{s['f1']:>8.4f} {s['support']:>8}")

    print()
    print("  Confusion Matrix (rows=真實, cols=預測):")
    header = "  " + " " * 18 + "  ".join(f"{c[:8]:>8}" for c in CLASSES)
    print(header)
    for i, cls_name in enumerate(CLASSES):
        row = "  " + f"{cls_name[:16]:<18}" + "  ".join(
            f"{m['confusion_matrix'][i, j]:>8}" for j in range(len(CLASSES)))
        print(row)


if __name__ == "__main__":
    main()
