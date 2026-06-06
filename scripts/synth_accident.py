"""產生台灣高公局合成車禍正樣本（compositing），可選併入 accident/train。

  # 1. 產生合成資料到 datasets/accident/synth/（先看品質）
  docker compose exec ray-head python scripts/synth_accident.py

  # 2. 確認品質後，併入 train（只進 train、不碰 val/test）
  docker compose exec ray-head python scripts/synth_accident.py --merge-train

流程定位（解 Accident domain-gap）：
  正樣本背景與車輛都用真高公局 → 與台灣負樣本/部署域同 domain，消除「哪國街景」
  捷徑。合成只供訓練；驗證仍用真事故片段（serve /inject）確認鑑別力。
"""

import argparse

from src.data.accident.synth import generate, merge_into_train


def main():
    ap = argparse.ArgumentParser(description="合成台灣高公局車禍正樣本")
    ap.add_argument("--freeway-images",
                    default="/workspace/datasets/freeway_yolo/images",
                    help="背景幀來源資料夾")
    ap.add_argument("--freeway-yolo",
                    default="/workspace/datasets/freeway_yolo",
                    help="freeway_yolo 根（images/ + labels/，用於裁車）")
    ap.add_argument("--out", default="/workspace/datasets/accident/synth",
                    help="合成輸出根（產生 accident/ non-accident/）")
    ap.add_argument("--n-accident", type=int, default=800,
                    help="合成事故正樣本數")
    ap.add_argument("--n-normal", type=int, default=800,
                    help="真實正常負樣本數")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge-train", action="store_true",
                    help="併入 datasets/accident/train/（只進 train）")
    ap.add_argument("--accident-root", default="/workspace/datasets/accident",
                    help="accident 資料根（--merge-train 用）")
    args = ap.parse_args()

    n_acc, n_non = generate(
        args.freeway_images, args.freeway_yolo, args.out,
        n_accident=args.n_accident, n_normal=args.n_normal, seed=args.seed)
    print(f"[合成完成] accident {n_acc} / non-accident {n_non} → {args.out}")

    if args.merge_train:
        m_acc, m_non = merge_into_train(args.out, args.accident_root)
        print(f"[併入 train] +accident {m_acc} / +non-accident {m_non} "
              f"→ {args.accident_root}/train")
        print("  注意：val/test 維持原土耳其 held-out，未受影響。")
    else:
        print("  （加 --merge-train 才會併入訓練集；先檢視 synth/ 品質）")


if __name__ == "__main__":
    main()
