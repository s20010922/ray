"""高公局影像自動標籤（進入點）。

teacher：官方 YOLO11x（COCO）+ SAHI 切片，治 352x240 低畫質遠距小車。
car/bus/truck/motorcycle 併為單類 Vehicle，輸出 YOLO 偵測資料集。

  # 先各鏡頭抽 5 張看效果（含 preview）
  docker compose exec ray-head python scripts/prelabel_freeway.py --sample 5

  # 確認後全量
  docker compose exec ray-head python scripts/prelabel_freeway.py
"""

import argparse

from src.data.freeway.prelabel import prelabel


def main():
    ap = argparse.ArgumentParser(description="高公局影像自動標籤（YOLO11x + SAHI）")
    ap.add_argument("--raw-root", default="/workspace/datasets/freeway_raw")
    ap.add_argument("--out-root", default="/workspace/datasets/freeway_yolo")
    ap.add_argument("--weights", default="yolo11x.pt", help="teacher 權重（COCO）")
    ap.add_argument("--conf", type=float, default=0.25, help="信心門檻（偏低保召回）")
    ap.add_argument("--slice", type=int, default=192, help="SAHI 切片邊長")
    ap.add_argument("--overlap", type=float, default=0.25, help="切片重疊比例")
    ap.add_argument("--hour-lo", type=int, default=6, help="時段下界（含）")
    ap.add_argument("--hour-hi", type=int, default=18, help="時段上界（含）")
    ap.add_argument("--sample", type=int, default=0, help="每鏡頭等間隔抽 N 張；0=全部")
    ap.add_argument("--no-preview", action="store_true", help="不輸出 preview")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    prelabel(args.raw_root, args.out_root, weights=args.weights, conf=args.conf,
             slice_size=args.slice, overlap=args.overlap,
             hour_lo=args.hour_lo, hour_hi=args.hour_hi, sample=args.sample,
             preview=not args.no_preview, device=args.device)


if __name__ == "__main__":
    main()
