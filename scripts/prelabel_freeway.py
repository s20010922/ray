"""高公局影像模型輔助預標（進入點）。

  # 先各鏡頭抽幾張看 base 在高公局上的實際效果（含 preview）
  docker compose exec ray-head python scripts/prelabel_freeway.py --sample 4

  # 確認可用後，全量預標
  docker compose exec ray-head python scripts/prelabel_freeway.py

輸出到 /workspace/datasets/freeway_yolo/（images/labels/preview/data.yaml）。
preview 給人眼掃過、labels 給 labelImg 修正。
"""

import argparse
import re
from pathlib import Path

from src.data.freeway.prelabel import prelabel
from src.infer.traffic import find_best_checkpoint, load_detector

_TS = re.compile(r"_(\d{8})_(\d{2})\d{4}_")   # 檔名 _YYYYMMDD_HHMMSS_


def _hour(path: Path):
    m = _TS.search(path.name)
    return int(m.group(2)) if m else None


def collect_images(raw_root: str, sample: int,
                   hour_lo: int = 6, hour_hi: int = 18) -> list:
    """掃 freeway_raw 各鏡頭子資料夾的 jpg。

    只取 [hour_lo, hour_hi] 時段（濾掉凌晨空路，那些預標全空、稀釋資料）。
    sample>0 則每鏡頭「等間隔」抽 N 張（涵蓋整段時間，避免只取最早一批）。
    """
    root = Path(raw_root)
    paths = []
    for cam_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not cam_dir.name.startswith("CCTV"):
            continue                      # 跳過 "all pic" 彙整夾，避免重複
        imgs = sorted(p for p in cam_dir.glob("*.jpg")
                      if (h := _hour(p)) is not None and hour_lo <= h <= hour_hi)
        if sample > 0 and len(imgs) > sample:
            step = len(imgs) / sample
            imgs = [imgs[int(i * step)] for i in range(sample)]
        paths.extend(imgs)
    return [str(p) for p in paths]


def main():
    ap = argparse.ArgumentParser(description="高公局影像模型輔助預標")
    ap.add_argument("--checkpoint", default=None,
                    help="model.pt；省略則自動找最新訓練的最後 checkpoint")
    ap.add_argument("--raw-root", default="/workspace/datasets/freeway_raw")
    ap.add_argument("--out-root", default="/workspace/datasets/freeway_yolo")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--sample", type=int, default=0,
                    help="每鏡頭等間隔抽 N 張（先試效果用）；0=全部")
    ap.add_argument("--hour-lo", type=int, default=6, help="時段下界（含）")
    ap.add_argument("--hour-hi", type=int, default=18, help="時段上界（含）")
    ap.add_argument("--no-roi", action="store_true",
                    help="不套 ROI 過濾（先做物件標記，標所有車）")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--coco", action="store_true",
                    help="改用官方 YOLO11 COCO 模型自動標(car/bus/truck→Vehicle)，"
                         "品質遠高於自訓 base")
    ap.add_argument("--coco-size", default="x",
                    help="COCO 模型大小 n/s/m/l/x（預設 x 最準）")
    args = ap.parse_args()

    if args.coco:
        from src.infer.coco_vehicle import detect as detect_fn
        from src.infer.coco_vehicle import load_coco_vehicle_model
        print(f"[載入] 官方 COCO 模型: yolo11{args.coco_size}.pt")
        model = load_coco_vehicle_model(args.coco_size)
    else:
        detect_fn = None
        ckpt = args.checkpoint or find_best_checkpoint()
        print(f"[載入] checkpoint: {ckpt}")
        model = load_detector(ckpt)

    paths = collect_images(args.raw_root, args.sample,
                           args.hour_lo, args.hour_hi)
    print(f"[資料] 待預標影像: {len(paths)} 張"
          f"（時段 {args.hour_lo}-{args.hour_hi} 時）")

    stats = prelabel(model, paths, args.out_root, conf=args.conf,
                     preview=not args.no_preview, use_roi=not args.no_roi,
                     detect_fn=detect_fn)
    print("\n=== 預標完成 ===")
    print(f"  影像: {stats['n_images']} 張")
    print(f"  總框數: {stats['n_boxes']}（平均 {stats['n_boxes']/max(1,stats['n_images']):.1f}/張）")
    print(f"  零框張數: {stats['n_empty']}")
    print(f"  輸出: {args.out_root}")


if __name__ == "__main__":
    main()
