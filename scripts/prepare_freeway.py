"""Stage 1 — Ray Data：freeway_yolo 分散式前處理（進入點）。

鏡頭級切分 + 對 train 做離線劣化增強（跨 3 節點 CPU 分散），輸出
freeway_prepared/（標準 ultralytics 結構）供後續 Tune／Train 使用。

  docker compose exec ray-head python scripts/prepare_freeway.py --aug 2
"""

import argparse

from src.core.cluster import init_ray
from src.data.freeway.pipeline import prepare


def main():
    ap = argparse.ArgumentParser(description="Stage 1 Ray Data 前處理")
    ap.add_argument("--root", default="/workspace/datasets/freeway_yolo")
    ap.add_argument("--out-root", default="/workspace/datasets/freeway_prepared")
    ap.add_argument("--test-cam", default="CCTV-N1-S-93.080-M",
                    help="held-out 測試鏡頭（整顆隔離）")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--aug", type=int, default=0,
                    help="每張 train 影像額外產生的劣化變體數（freeway 同域低畫質，"
                         "預設 0 不增強；ultralytics 訓練時已做即時增強）")
    args = ap.parse_args()

    init_ray()
    prepare(args.root, args.out_root, args.test_cam, args.val_ratio, args.aug)


if __name__ == "__main__":
    main()
