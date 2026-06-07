"""Stage 1 入口：AccidentBench 真實事故影片 → 軌跡運動特徵時序資料集(Ray Data)。

  docker compose exec ray-head python scripts/prepare_accident.py
  docker compose exec ray-head python scripts/prepare_accident.py --max-clips 20   # 小批驗證
"""

import argparse

import ray

from src.data.accident.pipeline import ACC_ROOT, META_PATH, TARGET_FPS, prepare


def main():
    ap = argparse.ArgumentParser(description="事故時序資料前處理 (Ray Data)")
    ap.add_argument("--meta", default=META_PATH)
    ap.add_argument("--root", default=ACC_ROOT)
    ap.add_argument("--out", default="/workspace/datasets/accident_seq")
    ap.add_argument("--weights", default="/workspace/datasets/weights/yolo11x.pt",
                    help="追蹤用偵測器")
    ap.add_argument("--target-fps", type=float, default=TARGET_FPS,
                    help="對齊部署等效幀率")
    ap.add_argument("--seq-len", type=int, default=20, help="時序視窗長度 T")
    ap.add_argument("--win-stride", type=int, default=5, help="切窗步長")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--max-clips", type=int, default=0, help="只取前 N 支(0=全部)")
    args = ap.parse_args()

    ray.init(address="auto", ignore_reinit_error=True)
    prepare(meta_path=args.meta, root=args.root, out_root=args.out,
            weights=args.weights, target_fps=args.target_fps,
            T=args.seq_len, win_stride=args.win_stride, conf=args.conf,
            max_clips=args.max_clips)


if __name__ == "__main__":
    main()
