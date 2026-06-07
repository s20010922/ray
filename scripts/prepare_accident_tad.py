"""Stage 1 入口：TAD 車禍偵測前處理（影片級切分 Ray Data）。

  docker compose exec ray-head python scripts/prepare_accident_tad.py
  docker compose exec ray-head python scripts/prepare_accident_tad.py --k 40
"""

import argparse

import ray

from src.data.accident_tad.pipeline import FRAMES_PER_VIDEO, TAD_ROOT, prepare


def main():
    ap = argparse.ArgumentParser(description="TAD 車禍前處理 (Ray Data)")
    ap.add_argument("--root", default=TAD_ROOT)
    ap.add_argument("--out", default="/workspace/datasets/accident_tad_seq")
    ap.add_argument("--k", type=int, default=FRAMES_PER_VIDEO,
                    help="每支影片均勻抽幀數")
    args = ap.parse_args()

    ray.init(address="auto", ignore_reinit_error=True)
    prepare(root=args.root, out_root=args.out, k=args.k)


if __name__ == "__main__":
    main()
