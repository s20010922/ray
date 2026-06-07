"""Stage 1 入口：逐幀事故圖 → 防洩漏切分的 224×224 資料集(Ray Data)。

  docker compose exec ray-head python scripts/prepare_accident_cnn.py
"""

import argparse

import ray

from src.data.accident_cnn.pipeline import CNN_ROOT, prepare


def main():
    ap = argparse.ArgumentParser(description="逐幀事故圖前處理 (Ray Data)")
    ap.add_argument("--root", default=CNN_ROOT)
    ap.add_argument("--out", default="/workspace/datasets/accident_cnn_seq")
    args = ap.parse_args()

    ray.init(address="auto", ignore_reinit_error=True)
    prepare(root=args.root, out_root=args.out)


if __name__ == "__main__":
    main()
