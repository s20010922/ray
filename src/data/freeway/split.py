"""freeway_yolo 切分為 train/val/test（鏡頭級隔離）。

同一鏡頭固定機位、相鄰幀高度相似，若跨 split 會洩漏。故保留 1 個完整鏡頭
做 held-out test，其餘鏡頭再分層切 train/val。輸出 ultralytics 用的
train.txt/val.txt/test.txt + dataset.yaml（label 由 images→labels 路徑推得）。
"""

import random
import re
from pathlib import Path

_CAM = re.compile(r"^(CCTV-[^_]+)_")


def _camera(name: str):
    m = _CAM.match(name)
    return m.group(1) if m else name


def split_paths(root: str = "/workspace/datasets/freeway_yolo",
                test_cam: str = "CCTV-N1-S-93.080-M",
                val_ratio: float = 0.2, seed: int = 42) -> dict:
    """鏡頭級切分，回傳 {'train':[Path...], 'val':[...], 'test':[...]}。

    test_cam 整顆鏡頭做 held-out test；其餘鏡頭分層切 train/val。
    """
    root = Path(root)
    imgs = sorted((root / "images").glob("*.jpg"))
    by_cam = {}
    for p in imgs:
        by_cam.setdefault(_camera(p.name), []).append(p)
    cams = sorted(by_cam)
    if test_cam not in by_cam:
        raise ValueError(f"test_cam {test_cam} 不在鏡頭清單 {cams}")

    rng = random.Random(seed)
    train, val, test = [], [], []
    for cam in cams:
        files = sorted(by_cam[cam])
        if cam == test_cam:
            test.extend(files)
            continue
        rng.shuffle(files)
        n_val = round(len(files) * val_ratio)
        val.extend(files[:n_val])
        train.extend(files[n_val:])
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def make_split(root: str = "/workspace/datasets/freeway_yolo",
               test_cam: str = "CCTV-N1-S-93.080-M",
               val_ratio: float = 0.2, seed: int = 42) -> str:
    """寫出 train/val/test 清單與 dataset.yaml，回傳 yaml 路徑。"""
    root = Path(root)
    sp = split_paths(root, test_cam, val_ratio, seed)
    train, val, test = sp["train"], sp["val"], sp["test"]

    for name, items in (("train", train), ("val", val), ("test", test)):
        (root / f"{name}.txt").write_text(
            "\n".join(str(p) for p in sorted(items)))

    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(
        f"path: {root}\ntrain: train.txt\nval: val.txt\ntest: test.txt\n"
        f"nc: 1\nnames: ['Vehicle']\n")

    print(f"[切分] 鏡頭級隔離 test={test_cam}")
    print(f"  train {len(train)}（{len(cams)-1} 鏡頭）"
          f"／val {len(val)}／test {len(test)}（1 鏡頭）→ {yaml_path}")
    return str(yaml_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/datasets/freeway_yolo")
    ap.add_argument("--test-cam", default="CCTV-N1-S-93.080-M")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    args = ap.parse_args()
    make_split(args.root, args.test_cam, args.val_ratio)
