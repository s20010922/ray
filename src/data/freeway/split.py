"""把 freeway_yolo（images+labels 平鋪）切成 ultralytics 訓練結構。

依「鏡頭」分層切 train/val（每個 cam 各自 8:2），避免某 cam 整批落到單邊
造成分布偏差。輸出標準 ultralytics 結構：

  out_root/
    images/train/*.jpg  images/val/*.jpg
    labels/train/*.txt  labels/val/*.txt
    data.yaml
"""

import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Tuple

from src.modeling.traffic import CLASSES


def make_det_split(src_root: str, out_root: str,
                   val_ratio: float = 0.2, seed: int = 42) -> Tuple[str, int, int]:
    """切 train/val 並複製成 ultralytics 結構，回傳 (data.yaml 路徑, n_train, n_val)。"""
    src, out = Path(src_root), Path(out_root)
    imgs = sorted((src / "images").glob("*.jpg"))

    groups = defaultdict(list)
    for p in imgs:
        groups[p.stem.split("_")[0]].append(p)   # 依 cctv_id 分組

    rng = random.Random(seed)
    train, val = [], []
    for cam, ps in groups.items():
        ps = ps[:]
        rng.shuffle(ps)
        k = int(len(ps) * val_ratio)
        val.extend(ps[:k])
        train.extend(ps[k:])

    for split, ps in [("train", train), ("val", val)]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for p in ps:
            shutil.copy2(p, out / "images" / split / p.name)
            lbl = src / "labels" / f"{p.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, out / "labels" / split / f"{p.stem}.txt")

    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\n\n"
        f"nc: {len(CLASSES)}\nnames:\n{names}\n")
    return str(yaml_path), len(train), len(val)
