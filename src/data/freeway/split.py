"""把 freeway_yolo（images+labels 平鋪）切成 ultralytics 訓練結構。

依「鏡頭」分層切 train/val/test（整個鏡頭為單位），避免同一鏡頭的幀
跨 split 造成資料洩漏。

test_ratio > 0 時輸出三分 split：
  out_root/
    images/{train,val,test}/*.jpg
    labels/{train,val,test}/*.txt
    data.yaml  （只含 train/val，供 ultralytics train 用）

test 鏡頭在訓練全程不可見，僅供最終 eval 用。
"""

import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Tuple

from src.modeling.traffic import CLASSES


def make_det_split(src_root: str, out_root: str,
                   val_ratio: float = 0.2, test_ratio: float = 0.1,
                   seed: int = 42) -> Tuple[str, int, int, int]:
    """切 train/val/test 並複製成 ultralytics 結構。

    鏡頭整批分配：先保留 test_ratio 的鏡頭為 test，剩餘鏡頭再切 val_ratio 為 val。

    Returns:
        (data.yaml 路徑, n_train, n_val, n_test)
    """
    src, out = Path(src_root), Path(out_root)
    imgs = sorted((src / "images").glob("*.jpg"))

    groups = defaultdict(list)
    for p in imgs:
        groups[p.stem.split("_")[0]].append(p)   # 依 cctv_id 分組

    cams = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(cams)

    # 先從尾端切出 test 鏡頭（整鏡頭隔離）
    n_test_cams = max(1, int(len(cams) * test_ratio)) if test_ratio > 0 else 0
    test_cams = set(cams[-n_test_cams:]) if n_test_cams else set()
    remain_cams = [c for c in cams if c not in test_cams]

    train, val, test = [], [], []
    for cam in remain_cams:
        ps = groups[cam][:]
        rng.shuffle(ps)
        k = int(len(ps) * val_ratio)
        val.extend(ps[:k])
        train.extend(ps[k:])
    for cam in test_cams:
        test.extend(groups[cam])

    splits = [("train", train), ("val", val)]
    if test:
        splits.append(("test", test))

    for split, ps in splits:
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
    return str(yaml_path), len(train), len(val), len(test)
