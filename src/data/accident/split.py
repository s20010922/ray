"""把 accident 資料夾重新切成 train/val/test 三份。

原始資料假設為 team edit 整理後的分類結構（已切或未切皆可）：
  src_root/{任意子資料夾}/{accident,non-accident}/*.jpg
  或
  src_root/{train,val}/{accident,non-accident}/*.jpg

做法：把所有圖先收集起來，依「類別」分層後打亂，再切 train/val/test。
類別分層確保三份的 accident:non-accident 比例一致。

輸出：
  out_root/{train,val,test}/{accident,non-accident}/*.jpg
"""

import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Tuple

from src.modeling.accident import CLASSES

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png")


def _collect_all(src_root: Path) -> dict:
    """遞迴掃 src_root，依類別名稱（accident / non-accident）收集所有圖。"""
    per_class: dict = defaultdict(list)
    for cls_name in CLASSES:
        for img_path in src_root.rglob(f"{cls_name}/*"):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                per_class[cls_name].append(img_path)
    return per_class


def make_cls_split(src_root: str, out_root: str,
                   val_ratio: float = 0.15, test_ratio: float = 0.15,
                   seed: int = 42) -> Tuple[int, int, int]:
    """重新切 accident 資料成 train/val/test，輸出到 out_root。

    每個類別獨立打亂再切（分層抽樣），確保三份比例一致。
    test set 在訓練全程不可見，僅供最終 eval 用。

    Returns:
        (n_train, n_val, n_test)
    """
    src, out = Path(src_root), Path(out_root)
    per_class = _collect_all(src)

    rng = random.Random(seed)
    n_train = n_val = n_test = 0

    for cls_name, paths in per_class.items():
        paths = paths[:]
        rng.shuffle(paths)
        n = len(paths)
        n_te = max(1, int(n * test_ratio)) if test_ratio > 0 else 0
        n_va = max(1, int(n * val_ratio))

        splits = {
            "test": paths[:n_te],
            "val":  paths[n_te:n_te + n_va],
            "train": paths[n_te + n_va:],
        }

        for split, ps in splits.items():
            dest_dir = out / split / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for p in ps:
                shutil.copy2(p, dest_dir / p.name)

        n_train += len(splits["train"])
        n_val   += len(splits["val"])
        n_test  += len(splits["test"])

    return n_train, n_val, n_test


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Accident 資料重切 train/val/test")
    ap.add_argument("--src", default="/workspace/datasets/accident_raw",
                    help="原始資料根（含 accident/non-accident 子目錄，可多層）")
    ap.add_argument("--out", default="/workspace/datasets/accident",
                    help="輸出根（覆蓋舊的 train/val/test）")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    args = ap.parse_args()

    n_tr, n_va, n_te = make_cls_split(args.src, args.out,
                                       args.val_ratio, args.test_ratio)
    total = n_tr + n_va + n_te
    print(f"[切分完成] train {n_tr} / val {n_va} / test {n_te}（共 {total} 張）")
    print(f"  比例 {n_tr/total:.0%} / {n_va/total:.0%} / {n_te/total:.0%}")
    print(f"  輸出 → {args.out}")
