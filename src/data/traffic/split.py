"""UA-DETRAC 依序列切 train/val/test 三份（held-out test set）。

與 sources.list_detrac_records 的差別：
  - sources 只切 train/val（兩份），給原本的訓練用。
  - 這支切 train/val/test（三份），test 序列在訓練全程不可見，僅供最終 eval。

切分粒度為「序列」（非幀），避免同一段影片的相鄰幀洩漏到不同 split。
切分前先對序列名稱排序再以固定 seed 打亂，確保可重現。

用法：
  # 看三份的序列數 / 幀數 / box 數
  python -m src.data.traffic.split --detrac-root /data/detrac

  # 訓練時：list_detrac_splits(...)["train"], [...]["val"]
  # 評估時：list_detrac_splits(...)["test"]
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.traffic.sources import _parse_sequence


def split_sequences(detrac_root: str = "/data/detrac",
                    val_ratio: float = 0.2,
                    test_ratio: float = 0.2,
                    seed: int = 42,
                    limit_sequences: Optional[int] = None) -> Dict[str, List[str]]:
    """把所有序列名稱依固定 seed 打亂後切 train/val/test。

    Returns:
        {"train": [...], "val": [...], "test": [...]}（皆為序列名稱）。
    """
    xml_base = (Path(detrac_root) / "DETRAC-Train-Annotations-XML"
                / "DETRAC-Train-Annotations-XML")
    seqs = sorted(p.stem for p in xml_base.glob("*.xml"))
    if limit_sequences:
        seqs = seqs[:limit_sequences]

    random.Random(seed).shuffle(seqs)
    n = len(seqs)
    n_test = max(1, int(n * test_ratio)) if test_ratio > 0 else 0
    n_val = max(1, int(n * val_ratio)) if val_ratio > 0 else 0

    return {
        "test":  seqs[:n_test],
        "val":   seqs[n_test:n_test + n_val],
        "train": seqs[n_test + n_val:],
    }


def list_detrac_splits(
        detrac_root: str = "/data/detrac",
        frame_stride: int = 10,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 42,
        limit_sequences: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """產生 UA-DETRAC 的 train / val / test records（依序列切）。

    Args:
        detrac_root: DETRAC 掛載根（含 DETRAC-Images/、DETRAC-Train-Annotations-XML/）。
        frame_stride: 每幾幀取 1（降冗餘）。
        val_ratio / test_ratio: 依序列切的比例。
        seed: 打亂序列用的固定種子（可重現）。
        limit_sequences: 只用前 N 個序列（先驗證用）；None=全部。

    Returns:
        {"train": [...records], "val": [...records], "test": [...records]}
        每筆 record 格式同 sources._parse_sequence。
    """
    root = Path(detrac_root)
    img_base = root / "DETRAC-Images" / "DETRAC-Images"
    xml_base = (root / "DETRAC-Train-Annotations-XML"
                / "DETRAC-Train-Annotations-XML")

    seq_split = split_sequences(detrac_root, val_ratio, test_ratio,
                                seed, limit_sequences)

    out: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for split, seqs in seq_split.items():
        for seq in seqs:
            recs = _parse_sequence(xml_base / f"{seq}.xml", img_base / seq,
                                   frame_stride)
            out[split].extend(recs)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="UA-DETRAC 三分切（train/val/test 統計）")
    ap.add_argument("--detrac-root", default="/data/detrac")
    ap.add_argument("--frame-stride", type=int, default=10)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    seq_split = split_sequences(args.detrac_root, args.val_ratio,
                                args.test_ratio, args.seed, args.limit)
    print("[序列切分]")
    for split in ("train", "val", "test"):
        print(f"  {split:5s}: {len(seq_split[split])} 序列")

    splits = list_detrac_splits(args.detrac_root, args.frame_stride,
                                args.val_ratio, args.test_ratio,
                                args.seed, args.limit)
    print("[幀 / box 統計]")
    for split in ("train", "val", "test"):
        recs = splits[split]
        n_box = sum(len(r["boxes_xyxy"]) for r in recs)
        print(f"  {split:5s}: {len(recs)} 幀 / {n_box} boxes")
