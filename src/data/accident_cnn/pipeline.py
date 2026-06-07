"""Stage 1 — Ray Data：逐幀事故圖 → 解碼/縮放/切分為 train/val/test。

資料 = 從影片連續抽出的 jpg(檔名連號)。**防洩漏是這關的核心**：
random split 會把同一場事故的相鄰幀同時放進 train 與 val → 數字虛高、換域即崩。
作法：同類別內依檔名連號切成「block(偽片段)」，以 block 為單位分層切分，
保證同一段連續幀不跨 split。

Ray Data 把 2 萬張圖在 3 節點上分散解碼+縮放成 224×224 uint8，
各 split 寫成 npz。標準化留到訓練時用 ImageNet mean/std(backbone 為 ImageNet 預訓練)。

容器內路徑：/data/accident_cnn/{Accident,NonAccident}/*/*.jpg
"""

import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import ray

CNN_ROOT = "/data/accident_cnn"
IMG_SIZE = 224
BLOCK = 30                 # 連續幀切成偽片段的長度(防相鄰幀洩漏)


def _list_class(root: str, sub: str, label: int) -> list:
    """列出某類別的圖，依檔名數字排序(還原時間順序)。"""
    base = Path(root) / sub / sub          # 解壓後是 Accident/Accident/*.jpg
    if not base.exists():
        base = Path(root) / sub
    files = list(base.glob("*.jpg")) + list(base.glob("*.png"))

    def _key(p):
        try:
            return int(p.stem)
        except ValueError:
            return p.stem
    files.sort(key=_key)
    return [{"path": str(p), "label": label} for p in files]


def _block_split(items: list, ratios=(0.7, 0.15, 0.15), seed=42) -> list:
    """把連號圖切成 BLOCK 大小的 block，以 block 為單位分層切分。"""
    rng = random.Random(seed)
    blocks = [items[i:i + BLOCK] for i in range(0, len(items), BLOCK)]
    rng.shuffle(blocks)
    n = len(blocks)
    n_tr = round(n * ratios[0])
    n_va = round(n * ratios[1])
    out = []
    for i, blk in enumerate(blocks):
        split = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
        for it in blk:
            out.append({**it, "split": split})
    return out


def _decode(row: dict) -> dict:
    img = cv2.imread(row["path"], cv2.IMREAD_COLOR)
    if img is None:                              # 壞檔 → 補黑
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return {"img": img.astype(np.uint8), "label": row["label"],
            "split": row["split"]}


def prepare(root: str = CNN_ROOT,
            out_root: str = "/workspace/datasets/accident_cnn_seq",
            seed: int = 42) -> str:
    """Ray Data 分散式前處理，回傳 out_root。"""
    out = Path(out_root)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    acc = _list_class(root, "Accident", 1)
    non = _list_class(root, "NonAccident", 0)
    print(f"[Ray Data] Accident {len(acc)} 張 | NonAccident {len(non)} 張")

    items = _block_split(acc, seed=seed) + _block_split(non, seed=seed + 1)
    random.Random(seed).shuffle(items)
    (out / "_total.txt").write_text(str(len(items)))     # 供 MONITOR 進度%

    ds = ray.data.from_items(items, override_num_blocks=48)
    rows = ds.map(_decode).take_all()                    # 3 節點分散解碼

    buckets = {"train": [], "val": [], "test": []}
    for r in rows:
        buckets[r["split"]].append(r)
    for split, rs in buckets.items():
        X = np.stack([r["img"] for r in rs]) if rs else \
            np.zeros((0, IMG_SIZE, IMG_SIZE, 3), np.uint8)
        y = np.array([r["label"] for r in rs], np.int64)
        np.savez(out / f"{split}.npz", X=X, y=y)
        print(f"[Ray Data] {split}: {len(y)} 張 (事故 {int(y.sum())}/{len(y)})")

    print(f"[Ray Data] 輸出 → {out}")
    return str(out)
