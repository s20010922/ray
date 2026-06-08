"""Stage 1 — Ray Data：TAD 車禍偵測資料集前處理（影片級切分，防洩漏）。

TAD frames 結構：
  TAD/frames/normal/Normal_NNN.mp4/{0.jpg,1.jpg,...}        ← 正常（負類 0）
  TAD/frames/abnormal/01_Accident_NNN.mp4/{...}             ← 車禍（正類 1）
  TAD/frames/abnormal/02_IllegalTurn.../...                 ← 其它異常（排除）

純車禍偵測：正類只取 `01_Accident_*`，負類取 normal，其餘 5 種異常不用。

防洩漏關鍵：**以影片為單位**分層切 train/val/test（同一支影片的幀不跨 split），
否則同片相鄰幀散到 train+test 會把分數灌爆（notebook 99% 即此坑）。
每支影片均勻抽 K 幀，避免 53 萬幀爆量、也壓低 normal(幀超多)的不平衡。
影像縮 224×224 uint8，標準化留到訓練時用 ImageNet mean/std。
"""

import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import ray

TAD_ROOT = "/workspace/datasets/Traffic Anomaly Dataset/TAD/frames"
IMG_SIZE = 224
FRAMES_PER_VIDEO = 60          # 每支影片均勻抽幀數（控總量 + 壓不平衡）


def _list_videos(root: str) -> list:
    """回傳 [(video_dir, label, vid_name), ...]：accident=1, normal=0。"""
    root = Path(root)
    vids = []
    for d in sorted((root / "normal").glob("*")):
        if d.is_dir():
            vids.append((str(d), 0, d.name))
    for d in sorted((root / "abnormal").glob("01_Accident_*")):
        if d.is_dir():
            vids.append((str(d), 1, d.name))
    return vids


def _video_split(vids: list, ratios=(0.7, 0.15, 0.15), seed=42) -> dict:
    """以影片為單位、依標籤分層切分 → {vid_name: split}。"""
    rng = random.Random(seed)
    smap = {}
    for label in (0, 1):
        grp = [v[2] for v in vids if v[1] == label]
        rng.shuffle(grp)
        n = len(grp)
        n_tr = round(n * ratios[0])
        n_va = round(n * ratios[1])
        for v in grp[:n_tr]:
            smap[v] = "train"
        for v in grp[n_tr:n_tr + n_va]:
            smap[v] = "val"
        for v in grp[n_tr + n_va:]:
            smap[v] = "test"
    return smap


def _sample_frames(video_dir: str, k: int) -> list:
    """均勻抽 k 張幀路徑（依檔名數字排序還原時間序）。"""
    fs = list(Path(video_dir).glob("*.jpg"))

    def _key(p):
        try:
            return int(p.stem)
        except ValueError:
            return 0
    fs.sort(key=_key)
    if not fs:
        return []
    if len(fs) <= k:
        return [str(p) for p in fs]
    idx = np.linspace(0, len(fs) - 1, k).round().astype(int)
    return [str(fs[i]) for i in idx]


def _decode(row: dict) -> dict:
    img = cv2.imread(row["path"], cv2.IMREAD_COLOR)
    if img is None:
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return {"img": img.astype(np.uint8), "label": row["label"],
            "split": row["split"], "vid": row["vid"]}


def prepare(root: str = TAD_ROOT,
            out_root: str = "/workspace/datasets/accident_tad_seq",
            k: int = FRAMES_PER_VIDEO, seed: int = 42) -> str:
    out = Path(out_root)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    vids = _list_videos(root)
    n_acc = sum(1 for v in vids if v[1] == 1)
    print(f"[Ray Data] 影片：normal {len(vids) - n_acc} + accident {n_acc} = {len(vids)}")
    smap = _video_split(vids, seed=seed)

    # 影片級切分後，每片均勻抽 K 幀展開成工作項（帶 split / vid）
    vid_id = {v[2]: i for i, v in enumerate(vids)}      # vid 名 → 整數 id（評估用）
    items = []
    for vdir, label, vname in vids:
        for p in _sample_frames(vdir, k):
            items.append({"path": p, "label": label,
                          "split": smap[vname], "vid": vid_id[vname]})
    random.Random(seed).shuffle(items)
    (out / "_total.txt").write_text(str(len(items)))
    print(f"[Ray Data] 抽幀後總樣本 {len(items)}（每片≤{k} 幀）")

    ds = ray.data.from_items(items, override_num_blocks=64)
    rows, prog = [], out / "_progress.txt"           # 邊處理邊寫進度（供 MONITOR %）
    for r in ds.map(_decode).iter_rows():
        rows.append(r)
        if len(rows) % 256 == 0:
            prog.write_text(str(len(rows)))
    prog.write_text(str(len(rows)))

    buckets = {"train": [], "val": [], "test": []}
    for r in rows:
        buckets[r["split"]].append(r)
    for split, rs in buckets.items():
        X = np.stack([r["img"] for r in rs]) if rs else \
            np.zeros((0, IMG_SIZE, IMG_SIZE, 3), np.uint8)
        y = np.array([r["label"] for r in rs], np.int64)
        vid = np.array([r["vid"] for r in rs], np.int64)
        np.savez(out / f"{split}.npz", X=X, y=y, vid=vid)
        nvid = len(set(vid.tolist()))
        print(f"[Ray Data] {split}: {len(y)} 幀 / {nvid} 片 (事故幀 {int(y.sum())})")

    # vid id → {name,label}（影片級評估對齊）
    import json
    meta = {str(vid_id[v[2]]): {"name": v[2], "label": v[1]} for v in vids}
    (out / "videos.json").write_text(json.dumps(meta, ensure_ascii=False))
    print(f"[Ray Data] 輸出 → {out}")
    return str(out)
