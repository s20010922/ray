"""Stage 1 — Ray Data：AccidentBench 真實事故影片 → 軌跡運動特徵時序資料集。

每支影片一個工作項：yolo11x+ByteTrack 追蹤 → 運動特徵 → 滑動視窗 → 時空打標。
偵測在單顆 GPU 上序列執行(concurrency=1)；每片各自寫 part npz，driver 合併成
train/val/test.npz，並用 train 集算 standardize 的 mean/std 存 scaler.npz。

每片依自己的 fps 動態決定 stride，把幀率對齊部署等效 ~10fps(高公局實測)，
讓速度/加速度數值尺度跨資料集一致。標籤用事故幀+事故框做時空判定：
捲入事故的車在事故時刻附近 = 正，其餘車/其餘時間 = 負。

容器內路徑(預設)：
  影片根   /data/accident
  metadata /data/accident/metadata-real.csv
"""

import shutil
from pathlib import Path

import numpy as np
import ray

import random

from src.data.accident.features import F, build_features, make_windows
from src.data.accident.label import (DEFAULT_FILTER, identify_culprits,
                                     load_clips)
from src.data.accident.tracking import track_video

ACC_ROOT = "/data/accident"
META_PATH = "/data/accident/metadata-real.csv"
TARGET_FPS = 10.0       # 對齊高公局部署等效幀率


class _Process:
    """Ray Data actor：載入一次 YOLO，逐片追蹤→特徵→切窗→寫 part。"""

    def __init__(self, weights, target_fps, T, win_stride, conf, parts_dir):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.target_fps = target_fps
        self.T = T
        self.win_stride = win_stride
        self.conf = conf
        self.parts = Path(parts_dir)
        self.parts.mkdir(parents=True, exist_ok=True)

    def __call__(self, clip: dict) -> dict:
        stride = max(1, round(clip["fps"] / self.target_fps))
        tracks = track_video(clip["video"], self.model,
                             stride=stride, conf=self.conf)
        feats = build_features(tracks)
        positives = identify_culprits(tracks, clip, stride, clip["fps"])
        X, y, centers = make_windows(feats, positives, self.T, self.win_stride)
        frames = centers * stride                 # 中心幀換回原生幀(對齊事故幀)
        part = str(self.parts / f"{Path(clip['name']).stem}.npz")
        np.savez(part, X=X, y=y, frames=frames)
        return {"name": clip["name"], "part": part, "n": int(len(y)),
                "pos": int(y.sum()), "tracks": len(tracks),
                "accident_frame": int(clip["accident_frame"])}


def _stratified_split(results: list, ratios=(0.7, 0.15, 0.15),
                      seed: int = 42) -> dict:
    """依各片是否含正樣本分層做影片級切分，保證每 split 都有事故正樣本。"""
    rng = random.Random(seed)
    pos = sorted([r["name"] for r in results if r["pos"] > 0])
    neg = sorted([r["name"] for r in results if r["pos"] == 0])
    smap = {}
    for group in (pos, neg):
        rng.shuffle(group)
        n = len(group)
        n_tr = round(n * ratios[0])
        n_va = round(n * ratios[1])
        for v in group[:n_tr]:
            smap[v] = "train"
        for v in group[n_tr:n_tr + n_va]:
            smap[v] = "val"
        for v in group[n_tr + n_va:]:
            smap[v] = "test"
    return smap


def prepare(meta_path: str = META_PATH, root: str = ACC_ROOT,
            out_root: str = "/workspace/datasets/accident_seq",
            weights: str = "yolo11x.pt", target_fps: float = TARGET_FPS,
            T: int = 20, win_stride: int = 5, conf: float = 0.2,
            max_clips: int = 0, flt: dict = None, seed: int = 42) -> str:
    """Ray Data 分散式前處理主流程，回傳 out_root。"""
    out = Path(out_root)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    parts_dir = out / "_parts"

    clips = load_clips(meta_path, root, flt or DEFAULT_FILTER)
    if max_clips:
        clips = clips[:max_clips]
    print(f"[Ray Data] 篩選後事故片 {len(clips)} 支(highway+day+畫質OK)")
    print(f"[Ray Data] 對齊 {target_fps:.0f}fps  T={T} 窗步={win_stride}")

    ds = ray.data.from_items(clips)
    results = ds.map(
        _Process, concurrency=1, num_gpus=1,
        fn_constructor_kwargs={
            "weights": weights, "target_fps": target_fps, "T": T,
            "win_stride": win_stride, "conf": conf,
            "parts_dir": str(parts_dir)},
    ).take_all()

    # 處理完才依實際正樣本數分層切分(保證每 split 有正樣本)
    smap = _stratified_split(results, seed=seed)
    n_pos_clip = sum(1 for r in results if r["pos"] > 0)
    by_name = {r["name"]: r for r in results}
    agg = {"train": [], "val": [], "test": []}
    for r in results:
        agg[smap[r["name"]]].append(r["name"])
    print(f"[Ray Data] 追蹤完成 {len(results)} 支(含事故正樣本的片 {n_pos_clip} 支)，合併…")

    stats = {}
    clip_index = {}                       # name → 全域 clip id（事件級評估用）
    for split, names in agg.items():
        Xs, ys, frs, cids = [], [], [], []
        for name in names:
            r = by_name[name]
            d = np.load(r["part"])
            if not len(d["y"]):
                continue
            cid = clip_index.setdefault(name, len(clip_index))
            Xs.append(d["X"])
            ys.append(d["y"])
            frs.append(d["frames"])
            cids.append(np.full(len(d["y"]), cid, np.int64))
        X = np.concatenate(Xs) if Xs else np.zeros((0, T, F), np.float32)
        y = np.concatenate(ys) if ys else np.zeros((0,), np.int64)
        fr = np.concatenate(frs) if frs else np.zeros((0,), np.int64)
        ci = np.concatenate(cids) if cids else np.zeros((0,), np.int64)
        np.savez(out / f"{split}.npz", X=X, y=y, frames=fr, clip=ci)
        stats[split] = (len(y), int(y.sum()))

    # clips.json：clip id → {name, accident_frame}（事件級評估對齊事故時刻）
    import json
    clips_meta = {str(cid): {"name": name,
                             "accident_frame": by_name[name]["accident_frame"]}
                  for name, cid in clip_index.items()}
    (out / "clips.json").write_text(json.dumps(clips_meta, ensure_ascii=False))

    tr = np.load(out / "train.npz")["X"]
    if len(tr):
        flat = tr.reshape(-1, F)
        np.savez(out / "scaler.npz", mean=flat.mean(0), std=flat.std(0) + 1e-6)

    shutil.rmtree(parts_dir, ignore_errors=True)
    for split, (n, pos) in stats.items():
        print(f"[Ray Data] {split}: {n} 樣本 (正樣本 {pos}/{n})")
    print(f"[Ray Data] 輸出 → {out}")
    return str(out)
