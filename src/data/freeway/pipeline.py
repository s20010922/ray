"""Stage 1 — Ray Data：freeway_yolo 分散式前處理 → 訓練就緒資料集。

對 train 影像做 geometry-preserving 劣化增強（JPEG 再壓縮、模糊、噪點、亮度／
對比抖動，皆不動框座標，標籤原樣複製），離線擴增小資料集；val／test 原樣複製、
不增強。整個 map 跨 3 節點 CPU 分散執行（head 的 GPU 留給 Ray Train），輸出標準
ultralytics 資料夾結構到 freeway_prepared/，交接給 Ray Train 的 ultralytics 消費。
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
import ray

from src.data.freeway.split import split_paths


def _degrade(img, seed: int):
    """低畫質劣化增強（保持幾何不變，框座標不需改）。"""
    rng = np.random.default_rng(seed)
    q = int(rng.integers(25, 60))                 # JPEG 再壓縮
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if rng.random() < 0.5:                         # 高斯模糊
        k = int(rng.choice([3, 3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.5:                         # 噪點
        noise = rng.normal(0, rng.uniform(3, 10), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    a, b = rng.uniform(0.85, 1.15), rng.uniform(-15, 15)   # 亮度/對比
    return np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)


def _process(job: dict) -> dict:
    """單一影像：讀圖→（選擇性劣化）→寫出 image + 複製 label。在 worker 上跑。"""
    img = cv2.imread(job["src_img"])
    if img is None:
        return {"split": job["split"], "ok": False}
    if job["degrade"]:
        img = _degrade(img, job["seed"])
    cv2.imwrite(job["dst_img"], img)
    if job["src_lbl"] and Path(job["src_lbl"]).exists():
        shutil.copy(job["src_lbl"], job["dst_lbl"])
    else:
        Path(job["dst_lbl"]).write_text("")        # 零框也留空檔
    return {"split": job["split"], "ok": True}


def _build_jobs(splits: dict, out: Path, aug_per_image: int) -> list:
    """把 split 結果展開成 Ray Data 的工作項（train 多 aug_per_image 個劣化變體）。"""
    jobs = []
    for split, paths in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for p in paths:
            lbl = p.parent.parent / "labels" / (p.stem + ".txt")
            n = aug_per_image if split == "train" else 0   # 只增強 train
            for k in range(n + 1):                          # k=0 原圖
                stem = p.stem if k == 0 else f"{p.stem}_aug{k}"
                jobs.append({
                    "split": split,
                    "src_img": str(p),
                    "src_lbl": str(lbl),
                    "dst_img": str(out / "images" / split / f"{stem}.jpg"),
                    "dst_lbl": str(out / "labels" / split / f"{stem}.txt"),
                    "degrade": k > 0,
                    "seed": abs(hash((p.stem, k))) % (2**31),
                })
    return jobs


def prepare(root: str = "/workspace/datasets/freeway_yolo",
            out_root: str = "/workspace/datasets/freeway_prepared",
            test_cam: str = "CCTV-N1-S-93.080-M",
            val_ratio: float = 0.2, aug_per_image: int = 0) -> str:
    """Ray Data 分散式前處理主流程，回傳 dataset.yaml 路徑。"""
    out = Path(out_root)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    splits = split_paths(root, test_cam, val_ratio)
    jobs = _build_jobs(splits, out, aug_per_image)
    aug_note = f"（×{aug_per_image+1} 劣化增強）" if aug_per_image else "（不增強）"
    print(f"[Ray Data] 切分 train {len(splits['train'])}{aug_note}"
          f"／val {len(splits['val'])}／test {len(splits['test'])}")
    print(f"[Ray Data] 展開 {len(jobs)} 個處理項，跨叢集分散執行…")

    ds = ray.data.from_items(jobs)
    results = ds.map(_process).take_all()
    n_ok = sum(r["ok"] for r in results)
    per = {}
    for r in results:
        per[r["split"]] = per.get(r["split"], 0) + (1 if r["ok"] else 0)
    print(f"[Ray Data] 完成 {n_ok}/{len(results)} 張 → {per}")

    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: 1\nnames: ['Vehicle']\n")
    print(f"[Ray Data] 輸出 {yaml_path}")
    return str(yaml_path)
