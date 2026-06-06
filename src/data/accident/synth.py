"""合成台灣高公局車禍正樣本（compositing / copy-paste）。

動機：Accident 分類器原本正樣本是土耳其 CCTV、負樣本與部署域卻是台灣高公局，
domain 不一致 → 模型學到「哪國街景」的捷徑而非「事故」本身，對台灣畫面無鑑別力。

解法：讓正樣本的**背景與車輛都是真實高公局**，只有「排列」是合成的——
拿 freeway_yolo 已標好的車輛框裁出真台灣車，貼到真台灣幀上，擺成事故姿態
（翻車 / 追撞 / 橫停佔道 / 連環追撞）。正負樣本同 domain，捷徑消失，模型被迫
去學事故的視覺結構。352×240 低畫質 + 訓練時的劣化增強會蓋掉貼合接縫，故
compositing 在此格外可行。

輸出（鏡像 Image/ 結構，可直接餵 split.py 或併入 train）：
  out/accident/*.jpg      合成事故（真背景 + 真車、事故排列）
  out/non-accident/*.jpg  真實高公局正常畫面（直接複製，與正樣本共用背景）

> 誠實註記：合成資料供**訓練**；驗證仍須用真事故（datasets/accident/video/
> accident 的 101 支真片段，經 serve /inject 注入）才能證明對真事故有鑑別力。
"""

import random
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

FW_W, FW_H = 352, 240                 # 高公局畫面尺寸
# 只取較大的近景車（小車放大會糊成補丁）；對比過濾剔除淡平的來源。
_MIN_CROP_W, _MIN_CROP_H = 42, 28
_MIN_CONTRAST = 24.0                   # 灰階標準差下限（太低＝平淡，貼起來像柔斑）
_MAX_UPSCALE = 1.3                     # 放大倍率上限（避免小車放大失真）


# ── 車輛素材池：從 freeway_yolo 的標註裁出真台灣車 ──────────────
def build_vehicle_pool(freeway_root: str, max_crops: int = 500,
                       seed: int = 42) -> List[np.ndarray]:
    """掃 freeway_yolo/{images,labels}，裁出夠大、對比足的車輛 crop（BGR）。"""
    root = Path(freeway_root)
    img_dir, lab_dir = root / "images", root / "labels"
    crops: List[np.ndarray] = []
    for lab in sorted(lab_dir.glob("*.txt")):
        img_path = next((img_dir / (lab.stem + e)
                         for e in (".jpg", ".jpeg", ".png")
                         if (img_dir / (lab.stem + e)).exists()), None)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        for line in lab.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            pw, ph = bw * w, bh * h
            if pw < _MIN_CROP_W or ph < _MIN_CROP_H:
                continue
            x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < _MIN_CROP_W or y2 - y1 < _MIN_CROP_H:
                continue
            crop = img[y1:y2, x1:x2].copy()
            # 對比過濾：灰階標準差太低＝平淡（天空/路面/過曝車），貼起來像柔斑
            if float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).std()) < _MIN_CONTRAST:
                continue
            crops.append(crop)
    rng = random.Random(seed)
    rng.shuffle(crops)
    return crops[:max_crops]


# ── 幾何工具 ──────────────────────────────────────────────────
def _rotate(crop: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """旋轉 crop，回傳 (旋轉後影像, 有效區遮罩)，畫布隨之放大。"""
    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    rot = cv2.warpAffine(crop, M, (nw, nh))
    mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (nw, nh))
    return rot, mask


def _fit(crop: np.ndarray, mask: np.ndarray, target_w: int,
         max_frac: float = 0.45) -> Tuple[np.ndarray, np.ndarray]:
    """縮放使寬約 target_w，限制放大倍率與畫面佔比（避免小車放大成糊斑）。"""
    h, w = crop.shape[:2]
    s = target_w / max(1, w)
    s = min(s, _MAX_UPSCALE, FW_W * max_frac / w, FW_H * max_frac / h)
    nw, nh = max(8, int(w * s)), max(8, int(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return (cv2.resize(crop, (nw, nh), interpolation=interp),
            cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST))


def _paste(base: np.ndarray, fg: np.ndarray, mask: np.ndarray,
           cx: int, cy: int) -> np.ndarray:
    """把 fg 依 mask 貼到 base 的 (cx,cy)：羽化 alpha——保留車身、只柔化邊緣。

    刻意不用 Poisson(seamlessClone)：它會把車的對比洗淡成幽靈糊影，而我們要
    車「看得出來但姿態異常」。羽化 alpha 內部全不透明、僅邊界漸變，接縫在
    低畫質 + 後續劣化下幾乎不可見。
    """
    H, W = base.shape[:2]
    fh, fw = fg.shape[:2]
    cx = int(np.clip(cx, fw // 2 + 1, W - fw // 2 - 1))
    cy = int(np.clip(cy, fh // 2 + 1, H - fh // 2 - 1))
    x1, y1 = cx - fw // 2, cy - fh // 2
    roi = base[y1:y1 + fh, x1:x1 + fw].astype(np.float32)
    a = (cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0)[..., None]
    base[y1:y1 + fh, x1:x1 + fw] = \
        (fg.astype(np.float32) * a + roi * (1 - a)).astype(np.uint8)
    return base


# ── 事故場景合成 ──────────────────────────────────────────────
def _road_xy(rng: random.Random) -> Tuple[int, int]:
    """道路區的隨機落點（畫面下半、水平中央帶）。"""
    return (int(rng.uniform(0.25, 0.75) * FW_W),
            int(rng.uniform(0.55, 0.88) * FW_H))


def _compose_accident(base: np.ndarray, pool: List[np.ndarray],
                      rng: random.Random) -> np.ndarray:
    """在真背景上擺出隨機事故姿態，回傳合成幀。"""
    img = base.copy()
    scenario = rng.choice(["rear_end", "overturned", "sideways", "pileup"])
    cx, cy = _road_xy(rng)
    tw = int(rng.uniform(0.14, 0.30) * FW_W)      # 目標車寬

    def put(angle, x, y, target_w):
        crop = rng.choice(pool)
        rot, mask = _rotate(crop, angle)
        rot, mask = _fit(rot, mask, target_w)
        return _paste(img, rot, mask, x, y)

    if scenario == "rear_end":                     # 追撞：兩車前後高度重疊
        img = put(rng.uniform(-8, 8), cx, cy, tw)
        img = put(rng.uniform(-12, 12), cx + int(tw * 0.5),
                  cy - int(tw * 0.35), int(tw * 0.9))
    elif scenario == "overturned":                 # 翻車：大角度傾倒
        img = put(rng.choice([90, 180, -90]) + rng.uniform(-15, 15),
                  cx, cy, int(tw * 1.1))
    elif scenario == "sideways":                    # 橫停佔道
        img = put(rng.uniform(70, 110), cx, cy, int(tw * 1.1))
    else:                                            # pileup：連環追撞，多車聚集
        for _ in range(rng.randint(3, 5)):
            img = put(rng.uniform(-40, 40),
                      cx + rng.randint(-tw, tw),
                      cy + rng.randint(-tw // 2, tw // 2),
                      int(tw * rng.uniform(0.7, 1.1)))
    return img


# ── 進入點 ────────────────────────────────────────────────────
def generate(freeway_images: str, freeway_yolo: str, out_root: str,
             n_accident: int = 800, n_normal: int = 800,
             seed: int = 42) -> Tuple[int, int]:
    """產生合成事故正樣本 + 收集真實正常負樣本。

    Args:
        freeway_images: 背景幀來源（高公局幀資料夾，含 *.jpg）。
        freeway_yolo:   freeway_yolo 根（含 images/ labels/，用於裁車）。
        out_root:       輸出根，產生 out/{accident,non-accident}/。
    Returns:
        (n_accident_written, n_normal_written)
    """
    rng = random.Random(seed)
    bg_paths = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png")
                      for p in Path(freeway_images).glob(ext))
    if not bg_paths:
        raise FileNotFoundError(f"背景幀資料夾為空：{freeway_images}")
    pool = build_vehicle_pool(freeway_yolo, seed=seed)
    if not pool:
        raise RuntimeError(f"車輛素材池為空：{freeway_yolo}（檢查 labels/）")

    out = Path(out_root)
    acc_dir = out / "accident"
    non_dir = out / "non-accident"
    acc_dir.mkdir(parents=True, exist_ok=True)
    non_dir.mkdir(parents=True, exist_ok=True)

    # 正樣本：合成事故（背景隨機重用）
    n_acc = 0
    for i in range(n_accident):
        base = cv2.imread(str(rng.choice(bg_paths)))
        if base is None:
            continue
        if base.shape[:2] != (FW_H, FW_W):
            base = cv2.resize(base, (FW_W, FW_H))
        synth = _compose_accident(base, pool, rng)
        if cv2.imwrite(str(acc_dir / f"synth_{i:05d}.jpg"), synth):
            n_acc += 1

    # 負樣本：真實正常畫面（與正樣本共用背景 → 逼模型學前景而非背景）
    rng.shuffle(bg_paths)
    n_non = 0
    for i, p in enumerate(bg_paths[:n_normal]):
        if cv2.imwrite(str(non_dir / f"real_{i:05d}.jpg"),
                       cv2.imread(str(p))):
            n_non += 1

    return n_acc, n_non


def merge_into_train(synth_root: str, accident_root: str) -> Tuple[int, int]:
    """把合成 train 資料併入 accident/train/（只進 train，不碰 val/test）。

    檔名加前綴避免覆蓋，方便日後辨識/移除。
    Returns: (併入正樣本數, 併入負樣本數)
    """
    synth, train = Path(synth_root), Path(accident_root) / "train"
    counts = []
    # 前綴：合成正樣本 synth_*、真實負樣本 real_*（土耳其原檔不帶這些前綴）
    prefix = {"accident": "synth_", "non-accident": "real_"}
    for cls in ("accident", "non-accident"):
        dst = train / cls
        dst.mkdir(parents=True, exist_ok=True)
        # 先清掉上次併入的合成檔（避免重跑累積／殘留），土耳其原檔不受影響
        for old in dst.glob(prefix[cls] + "*"):
            old.unlink()
        n = 0
        for p in (synth / cls).glob("*"):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                shutil.copy2(p, dst / p.name)
                n += 1
        counts.append(n)
    return tuple(counts)
