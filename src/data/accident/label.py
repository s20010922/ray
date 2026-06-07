"""AccidentBench 真實事故標註解析(metadata-real.csv)。

每支影片給:
  accident_frame  事故發生幀(原生幀號)
  x1,y1,x2,y2     事故區域框(正規化 0~1)
  center_x/y      事故中心(正規化)
  type            事故型態(rear-end / t-bone / single ...)
  scene_layout / day_time / quality  場景/時段/畫質(供篩選對齊高公局)

時空打標:某台車的時序視窗 = 正樣本，當「中心幀落在事故幀附近 ±窗」
且「該車當下位置落在(放大的)事故框內」→ 即捲入事故的車。其餘車/其餘時間為負。
"""

from pathlib import Path

import pandas as pd

# 對齊高公局的預設篩選：高速公路 + 白天 + 畫質非最差
DEFAULT_FILTER = dict(scene_layout=["highway"], day_time=["day"],
                      exclude_quality=["Very_Poor", "Poor"])


def load_clips(csv_path: str, root: str, flt: dict = None) -> list:
    """讀 metadata，套用篩選，回傳 clip dict 清單(只保留檔案存在者)。"""
    flt = flt or DEFAULT_FILTER
    df = pd.read_csv(csv_path)
    if flt.get("scene_layout"):
        df = df[df.scene_layout.isin(flt["scene_layout"])]
    if flt.get("day_time"):
        df = df[df.day_time.isin(flt["day_time"])]
    if flt.get("exclude_quality"):
        df = df[~df.quality.isin(flt["exclude_quality"])]

    clips = []
    for _, r in df.iterrows():
        p = Path(root) / r["path"]
        if not p.exists():
            continue
        fps = (float(r["no_frames"]) / float(r["duration"])
               if r["duration"] else 30.0)
        clips.append({
            "video": str(p),
            "name": Path(r["path"]).name,
            "accident_frame": int(r["accident_frame"]),
            "bbox": (float(r["x1"]), float(r["y1"]),
                     float(r["x2"]), float(r["y2"])),
            "type": r["type"],
            "fps": round(fps, 2),
        })
    return clips


def _iou(cx, cy, w, h, box):
    """軌跡框(中心式)與事故框(x1y1x2y2)的 IoU。"""
    ax1, ay1, ax2, ay2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    x1, y1, x2, y2 = box
    ix1, iy1 = max(ax1, x1), max(ay1, y1)
    ix2, iy2 = min(ax2, x2), min(ay2, y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (ax2 - ax1) * (ay2 - ay1) + (x2 - x1) * (y2 - y1) - inter
    return inter / ua if ua > 0 else 0.0


def identify_culprits(tracks: dict, clip: dict, stride: int, native_fps: float,
                      max_culprits: int = 2, min_iou: float = 0.1,
                      half_s: float = 1.0):
    """找肇事軌跡 → 回傳 {tid: (lo_eff, hi_eff)} 正樣本等效幀範圍。

    肇事車 = 事故時刻其框與事故框 IoU 最高的軌跡(碰撞取前 max_culprits 台)。
    正樣本只取撞擊前後 ±half_s 秒，排除正常過路車造成的標籤雜訊。
    """
    af_eff = clip["accident_frame"] / stride
    box = clip["bbox"]
    half = half_s * native_fps / stride
    tol = half + 2                       # 軌跡需出現在事故時刻附近

    scored = []
    for tid, seq in tracks.items():
        t, cx, cy, w, h = min(seq, key=lambda p: abs(p[0] - af_eff))
        if abs(t - af_eff) > tol:
            continue
        io = _iou(cx, cy, w, h, box)
        if io >= min_iou:
            scored.append((io, tid))
    scored.sort(reverse=True)
    rng = (af_eff - half, af_eff + half)
    return {tid: rng for _, tid in scored[:max_culprits]}
