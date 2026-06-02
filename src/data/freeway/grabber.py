"""高公局 CCTV 即時影像取得（MJPEG 抽幀）。

資料取得方式參考 focus_poller.py：
  - catalog：每支鏡頭 = (cctv_id, label, stream_url)
  - grab_jpeg_frame(stream_url) -> bytes：從 MJPEG 串流抽「單張」JPEG

grab_jpeg_frame 是最底層的共用單位：
  - 收集 fine-tune 資料 → 定時呼叫它、把回傳的 bytes 存檔
  - 即時推論（focus_poller）→ 呼叫它、把 bytes 直接 POST 給 Ray Serve

⚠️ 抓下來的畫面「沒有標註」。要 fine-tune 得人工標：
  traffic 框車（bbox）、accident 標 accident/non-accident。
即時串流幾乎抓不到「車禍」正樣本，accident 正樣本需另尋來源；
這裡抓的主要餵 traffic、以及 accident 的 non-accident 負樣本。
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import requests

# 政府網站常以 WAF 檢查來源，附上瀏覽器式標頭。
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
    "Referer": "https://cctvn.freeway.gov.tw/",
}

_BASE = "https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera={cam}"


@dataclass
class Camera:
    """一支 CCTV 鏡頭。stream_url 是 MJPEG 串流位址。"""
    cctv_id: str
    label: str
    stream_url: str


# focus 的 5 支鏡頭（使用者指定）。
FOCUS_CAMERAS: List[Camera] = [
    Camera("CCTV-N1-S-34.018-M",  "國1南 34K+020 高公局交流道",  _BASE.format(cam="13400")),
    Camera("CCTV-N3-S-40.980-M",  "國3南 40K+980 土城路段",      _BASE.format(cam="34090")),
    Camera("CCTV-N1-N-37.050-M",  "國1北 37K+050 泰山路段",      _BASE.format(cam="13700")),
    Camera("CCTV-N1H-S-17.450-M", "國1高架南 17K+450 內湖交流道", _BASE.format(cam="11740")),
    Camera("CCTV-N1-S-93.080-M",  "國1南 93K+080 新竹路段",      _BASE.format(cam="19300")),
]


def grab_jpeg_frame(stream_url: str, timeout: float = 8.0,
                    headers: Optional[dict] = None) -> bytes:
    """從 MJPEG 串流抽出「第一張完整 JPEG」，回傳原始 bytes。

    MJPEG 是連續的 JPEG，這裡邊讀邊找 JPEG 的起始 (FFD8) / 結束 (FFD9)
    標記，湊到一張完整的就回傳。

    Raises:
        requests.RequestException: 連線失敗（含被 WAF 擋的 403）。
        RuntimeError: 串流讀完仍湊不出一張完整 JPEG。
    """
    r = requests.get(stream_url, stream=True,
                     headers=headers or HEADERS, timeout=timeout)
    r.raise_for_status()
    buf = b""
    try:
        for chunk in r.iter_content(chunk_size=8192):
            buf += chunk
            start = buf.find(b"\xff\xd8")   # SOI
            end = buf.find(b"\xff\xd9")     # EOI
            if start != -1 and end != -1 and end > start:
                return buf[start:end + 2]
    finally:
        r.close()
    raise RuntimeError("串流中找不到完整 JPEG")


def save_frame(jpg: bytes, out_dir: Path, cam: Camera, idx: int) -> Path:
    """把一張 JPEG bytes 存成檔（含鏡頭與時間戳，方便日後標註對照）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{cam.cctv_id}_{ts}_{idx:04d}.jpg"
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("JPEG 解碼失敗")
    cv2.imwrite(str(path), img)
    return path


def collect_for_finetune(cameras: List[Camera],
                         out_root: str = "/workspace/datasets/freeway_raw",
                         per_camera: int = 50,
                         interval_sec: float = 5.0) -> dict:
    """對每支鏡頭定時抽幀存檔，累積 fine-tune 原始資料。

    即時畫面連續幀太相似，靠 interval_sec 間隔取樣才有多樣性。
    要更高多樣性（不同車流/天氣），應分散在一天不同時段多次執行，
    而不是一次連抓一大批。

    Args:
        per_camera: 每支鏡頭這一輪抓幾張。
        interval_sec: 每張間隔秒數。

    Returns:
        {cctv_id: 實際存下的張數}
    """
    out = Path(out_root)
    stats = {}
    for cam in cameras:
        saved = 0
        for i in range(per_camera):
            try:
                jpg = grab_jpeg_frame(cam.stream_url)
                save_frame(jpg, out / cam.cctv_id, cam, i)
                saved += 1
            except Exception as e:
                print(f"[grab] {cam.cctv_id} 第{i}張失敗：{type(e).__name__}: {e}")
                break
            time.sleep(interval_sec)
        stats[cam.cctv_id] = saved
        print(f"[grab] {cam.cctv_id} {cam.label}：{saved}/{per_camera} 張")
    return stats


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="高公局 CCTV MJPEG 抓取器")
    ap.add_argument("--out-root", default="/workspace/datasets/freeway_raw")
    ap.add_argument("--per-camera", type=int, default=50)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--camera", default=None, help="只抓單一鏡頭 ID（測試用）")
    ap.add_argument("--num", type=int, default=1)
    args = ap.parse_args()

    if args.camera:
        cam = next((c for c in FOCUS_CAMERAS if c.cctv_id == args.camera), None)
        if cam is None:
            raise SystemExit(f"未知鏡頭：{args.camera}")
        n = 0
        for i in range(args.num):
            try:
                save_frame(grab_jpeg_frame(cam.stream_url),
                           Path(args.out_root) / cam.cctv_id, cam, i)
                n += 1
            except Exception as e:
                print(f"失敗：{type(e).__name__}: {e}")
                break
            time.sleep(args.interval)
        print(f"✅ 鏡頭 {args.camera} 抓到 {n} 張")
    else:
        s = collect_for_finetune(FOCUS_CAMERAS, args.out_root,
                                 args.per_camera, args.interval)
        print(f"✅ 完成：{s}")
