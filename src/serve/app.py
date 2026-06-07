"""高公局即時車流監控 Ray Serve 後端（餵 dashboard.html）。

UI 契約（每鏡頭、每 pollMs 輪詢一次）：
  GET /live_focus/<cctv_id>.jpg   → 畫好框的最新標註幀（JPEG）
  GET /live_focus/<cctv_id>.json  → {num_detections, count_level, density_level,
                                     captured_at}
  GET /                           → dashboard.html（同源，免 CORS proxy）

設計：
  - 單 replica（占 1 GPU），啟動載入 freeway yolo11s best.pt（ultralytics 原生格式）。
  - 背景 asyncio 迴圈：對 5 支鏡頭輪流 grab_jpeg_frame → 偵測 → 更新快取。
    抓幀/推論是阻塞操作，丟到 thread executor 跑，不卡事件迴圈。

註：車禍偵測（分類器 + 靜止追蹤 + 片段注入）與 ROI 幾何過濾已於重新規劃時移除，
    待 accident 模型（DoTA/Roboflow）重建後再接回。本服務目前只做全幅車流偵測與
    密度分級。
"""

import asyncio
import time
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from ray import serve

from src.data.freeway.grabber import FOCUS_CAMERAS, grab_jpeg_frame

# 車流分級門檻（依 ROI 內車輛數 / 佔用面積比；目測初版，可再校）
_COUNT_BANDS = [(8, "LOW"), (20, "MED"), (10**9, "HIGH")]
_DENSITY_BANDS = [(0.10, "LOW"), (0.25, "MED"), (1.01, "HIGH")]

app = FastAPI()


def _level(value: float, bands) -> str:
    for thr, label in bands:
        if value < thr:
            return label
    return bands[-1][1]


@serve.deployment(ray_actor_options={"num_gpus": 1, "num_cpus": 2})
@serve.ingress(app)
class TrafficMonitor:
    def __init__(self,
                 detector_weights: str =
                 "/workspace/ray_results/freeway_final/weights/best.pt",
                 poll_interval: float = 4.0,
                 conf: float = 0.4,
                 imgsz: int = 640,
                 device: str = "cuda"):
        from ultralytics import YOLO

        self.poll_interval = poll_interval
        self.conf = conf
        self.imgsz = imgsz
        self.device = device   # cuda；demo 監控訓練時用 cpu 釋出 GPU

        # 車流偵測：freeway best.pt 是 ultralytics 原生格式，直接 YOLO 載
        self.detector = YOLO(detector_weights)

        # 每鏡頭快取：最新標註 jpg bytes + json dict
        self.cache: Dict[str, dict] = {}

        self._dashboard = (Path(__file__).parent / "dashboard.html").read_text(
            encoding="utf-8")

        # 啟動背景輪詢
        self._task = asyncio.create_task(self._poll_loop())

    # ── 推論單張（阻塞，跑在 executor）─────────────────
    def _infer_frame(self, cam_id: str, img: np.ndarray) -> dict:
        h, w = img.shape[:2]

        res = self.detector.predict(img, imgsz=self.imgsz, conf=self.conf,
                                    device=self.device, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None \
            else np.zeros((0, 4), np.float32)
        scores = res.boxes.conf.cpu().numpy() if res.boxes is not None \
            else np.zeros((0,), np.float32)

        n = int(len(boxes))
        area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes)
        density = float(area / (w * h)) if (w * h) else 0.0

        # 畫框（全幅，不疊文字——車數改由前端各格底下顯示）
        vis = img.copy()
        for (x1, y1, x2, y2), s in zip(boxes.astype(int), scores):
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 136), 2)
        ok, enc = cv2.imencode(".jpg", vis)
        jpg = enc.tobytes() if ok else b""

        return {
            "jpg": jpg,
            "json": {
                "num_detections": n,
                "count_level": _level(n, _COUNT_BANDS),
                "density_level": _level(density, _DENSITY_BANDS),
                "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

    # ── 取得一幀：抓即時串流（阻塞）──
    def _grab_frame(self, stream_url: str):
        jpg = grab_jpeg_frame(stream_url)
        return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

    # ── 背景輪詢迴圈 ──────────────────────────────────
    async def _poll_loop(self):
        loop = asyncio.get_event_loop()
        while True:
            t0 = time.time()
            for cam in FOCUS_CAMERAS:
                try:
                    img = await loop.run_in_executor(
                        None, self._grab_frame, cam.stream_url)
                    if img is None:
                        continue
                    result = await loop.run_in_executor(
                        None, self._infer_frame, cam.cctv_id, img)
                    self.cache[cam.cctv_id] = result
                except Exception as e:
                    print(f"[serve] {cam.cctv_id} 失敗：{type(e).__name__}: {e}")
            dt = time.time() - t0
            await asyncio.sleep(max(0.0, self.poll_interval - dt))

    # ── HTTP endpoints（對齊 UI 契約）─────────────────
    @app.get("/")
    def index(self):
        return HTMLResponse(self._dashboard,
                            headers={"Cache-Control": "no-cache"})

    @app.get("/live_focus/{name}")
    def live_focus(self, name: str):
        cam_id, _, ext = name.rpartition(".")
        entry = self.cache.get(cam_id)
        if entry is None:
            return JSONResponse({"error": "no data yet"}, status_code=503)
        if ext == "jpg":
            return Response(entry["jpg"], media_type="image/jpeg",
                            headers={"Cache-Control": "no-cache"})
        return JSONResponse(entry["json"],
                            headers={"Cache-Control": "no-cache"})


def build_app(args: dict = None):
    """Serve 應用工廠（給 serve.run / config 用）。"""
    args = args or {}
    return TrafficMonitor.bind(**args)
