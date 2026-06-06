"""高公局即時監控 Ray Serve 後端（餵 team edit 的 smart-traffic-ui）。

UI 契約（每鏡頭、每 pollMs 輪詢一次）：
  GET /live_focus/<cctv_id>.jpg   → 畫好框的最新標註幀（JPEG）
  GET /live_focus/<cctv_id>.json  → {num_detections, count_level, density_level,
                                     is_accident, accident_conf, captured_at}
  GET /                           → dashboard.html（同源，免 CORS proxy）

設計：
  - 單 replica（占 1 GPU），啟動時載入兩個 model：
      Traffic 偵測 = freeway fine-tune best.pt（ultralytics 原生格式）
      Accident 分類 = Ray Train checkpoint（model.pt）
  - 背景 asyncio 迴圈：對 5 支鏡頭輪流 grab_jpeg_frame → 推論 → 更新快取。
    抓幀/推論是阻塞操作，丟到 thread executor 跑，不卡事件迴圈。
  - 車禍「連續確認」：單幀易誤判，需連續 N 幀都判 accident 且高信心才算事件。
  - ROI：推論階段幾何過濾（只算主車道區的車），對齊 roi.py。
"""

import asyncio
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from ray import serve

from src.data.freeway.grabber import FOCUS_CAMERAS, grab_jpeg_frame
from src.data.freeway.roi import draw_roi, filter_by_roi, get_roi
from src.infer.accident import classify, load_classifier
from src.modeling.accident import CLASSES as ACC_CLASSES

# 車流分級門檻（依 ROI 內車輛數 / 佔用面積比；目測初版，可再校）
_COUNT_BANDS = [(8, "LOW"), (20, "MED"), (10**9, "HIGH")]
_DENSITY_BANDS = [(0.10, "LOW"), (0.25, "MED"), (1.01, "HIGH")]

# 車禍連續確認：最近 _ACC_WINDOW 幀中需連續 _ACC_CONSEC 幀 accident 且 conf≥門檻
# 門檻預設拉高（治 Accident domain-gap 誤報；門檻可由啟動參數覆寫）。
_ACC_WINDOW = 5
_ACC_CONSEC = 3
_ACC_CONF_TH_DEFAULT = 0.97

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
                 accident_ckpt: str = None,
                 poll_interval: float = 4.0,
                 conf: float = 0.4,
                 imgsz: int = 960,
                 use_roi: bool = True,
                 accident_conf_th: float = _ACC_CONF_TH_DEFAULT,
                 clip_dir: str =
                 "/workspace/datasets/accident/video/accident",
                 device: str = "cuda"):
        from ultralytics import YOLO

        from src.infer.accident import find_best_accident_checkpoint

        self.poll_interval = poll_interval
        self.conf = conf
        self.imgsz = imgsz
        self.use_roi = use_roi
        self.accident_conf_th = accident_conf_th
        self.clip_dir = Path(clip_dir)
        self.device = device   # cuda；demo 監控訓練時用 cpu 釋出 GPU

        # 車禍片段注入狀態：{cam_id: cv2.VideoCapture}（驗證用，手動觸發）
        self.inject_cap: Dict[str, "cv2.VideoCapture"] = {}
        self.inject_clip: Dict[str, str] = {}   # {cam_id: 片段檔名（狀態顯示）}

        # Traffic 偵測：freeway best.pt 是 ultralytics 原生格式，直接 YOLO 載
        self.detector = YOLO(detector_weights)

        # Accident 分類：Ray Train checkpoint（state_dict）
        ckpt = accident_ckpt or find_best_accident_checkpoint()
        self.classifier, self.acc_device = load_classifier(ckpt, device=device)

        # 每鏡頭快取：最新標註 jpg bytes + json dict
        self.cache: Dict[str, dict] = {}
        # 每鏡頭最近幾幀的 accident 判定（連續確認用）
        self.acc_hist: Dict[str, Deque] = defaultdict(
            lambda: deque(maxlen=_ACC_WINDOW))

        self._dashboard = (Path(__file__).parent / "dashboard.html").read_text(
            encoding="utf-8")

        # 啟動背景輪詢
        self._task = asyncio.create_task(self._poll_loop())

    # ── 推論單張（阻塞，跑在 executor）─────────────────
    def _infer_frame(self, cam_id: str, img: np.ndarray) -> dict:
        h, w = img.shape[:2]

        # Traffic 偵測
        res = self.detector.predict(img, imgsz=self.imgsz, conf=self.conf,
                                    device=self.device, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None \
            else np.zeros((0, 4), np.float32)
        scores = res.boxes.conf.cpu().numpy() if res.boxes is not None \
            else np.zeros((0,), np.float32)

        # ROI 幾何過濾（只算主車道區）
        roi = get_roi(cam_id) if self.use_roi else None
        if roi is not None:
            boxes, scores = filter_by_roi(boxes, scores, roi, w, h)

        n = int(len(boxes))
        # 密度 = ROI（或全幅）內車框總面積佔比
        area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes)
        density = float(area / (w * h)) if (w * h) else 0.0

        # Accident 分類（整幅）→ P(accident)
        pred, conf = classify(self.classifier, img, self.acc_device)
        acc_prob = conf if ACC_CLASSES[pred] == "accident" else 1.0 - conf

        # 連續確認（門檻可調，治 domain-gap 誤報）
        self.acc_hist[cam_id].append(acc_prob >= self.accident_conf_th)
        hist = list(self.acc_hist[cam_id])
        is_acc = (len(hist) >= _ACC_CONSEC and all(hist[-_ACC_CONSEC:]))

        # 畫框（+ ROI 邊界）
        vis = img.copy()
        for (x1, y1, x2, y2), s in zip(boxes.astype(int), scores):
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 136), 2)
        if roi is not None:
            vis = draw_roi(vis, roi)
        cv2.putText(vis, f"vehicles: {n}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        if is_acc:
            cv2.putText(vis, f"ACCIDENT p={acc_prob:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 32, 255), 2)
        ok, enc = cv2.imencode(".jpg", vis)
        jpg = enc.tobytes() if ok else b""

        return {
            "jpg": jpg,
            "json": {
                "num_detections": n,
                "count_level": _level(n, _COUNT_BANDS),
                "density_level": _level(density, _DENSITY_BANDS),
                "is_accident": bool(is_acc),
                "accident_conf": round(float(acc_prob), 4),
                "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

    # ── 取得一幀：注入中讀車禍片段，否則抓即時串流（阻塞）──
    def _grab_frame(self, cam_id: str, stream_url: str):
        cap = self.inject_cap.get(cam_id)
        if cap is not None:
            ok, frame = cap.read()
            if not ok:                       # 片段播完 → 循環回第一幀
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if ok:
                cv2.putText(frame, "INJECTED CLIP", (10, frame.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return frame
        # 即時串流
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
                        None, self._grab_frame, cam.cctv_id, cam.stream_url)
                    if img is None:
                        continue
                    result = await loop.run_in_executor(
                        None, self._infer_frame, cam.cctv_id, img)
                    self.cache[cam.cctv_id] = result
                except Exception as e:
                    print(f"[serve] {cam.cctv_id} 失敗：{type(e).__name__}: {e}")
            # 補足輪詢間隔
            dt = time.time() - t0
            await asyncio.sleep(max(0.0, self.poll_interval - dt))

    # ── HTTP endpoints（對齊 UI 契約）─────────────────
    @app.get("/")
    def index(self):
        return HTMLResponse(self._dashboard)

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

    # ── 車禍片段注入（驗證用，手動觸發）──────────────
    @app.get("/clips")
    def list_clips(self):
        """列出可用車禍片段 + 目前注入狀態。"""
        clips = sorted(p.name for p in self.clip_dir.glob("*.mp4"))
        return JSONResponse({"clips": clips, "active": self.inject_clip})

    @app.post("/inject/{cam_id}")
    def inject(self, cam_id: str, clip: str = None):
        """讓某鏡頭改播車禍片段（clip 省略則隨機挑一支）。"""
        import random

        valid = {c.cctv_id for c in FOCUS_CAMERAS}
        if cam_id not in valid:
            return JSONResponse({"error": f"未知鏡頭 {cam_id}"}, status_code=404)

        clips = sorted(p.name for p in self.clip_dir.glob("*.mp4"))
        if not clips:
            return JSONResponse({"error": f"無片段於 {self.clip_dir}"},
                                status_code=404)
        name = clip or random.choice(clips)
        path = self.clip_dir / name
        if not path.exists():
            return JSONResponse({"error": f"找不到片段 {name}"}, status_code=404)

        old = self.inject_cap.pop(cam_id, None)
        if old is not None:
            old.release()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return JSONResponse({"error": f"無法開啟 {name}"}, status_code=500)
        self.inject_cap[cam_id] = cap
        self.inject_clip[cam_id] = name
        self.acc_hist[cam_id].clear()        # 重置連續確認
        return JSONResponse({"status": "injecting", "cam": cam_id, "clip": name})

    @app.post("/inject/{cam_id}/clear")
    def inject_clear(self, cam_id: str):
        """取消注入，恢復即時串流。"""
        cap = self.inject_cap.pop(cam_id, None)
        if cap is not None:
            cap.release()
        self.inject_clip.pop(cam_id, None)
        self.acc_hist[cam_id].clear()
        return JSONResponse({"status": "cleared", "cam": cam_id})


def build_app(args: dict = None):
    """Serve 應用工廠（給 serve.run / config 用）。"""
    args = args or {}
    return TrafficMonitor.bind(**args)
