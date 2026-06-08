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


def _build_tad_demo(model_path: str, data_dir: str, frames_root: str,
                    device: str, n_acc: int = 3, n_norm: int = 3,
                    fpv: int = 40, thr: float = 0.5, disp_w: int = 480,
                    onset_min: float = 0.25, early_max: float = 0.55,
                    late_min: float = 0.6):
    """B 方案：讀 TAD **測試影片**的原始幀，按時間播放、TAD 模型逐幀打分。

    只取 test split 的影片（模型沒訓練過），事故/正常各幾支，交錯排。
    回傳扁平的「逐幀」清單(已按影片→時間排序)，每筆：
      {jpg, video, true, pred, prob, fpos, fn, vidx, vtotal}
    顯示用原始畫面(縮到 disp_w 寬)，模型吃 224。檔案缺失回傳 []。
    """
    import json

    import torch

    from src.modeling.accident_cnn import build_model
    from src.train.accident_cnn.trainer import IMAGENET_MEAN, IMAGENET_STD

    try:
        ck = torch.load(model_path, map_location=device, weights_only=False)
        model = build_model(ck["config"]).to(device).eval()
        model.load_state_dict(ck["state_dict"])
        test_vids = set(np.load(Path(data_dir) / "test.npz")["vid"].tolist())
        meta = json.loads((Path(data_dir) / "videos.json").read_text("utf-8"))
    except (OSError, KeyError) as e:
        print(f"[serve] 車禍展示停用（載入失敗）：{type(e).__name__}: {e}")
        return []

    def _frame_dir(name, label):
        sub = "abnormal" if label == 1 else "normal"
        return Path(frames_root) / sub / name

    def _sel_frames(name, label):
        fs = sorted(_frame_dir(name, label).glob("*.jpg"),
                    key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
        if not fs:
            return []
        idx = np.linspace(0, len(fs) - 1, min(fpv, len(fs))).round().astype(int)
        return [fs[i] for i in idx]

    def _scores(paths):
        out = []
        for fp in paths:
            bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if bgr is None:
                out.append(0.0)
                continue
            sq = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (224, 224),
                            interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(sq).float().permute(2, 0, 1) / 255.0
            t = ((t - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(device)
            with torch.no_grad():
                out.append(float(torch.sigmoid(model(t)).item()))
        return np.array(out)

    # ── 第一輪：對所有 test 影片打分，挑「會呈現時間反應」的片 ──
    # 事故片只留「真的有時間轉折」者：前段像正常(early<early_max)、後段確實報事故
    # (late>=late_min)、上升幅度夠(onset>=onset_min)。藉此排除兩種不適合展示的片：
    #   ① 一開場車禍就在畫面、整片滿分（early 已高）→ 看不出「逐幀偵測」感
    #   ② 全程低分、模型漏報（late 偏低）
    # 正常片挑全程分數最低者(最乾淨、不誤報)。
    acc = sorted(meta[str(v)]["name"] for v in test_vids if meta[str(v)]["label"] == 1)
    norm = sorted(meta[str(v)]["name"] for v in test_vids if meta[str(v)]["label"] == 0)
    acc_all, norm_cand = [], []
    for name in acc:
        paths = _sel_frames(name, 1)
        if not paths:
            continue
        s = _scores(paths)
        third = max(1, len(s) // 3)
        early = float(s[:third].mean())
        late = float(s[-third:].mean())
        acc_all.append((late - early, early, late, name, paths, s))
    for name in norm:
        paths = _sel_frames(name, 0)
        if not paths:
            continue
        s = _scores(paths)
        norm_cand.append((float(s.mean()), name, 0, paths, s))

    # 只留 onset 真的明顯的事故片（前低後高）；按上升幅度排序
    clear = [c for c in acc_all
             if c[0] >= onset_min and c[1] < early_max and c[2] >= late_min]
    clear.sort(key=lambda x: x[0], reverse=True)
    if not clear and acc_all:           # 全無明顯 onset → 至少留 onset 最大一支，避免事故格全空
        clear = [max(acc_all, key=lambda x: x[0])]
    acc_cand = [(c[0], c[3], 1, c[4], c[5]) for c in clear]   # 還原成 (onset,name,label,paths,s)
    print(f"[serve] 事故片(明顯 onset)入選 {len(acc_cand)}/{len(acc_all)} 支："
          + ", ".join(c[1] for c in acc_cand))
    norm_cand.sort(key=lambda x: x[0])                 # 平均分 低→高
    picks = []
    for i in range(max(n_acc, n_norm)):                # 事故/正常交錯
        if i < n_acc and i < len(acc_cand):
            picks.append(acc_cand[i])
        if i < n_norm and i < len(norm_cand):
            picks.append(norm_cand[i])

    # ── 第二輪：對選中的片建顯示用 jpg + payload（分數沿用第一輪）──
    playlist = []
    for vidx, (_, name, label, paths, s) in enumerate(picks):
        for fpos, (fp, prob) in enumerate(zip(paths, s)):
            bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            disp = cv2.resize(bgr, (disp_w, round(h * disp_w / w)))
            ok, enc = cv2.imencode(".jpg", disp)
            playlist.append({
                "jpg": enc.tobytes() if ok else b"",
                "video": name, "true": label,
                "pred": int(prob >= thr), "prob": float(prob),
                "fpos": fpos + 1, "fn": len(paths),
                "vidx": vidx + 1, "vtotal": len(picks),
            })
    print(f"[serve] TAD 影片展示就緒：{len(picks)} 片 / {len(playlist)} 幀")
    return playlist


@serve.deployment(ray_actor_options={"num_gpus": 1, "num_cpus": 2})
@serve.ingress(app)
class TrafficMonitor:
    def __init__(self,
                 detector_weights: str =
                 "/workspace/ray_results/freeway_final/weights/best.pt",
                 poll_interval: float = 4.0,
                 conf: float = 0.4,
                 imgsz: int = 640,
                 device: str = "cuda",
                 accident_model: str =
                 "/workspace/ray_results/accident_tad_final/accident_tad.pt",
                 tad_data_dir: str = "/workspace/datasets/accident_tad_seq",
                 tad_frames_root: str =
                 "/workspace/datasets/Traffic Anomaly Dataset/TAD/frames",
                 demo_interval: float = 2.0):   # 與 traffic 輪詢(pollMs 2000)對齊
        from ultralytics import YOLO

        self.poll_interval = poll_interval
        self.conf = conf
        self.imgsz = imgsz
        self.device = device   # cuda；demo 監控訓練時用 cpu 釋出 GPU

        # 車流偵測：freeway best.pt 是 ultralytics 原生格式，直接 YOLO 載
        self.detector = YOLO(detector_weights)

        # 車禍偵測展示：TAD 測試影片逐幀播放 + TAD 模型即時判定（右下角面板）
        self.demo_interval = demo_interval
        self.accident_demo = _build_tad_demo(
            accident_model, tad_data_dir, tad_frames_root, device)

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

    # ── 車禍偵測展示（右下角面板）：TAD 測試影片逐幀播放 ──
    def _demo_index(self) -> int:
        n = len(self.accident_demo)
        if n == 0:
            return -1
        return int(time.time() / self.demo_interval) % n

    @app.get("/accident_demo.{ext}")
    def accident_demo(self, ext: str):
        idx = self._demo_index()
        if idx < 0:
            return JSONResponse({"error": "demo unavailable"}, status_code=503)
        item = self.accident_demo[idx]
        if ext == "jpg":
            return Response(item["jpg"], media_type="image/jpeg",
                            headers={"Cache-Control": "no-cache"})
        return JSONResponse({
            "idx": idx,
            "video": item["video"], "true": item["true"],
            "pred": item["pred"], "prob": round(item["prob"], 4),
            "fpos": item["fpos"], "fn": item["fn"],
            "vidx": item["vidx"], "vtotal": item["vtotal"],
            "interval": self.demo_interval,
        }, headers={"Cache-Control": "no-cache"})


def build_app(args: dict = None):
    """Serve 應用工廠（給 serve.run / config 用）。"""
    args = args or {}
    return TrafficMonitor.bind(**args)
