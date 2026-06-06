"""獨立 Ray 叢集監控網頁（不依賴 Ray Serve）。

只當「觀察者」連上叢集，查節點負載與各元件活動，提供總覽頁。
因此從零開始（叢集一啟動）就能看，與 serve／訓練是否在跑無關。

  # 叢集起來後即可開（不需先跑 serve）
  docker compose exec ray-head python scripts/monitor.py
  # 瀏覽器：http://localhost:8501/

對照：
  scripts/serve_dashboard.py  → 即時推論服務（5 鏡頭畫面，:8000）
  scripts/monitor.py          → 叢集監控總覽（本檔，:8501）
"""

import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from src.core.cluster import init_ray
from src.monitor.state import cluster_state, components_state

app = FastAPI()
_HTML = (Path(__file__).resolve().parent.parent
         / "src" / "monitor" / "overview.html").read_text(encoding="utf-8")
_NOCACHE = {"Cache-Control": "no-cache"}


@app.get("/")
def index():
    return HTMLResponse(_HTML)


@app.get("/cluster.json")
def cluster():
    return JSONResponse(cluster_state(), headers=_NOCACHE)


@app.get("/components.json")
def components():
    return JSONResponse(components_state(), headers=_NOCACHE)


def main():
    ap = argparse.ArgumentParser(description="Ray 叢集監控總覽（獨立服務）")
    ap.add_argument("--port", type=int, default=8501)
    args = ap.parse_args()

    init_ray()   # 以輕量 driver 接上現有叢集（只做唯讀狀態查詢）
    print(f"=== RAY MONITOR 已啟動 ===")
    print(f"  總覽頁: http://localhost:{args.port}/")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
