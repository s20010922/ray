"""啟動高公局即時監控 Ray Serve（餵 smart-traffic-ui）。

  # 啟動（前景；Ctrl-C 結束）
  docker compose exec ray-head python scripts/serve_dashboard.py

  # 指定 model / 關 ROI / 調輪詢間隔
  docker compose exec ray-head python scripts/serve_dashboard.py \\
      --detector /workspace/ray_results/freeway_final/weights/best.pt \\
      --poll-interval 4.0 --no-roi

啟動後開瀏覽器：
  http://localhost:8000/                  → 監控儀表板
  http://localhost:8000/live_focus/<id>.jpg / .json
"""

import argparse
import time

from ray import serve

from src.core.cluster import init_ray
from src.serve.app import TrafficMonitor


def main():
    ap = argparse.ArgumentParser(description="高公局即時監控 Ray Serve")
    ap.add_argument("--detector",
                    default="/workspace/ray_results/freeway_final/weights/best.pt",
                    help="Traffic 偵測權重（ultralytics best.pt）")
    ap.add_argument("--accident-ckpt", default=None,
                    help="Accident 分類 checkpoint；省略則自動找最新")
    ap.add_argument("--poll-interval", type=float, default=4.0)
    ap.add_argument("--conf", type=float, default=0.4, help="偵測信心門檻")
    ap.add_argument("--accident-conf-th", type=float, default=0.97,
                    help="車禍報警門檻（拉高治 domain-gap 誤報；連續 3 幀超過才報）")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--no-roi", action="store_true", help="關閉 ROI 幾何過濾")
    ap.add_argument("--clip-dir",
                    default="/workspace/datasets/accident/video/accident",
                    help="車禍片段資料夾（注入驗證用）")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    init_ray()
    serve.start(http_options={"host": "0.0.0.0", "port": args.port})

    monitor = TrafficMonitor.bind(
        detector_weights=args.detector,
        accident_ckpt=args.accident_ckpt,
        poll_interval=args.poll_interval,
        conf=args.conf,
        imgsz=args.imgsz,
        use_roi=not args.no_roi,
        accident_conf_th=args.accident_conf_th,
        clip_dir=args.clip_dir,
    )
    serve.run(monitor, name="traffic_monitor", route_prefix="/")

    print(f"=== Serve 已啟動 ===")
    print(f"  儀表板: http://localhost:{args.port}/")
    print(f"  API   : http://localhost:{args.port}/live_focus/<cctv_id>.jpg / .json")
    print("  Ctrl-C 結束")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n停止 serve。")
        serve.shutdown()


if __name__ == "__main__":
    main()
