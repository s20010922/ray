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
import os
import signal
import time

from ray import serve

from src.core.cluster import init_ray
from src.serve.app import TrafficMonitor


def _kill_stale_serve_drivers():
    """清掉前次 session 殘留的 serve_dashboard driver。

    serve_dashboard.py 末尾有 while-sleep 迴圈常駐，`serve shutdown` 只會移除
    Ray Serve 部署、不會殺掉這些 driver 行程。殘留多個 driver 會互搶 serve.run
    導致新 serve 起不來。啟動時先掃 /proc 殺掉其它同名行程（排除自己與父行程）。
    """
    me, parent = os.getpid(), os.getppid()
    killed = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        p = int(pid)
        if p in (me, parent):
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\x00", b" ").decode("utf-8", "ignore")
        except OSError:
            continue
        if "serve_dashboard.py" in cmd:
            try:
                os.kill(p, signal.SIGKILL)
                killed.append(p)
            except OSError:
                pass
    if killed:
        print(f"[serve] 清掉殘留 driver：{killed}")
        time.sleep(2)


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
                    help="整幅分類器輔助門檻（domain-gap 不可靠，僅報告用）")
    ap.add_argument("--stall-frames", type=int, default=3,
                    help="車道內車輛連續靜止幾幀視為事故（poll 2s 時 3≈6 秒）")
    ap.add_argument("--move-frac", type=float, default=0.15,
                    help="中心位移 < move-frac × box 對角線 → 判定未移動")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--no-roi", action="store_true", help="關閉 ROI 幾何過濾")
    ap.add_argument("--clip-dir",
                    default="/workspace/datasets/accident/video/accident",
                    help="車禍片段資料夾（注入驗證用）")
    ap.add_argument("--no-gpu", action="store_true",
                    help="CPU 推論並釋出 GPU（demo 時邊訓練邊看 RAY MONITOR 用）")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    _kill_stale_serve_drivers()          # 先清殘留 driver，避免互搶
    init_ray()
    try:
        serve.shutdown()                 # 清掉任何既有部署，確保乾淨重啟
    except Exception:
        pass
    serve.start(http_options={"host": "0.0.0.0", "port": args.port})

    gpus = 0 if args.no_gpu else 1
    device = "cpu" if args.no_gpu else "cuda"
    monitor = TrafficMonitor.options(
        ray_actor_options={"num_gpus": gpus, "num_cpus": 2}
    ).bind(
        detector_weights=args.detector,
        accident_ckpt=args.accident_ckpt,
        poll_interval=args.poll_interval,
        conf=args.conf,
        imgsz=args.imgsz,
        use_roi=not args.no_roi,
        accident_conf_th=args.accident_conf_th,
        stall_frames=args.stall_frames,
        move_frac=args.move_frac,
        clip_dir=args.clip_dir,
        device=device,
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
