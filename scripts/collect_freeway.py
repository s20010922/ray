"""定時收集高公局 CCTV 影像（待命腳本）。

⚠️ 需要「能存取高公局的網路環境」。目前從本機/容器會被 WAF 擋 403，
   此腳本會持續重試並記錄失敗，等存取問題解決即可正常累積資料。

長駐執行：每隔 round_interval 秒對 focus 鏡頭抓一輪，跨時段累積
多樣化的 fine-tune 原始資料（多樣性靠「分時段多輪」而非「一次連抓」）。

背景啟動：
  docker compose exec -d ray-head python scripts/collect_freeway.py \
      --per-round 10 --frame-interval 8 --round-interval 1800

  （每 30 分鐘一輪、每輪每鏡頭抓 10 張、張間隔 8 秒）
"""

import argparse
import time
from pathlib import Path

from src.data.freeway.grabber import FOCUS_CAMERAS, grab_jpeg_frame, save_frame


def parse_args():
    p = argparse.ArgumentParser(description="定時收集高公局 CCTV 影像")
    p.add_argument("--out-root", default="/workspace/datasets/freeway_raw")
    p.add_argument("--per-round", type=int, default=10,
                   help="每輪每支鏡頭抓幾張")
    p.add_argument("--frame-interval", type=float, default=8.0,
                   help="同一輪內每張的間隔秒數")
    p.add_argument("--round-interval", type=float, default=1800.0,
                   help="每輪之間的間隔秒數（預設 30 分鐘）")
    p.add_argument("--target-per-camera", type=int, default=200,
                   help="每支鏡頭目標張數，到了就停（0=不限、持續跑）")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_root)
    counts = {cam.cctv_id: 0 for cam in FOCUS_CAMERAS}

    print(f"[collect] 啟動：{len(FOCUS_CAMERAS)} 鏡頭  "
          f"每輪{args.per_round}張/鏡頭  輪距{args.round_interval}s  "
          f"目標{args.target_per_camera}張/鏡頭  -> {out}")

    round_no = 0
    while True:
        round_no += 1
        for cam in FOCUS_CAMERAS:
            if args.target_per_camera and counts[cam.cctv_id] >= args.target_per_camera:
                continue
            ok = 0
            for _ in range(args.per_round):
                try:
                    jpg = grab_jpeg_frame(cam.stream_url)
                    save_frame(jpg, out / cam.cctv_id, cam, counts[cam.cctv_id])
                    counts[cam.cctv_id] += 1
                    ok += 1
                except Exception as e:
                    print(f"[collect] {cam.cctv_id} 失敗：{type(e).__name__}: {e}")
                    break
                time.sleep(args.frame_interval)
            print(f"[collect] 第{round_no}輪 {cam.cctv_id}：+{ok}  "
                  f"累計 {counts[cam.cctv_id]}/{args.target_per_camera}")

        # 全部達標就結束
        if args.target_per_camera and all(
                c >= args.target_per_camera for c in counts.values()):
            print(f"[collect] ✅ 全部達標：{counts}")
            break

        print(f"[collect] 第{round_no}輪結束，休息 {args.round_interval}s …")
        time.sleep(args.round_interval)


if __name__ == "__main__":
    main()
