"""一鍵按順序跑完整流程，並即時把目前階段寫給 RAY MONITOR 點亮步驟圖。

階段順序（與 src/monitor/state.py 的 _PIPELINE_STAGES 一致）：
  1. synth             合成台灣車禍資料 → 併入 accident/train
  2. train_accident    Ray Train 訓練車禍分類（最佳超參已是預設）
  3. train_traffic     Ray Train 訓練車流偵測
  4. finetune_freeway  ultralytics 微調高公局偵測
  5. eval              三模型 held-out test 評估
  6. serve             啟動即時推論服務（背景常駐）

  # 容器內一鍵全跑（約 1 小時，單 GPU 依序訓練）
  docker compose exec ray-head python scripts/run_pipeline.py

  # 跳過合成資料（沿用現有 train）/ 不啟動 serve / 只從某階段起跑
  docker compose exec ray-head python scripts/run_pipeline.py --skip-synth --no-serve
  docker compose exec ray-head python scripts/run_pipeline.py --from train_traffic

每階段開始前會更新 /workspace/ray_results/_pipeline_state.json，RAY MONITOR
（:8501）右側的流程步驟圖即時亮起當前階段、標記已完成。
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# 與 src/monitor/state.py 的 _PIPELINE_STAGES 對齊
STAGES = [
    ("synth", "合成資料"),
    ("train_accident", "Accident 訓練"),
    ("train_traffic", "Traffic 訓練"),
    ("finetune_freeway", "Freeway 微調"),
    ("eval", "三模型評估"),
    ("serve", "上線 Serve"),
]

STATE_FILE = Path("/workspace/ray_results/_pipeline_state.json")
LOG_DIR = Path("/workspace/ray_results")
PY = sys.executable


def write_state(current, done, status="running"):
    STATE_FILE.write_text(json.dumps({
        "stages": [{"id": i, "label": l} for i, l in STAGES],
        "current": current,
        "done": done,
        "status": status,
        "updated": time.time(),
    }, ensure_ascii=False), encoding="utf-8")


def run(cmd, log_name):
    """前景執行一個步驟，輸出寫 log；回傳 returncode。"""
    log = LOG_DIR / log_name
    print(f"\n=== $ {' '.join(cmd)}  （log: {log}）", flush=True)
    with open(log, "w", encoding="utf-8") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def step_synth(args):
    return run([PY, "scripts/synth_accident.py", "--merge-train",
                "--n-accident", str(args.synth_n),
                "--n-normal", str(args.synth_n)], "_pipe_synth.log")


def step_train_accident(args):
    return run([PY, "scripts/train_accident.py",
                "--epochs", str(args.accident_epochs)], "_pipe_train_accident.log")


def step_train_traffic(args):
    return run([PY, "scripts/train_traffic.py",
                "--epochs", str(args.traffic_epochs)], "_pipe_train_traffic.log")


def step_finetune_freeway(args):
    return run([PY, "scripts/finetune_freeway.py",
                "--epochs", str(args.freeway_epochs)], "_pipe_finetune_freeway.log")


def step_eval(args):
    rc = 0
    for s in ("eval_accident", "eval_traffic", "eval_freeway"):
        rc |= run([PY, f"scripts/{s}.py"], f"_pipe_{s}.log")
    return rc


def step_serve(args):
    """背景啟動 serve（不阻塞流程）。"""
    log = LOG_DIR / "_pipe_serve.log"
    print(f"\n=== 背景啟動 serve（log: {log}）", flush=True)
    with open(log, "w", encoding="utf-8") as f:
        subprocess.Popen(
            [PY, "scripts/serve_dashboard.py", "--accident-conf-th", "0.97",
             "--no-roi", "--poll-interval", "2.0"],
            stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(3)
    return 0


STEP_FUNCS = {
    "synth": step_synth,
    "train_accident": step_train_accident,
    "train_traffic": step_train_traffic,
    "finetune_freeway": step_finetune_freeway,
    "eval": step_eval,
    "serve": step_serve,
}


def main():
    ap = argparse.ArgumentParser(description="一鍵按順序跑完整流程")
    ap.add_argument("--skip-synth", action="store_true", help="跳過合成資料步驟")
    ap.add_argument("--no-serve", action="store_true", help="最後不啟動 serve")
    ap.add_argument("--from", dest="from_stage", default=None,
                    help="從指定階段起跑（synth/train_accident/.../serve）")
    ap.add_argument("--synth-n", type=int, default=600)
    ap.add_argument("--accident-epochs", type=int, default=50)
    ap.add_argument("--traffic-epochs", type=int, default=30)
    ap.add_argument("--freeway-epochs", type=int, default=100)
    args = ap.parse_args()

    order = [sid for sid, _ in STAGES]
    skip = set()
    if args.skip_synth:
        skip.add("synth")
    if args.no_serve:
        skip.add("serve")

    start_idx = order.index(args.from_stage) if args.from_stage in order else 0
    done = order[:start_idx]                 # 起跑點之前視為已完成

    print(f"=== 流程開始：{order[start_idx:]}（跳過 {sorted(skip) or '無'}）", flush=True)
    t0 = time.time()
    for sid in order[start_idx:]:
        if sid in skip:
            done.append(sid)
            continue
        write_state(sid, done, "running")
        ts = time.time()
        rc = STEP_FUNCS[sid](args)
        dt = time.time() - ts
        if rc != 0:
            print(f"\n✗ 階段 [{sid}] 失敗（rc={rc}，{dt:.0f}s）。流程中止。", flush=True)
            write_state(sid, done, "error")
            sys.exit(1)
        print(f"✓ 階段 [{sid}] 完成（{dt:.0f}s）", flush=True)
        done.append(sid)

    write_state(None, done, "done")
    print(f"\n=== 全流程完成，共 {time.time() - t0:.0f}s ===", flush=True)
    print("  Serve : http://localhost:8000/    MONITOR: http://localhost:8501/")


if __name__ == "__main__":
    main()
