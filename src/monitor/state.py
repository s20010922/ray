"""Ray 叢集即時狀態查詢（給獨立監控頁用，不依賴 Ray Serve）。

cluster_state()    → 節點清單 + 每節點負載（實體 + Ray 邏輯任務數）+ 叢集資源
components_state()  → Data / Train / Tune / Serve 各元件即時活動

所有資訊都從「外部觀察」叢集而來（ray.nodes / 狀態 API / dashboard），
因此監控程式本身只是個輕量 driver，與被監控的 serve / 訓練任務無關。
"""

import glob
import json
import os
import subprocess
import time
import urllib.request
from collections import Counter, defaultdict, deque

import ray

# Ray Data / Serve 沒有 epoch 指標檔，改維護「滾動歷史」：每約 2s 記一筆時間戳，
# 讓卡片 log 隨時間捲動且不消失（歷史保留在記憶體，閒置時凍結顯示最後狀態）。
_DATA_HIST = deque(maxlen=14)
_DATA_HIST_T = [0.0]


def _data_log_rolling(data_ops):
    """更新並回傳 Ray Data 的滾動歷史 log。"""
    now = time.time()
    if data_ops and now - _DATA_HIST_T[0] >= 1.8:
        _DATA_HIST_T[0] = now
        summ = " · ".join(
            f"{k.split('(')[0].split('.')[0]}×{v}"
            for k, v in data_ops.most_common(3))
        _DATA_HIST.append(f"{time.strftime('%H:%M:%S')}  {summ}")
    return list(_DATA_HIST)


def _serve_camera_log(active):
    """Serve log：直接抓 serve 的 5 鏡頭即時指標（車數/密度/有無事故）。

    best-effort——serve 沒開或某鏡頭尚無資料就略過；只在 serve 有 replica 時嘗試，
    逾時設短避免拖慢監控。比心跳更有資訊：直接看到推論服務每支鏡頭在偵測什麼。
    """
    if not active:
        return []
    try:
        from src.data.freeway.grabber import FOCUS_CAMERAS
    except Exception:
        return []
    out = []
    for cam in FOCUS_CAMERAS:
        try:
            url = f"http://localhost:8000/live_focus/{cam.cctv_id}.json"
            with urllib.request.urlopen(url, timeout=0.6) as r:
                d = json.load(r)
            flag = "⚠事故" if d.get("is_accident") else "正常"
            out.append(f"{cam.cctv_id[:20]}  車{d.get('num_detections', 0)} · "
                       f"密度{d.get('density_level', '?')} · {flag}")
        except Exception:
            continue
    return out


def _gpu_util_smi():
    """直接用 nvidia-smi 取 GPU 即時利用率（%）。

    Ray Dashboard 的 utilizationGpu 在 docker 環境更新極慢／會卡住（實測訓練
    GPU 已 86% 仍回報 34%）；改用 nvidia-smi 取即時值。head 是唯一 GPU 節點、
    監控也跑在 head 上，故本機 smi 即代表 head GPU。
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
        vals = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
        return vals[0] if vals else None
    except Exception:
        return None


def _dashboard_nodes():
    """抓 Ray Dashboard 的每節點實體負載（CPU%/GPU 利用率/記憶體/object store）。"""
    try:
        with urllib.request.urlopen(
                "http://localhost:8265/nodes?view=summary", timeout=3) as r:
            return {n["raylet"]["nodeId"]: n
                    for n in json.load(r)["data"]["summary"]}
    except Exception as e:
        print(f"[monitor] dashboard 查詢失敗：{type(e).__name__}: {e}")
        return {}


def _running_tasks_per_node():
    """每節點正在跑的 task 數（Ray 邏輯負載；Ray Data 階段會即時跳動）。"""
    per_node = Counter()
    try:
        from ray.util.state import list_tasks
        for t in list_tasks(filters=[("state", "=", "RUNNING")], limit=4000):
            if t.node_id:
                per_node[t.node_id] += 1
    except Exception as e:
        print(f"[monitor] task 查詢失敗：{type(e).__name__}: {e}")
    return per_node


def _recent_metrics(experiments, limit=12):
    """某類實驗最新一次 run 的近幾筆 epoch 報告（給卡片中間當即時 log）。

    Ray Train 每次 report 會往 run 目錄的 result.json 追加一行 JSON；取
    mtime 最新的 result.json 末幾行，解析成可讀的指標行（accident 報 val_acc、
    traffic 報 val_loss）。
    """
    files = []
    for e in experiments:
        files += glob.glob(
            f"/workspace/ray_results/{e}/**/result.json", recursive=True)
    if not files:
        return []
    try:
        newest = max(files, key=os.path.getmtime)
        with open(newest, encoding="utf-8") as f:
            rows = [r for r in f.read().splitlines() if r.strip()]
    except OSError:
        return []

    out = []
    for r in rows[-limit:]:
        try:
            d = json.loads(r)
        except json.JSONDecodeError:
            continue
        ep = d.get("epoch")
        seg = [f"ep{ep:>2}" if ep is not None
               else f"it{d.get('training_iteration', '?')}"]
        if d.get("train_loss") is not None:
            seg.append(f"loss {d['train_loss']:.3f}")
        if d.get("val_acc") is not None:
            seg.append(f"val_acc {d['val_acc']:.3f}")
        elif d.get("val_loss") is not None:
            seg.append(f"val_loss {d['val_loss']:.2f}")
        out.append("  ".join(seg))
    return out


_PIPE_STATE_FILE = "/workspace/ray_results/_pipeline_state.json"

# 完整流程的標準階段（run_pipeline.py 與監控步驟圖共用此順序）
_PIPELINE_STAGES = [
    ("synth", "合成資料"),
    ("train_accident", "Accident 訓練"),
    ("train_traffic", "Traffic 訓練"),
    ("finetune_freeway", "Freeway 微調"),
    ("eval", "三模型評估"),
    ("serve", "上線 Serve"),
]


def pipeline_state():
    """流程步驟圖狀態：優先用 run_pipeline.py 寫的 state file；否則由執行中 job 推斷。

    回傳 {stages:[{id,label}], current, done:[id], status}。
    """
    default_stages = [{"id": i, "label": l} for i, l in _PIPELINE_STAGES]

    # 先讀 run_pipeline.py 寫的 state file（可能不存在）
    st = None
    try:
        if os.path.exists(_PIPE_STATE_FILE):
            with open(_PIPE_STATE_FILE, encoding="utf-8") as f:
                st = json.load(f)
    except Exception:
        st = None

    # 1) 進行中的 pipeline（近 120s 內更新）→ 最優先
    if st and time.time() - st.get("updated", 0) < 120:
        return {
            "stages": st.get("stages") or default_stages,
            "current": st.get("current"),
            "done": st.get("done", []),
            "status": st.get("status", "running"),
        }

    # 2) 由執行中 job 推斷（單獨跑某腳本時也能亮對應階段）
    ids = [i for i, _ in _PIPELINE_STAGES]
    kind, label = _active_job()
    job_to_stage = {
        ("train", "Accident 分類"): "train_accident",
        ("train", "Traffic 偵測"): "train_traffic",
        ("finetune", "Traffic（Freeway 微調）"): "finetune_freeway",
    }
    cur = job_to_stage.get((kind, label))
    if cur is None and _serve_alive():
        cur = "serve"
    if cur:
        return {"stages": default_stages, "current": cur,
                "done": ids[:ids.index(cur)], "status": "running"}

    # 3) 沒有進行中的工作 → 顯示上次 pipeline 的完成狀態（黏著 finish 提示）
    if st and st.get("status") == "done":
        return {"stages": st.get("stages") or default_stages,
                "current": None, "done": st.get("done", []), "status": "done"}

    return {"stages": default_stages, "current": None, "done": [], "status": "idle"}


def _serve_alive():
    """是否有 Ray Serve replica 在跑（推斷 serve 階段用）。"""
    try:
        from ray.util.state import list_actors
        for a in list_actors(filters=[("state", "=", "ALIVE")], limit=500):
            if "servereplica" in (a.class_name or "").lower():
                return True
    except Exception:
        pass
    return False


def _active_job():
    """從執行中 job 的進入點判斷 (kind, label)。

    kind ∈ {'train', 'tune', 'finetune', ''}；label 是模型/案別中文。
    關鍵：tune_* 內部是「Ray Tune 包 TorchTrainer」，會產生 RayTrainWorker
    actor。若只看 actor 會把 Tune 誤認成 Train。改以 job 進入點為準歸屬：
    tune_* → Tune（不重複點亮 Train）；train_* → Train。

    Freeway 不是獨立案，是 Traffic 案的 Freeway 微調版，故 label 收進 Traffic。
    """
    try:
        from ray.util.state import list_jobs
        for j in list_jobs(limit=200):
            if "RUNNING" not in str(getattr(j, "status", "")).upper():
                continue
            ep = (getattr(j, "entrypoint", "") or "").lower()
            if "train_accident" in ep:
                return ("train", "Accident 分類")
            if "train_traffic" in ep:
                return ("train", "Traffic 偵測")
            if "tune_accident" in ep:
                return ("tune", "Accident 分類")
            if "tune_freeway" in ep:
                return ("tune", "Traffic（Freeway 微調）")
            if "finetune_freeway" in ep:
                return ("finetune", "Traffic（Freeway 微調）")
    except Exception as e:
        print(f"[monitor] job 查詢失敗：{type(e).__name__}: {e}")
    return ("", "")


def cluster_state():
    """節點清單 + 每節點負載 + 叢集資源總量。"""
    dash = _dashboard_nodes()
    tasks_per_node = _running_tasks_per_node()
    gpu_smi = _gpu_util_smi()        # 即時 GPU 利用率（取代 dashboard 的卡住值）

    nodes = []
    for n in ray.nodes():
        res = n.get("Resources", {})
        gpu = res.get("GPU", 0)
        cpu = res.get("CPU", 0)
        nid = n.get("NodeID", "")
        dn = dash.get(nid, {})
        rl = dn.get("raylet", {})
        gpus = dn.get("gpus") or []
        # GPU 節點（head）用 nvidia-smi 即時值；dashboard 的 utilizationGpu 會卡住
        if gpu and gpu_smi is not None:
            gpu_pct = gpu_smi
        else:
            gpu_pct = gpus[0].get("utilizationGpu", 0) if gpus else 0
        mem = dn.get("mem") or []
        obj_used = rl.get("objectStoreUsedMemory", 0)
        obj_avail = rl.get("objectStoreAvailableMemory", 0)
        obj_tot = obj_used + obj_avail
        ray_tasks = int(tasks_per_node.get(nid, 0))
        nodes.append({
            "id": nid[:12],
            "alive": bool(n.get("Alive", False)),
            "address": n.get("NodeManagerAddress", ""),
            "cpu": cpu,
            "gpu": gpu,
            "cpu_pct": round(dn.get("cpu", 0), 1),
            "gpu_pct": round(gpu_pct or 0, 1),
            "mem_pct": round(mem[2], 1) if len(mem) > 2 else 0,
            "mem_gb": round(res.get("memory", 0) / 2**30, 1),
            "obj_pct": round(obj_used / obj_tot * 100, 1) if obj_tot else 0,
            "obj_gb": round(obj_tot / 2**30, 1),
            "ray_tasks": ray_tasks,                                 # 邏輯負載
            "ray_pct": round(min(ray_tasks / cpu * 100, 100), 1) if cpu else 0,
            "role": "Head" if rl.get("isHeadNode", bool(gpu)) else "Worker",
        })
    nodes.sort(key=lambda x: (x["role"] != "Head", x["id"]))

    total = ray.cluster_resources()
    avail = ray.available_resources()
    obj_tot = total.get("object_store_memory", 0)
    obj_used = obj_tot - avail.get("object_store_memory", 0)
    return {
        "node_count": sum(1 for x in nodes if x["alive"]),
        "nodes": nodes,
        "cpu_total": total.get("CPU", 0),
        "cpu_used": round(total.get("CPU", 0) - avail.get("CPU", 0), 1),
        "gpu_total": total.get("GPU", 0),
        "gpu_used": round(total.get("GPU", 0) - avail.get("GPU", 0), 2),
        "obj_total_gb": round(obj_tot / 2**30, 1),
        "obj_used_gb": round(obj_used / 2**30, 2),
        "obj_pct": round(obj_used / obj_tot * 100, 1) if obj_tot else 0,
    }


def components_state():
    """各 Ray 元件即時活動（從狀態 API 觀察 task / actor）。"""
    data_tasks = train_n = tune_n = serve_n = 0
    data_ops = Counter()
    try:
        from ray.util.state import list_actors, list_tasks
        for t in list_tasks(filters=[("state", "=", "RUNNING")], limit=4000):
            raw = t.name or t.func_or_class_name or ""
            nm = raw.lower()
            if any(k in nm for k in ("mapbatches", "_preprocess", "readrange",
                                      "split", "map(", "streaming")):
                data_tasks += 1
                data_ops[raw.split("->")[-1].strip()[:32] or "data task"] += 1
        for a in list_actors(filters=[("state", "=", "ALIVE")], limit=500):
            nm = (a.class_name or "").lower()
            if "trainworker" in nm or "torchtrainer" in nm:
                train_n += 1
            elif "tune" in nm or "trainable" in nm or "implicitfunc" in nm:
                tune_n += 1
            elif "servereplica" in nm:
                serve_n += 1
    except Exception as e:
        print(f"[monitor] 元件查詢失敗：{type(e).__name__}: {e}")

    kind, case = _active_job()           # 以執行中 job 歸屬（tune 內含 train）
    tag = f"（{case}）" if case else ""
    # tune_* 的 trial 內部用 TorchTrainer worker；歸給 Tune，不重複點亮 Train
    train_active = train_n > 0 and kind != "tune"
    tune_active = kind == "tune" or tune_n > 0

    def comp(active, doing, detail, log=None):
        return {"active": active, "doing": doing, "detail": detail,
                "log": log or []}

    data_log = _data_log_rolling(data_ops)          # 滾動歷史（時間戳 + 運算元）
    train_log = _recent_metrics(["accident", "traffic"])
    tune_log = _recent_metrics(["accident_tune", "freeway_tune"])

    return {
        "data": comp(
            data_tasks > 0,
            f"處理中 {data_tasks} 個 batch task{tag}" if data_tasks
            else "閒置（訓練時啟動串流前處理）",
            "解碼 → 劣化增強 → resize → 多節點平行前處理",
            data_log),
        "train": comp(
            train_active,
            f"{train_n} 個 worker 訓練中{tag}" if train_active
            else "閒置（待 train_accident / train_traffic）",
            f"TorchTrainer · 目前：{case}（在 head GPU 上）"
            if kind == "train" and case
            else "TorchTrainer：Accident 分類、Traffic 偵測（跑在 head GPU）",
            train_log),
        "tune": comp(
            tune_active,
            f"搜尋中{tag} · trial 訓練中" if tune_active
            else "閒置（待 tune_accident / tune_freeway）",
            f"ASHA 超參搜尋 · 目前：{case}（內部用 TorchTrainer）"
            if tune_active and case
            else "ASHA 超參搜尋：兩案皆可（Traffic 的 Freeway 微調、Accident 分類）",
            tune_log),
        "serve": comp(
            serve_n > 0,
            f"{serve_n} 個 replica 運行中" if serve_n
            else "未啟動（serve_dashboard.py）",
            "即時推論服務：5 鏡頭車流／車禍偵測",
            _serve_camera_log(serve_n > 0)),
    }
