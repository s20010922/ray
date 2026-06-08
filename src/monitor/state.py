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


def _friendly_op(name: str) -> str:
    """Ray Data 運算元名稱 → 白話。"""
    n = name.lower()
    if "map" in n or "_process" in n or "batch" in n:
        return "前處理影像"
    if "read" in n or "from_items" in n or "fromitems" in n:
        return "讀取影像"
    if "split" in n:
        return "切分資料"
    if "write" in n:
        return "寫出資料"
    return name.split("(")[0].split(".")[0]


def _data_log_rolling(data_ops):
    """更新並回傳 Ray Data 的滾動歷史 log（白話 + 並行任務數）。"""
    now = time.time()
    if data_ops and now - _DATA_HIST_T[0] >= 0.8:
        _DATA_HIST_T[0] = now
        summ = " · ".join(
            f"{_friendly_op(k)} {v} 個並行"
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
            out.append(f"{cam.cctv_id[:20]}  車{d.get('num_detections', 0)} · "
                       f"數量{d.get('count_level', '?')} · "
                       f"密度{d.get('density_level', '?')}")
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


def _gpu_mem_smi():
    """nvidia-smi 即時 GPU 記憶體 (已用GiB, 總量GiB)；dashboard 的 mem 在 docker
    會卡住（實測 4 次抓全一樣），故跟 util 一樣直讀 smi。訓練時 VRAM 才是真正
    浮動的記憶體（主機 RAM 載完資料後幾乎不動）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
        vals = [int(x) for x in out.stdout.replace(",", " ").split()
                if x.strip().isdigit()]
        if len(vals) >= 2:
            return round(vals[0] / 1024, 1), round(vals[1] / 1024, 1)
    except Exception:
        pass
    return None, None


def _cpu_pct_live():
    """主機實體 CPU 使用率(%)；dashboard 的 cpu 在 docker 會卡住（實測凍 3.2%），
    改用 psutil 直讀。GPU tune 的 trial 只佔 GPU 不佔 CPU，Ray 邏輯 CPU=0 會誤導，
    故 cluster 摘要改顯示實體使用率。"""
    try:
        import psutil
        return psutil.cpu_percent(interval=0.2)
    except Exception:
        return None


@ray.remote(num_cpus=0)
def _node_probe():
    """在「所在節點」上實量，回報該容器自己的 CPU% 與記憶體用量。

    記憶體讀 **cgroup**（`memory.current` 扣掉可回收的 file cache）——這是「這個
    容器吃了多少」的隔離計量，head/worker 各自不同；不像 psutil.virtual_memory()
    會讀到整台共享 VM 而三節點相同。讀不到 cgroup 才退回 psutil。分母（總量）由
    呼叫端用 Ray 的 --memory 預算（12G/4G/4G），故這裡只回已用量。
    """
    import psutil

    def _cgroup_used():
        # cgroup v2：memory.current - inactive_file；v1：usage_in_bytes - total_inactive_file
        for cur_p, stat_p, key in (
            ("/sys/fs/cgroup/memory.current",
             "/sys/fs/cgroup/memory.stat", "inactive_file "),
            ("/sys/fs/cgroup/memory/memory.usage_in_bytes",
             "/sys/fs/cgroup/memory/memory.stat", "total_inactive_file ")):
            try:
                cur = int(open(cur_p).read().strip())
            except OSError:
                continue
            cache = 0
            try:
                for ln in open(stat_p):
                    if ln.startswith(key):
                        cache = int(ln.split()[1])
                        break
            except OSError:
                pass
            return max(cur - cache, 0)
        return None

    used = _cgroup_used()
    if used is None:
        used = psutil.virtual_memory().used      # fallback：讀不到 cgroup
    return {
        "cpu_pct": round(psutil.cpu_percent(interval=0.15), 1),
        "mem_used_gb": round(used / 2**30, 2),
    }


def _node_probes():
    """對每個存活節點派 _node_probe（NodeAffinity 釘節點），收齊真值。

    回傳 {node_id: {cpu_pct, mem_used_gb, mem_total_gb, mem_pct}}；任何節點失敗
    或逾時就略過該節點，呼叫端 fallback 回 dashboard 舊值。
    """
    try:
        from ray.util.scheduling_strategies import (
            NodeAffinitySchedulingStrategy)
    except Exception:
        return {}
    ref_to_nid = {}
    for n in ray.nodes():
        if not n.get("Alive"):
            continue
        nid = n.get("NodeID", "")
        try:
            ref = _node_probe.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    nid, soft=False)).remote()
            ref_to_nid[ref] = nid
        except Exception:
            continue
    out = {}
    if not ref_to_nid:
        return out
    try:
        ready, _ = ray.wait(list(ref_to_nid), num_returns=len(ref_to_nid),
                            timeout=2.5)
        for ref in ready:
            try:
                out[ref_to_nid[ref]] = ray.get(ref)
            except Exception:
                continue
    except Exception:
        pass
    return out


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


def _recent_metrics(experiments, limit=12, base="/workspace/ray_results"):
    """某類實驗最新一次 run 的近幾筆 epoch 報告（給卡片中間當即時 log）。

    Ray Train / ultralytics Tune 每 epoch 會往 run 目錄的 result.json 追加一行
    JSON；取 mtime 最新的 result.json 末幾行，解析成可讀的指標行。experiments
    可含萬用字元（如 "tune*"），base 指定根目錄（Tune 在 runs/detect 下）。
    """
    files = []
    for e in experiments:
        files += glob.glob(f"{base}/{e}/**/result.json", recursive=True)
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
        # mAP：Ray Train 報 mAP50/mAP50_95；ultralytics Tune 報 metrics/mAP50(B)
        m50 = d.get("mAP50", d.get("metrics/mAP50(B)"))
        m5095 = d.get("mAP50_95", d.get("metrics/mAP50-95(B)"))
        if m50 is not None:
            seg.append(f"mAP50 {m50:.3f}")
        if m5095 is not None:
            seg.append(f"mAP50-95 {m5095:.3f}")
        # 事故時序模型：Ray Tune/Train 報 ap/recall/f1
        if d.get("ap") is not None:
            seg.append(f"AP {d['ap']:.3f}")
        if d.get("recall") is not None:
            seg.append(f"recall {d['recall']:.2f}")
        if d.get("f1") is not None:
            seg.append(f"f1 {d['f1']:.2f}")
        if d.get("val_acc") is not None:
            seg.append(f"val_acc {d['val_acc']:.3f}")
        elif d.get("val_loss") is not None:
            seg.append(f"val_loss {d['val_loss']:.2f}")
        out.append("  ".join(seg))
    return out


def _bar(pct, width=16):
    """文字進度條，如 ██████░░░░░░░░░░。"""
    fill = int(round(width * pct / 100))
    return "█" * fill + "░" * (width - fill)


_DATA_DIRS = [
    "/workspace/datasets/freeway_prepared",
    "/workspace/datasets/accident_tad_seq",
    "/workspace/datasets/accident_cnn_seq",
]


def _data_progress():
    """Ray Data 前處理進度 (done, total, pct)；無進行中則 None。

    pipeline 處理時往輸出目錄寫 _total.txt（總項數）與 _progress.txt（已完成數，
    每幾百項刷一次）。只在「_progress 仍在更新（mtime<30s）且 done<total」時回傳，
    完成或過期就回 None 讓卡片轉閒置。
    """
    for d in _DATA_DIRS:
        tp = os.path.join(d, "_total.txt")
        pp = os.path.join(d, "_progress.txt")
        if not (os.path.exists(tp) and os.path.exists(pp)):
            continue
        try:
            if time.time() - os.path.getmtime(pp) > 30:
                continue                      # 太久沒更新 → 視為非進行中
            total = int(open(tp, encoding="utf-8").read().strip())
            done = int(open(pp, encoding="utf-8").read().strip())
        except Exception:
            continue
        if total > 0 and done < total:
            return done, total, round(100 * done / total)
    return None


_PIPE_STATE_FILE = "/workspace/ray_results/_pipeline_state.json"

# Ray 標準四階段流程（freeway：Ray Data → Tune → Train → Serve）
_PIPELINE_STAGES = [
    ("prepare", "① Ray Data 前處理"),
    ("tune", "② Ray Tune 超參搜尋"),
    ("train", "③ Ray Train 正式訓練"),
    ("serve", "④ Ray Serve 上線"),
]


def _exists(p):
    try:
        return os.path.exists(p)
    except Exception:
        return False


def pipeline_state():
    """兩條流程狀態：車流（freeway，已完成可上線）與車禍（accident）。

    回傳 {pipelines:[{name, stages, current, done:[id], ready, status}, ...]}。
    """
    stages = [{"id": i, "label": l} for i, l in _PIPELINE_STAGES]
    ids = [i for i, _ in _PIPELINE_STAGES]
    job_to_stage = {"data": "prepare", "tune": "tune", "train": "train"}
    kind, case = _active_job()
    cur_stage = job_to_stage.get(kind)

    # ── 車流（Freeway）：依產物與執行中任務推斷（可清空，不再寫死）──
    fw_serve = _serve_alive()
    fw_data = _exists("/workspace/datasets/freeway_prepared/dataset.yaml")
    fw_model = _exists("/workspace/ray_results/freeway_final/weights/best.pt")
    fw_done = ["prepare"] if fw_data else []
    if fw_model:
        fw_done = ["prepare", "tune", "train"]
    traffic = {"name": "車流偵測（Freeway）", "stages": stages,
               "current": None, "ready": None,
               "done": fw_done, "status": "idle"}
    if fw_serve and fw_model:
        traffic.update(done=["prepare", "tune", "train", "serve"], status="done")
    elif fw_model:
        traffic.update(ready="serve", status="ready")   # 訓練完成、待上線
    elif fw_done:
        traffic["status"] = "partial"
    if case == "freeway" and cur_stage:     # freeway 任務正在跑 → 反映進度
        traffic.update(current=cur_stage, ready=None, status="running",
                       done=ids[:ids.index(cur_stage)])

    # ── 車禍（Accident）：依產物與執行中任務推斷（涵蓋 TAD / CNN / 軌跡）──
    acc_data = any(_exists(p) for p in (
        "/workspace/datasets/accident_tad_seq/train.npz",
        "/workspace/datasets/accident_cnn_seq/train.npz",
        "/workspace/datasets/accident_seq/train.npz"))
    acc_model = any(_exists(p) for p in (
        "/workspace/ray_results/accident_tad_final/accident_tad.pt",
        "/workspace/ray_results/accident_cnn_final/accident_cnn.pt",
        "/workspace/ray_results/accident_final/accident_seq.pt"))
    acc_done = []
    if acc_data:
        acc_done.append("prepare")
    if acc_model:
        acc_done = ["prepare", "tune", "train"]
    # 事故模型部署在 serve replica（右下角 TAD 影片 demo）→ serve 上線
    acc_serve = _serve_alive() and acc_model

    acc_cur, acc_ready, acc_status = None, None, "idle"
    if case == "accident" and cur_stage:
        acc_cur = cur_stage
        acc_done = ids[:ids.index(cur_stage)]
        acc_status = "running"
    elif acc_serve:
        acc_done = ["prepare", "tune", "train", "serve"]
        acc_status = "done"
    elif "train" in acc_done:
        acc_ready = "serve"             # 訓練完成、待上線
        acc_status = "ready"
    elif acc_done:
        acc_status = "partial"
    accident = {"name": "車禍偵測（Accident）", "stages": stages,
                "current": acc_cur, "ready": acc_ready,
                "done": acc_done, "status": acc_status}

    return {"pipelines": [traffic, accident]}


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
    """從執行中 job 的進入點判斷 (kind, case)。

    kind ∈ {'data', 'tune', 'train', ''}；case ∈ {'freeway', 'accident', ''}。
    關鍵：tune_* 內部是「Ray Tune 包 TorchTrainer」，會產生 RayTrainWorker
    actor。若只看 actor 會把 Tune 誤認成 Train。改以 job 進入點為準歸屬：
    prepare_* → Ray Data；tune_* → Tune（不重複點亮 Train）；train_* → Train。
    """
    try:
        from ray.util.state import list_jobs
        for j in list_jobs(limit=200):
            if "RUNNING" not in str(getattr(j, "status", "")).upper():
                continue
            ep = (getattr(j, "entrypoint", "") or "").lower()
            case = "accident" if "accident" in ep else "freeway"
            if "prepare_" in ep:
                return ("data", case)
            if "tune_" in ep:
                return ("tune", case)
            if "train_" in ep:
                return ("train", case)
    except Exception as e:
        print(f"[monitor] job 查詢失敗：{type(e).__name__}: {e}")
    return ("", "")


def _active_entrypoint():
    """執行中 job 的 entrypoint 字串（給解析 --epochs/--samples 用）。"""
    try:
        from ray.util.state import list_jobs
        for j in list_jobs(limit=200):
            if "RUNNING" in str(getattr(j, "status", "")).upper():
                return getattr(j, "entrypoint", "") or ""
    except Exception:
        pass
    return ""


def _flag_int(ep, names, default):
    """從 entrypoint 解析 `--flag N`（找不到回 default）。"""
    import re
    for n in names:
        m = re.search(rf"{re.escape(n)}[=\s]+(\d+)", ep)
        if m:
            return int(m.group(1))
    return default


def _csv_rows(path):
    """results.csv 資料列數（= 已完成 epoch 數，扣表頭）；讀不到回 0。"""
    try:
        with open(path, encoding="utf-8") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except OSError:
        return 0


def _json_rows(experiments, base="/workspace/ray_results"):
    """某類 run 最新 result.json 的列數（= 已完成 epoch 數）；無則 0。"""
    files = []
    for e in experiments:
        files += glob.glob(f"{base}/{e}/**/result.json", recursive=True)
    if not files:
        return 0
    try:
        newest = max(files, key=os.path.getmtime)
        with open(newest, encoding="utf-8") as f:
            return sum(1 for r in f.read().splitlines() if r.strip())
    except OSError:
        return 0


def _train_pct(kind, case, ep):
    """正式訓練進度 %（已完成 epoch / 目標 epoch）；無法判定回 None。"""
    if kind != "train":
        return None
    if case == "freeway":
        done = _csv_rows("/workspace/ray_results/freeway_final/results.csv")
        total = _flag_int(ep, ["--epochs"], 100)
    else:
        done = _json_rows(["accident_tad_final_raytrain",
                           "accident_cnn_final_raytrain"])
        total = _flag_int(ep, ["--epochs"], 15)
    return min(round(100 * done / total), 100) if total else None


def _tune_pct(kind, case, ep):
    """超參搜尋進度 %（已完成 trial / 目標數）；無法判定回 None（最佳努力）。"""
    if kind != "tune":
        return None
    if case == "freeway":
        total = _flag_int(ep, ["--iterations"], 12)
        done = len(glob.glob(
            "/workspace/runs/detect/tune*/**/result.json", recursive=True))
    else:
        total = _flag_int(ep, ["--samples"], 12)
        done = len(glob.glob(
            "/workspace/ray_results/accident_cnn_tune/*/result.json")) \
            or len(glob.glob(
                "/workspace/ray_results/accident_tad_tune/*/result.json"))
    if total and done:
        return min(round(100 * done / total), 100)
    return None


def cluster_state():
    """節點清單 + 每節點負載 + 叢集資源總量。"""
    dash = _dashboard_nodes()
    tasks_per_node = _running_tasks_per_node()
    gpu_smi = _gpu_util_smi()        # 即時 GPU 利用率（取代 dashboard 的卡住值）
    vram_used, vram_total = _gpu_mem_smi()   # 即時 VRAM（取代卡住的 dashboard mem）
    probes = _node_probes()          # 每節點實跑 psutil 的真值（取代凍住的 mem）

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
        # 記憶體：分母 = Ray --memory 預算（12G/4G/4G，各節點不同）；
        # 分子 = 該容器 cgroup 真實用量（探針回報，head/worker 各自不同）。
        # 探針失敗才退回 dashboard 的 mem[已用]。
        mem_total_gb = round(res.get("memory", 0) / 2**30, 1)      # Ray 預算
        pr = probes.get(nid)
        if pr:
            mem_used_gb = pr["mem_used_gb"]            # cgroup 容器真實用量
            node_cpu_pct = pr["cpu_pct"]
        else:
            mem = dn.get("mem") or []     # [總量, 可用, 主機%, 已用]
            mem_used_gb = round(mem[3] / 2**30, 2) if len(mem) > 3 else 0.0
            node_cpu_pct = round(dn.get("cpu", 0), 1)
        mem_pct = round(min(mem_used_gb / mem_total_gb * 100, 100), 1) \
            if mem_total_gb else 0
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
            "cpu_pct": node_cpu_pct,
            "gpu_pct": round(gpu_pct or 0, 1),
            "mem_pct": mem_pct,
            "mem_gb": mem_total_gb,
            "mem_used_gb": mem_used_gb,
            # VRAM（只 GPU 節點有；nvidia-smi 即時值，訓練時會浮動）
            "vram_used_gb": vram_used if (gpu and vram_used is not None) else 0,
            "vram_total_gb": vram_total if (gpu and vram_total) else 0,
            "vram_pct": round(vram_used / vram_total * 100, 1)
                        if (gpu and vram_total) else 0,
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
    cpu_total = total.get("CPU", 0)
    cpu_live = _cpu_pct_live()       # 實體使用率（取代會誤導的 Ray 邏輯保留）
    cpu_used = round(cpu_live / 100 * cpu_total) if cpu_live is not None \
        else round(cpu_total - avail.get("CPU", 0))
    return {
        "node_count": sum(1 for x in nodes if x["alive"]),
        "nodes": nodes,
        "cpu_total": cpu_total,
        "cpu_used": cpu_used,
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
    case_zh = {"freeway": "Freeway 偵測", "accident": "事故偵測"}.get(case, "")
    tag = f"（{case_zh}）" if case_zh else ""
    # tune_* 的 trial 內部用 TorchTrainer worker；歸給 Tune，不重複點亮 Train
    train_active = train_n > 0 and kind != "tune"
    tune_active = kind == "tune" or tune_n > 0

    def comp(active, doing, detail, log=None):
        return {"active": active, "doing": doing, "detail": detail,
                "log": log or []}

    data_log = _data_log_rolling(data_ops)          # 滾動歷史（時間戳 + 運算元）
    prog = _data_progress()                          # Ray Data 前處理進度 %
    if prog:
        done, total, pct = prog
        data_log = [f"進度 {done}/{total} 項　{pct}%　{_bar(pct)}"] + data_log

    ep_str = _active_entrypoint()                    # 解析 --epochs/--samples
    train_pct = _train_pct(kind, case, ep_str)       # 正式訓練 %
    tune_pct = _tune_pct(kind, case, ep_str)         # 超參搜尋 %

    def _pct(p):                                     # 進度條後綴（無 % 則空字串）
        return f"　{p}%　{_bar(p)}" if p is not None else ""
    # 依案別選 log 來源：事故走 Ray Tune/Train（ray_results）；freeway 走 ultralytics
    if case == "accident":
        # 涵蓋 TAD 主線、圖片集 CNN、軌跡對照；取 mtime 最新的 run（正在跑的會顯示）
        tune_log = _recent_metrics(
            ["accident_tad_tune", "accident_cnn_tune", "accident_tune"])
        train_log = _recent_metrics(
            ["accident_tad_final_raytrain", "accident_tad_final",
             "accident_cnn_final_raytrain", "accident_cnn_final",
             "accident_final_raytrain", "accident_final"])
    else:
        tune_log = _recent_metrics(["tune*"], base="/workspace/runs/detect")
        train_log = _recent_metrics(["freeway_final_raytrain", "freeway_final"])

    return {
        "data": comp(
            data_tasks > 0 or prog is not None,
            (f"處理中 {prog[0]}/{prog[1]} 項{tag}{_pct(prog[2])}" if prog
             else f"處理中 {data_tasks} 個 batch task{tag}" if data_tasks
             else "閒置（等待前處理）"),
            "把資料分成 訓練／驗證／測試 三份",
            data_log),
        "tune": comp(
            tune_active,
            f"搜尋中{tag} · trial 訓練中{_pct(tune_pct)}" if tune_active
            else "閒置（等待超參搜尋）",
            f"正在自動試參數，找最準的設定 · {case_zh}"
            if tune_active and case_zh
            else "自動試多組參數設定，挑出表現最準的一組",
            tune_log),
        "train": comp(
            train_active,
            f"{train_n} 個 worker 訓練中{tag}{_pct(train_pct)}" if train_active
            else "閒置（等待正式訓練）",
            f"正在用最佳設定正式訓練 · {case_zh}（用 GPU）"
            if kind == "train" and case_zh
            else "用最佳設定正式訓練模型",
            train_log),
        "serve": comp(
            serve_n > 0,
            f"{serve_n} 個 replica 運行中　100%　{_bar(100)}" if serve_n
            else "未啟動（待 serve_dashboard.py）",
            "把訓練好的模型上線，5 個鏡頭即時偵測車流／車禍",
            _serve_camera_log(serve_n > 0)),
    }
