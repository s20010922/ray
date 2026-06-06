"""Ray 叢集即時狀態查詢（給獨立監控頁用，不依賴 Ray Serve）。

cluster_state()    → 節點清單 + 每節點負載（實體 + Ray 邏輯任務數）+ 叢集資源
components_state()  → Data / Train / Tune / Serve 各元件即時活動

所有資訊都從「外部觀察」叢集而來（ray.nodes / 狀態 API / dashboard），
因此監控程式本身只是個輕量 driver，與被監控的 serve / 訓練任務無關。
"""

import json
import urllib.request
from collections import Counter, defaultdict

import ray


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


def cluster_state():
    """節點清單 + 每節點負載 + 叢集資源總量。"""
    dash = _dashboard_nodes()
    tasks_per_node = _running_tasks_per_node()

    nodes = []
    for n in ray.nodes():
        res = n.get("Resources", {})
        gpu = res.get("GPU", 0)
        cpu = res.get("CPU", 0)
        nid = n.get("NodeID", "")
        dn = dash.get(nid, {})
        rl = dn.get("raylet", {})
        gpus = dn.get("gpus") or []
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
    try:
        from ray.util.state import list_actors, list_tasks
        for t in list_tasks(filters=[("state", "=", "RUNNING")], limit=4000):
            nm = ((t.name or "") + (t.func_or_class_name or "")).lower()
            if any(k in nm for k in ("mapbatches", "_preprocess", "readrange",
                                      "split", "map(", "streaming")):
                data_tasks += 1
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

    def comp(active, doing, detail):
        return {"active": active, "doing": doing, "detail": detail}

    return {
        "data": comp(
            data_tasks > 0,
            f"處理中 {data_tasks} 個 batch task" if data_tasks
            else "閒置（訓練時啟動串流前處理）",
            "解碼 → 劣化增強 → resize → 多節點平行前處理"),
        "train": comp(
            train_n > 0,
            f"{train_n} 個 worker 訓練中" if train_n
            else "閒置（待 train_accident / train_traffic）",
            "TorchTrainer：Accident 分類、Traffic 偵測"),
        "tune": comp(
            tune_n > 0,
            f"{tune_n} 個 trial 搜尋中" if tune_n
            else "閒置（待 tune_freeway）",
            "ASHA 超參搜尋：Freeway 微調"),
        "serve": comp(
            serve_n > 0,
            f"{serve_n} 個 replica 運行中" if serve_n
            else "未啟動（serve_dashboard.py）",
            "即時推論服務：5 鏡頭車流／車禍偵測"),
    }
