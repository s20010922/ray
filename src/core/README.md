# `src/core` — Ray 叢集連線

這個模組只負責一件事：**讓容器內的腳本接上 Ray 叢集**。
其他模組（`data` / `train` / `tune` / `serve`）動工前都會先呼叫這裡的 `init_ray()`。

## 前提：一切都在 Docker 容器內跑

本專案的執行方式固定是：

```powershell
docker compose exec ray-head python scripts/你的腳本.py
```

容器啟動時，`ray-head` 服務已經用這行把資源池建好（[docker-compose.yml](../../docker-compose.yml)）：

```
ray start --head --num-cpus=16 --num-gpus=1 ...
```

並設了環境變數 `RAY_ADDRESS=auto`。

➡️ 所以 core **不需要決定要配多少資源**——16 核 + 1 GPU 由容器保證。
core 要做的就只是「接上」這個現成的叢集。

## 檔案

| 檔案 | 作用 |
|---|---|
| `cluster.py` | `init_ray()` — 接上 ray-head，印出資源池 |
| `__init__.py` | 套件標記（空檔） |

## `init_ray()` 怎麼運作

```
init_ray()
   │
   ├─ ray 已連線?  ── 是 ─→ 直接返回（重複呼叫安全）
   │
   └─ 否 ─→ ray.init(address=RAY_ADDRESS)   # RAY_ADDRESS=auto
              接上容器內正在跑的 ray-head
              │
              └─ verbose=True 時印出資源池：CPU=16 GPU=1
```

- **`address="auto"`**：Ray 自動找到本機正在跑的 head，不用手寫 IP。
- **重複呼叫安全**：已連線就 return，多次呼叫不會出錯。
- **`verbose`**：印出 `cluster_resources()`，方便確認接到的是預期的 16 核 + 1 GPU。

## 用法

任何 entry point 開頭都長這樣：

```python
from src.core.cluster import init_ray

init_ray()          # 接上叢集
# ... 之後才 import data / train / serve 做事 ...
```

實測輸出：

```
[init_ray] 已接上叢集 | CPU=16.0 GPU=1.0
```

## 與其他模組的關係

```
core.init_ray()  ←─ data / train / tune / serve 全部先依賴它
       │
       └─ 接上後，所有任務共用同一個 16核+1GPU 的資源池，
          由 Ray（領班）統一調度：誰先跑、誰排隊、用哪個資源。
```

> 「共用同一叢集」= 所有腳本都連到同一顆 ray-head，一起排隊用同一池
> 16 核 + 1 GPU，Ray 統一分配、不超賣。

## 設計取捨

| 決定 | 原因 |
|---|---|
| 只做 Attach、不自己開叢集 | 任務都在容器跑，資源池已由 ray-head 定好 |
| core 不碰 CPU/GPU 數字 | 避免「容器設一套、core 又設一套」兩處不一致 |
| `address` 預設 `"auto"` | 容器環境變數已是 auto，零設定即可連上 |
