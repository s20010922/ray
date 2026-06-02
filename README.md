# Ray｜高速公路 CCTV 車流 / 車禍偵測

用 Ray（Data / Train / Tune / Serve）在 Docker 上，建一套以 YOLO11 為核心的
交通偵測系統：偵測高公局即時 CCTV 的**車流**與**車禍**。

> 本文件是專案的單一說明入口，分章節對應各模組程式碼。

---

## 目錄

1. [專案目標](#1-專案目標)
2. [環境與啟動](#2-環境與啟動)
3. [架構總覽](#3-架構總覽)
4. [core — 叢集連線](#4-core--叢集連線)
5. [modeling — YOLO11 模型](#5-modeling--yolo11-模型)
6. [data — 資料來源與管線](#6-data--資料來源與管線)
7. [訓練策略](#7-訓練策略)
8. [現況與待辦](#8-現況與待辦)

---

## 1. 專案目標

| 任務 | 型態 | 模型 | 最終部署 |
|---|---|---|---|
| 車流偵測 | 物件偵測 | YOLO11n | 高公局 CCTV 即時車流密度 |
| 車禍偵測 | 影像分類 | YOLO11n-cls | 高公局 CCTV 即時事故判斷 |

資料來源最終是**高公局高速公路 CCTV**（固定監控、352×240 低畫質、台灣道路）。

---

## 2. 環境與啟動

**硬體**：RTX 3060 Ti（8GB）、12 核 / 20 緒、32GB RAM。
**容器**：CUDA 12.1 + Python 3.10 + Ray 2.40 + PyTorch（見 [Dockerfile](Dockerfile)）。

所有任務都在容器內跑。`ray-head` 啟動時建好資源池（CPU 16 / GPU 1），
設定見 [docker-compose.yml](docker-compose.yml)。

```powershell
docker compose up -d ray-head                          # 啟動叢集
# dashboard: http://localhost:8265
docker compose exec ray-head python scripts/你的腳本.py  # 跑任務
docker compose down                                     # 關閉
```

**資料掛載**（docker-compose.yml）：

| 主機 | 容器 | 用途 |
|---|---|---|
| `F:/dataset` | `/data/detrac`（唯讀） | UA-DETRAC 原始資料 |
| `./datasets` | `/workspace/datasets`（可寫） | 轉檔資料、抓到的 CCTV |
| `./src` | `/workspace/src`（唯讀） | 程式碼 |

> ⚠️ 圖片**不能**存進唯讀的 `src/`，一律存 `datasets/`。

---

## 3. 架構總覽

```
src/
├── core/         叢集連線（init_ray）
├── modeling/     YOLO11 模型載入（traffic / accident）
└── data/
    ├── augment.py 劣化增強（兩案共用，模擬高公局低畫質）
    ├── traffic/   車流：UA-DETRAC → Ray Data pipeline（偵測）
    ├── accident/  車禍：Roboflow CCTV → Ray Data pipeline（分類）
    └── freeway/   高公局 CCTV 即時影像（來源）
scripts/          進入點（collect_freeway.py …）
datasets/         資料（不進 git）
ray_results/      Ray Train/Tune 產出
```

依賴關係：`core.init_ray()` 先就緒 → `data` 串流資料 → `train` 吃資料訓練
（用到 `modeling` 的模型）→ `serve` 推論。

---

## 4. core — 叢集連線

程式：[src/core/cluster.py](src/core/cluster.py)

唯一職責：**接上容器內正在跑的 `ray-head`**。資源池（16 CPU / 1 GPU）由
容器的 `ray start` 保證，core 不碰資源數字。

```python
from src.core.cluster import init_ray
init_ray()      # 有 RAY_ADDRESS=auto → 接上現成叢集；重複呼叫安全
```

| 設計 | 原因 |
|---|---|
| 只做 attach、不自己開叢集 | 任務都在容器跑，資源池已由 ray-head 定好 |
| 不碰 CPU/GPU 數字 | 避免「容器一套、core 又一套」不一致 |

---

## 5. modeling — YOLO11 模型

程式：[traffic.py](src/modeling/traffic.py)、[accident.py](src/modeling/accident.py)

沿用 team edit 的任務設定，模型升級到 YOLO11n。

| 檔案 | 任務 | 模型 | imgsz | 類別 |
|---|---|---|---|---|
| `traffic.py` | 偵測 | `yolo11n.pt` | 640 | `Vehicle`（1 類）|
| `accident.py` | 分類 | `yolo11n-cls.pt` | 224 | `accident` / `non-accident` |

```python
from src.modeling.traffic import load_traffic_model, IMG_SIZE, CLASSES
det = load_traffic_model()                    # 官方預訓練
det = load_traffic_model(".../best.pt")        # 自訓練微調模型
```

**重要**：預設載的是官方預訓練（COCO / ImageNet），是「起點」非「成品」。
`CLASSES` 是「目標類別」，要訓練後才成立。`yolo11n-cls.pt` 完全不認得車禍，
非訓練不可。

---

## 6. data — 資料來源與管線

### 6.1 概觀

`traffic` / `accident` 是**任務**，`freeway` 是**來源**。一段高公局畫面可
同時餵兩個任務：

```
高公局 CCTV (freeway/)  ──→ traffic  車流偵測（偵測）
                        └─→ accident 車禍判斷（分類）
```

兩案結構**對稱**（各有 sources + pipeline），**劣化增強 [augment.py](src/data/augment.py)
共用**——兩案都部署在高公局低畫質 CCTV，需要同樣的低畫質適應。

| 子模組 | 內容 | 狀態 |
|---|---|---|
| `traffic/` | UA-DETRAC → Ray Data pipeline（偵測） | ✅ 可用 |
| `accident/` | Roboflow CCTV → Ray Data pipeline（分類） | ✅ 可用 |
| `freeway/` | CCTV MJPEG 抓取 | ✅ 可用 |
| `augment.py` | 劣化增強（兩案共用） | ✅ 可用 |

### 6.2 traffic — UA-DETRAC + Ray Data pipeline

**資料集**：UA-DETRAC（14 萬張、100 序列、交通監控視角，貼近高公局）。
標註是自訂 XML（box 像素 + 車種），需轉換。

走 **Ray Data 串流管線**（路 B），真正用到 Ray 的串流 + 多 CPU 平行：

```
DETRAC XML
  └ sources.list_detrac_records(frame_stride=10)   ← 解析 + 抽幀 → records(~1.4萬)
       └ pipeline.build_ray_dataset(augment=True)   ← Ray Data 串流，多 CPU 平行
            ├ 解碼 → 劣化增強 → 水平翻轉 → resize 640 → 正規化 → pad
            └ 輸出 image(3,640,640) + boxes_xywhn(100,4) + labels(100)
```

| 程式 | 職責 |
|---|---|
| [sources.py](src/data/traffic/sources.py) | XML → records（含**抽幀**降冗餘、依序列切 train/val）|
| [pipeline.py](src/data/traffic/pipeline.py) | Ray Data 串流：解碼 / 劣化增強（共用 [augment](src/data/augment.py)）/ 翻轉 / resize / pad |
| [detrac_to_yolo.py](src/data/traffic/detrac_to_yolo.py) | 另一條路：轉成 YOLO 檔案格式（給 ultralytics 內建訓練用）|

**劣化增強**（只在 train，val 不增強）：降解析度、JPEG 壓縮、模糊（高斯/中值/平均/動態）、
噪點、亮度對比——讓模型訓練時就見過低畫質，上線才認得高公局的糊畫面。

關鍵參數：`frame_stride=10`（抽幀，約 1.4 萬張）、`MAX_BOXES=100`、
`val_ratio=0.2`（依序列切，不洩漏）、單類 Vehicle。

### 6.3 accident — 車禍分類

**資料集**：[datasets/accident](datasets/accident)（Roboflow 匯出、土耳其 Adıyaman
CCTV 監控視角，貼近高公局）。已是分好的分類資料夾，**不需轉換器**。

| 程式 | 職責 |
|---|---|
| [sources.py](src/data/accident/sources.py) | 列檔 `{train,val}/{accident,non-accident}/` → records `{image_path, label}` |
| [pipeline.py](src/data/accident/pipeline.py) | Ray Data 串流：解碼 / 劣化增強（共用 [augment](src/data/augment.py)）/ 翻轉 / resize 224 |

規模：原始 424 張、train 527 / val 95、測試影片 202 部。
label：`0=accident, 1=non-accident`。

比 traffic 簡單：分類**沒有 bbox**，翻轉不需變換座標、也不用 pad。

**車禍偵測的時序邏輯**（部署時）：單幀分類易誤判，要**連續多幀都高信心
判為 accident** 才算偵測到車禍事件（並記錄發生時間）。這個「連續確認」邏輯
之後寫在 serve（即時推論）。

### 6.4 freeway — 高公局 CCTV

程式：[grabber.py](src/data/freeway/grabber.py)、收集器 [scripts/collect_freeway.py](scripts/collect_freeway.py)

核心是 `grab_jpeg_frame(stream_url)`——從 MJPEG 串流抽單張 JPEG（找 FFD8/FFD9）。
收集與即時推論共用它。

```powershell
# 背景定時收集，每鏡頭目標 200 張，達標自動停
docker compose exec -d ray-head python scripts/collect_freeway.py --target-per-camera 200
```

5 支 focus 鏡頭（國1/國3，串流 `https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=<id>`）。

> ⚠️ 是 **`abs2mjpg`** 不是 `abs2jpg`——路徑錯會回 403，易誤判成被擋。

**限制**：CCTV 影像 352×240、**無標註**。要 fine-tune 得人工標（traffic 框車、
accident 標類別）；即時幾乎抓不到車禍正樣本，主要供 traffic 與 accident 的負樣本。

---

## 7. 訓練策略

### 路線圖

```
官方 yolo11n 預訓練
   │ ① 用 UA-DETRAC + 劣化增強 訓練（學會認車、不怕糊）
   ▼
基礎模型（mAP@0.5 ~0.8 即可，別過擬合 UA-DETRAC）
   │ ② 用標註過的高公局畫面 fine-tune（domain adaptation）
   ▼
貼合高公局的成品 → serve 即時推論
```

### 為什麼這樣設計

| 事實 | 對策 |
|---|---|
| 高公局 352×240 低畫質，UA-DETRAC 960×540 清晰 | **劣化增強**模擬低畫質 |
| UA-DETRAC 影片拆幀、相鄰幀重複 | **抽幀** stride 10 |
| 這是基礎模型，之後 fine-tune | 堪用即可，力氣留給 fine-tune |
| 高公局即時串流無標註 | 先用公開資料訓練，少量人工標註再微調 |

### 資料量與訓練程度

- **車流資料量**：全 100 序列（多樣性）× 抽幀 10 ≈ 1.4 萬張。
- **訓練**：`epochs=100, patience=30, imgsz=640, batch=16`，通常 40~70 epoch 收斂。
- **fine-tune 資料**：每鏡頭 100~200 張（多樣性 > 數量，要跨日夜/尖離峰/晴雨）。

---

## 8. 現況與待辦

| 模組 | 狀態 |
|---|---|
| core | ✅ 完成 |
| modeling | ✅ 完成（traffic / accident 模型載入）|
| data/traffic | ✅ Ray Data pipeline（sources / pipeline，偵測）|
| data/accident | ✅ Ray Data pipeline（sources / pipeline，分類）|
| data/augment | ✅ 劣化增強（兩案共用）|
| data/freeway | ✅ 抓取 + 收集完成（已收 1001 張 CCTV）|
| train | ⏳ 待建（Ray Train TorchTrainer 吃 pipeline 訓練）|
| serve | ⏳ 待建（Ray Serve 即時推論 + 車禍連續確認）|
| tune | ⏳ 待建（Ray Tune 超參搜尋）|

**下一步**：實作 `train`——兩案的 Ray Data pipeline 都就緒，用 Ray Train（TorchTrainer）
吃 pipeline 訓練 YOLO11n（traffic 偵測）/ YOLO11n-cls（accident 分類）。
