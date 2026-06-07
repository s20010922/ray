# 高速公路 CCTV 車流／車禍即時偵測系統

以 **Ray**（Data／Tune／Train／Serve）與 **YOLO11** 為核心、在 Docker 中建置的交通智慧
監控系統。針對台灣高速公路局（高公局）即時 CCTV（固定機位、MJPEG 串流），提供
**車流密度偵測**與**車禍事件偵測**兩條獨立流水線，並以即時儀表板與叢集監控呈現。

叢集以**三個節點**運行（1 head + 2 worker，同一台機器的多容器，僅透過網路溝通），
兩條流程**各自完整走 Ray 標準四階段**：`Ray Data → Ray Tune → Ray Train → Ray Serve`。

---

## 目錄

1. [系統總覽](#1-系統總覽)
2. [兩條流程的設計](#2-兩條流程的設計)
3. [環境需求](#3-環境需求)
4. [快速開始](#4-快速開始)
5. [車流偵測流程（Freeway）](#5-車流偵測流程freeway)
6. [車禍偵測流程（Accident）](#6-車禍偵測流程accident)
7. [部署架構](#7-部署架構)
8. [叢集監控（RAY MONITOR）](#8-叢集監控ray-monitor)
9. [專案結構](#9-專案結構)
10. [效能與現況](#10-效能與現況)
11. [指令速查](#11-指令速查)

---

## 1. 系統總覽

### 1.1 兩個模型、兩條流程

| 流程 | 模型 | 型態 | 訓練資料 | 狀態 |
|------|------|------|----------|------|
| **車流偵測**（Freeway） | YOLO11s | 物件偵測（車輛） | 高公局 CCTV 自動標註（SAHI + yolo11x teacher） | ✅ 訓練完成，可上線 |
| **車禍偵測**（Accident） | LSTM／GRU／1D-CNN | 軌跡時序分類 | AccidentBench 真實高速公路事故影片 | 🔄 流程已跑通，模型優化中 |

兩者**共用同一個偵測前端**（YOLO 偵測 + ByteTrack 追蹤），再分流：一邊算車流計數／
密度，一邊把軌跡轉成運動特徵餵時序模型判斷事故。

### 1.2 Ray 技術堆疊

| 元件 | 車流（Freeway） | 車禍（Accident） |
|------|------|------|
| **Ray Data** | 影像切分／前處理（CPU 多節點） | yolo11x+ByteTrack 追蹤抽軌跡（GPU 序列） |
| **Ray Tune** | ASHA 搜偵測超參（ultralytics，GPU） | ASHA 搜時序超參（**CPU 三節點平行**） |
| **Ray Train** | TorchTrainer 編排 yolo11s 訓練（GPU） | TorchTrainer 訓練時序模型（GPU/CPU） |
| **Ray Serve** | 多鏡頭即時偵測儀表板（:8000） | 接在偵測前端後的時序判斷（規劃中） |

> **獨立的 RAY MONITOR**（[scripts/monitor.py](scripts/monitor.py)）以輕量 driver 連上叢集，
> 純觀察各 Ray 元件活動與節點負載，**不依賴 Serve、不佔 GPU**，叢集一啟動即可看，
> 右側 Pipeline 同時顯示**車流／車禍兩條流程**的階段進度。

---

## 2. 兩條流程的設計

### 2.1 核心理念：偵測器可換、時序模型不變

車禍時序模型**只吃抽象運動數字**（速度／加速度／航向變化／最近車距／停滯…），
**不吃像素**。像素層的差異（畫質、角度、解析度）在「偵測＋追蹤」那關就被吸收，
流到時序模型的只剩運動特徵：

```
AccidentBench 影片 ──YOLO(yolo11x)──┐
                                    ├─► [速度,加速度,車距,…] ─► 同一個時序模型
高公局 CCTV        ──YOLO(11s)──────┘
```

因此**用 AccidentBench 訓練的時序模型，能直接吃高公局的軌跡** —— 部署時只要把前端
偵測器換成高公局自己的 yolo11s，運動特徵的「數字長相」一致即可遷移。

### 2.2 幀率對齊

高公局 `cctvn.freeway.gov.tw` 的 MJPEG 串流實測約 **9–11 fps**。訓練時把 AccidentBench
（多為 30fps）依各片 fps 動態降採樣到**等效 10fps**，讓速度／加速度的數值尺度與部署
一致（[pipeline.py](src/data/accident/pipeline.py) 的 `target_fps`）。

---

## 3. 環境需求

### 3.1 硬體

| 項目 | 規格 |
|------|------|
| GPU | NVIDIA（head 持有 1 顆；workers 為 CPU-only） |
| CPU／RAM | 16 核（head 8 + 2×worker 4）／充足記憶體 |

### 3.2 軟體

容器映像（[Dockerfile](Dockerfile)）：CUDA + Python 3.10 + Ray + Ultralytics + PyTorch +
OpenCV（完整見 [requirements.txt](requirements.txt)）。

> ByteTrack 追蹤需 `lap`/`lapx`；離線環境可在 head 容器內以 wheel 安裝
> （`pip install --no-index <lapx wheel>`）。

### 3.3 資料掛載（[docker-compose.yml](docker-compose.yml)）

| 主機路徑 | 容器路徑 | 權限 | 用途 |
|----------|----------|------|------|
| `./ACCIDENT` | `/data/accident` | 唯讀 | AccidentBench 真實事故影片 + metadata |
| `./datasets` | `/workspace/datasets` | 可寫 | 轉檔資料、權重、時序資料集 |
| `./src` | `/workspace/src` | 唯讀 | 原始碼 |
| `./scripts` | `/workspace/scripts` | 唯讀 | 進入點腳本 |
| `./ray_results` | `/workspace/ray_results` | 可寫 | 訓練／搜參輸出 |

### 3.4 對外連接埠

| 連接埠 | 服務 |
|--------|------|
| 8265 | Ray Dashboard（原生）|
| 8000 | Ray Serve HTTP（車流相機儀表板）|
| 8501 | RAY MONITOR（叢集監控總覽，兩條流程）|

---

## 4. 快速開始

```powershell
# 1. 啟動 3 節點 Ray 叢集（1 head + 2 worker，共 CPU 16 / GPU 1）
docker compose up -d ray-head ray-worker-1 ray-worker-2

# 2. 開叢集監控總覽（不需 GPU，叢集一起來就能看）
docker compose exec -d ray-head python scripts/monitor.py
#    瀏覽器：http://localhost:8501/

# 3. 車流流程（Freeway）— 已訓練完成的模型在 ray_results/freeway_final/
#    （如需重跑，見 §5）

# 4. 車禍流程（Accident）— 三階段
docker compose exec ray-head python scripts/prepare_accident.py        # ① Ray Data
docker compose exec ray-head python scripts/tune_accident.py --samples 24   # ② Ray Tune（CPU 三節點）
docker compose exec ray-head python scripts/train_accident.py --kind gru --hidden 128 --layers 2  # ③ Ray Train
docker compose exec ray-head python scripts/eval_accident.py           # 評估（窗級 + 事件級）

# 5. 車流即時儀表板（佔 GPU）
docker compose exec -d ray-head python scripts/serve_dashboard.py
#    瀏覽器：http://localhost:8000/

docker compose down   # 關閉叢集
```

> **GPU 配置**：整個叢集只有 1 顆 GPU、由 head 持有。GPU 綁定的工作（yolo 追蹤／偵測訓練／
> serve 推論）只能在 head；CPU 可平行的工作（**車禍 Ray Tune**）才會用到兩個 worker。

---

## 5. 車流偵測流程（Freeway）

以高公局自身 CCTV 訓練的 yolo11s 車輛偵測器，完整走 Ray 四階段。

| 階段 | 腳本 | 說明 |
|------|------|------|
| 前置標註 | [prelabel_freeway.py](scripts/prelabel_freeway.py) | yolo11x **teacher** + **SAHI 切片**自動標註高公局影像（352×240 小圖需 192×192 切片偵測遠處小車）|
| ① Ray Data | [prepare_freeway.py](scripts/prepare_freeway.py) | 鏡頭級切分 train/val/test（整鏡頭隔離避免洩漏），CPU 多節點前處理 |
| ② Ray Tune | [tune_freeway.py](scripts/tune_freeway.py) | ultralytics `model.tune(use_ray=True)` ASHA 搜 lr/scale/mosaic… |
| ③ Ray Train | [train_freeway.py](scripts/train_freeway.py) | `TorchTrainer` 編排 yolo11s 訓練（`optimizer=AdamW` 固定，讓搜到的 lr 生效）|

- 起點權重 yolo11s，單類 Vehicle，imgsz 640。
- 鏡頭級隔離：保留 1 整支鏡頭做 held-out test。
- 成果：**mAP50 0.847 / mAP50-95 0.744**，模型存於 `ray_results/freeway_final/weights/best.pt`。

---

## 6. 車禍偵測流程（Accident）

軌跡時序模型，資料來源為 **AccidentBench** 真實高速公路事故影片。

### 6.1 資料來源與篩選

AccidentBench（`/data/accident`）含 2027 支真實事故影片 + 豐富 metadata（事故幀、事故框
`x1y1x2y2`、事故型態、場景、晝夜、畫質）。以 metadata 篩出**對齊高公局**的子集：
`scene_layout=highway` ＋ `day_time=day` ＋ 畫質非最差 → **271 支**乾淨可偵測影片。

### 6.2 四階段

| 階段 | 腳本／模組 | 說明 |
|------|------|------|
| ① Ray Data | [prepare_accident.py](scripts/prepare_accident.py)<br>[data/accident/](src/data/accident/) | 每片 yolo11x+ByteTrack 追蹤 → 運動特徵 → 滑動視窗。GPU 序列執行，輸出 `datasets/accident_seq/{train,val,test}.npz` |
| ② Ray Tune | [tune_accident.py](scripts/tune_accident.py) | ASHA 搜 LSTM/GRU/CNN × hidden/layers/lr/dropout。**每試驗 2 CPU → head+2 worker 平行** |
| ③ Ray Train | [train_accident.py](scripts/train_accident.py) | `TorchTrainer` 用最佳超參正式訓練，輸出自帶 scaler 的 checkpoint |
| 評估 | [eval_accident.py](scripts/eval_accident.py) | **窗級 AP + 事件級**（每片是否在事故時刻被觸發、背景誤報率）|

### 6.3 特徵工程（[features.py](src/data/accident/features.py)）

每台車逐幀算 10 維運動特徵：速率、速度分量、加速度、航向角變化、框面積與變化率、
最近他車距離與變化、停滯旗標。座標一律以畫面寬高**正規化**（解析度無關），位置只用於
打標、不進特徵矩陣（維持 domain 不變）。

### 6.4 正樣本標註（[label.py](src/data/accident/label.py)）

關鍵在排除標籤雜訊：以「事故時刻與事故框 **IoU 最高的軌跡**」鎖定肇事車（碰撞取前 2 台），
正樣本**只取肇事車撞擊前後 ±1 秒**，避免把正常過路車標成事故。

### 6.5 不平衡處理

事故正樣本稀少（約 1%）。訓練用 `BCEWithLogitsLoss(pos_weight)` 補償，模型挑選看
驗證集 **Average Precision（AP）**（對不平衡比 accuracy 有意義），切分採「依正樣本數
分層」保證每個 split 都有事故樣本。

---

## 7. 部署架構

```
                高公局 CCTV（MJPEG ~10fps）
                        │ 連續抓幀
                ┌───────▼────────┐
                │ YOLO 偵測(11s) │  ← 共用前端，吃像素
                │ freeway best.pt│
                └───────┬────────┘
                ┌───────▼────────┐
                │ ByteTrack 追蹤 │  ← 串軌跡
                └───┬────────┬───┘
            走車流  │        │  走車禍
        ┌───────────▼┐    ┌──▼─────────────┐
        │ 計數／密度  │    │ 特徵工程(純數學)│
        └───────────┬┘    └──┬─────────────┘
                    │        │ (T,F) 序列
                    │     ┌──▼──────────────┐
                    │     │ 時序模型(LSTM)  │  ← 吃數字，跨域
                    │     │ accident_seq.pt │
                    │     └──┬──────────────┘
                ┌───▼────────▼───┐
                │ Ray Serve JSON │ → :8000 儀表板
                └────────────────┘
```

- **偵測+追蹤只跑一次**（共用前端），之後分流給車流與車禍，省算力。
- 部署用高公局 yolo11s（domain 對），時序模型不變（吃抽象數字）。
- 車禍偵測需連續高 fps 抓幀（~10fps），與車流的低頻輪詢不同（Serve 整合為後續工作）。

---

## 8. 叢集監控（RAY MONITOR）

獨立服務（[scripts/monitor.py](scripts/monitor.py)），唯讀查詢叢集狀態、不佔 GPU。

| 區塊 | 內容 |
|------|------|
| **叢集節點** | active 節點數、CPU/GPU/Object Store 總量、每節點負載與 Ray 任務數 |
| **Ray 元件活動** | Data／Tune／Train／Serve 即時 active/idle 與正在做什麼（含即時 log）|
| **Pipeline（兩條）** | **車流**（已完成可上線）與**車禍**（隨任務即時亮階段）的步驟圖 |

端點：`GET /`、`/cluster.json`、`/components.json`、`/pipeline.json`。

> Pipeline 狀態自動推斷（不寫死）：車流偵測到模型已存在即標完成／可上線；車禍以
> `accident_seq/train.npz`、`accident_final/accident_seq.pt` 是否存在 + 執行中 job 判定階段。
> Tune log 依案別讀對應來源（freeway 走 ultralytics、accident 走 Ray Tune），解析
> mAP 或 AP/recall/f1。

---

## 9. 專案結構

```
src/
├── core/          叢集連線（init_ray）
├── modeling/      accident.py：時序模型 LSTM/GRU/1D-CNN
├── data/
│   ├── freeway/   高公局抓取(grabber)、SAHI 自動標註(prelabel)、鏡頭切分、Ray Data
│   └── accident/  tracking（YOLO+ByteTrack）、features（運動特徵）、
│                  label（肇事車標註）、pipeline（Ray Data 主流程）
├── train/
│   ├── freeway/   Ray Train 偵測訓練
│   └── accident/  trainer.py：時序模型訓練核心（Tune/Train 共用）
├── serve/         Ray Serve 車流相機推論（app.py + dashboard.html）
└── monitor/       RAY MONITOR（state.py + overview.html，兩條 pipeline）

scripts/           prepare_/tune_/train_freeway、prepare_/tune_/train_/eval_accident、
                   prelabel_freeway、serve_dashboard、monitor
datasets/          資料、權重、accident_seq 時序資料集（不納入版控）
ray_results/       訓練與搜參輸出（不納入版控）
```

---

## 10. 效能與現況

| 模型 | Test Set | 主要指標 | 狀態 |
|------|----------|----------|------|
| **Freeway**（yolo11s） | 1 整支鏡頭（鏡頭級隔離）| **mAP50 0.847 / mAP50-95 0.744** | ✅ 完成、可上線 |
| **Accident**（時序） | 影片級隔離（窗級 + 事件級）| 流程已端到端驗證；窗級 AP 優化中 | 🔄 輕量測試版 |

說明：

- **Freeway** 已完成 Ray 四階段並可部署，是系統的偵測前端骨幹。
- **Accident** 的四階段（Data→Tune→Train→Eval）已完整跑通；目前為**輕量測試模型**。
  因事故正樣本稀少 + 弱標籤，窗級 AP 仍偏低，正以**收緊正樣本（肇事車 ±1s）**與
  **事件級評估**提升鑑別力。架構（運動特徵 + 時序模型）與資料無關，未來換高公局真實
  事故或擴大資料皆不需改架構。

> 設計取捨、資料集評選（UCF-Crime／CADP／TUMTraf／AccidentBench）與除錯過程見
> [NOTE.md](NOTE.md)。

---

## 11. 指令速查

| 階段 | 指令（前綴 `docker compose exec ray-head`）|
|------|------|
| 啟動 3 節點叢集 | `docker compose up -d ray-head ray-worker-1 ray-worker-2` |
| 叢集監控總覽 | `python scripts/monitor.py`（:8501，不佔 GPU）|
| **車流** ① Ray Data | `python scripts/prepare_freeway.py` |
| **車流** ② Ray Tune | `python scripts/tune_freeway.py` |
| **車流** ③ Ray Train | `python scripts/train_freeway.py` |
| **車禍** ① Ray Data | `python scripts/prepare_accident.py` |
| **車禍** ② Ray Tune | `python scripts/tune_accident.py --samples 24`（CPU 三節點平行）|
| **車禍** ③ Ray Train | `python scripts/train_accident.py --kind gru --hidden 128 --layers 2` |
| **車禍** 評估 | `python scripts/eval_accident.py`（窗級 + 事件級）|
| 車流即時儀表板 | `python scripts/serve_dashboard.py`（:8000，佔 GPU）|
| 關閉叢集 | `docker compose down` |

---

## 附錄：開發筆記

設計演進、資料集評選與除錯過程記錄於 [NOTE.md](NOTE.md)。
