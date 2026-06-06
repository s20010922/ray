# 高速公路 CCTV 車流／車禍即時偵測系統

以 **Ray**（Data／Train／Tune／Serve）與 **YOLO11** 為核心，在 Docker 環境中建置的
交通智慧監控系統。針對台灣高速公路局即時 CCTV（固定機位、352×240 低畫質），
提供**車流密度偵測**與**車禍事件判斷**，並透過即時儀表板呈現。

叢集以**三個節點**運行（1 head + 2 worker，同一台機器的多容器），並提供兩個網頁：
**即時推論儀表板**（Serve，相機畫面）與 **RAY MONITOR**（獨立的叢集監控總覽）。

---

## 目錄

1. [系統總覽](#1-系統總覽)
2. [環境需求](#2-環境需求)
3. [快速開始](#3-快速開始)
4. [資料準備](#4-資料準備)
5. [模型訓練](#5-模型訓練)
6. [模型評估](#6-模型評估)
7. [服務部署（Serve 推論 + RAY MONITOR）](#7-服務部署serve-推論--ray-monitor)
8. [專案結構](#8-專案結構)
9. [效能總結](#9-效能總結)
10. [已知限制](#10-已知限制)
11. [指令速查](#11-指令速查)

---

## 1. 系統總覽

### 1.1 三項任務

本系統包含兩個獨立 base 模型與一個微調模型：

| 任務 | 型態 | 模型 | 訓練資料 | 角色 |
|------|------|------|----------|------|
| **車禍偵測**（Accident） | 影像分類 | YOLO11n-cls | Roboflow CCTV 車禍圖 | 獨立 base 模型 |
| **車流偵測**（Traffic） | 物件偵測 | YOLO11n | UA-DETRAC | 偵測 base 模型 |
| **高公局微調**（Freeway） | 物件偵測 | YOLO11n | 高公局 CCTV（自動標註） | Traffic base 的蒸餾微調版 |

> Freeway 並非第三個獨立模型，而是以知識蒸餾（teacher `yolo11x` 自動標註高公局影像，
> student `yolo11n` 微調）讓 Traffic base 適應高公局場景的產物。

### 1.2 Ray 技術堆疊

| 元件 | 用途 |
|------|------|
| **Ray Data** | 串流式資料管線（解碼、劣化增強、多 CPU 平行前處理） |
| **Ray Train** | 分散式訓練（`TorchTrainer`，Accident 與 Traffic base） |
| **Ray Tune** | 超參數搜尋（Freeway 微調，ASHA 排程器） |
| **Ray Serve** | 即時推論服務（多鏡頭輪詢、畫框、HTTP API、相機儀表板） |

> 另有獨立的 **RAY MONITOR**（[scripts/monitor.py](scripts/monitor.py)）以輕量 driver
> 連上叢集，純觀察各 Ray 元件活動與節點負載，**不依賴 Serve、不佔 GPU**，叢集一啟動即可看。

### 1.3 資料流

```
【車禍分類】
Roboflow 車禍圖 ──► Ray Data ──► Ray Train ──► Accident base ─────────────┐
                   （劣化增強）                （分類，不微調）            │
                                                                          │
【車流偵測】                                                              ▼
UA-DETRAC ──► Ray Data ──► Ray Train ──► Traffic base ─┐          Ray Serve
            （劣化增強）                （偵測）         │          即時監控
                                                        ▼          儀表板
高公局 CCTV ──► yolo11x 自動標註 ──► Ray Tune 微調 ──► Freeway 模型 ───────┘
               （知識蒸餾 teacher）   （超參搜尋）    （Traffic base 微調版）
```

- **Accident base**（分類）獨立訓練，因高公局無車禍影片**不微調**，直接進 Serve。
- **Traffic base**（偵測）經高公局影像知識蒸餾微調為 **Freeway 模型**後進 Serve。
- 兩個模型在 Serve 各司其職：Freeway 偵測車流、Accident 判斷事故。

---

## 2. 環境需求

### 2.1 硬體

| 項目 | 規格 |
|------|------|
| GPU | NVIDIA RTX 3060 Ti（8 GB VRAM） |
| CPU／RAM | 12 核 20 緒／32 GB |

### 2.2 軟體

容器映像（見 [Dockerfile](Dockerfile)）：CUDA 12.1 + Python 3.10 + 下列主要套件
（完整清單見 [requirements.txt](requirements.txt)）：

| 套件 | 版本 |
|------|------|
| PyTorch | 2.5.1+cu121 |
| Ray | 2.40.0（`[default,train,data,serve]`） |
| Ultralytics | 8.3.40 |
| OpenCV | 4.10.0.84 |

### 2.3 資料掛載（[docker-compose.yml](docker-compose.yml)）

| 主機路徑 | 容器路徑 | 權限 | 用途 |
|----------|----------|------|------|
| `F:/dataset` | `/data/detrac` | 唯讀 | UA-DETRAC 原始資料 |
| `./datasets` | `/workspace/datasets` | 可寫 | 轉檔資料、抓取的 CCTV 影像 |
| `./src` | `/workspace/src` | 唯讀 | 原始碼 |
| `./scripts` | `/workspace/scripts` | 唯讀 | 進入點腳本 |
| `./ray_results` | `/workspace/ray_results` | 可寫 | 訓練／微調輸出 |

### 2.4 對外連接埠

| 連接埠 | 服務 |
|--------|------|
| 8265 | Ray Dashboard（原生）|
| 8000 | Ray Serve HTTP（相機儀表板 + API）|
| 8501 | RAY MONITOR（叢集監控總覽）|
| 6006 | TensorBoard（訓練曲線／Tune HParams）|

---

## 3. 快速開始

```powershell
# 1. 啟動 3 節點 Ray 叢集（1 head + 2 worker，共 CPU 16 / GPU 1）
docker compose up -d ray-head ray-worker-1 ray-worker-2

# 2. 開叢集監控總覽（不需 GPU，叢集一起來就能看）
docker compose exec -d ray-head python scripts/monitor.py
#    開啟瀏覽器：http://localhost:8501/

# 3. 切分資料（隔離 held-out test set）
docker compose exec ray-head python -m src.data.accident.split `
    --src /workspace/datasets/accident/Image --out /workspace/datasets/accident

# 4. 訓練（擇一；可在 RAY MONITOR 觀察 Ray Data/Train 跨節點負載）
docker compose exec ray-head python scripts/train_accident.py --epochs 50
docker compose exec ray-head python scripts/train_traffic.py  --epochs 30

# 5. 評估（在 held-out test set 上）
docker compose exec ray-head python scripts/eval_accident.py
docker compose exec ray-head python scripts/eval_traffic.py

# 6. 啟動即時推論儀表板（相機畫面，佔 GPU）
docker compose exec -d ray-head python scripts/serve_dashboard.py
#    開啟瀏覽器：http://localhost:8000/

# 關閉叢集
docker compose down
```

所有任務皆於容器內執行。三個網頁：相機 http://localhost:8000/、
叢集監控 http://localhost:8501/、Ray 原生 Dashboard http://localhost:8265。

> **GPU 配置**：整個叢集只有 1 顆 GPU，由 head 持有。Serve 推論與訓練都需要 GPU，
> 故兩者不可同時搶用——demo「邊訓練邊看監控」時，用 `serve_dashboard.py --no-gpu`
> 讓 Serve 改用 CPU 推論、把 GPU 讓給訓練（RAY MONITOR 本就不需 GPU，照常運作）。

---

## 4. 資料準備

### 4.1 Held-out Test Set 設計

為提供可信的評估指標，三組資料集皆切分為 **train／val／test** 三份，其中 **test
集在訓練全程完全不可見**，僅供最終評估使用一次。隔離粒度依資料特性而異，以避免
相似樣本跨 split 洩漏：

| 任務 | 切分程式 | 隔離粒度 | 切分結果 |
|------|----------|----------|----------|
| Accident | [accident/split.py](src/data/accident/split.py) | 圖片級（按類別分層） | train 300／val 62／test 62（皆 1:1 平衡） |
| Traffic | [traffic/split.py](src/data/traffic/split.py) | 序列級（同序列幀不跨 split） | train 36／val 12／test 12 序列 |
| Freeway | [freeway/split.py](src/data/freeway/split.py) | 鏡頭級（同鏡頭幀不跨 split） | train 384／val 96／test 120（整鏡頭隔離） |

### 4.2 資料來源

| 資料集 | 內容 | 說明 |
|--------|------|------|
| UA-DETRAC | 100 序列交通監控影像 | 視角貼近高公局；自訂 XML 標註，依序列抽幀降冗餘 |
| Roboflow Accident | 平衡車禍／非車禍分類圖 | 原始 `Image/` 共 424 張（212:212） |
| 高公局 CCTV | 5 支 focus 鏡頭 MJPEG | 由 grabber 抓取，yolo11x 自動標註 |

### 4.3 劣化增強

訓練時對影像施加降解析度、JPEG 壓縮、模糊、噪點、亮度對比抖動
（[src/data/augment.py](src/data/augment.py)），模擬高公局低畫質 CCTV，使模型在部署
時能適應糊化畫面。**僅作用於 train，val／test 不增強。**

---

## 5. 模型訓練

### 5.1 車禍分類（Ray Train）

```powershell
docker compose exec ray-head python scripts/train_accident.py --epochs 50
```

- 起點權重 `yolo11n-cls.pt`，輸入 224×224，二元分類。
- 僅載入 train／val，test 完全隔離；依 `val_acc` 保留最佳 checkpoint。

### 5.2 車流偵測（Ray Train）

```powershell
# 骨架驗證（少序列、小 epoch）
docker compose exec ray-head python scripts/train_traffic.py --limit 5 --epochs 2
# 正式訓練
docker compose exec ray-head python scripts/train_traffic.py --epochs 30
```

- 起點權重 `yolo11n.pt`，輸入 640×640，單類 Vehicle。
- 依序列三分，僅載入 train／val；依 `val_loss` 保留最佳 checkpoint。

### 5.3 高公局微調（Ultralytics + Ray Tune）

```powershell
# (1) yolo11x 自動標註高公局白天影像（知識蒸餾的 teacher）
docker compose exec ray-head python scripts/prelabel_freeway.py --coco --no-roi

# (2) 微調（imgsz 960、letterbox、整鏡頭隔離 test）
docker compose exec ray-head python scripts/finetune_freeway.py --epochs 100 --test-ratio 0.1

# (3) 超參搜尋（選用，ASHA）
docker compose exec ray-head python scripts/tune_freeway.py --iterations 12 --epochs 30
```

微調採 Ultralytics 原生訓練，內建 mAP 評估、early-stopping、letterbox。`finetune_freeway.py`
的預設超參即由 Ray Tune 搜尋所得。

> 單 GPU 環境下各訓練任務不可同時執行。所有 checkpoint 輸出至 `ray_results/<任務>/`。

---

## 6. 模型評估

三個評估腳本各對應其 held-out test set 與模型格式：

| 腳本 | 模型格式 | Test 來源 | 評估指標 |
|------|----------|-----------|----------|
| [eval_accident.py](scripts/eval_accident.py) | Ray Train checkpoint | `accident/test` | accuracy／precision／recall／F1／混淆矩陣 |
| [eval_traffic.py](scripts/eval_traffic.py) | Ray Train checkpoint | DETRAC test 序列 | mAP@0.5（自寫 VOC all-point 積分） |
| [eval_freeway.py](scripts/eval_freeway.py) | Ultralytics `best.pt` | `freeway_det/test` | mAP@0.5／mAP@0.5:0.95（Ultralytics 原生 val） |

```powershell
docker compose exec ray-head python scripts/eval_accident.py
docker compose exec ray-head python scripts/eval_traffic.py
docker compose exec ray-head python scripts/eval_freeway.py
```

評估結果見 [§9 效能總結](#9-效能總結)。

---

## 7. 服務部署（Serve 推論 + RAY MONITOR）

系統提供兩個獨立網頁：**Serve 相機儀表板**（:8000，跑模型、佔 GPU）與
**RAY MONITOR**（:8501，純觀察叢集、不佔 GPU）。兩者互不依賴。

### 7.1 Serve 相機儀表板 — 啟動

```powershell
docker compose exec ray-head python scripts/serve_dashboard.py `
    --poll-interval 2.0 --accident-conf-th 0.97
# 開啟瀏覽器：http://localhost:8000/
```

| 參數 | 預設 | 說明 |
|------|------|------|
| `--poll-interval` | 4.0 | 每輪輪詢間隔（秒） |
| `--conf` | 0.4 | 車輛偵測信心門檻 |
| `--accident-conf-th` | 0.97 | 車禍報警門檻（連續 3 幀超過才報） |
| `--no-roi` | 關 | 關閉 ROI 幾何過濾，偵測全幅車輛 |
| `--no-gpu` | 關 | CPU 推論並釋出 GPU（demo 邊訓練邊看用） |

### 7.2 Serve 架構

Ray Serve 以單 replica（預設佔 1 GPU）常駐，背景非同步迴圈對 5 支鏡頭依序執行：

1. 由 MJPEG 串流抓取最新幀（[grabber.py](src/data/freeway/grabber.py)）
2. Traffic 偵測 → 畫框、計數
3. （選用）ROI 幾何過濾 → 計算車流數量／密度分級
4. Accident 分類 → P(accident)，並做**連續確認**（最近 5 幀需連續 3 幀超過門檻才判定為車禍事件）

結果快取為每鏡頭的標註影像與 JSON 指標。

### 7.3 HTTP API

| 端點 | 方法 | 回應 |
|------|------|------|
| `/` | GET | 監控儀表板（HTML，同源免 CORS proxy） |
| `/live_focus/<cctv_id>.jpg` | GET | 畫好框的最新標註幀（JPEG） |
| `/live_focus/<cctv_id>.json` | GET | 偵測指標 JSON |
| `/clips` | GET | 列出可用車禍片段 + 目前注入狀態 |
| `/inject/<cctv_id>` | POST | 讓該鏡頭改播車禍片段（`?clip=12.mp4`，省略則隨機）|
| `/inject/<cctv_id>/clear` | POST | 取消注入，恢復即時串流 |

JSON 欄位：

```json
{
  "num_detections": 7,
  "count_level": "LOW",
  "density_level": "LOW",
  "is_accident": false,
  "accident_conf": 0.86,
  "captured_at": "2026-06-03 20:40:08"
}
```

### 7.4 前端儀表板

前端 [dashboard.html](src/serve/dashboard.html) 為 5 宮格即時監控介面，包含車流統計、
事件日誌與車禍警報彈窗，由 Ray Serve 同源提供，每 2 秒更新一次。

介面在 team edit 基礎上做過**可讀性改版**：等寬字改用 Cascadia Code/Consolas、提亮
暗底文字對比、放大鏡頭名稱與車流數字（→20px）與事件日誌（→12px），並加寬日誌的
鏡頭欄避免換行，使投影／錄影 demo 時遠看仍清晰。

> **注意**：須透過 `http://localhost:8000/` 開啟（非直接開啟 HTML 檔案），否則前端
> 的 API 請求無法連至後端。

### 7.5 車禍片段注入（驗證用）

為驗證車禍偵測鏈路，可將真實車禍片段（`datasets/accident/video/accident/` 共 101 支）
注入任一鏡頭，取代即時串流送入推論：

```powershell
# 列出片段
curl http://localhost:8000/clips
# 注入（省略 clip 則隨機）
curl -X POST "http://localhost:8000/inject/CCTV-N1-S-34.018-M?clip=12.mp4"
# 恢復即時
curl -X POST "http://localhost:8000/inject/CCTV-N1-S-34.018-M/clear"
```

> 此驗證實測結果見 [§10 已知限制](#10-已知限制)——車禍片段的 `accident_conf` 與正常
> 畫面重疊，證實 Accident 模型在此資料分佈無鑑別力（屬待追蹤問題）。

### 7.6 RAY MONITOR — 叢集監控總覽

獨立服務（[scripts/monitor.py](scripts/monitor.py)），以輕量 driver 連上叢集，
**只做唯讀狀態查詢，不載入模型、不佔 GPU**，因此叢集一啟動即可開、與 Serve／
訓練是否在跑無關。

```powershell
docker compose exec -d ray-head python scripts/monitor.py
# 開啟瀏覽器：http://localhost:8501/
```

| 區塊 | 內容 | 資料來源 |
|------|------|----------|
| **叢集節點** | active 節點數、CPU/GPU/Object Store 總量、每節點負載 | `ray.nodes()` + Dashboard `/nodes` |
| **每節點負載** | CPU%／GPU%／MEM%／OBJ% + **Ray 任務數**（Ray Data 時即時跳動） | 狀態 API（running task 依節點分組）|
| **Ray 元件活動** | Data／Train／Tune／Serve 即時 active/idle 與正在做什麼 | `ray.util.state`（list_tasks / list_actors）|

端點：`GET /`（總覽頁）、`GET /cluster.json`（節點負載）、`GET /components.json`（元件活動）。

> **設計重點**：node 卡片以「每節點 running task 數」當 Ray 邏輯負載——比 Dashboard 的
> 物理 CPU%（更新慢）更靈敏，訓練啟動 Ray Data 時 worker 節點會立刻顯示任務數上升、
> Object Store 跨節點填充，直觀呈現多節點分工。

---

## 8. 專案結構

```
src/
├── core/          叢集連線（init_ray）
├── modeling/      YOLO11 模型載入（traffic／accident）
├── data/
│   ├── augment.py 劣化增強（兩案共用）
│   ├── traffic/   UA-DETRAC → Ray Data 管線、序列切分
│   ├── accident/  Roboflow → Ray Data 管線、分層切分
│   └── freeway/   高公局抓取、ROI、自動標註、鏡頭切分
├── train/
│   ├── traffic/   Ray Train 偵測訓練
│   └── accident/  Ray Train 分類訓練
├── infer/         推論（traffic／accident／coco_vehicle）
├── eval/          評估指標（mAP@0.5、分類指標）
├── serve/         Ray Serve 相機推論（app.py + dashboard.html）
└── monitor/       RAY MONITOR 叢集監控（state.py + overview.html）

scripts/           進入點：train_* / eval_* / finetune_freeway / tune_freeway /
                   serve_dashboard / monitor / collect_freeway / prelabel_freeway
datasets/          資料（不納入版控）
ray_results/       訓練與微調輸出（不納入版控）
```

> **服務分工**：`serve/` 跑模型推論（佔 GPU，相機畫面）；`monitor/` 純觀察叢集
> （不佔 GPU，從零可看）。兩者獨立，可單獨或同時運行。

---

## 9. 效能總結

於 held-out test set（訓練全程不可見）上的評估結果：

| 模型 | Test Set | 主要指標 | 補充 |
|------|----------|----------|------|
| **Accident** | 62 張（圖片級） | accuracy **88.7%**／macro F1 **0.886** | accident recall 0.774、non-accident 零誤報 |
| **Traffic** | 12 序列 1417 幀（序列級） | mAP@0.5 **82.0%** | recall 0.879 |
| **Freeway** | 120 張 1 鏡頭（鏡頭級） | mAP@0.5 **94.1%**／mAP@0.5:0.95 **0.890** | precision 0.955、recall 0.891 |

說明：

- **Accident**：混淆矩陣顯示車禍 recall 偏低（31 件漏 7 件），系統不會誤報，但有漏報
  傾向——此為乾淨 test set 才能揭露的真實表現。
- **Traffic／Freeway**：test 指標高於部署前的 val，因 held-out test 為完全未見的序列／
  鏡頭，能真實反映泛化能力。Freeway test 鏡頭 mAP 甚至高於訓練鏡頭 val（0.92），顯示
  對未見鏡頭泛化良好。

---

## 10. 已知限制

| 限制 | 說明 | 因應 |
|------|------|------|
| **🔴 Accident 模型無鑑別力（追蹤中）** | 分類器僅在土耳其 CCTV 車禍資料訓練，對台灣高公局有嚴重 domain gap；因高公局無車禍影片，無法 fine-tune。**實測（見下）證實它分不出車禍與正常畫面**。 | 即時服務以高門檻（0.97）抑制誤報，但等同也偵測不到真車禍。根本解需補充台灣道路車禍正樣本後重訓。 |
| **Freeway 標註為自動產生** | test 鏡頭 GT 由 yolo11x 自動標註，故指標反映「student 對 teacher 的逼近程度」，非對人工真值。 | 蒸餾框架下仍為有效的泛化指標；如需絕對精度應另建人工標註 test。 |
| **單 GPU 序列推論** | 即時服務 5 鏡頭共用 1 GPU 依序推論，每鏡頭實際刷新約 3 秒。 | 如需更高頻率可改批次／多 replica 平行推論。 |

### 車禍片段注入驗證結果（2026-06-03）

以 [§7.5 注入機制](#75-車禍片段注入驗證用) 將真實車禍片段送入推論，量測 `accident_conf`：

| 輸入來源 | accident_conf |
|----------|---------------|
| 正常台灣高公局（即時串流） | ~0.88 |
| 車禍片段 19.mp4 | 0.78 |
| 車禍片段 1.mp4 | 0.85 |
| 車禍片段 50.mp4 | 0.88 |
| 車禍片段 100.mp4 | 0.46 |

**結論**：車禍片段的分數（0.46–0.88）與正常畫面（~0.88）完全重疊，甚至偏低，代表
Accident 模型輸出與「是否真有車禍」不相關——在此部署資料分佈上**無鑑別力**。比單純
「誤報」更根本，是 Accident 案的核心缺口。

**待辦**：取得台灣道路車禍正樣本 → 讓 Accident 也能 fine-tune（對齊 Traffic 走知識蒸餾
或人工標註）→ 重訓後重跑本注入驗證確認鑑別力。

---

## 11. 指令速查

| 階段 | 指令 |
|------|------|
| 啟動 3 節點叢集 | `docker compose up -d ray-head ray-worker-1 ray-worker-2` |
| 叢集監控總覽 | `python scripts/monitor.py`（:8501，不佔 GPU）|
| 切分 Accident | `python -m src.data.accident.split --src .../Image --out .../accident` |
| 訓練 Accident | `python scripts/train_accident.py --epochs 50` |
| 訓練 Traffic | `python scripts/train_traffic.py --epochs 30` |
| 微調 Freeway | `python scripts/finetune_freeway.py --epochs 100 --test-ratio 0.1` |
| 超參搜尋 | `python scripts/tune_freeway.py --iterations 12 --epochs 30` |
| 評估 Accident／Traffic／Freeway | `python scripts/eval_accident.py`（traffic／freeway 同）|
| 抓取 CCTV | `python scripts/collect_freeway.py --target-per-camera 200` |
| 自動標註 | `python scripts/prelabel_freeway.py --coco --no-roi` |
| 相機推論服務 | `python scripts/serve_dashboard.py --poll-interval 2.0` |
| 邊訓練邊看監控 | `python scripts/serve_dashboard.py --no-gpu`（讓出 GPU 給訓練）|
| 關閉叢集 | `docker compose down` |

> 所有指令前綴 `docker compose exec ray-head`（背景常駐服務加 `-d`）。

---

## 附錄：開發筆記

本專案的設計演進、取捨理由與除錯過程記錄於 [NOTE.md](NOTE.md)。
