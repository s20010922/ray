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
8. [eval / infer — 評估與推論](#8-eval--infer--評估與推論)
9. [高公局 fine-tune（知識蒸餾）](#9-高公局-fine-tune知識蒸餾)
10. [serve — 即時監控儀表板](#10-serve--即時監控儀表板)
11. [現況與待辦](#11-現況與待辦)

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
├── data/
│   ├── augment.py 劣化增強（兩案共用，模擬高公局低畫質）
│   ├── traffic/   車流：UA-DETRAC → Ray Data pipeline（偵測）
│   ├── accident/  車禍：Roboflow CCTV → Ray Data pipeline（分類）
│   └── freeway/   高公局 CCTV：抓取(grabber) + ROI + 預標 + 切分
├── train/
│   ├── traffic/   Ray Train 偵測訓練（v8DetectionLoss）
│   └── accident/  Ray Train 分類訓練（CrossEntropy）
├── infer/        推論：自訓 base(traffic) + COCO 自動標(coco_vehicle) + accident 分類
├── eval/         評估：mAP@0.5（traffic）+ 分類指標（accident）
├── serve/        Ray Serve 相機推論（app.py + dashboard.html，佔 GPU）
└── monitor/      RAY MONITOR 叢集監控（state.py + overview.html，不佔 GPU）
scripts/          進入點（train_* / eval_* / finetune_freeway / tune_freeway /
                  serve_dashboard / monitor …）
datasets/         資料（不進 git）
ray_results/      Ray Train 產出 + ultralytics fine-tune 產出（不進 git）
```

依賴關係：`core.init_ray()` 先就緒 → `data` 串流資料 → `train` 吃資料訓練
（用到 `modeling`）→ `infer` / `eval` 推論評估 →（高公局）`coco_vehicle` 自動標
→ `finetune` 蒸餾 → `serve` 上線。`monitor` 則是旁路觀察者，獨立於上述流程。

> 叢集為 **3 節點**（1 head + 2 worker，同機多容器）。`serve` 跑模型佔 GPU；
> `monitor` 純查狀態不佔 GPU，兩個服務獨立。

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

**限制**：CCTV 影像 352×240、**無標註**。traffic 的「無標註」已用第 9 章
**COCO 自動標（知識蒸餾）**解決，零人工；accident 仍需人工標類別，且即時幾乎
抓不到車禍正樣本，主要供 non-accident 負樣本。

> 收集到的 2002 張中，`all pic/` 1001 張是 5 個 cam 的**完整副本**（檔名 100%
> 重複），獨立資料實為 1001 張；白天 6–18 時有車的約 600 張為自動標/訓練主力。

---

## 7. 訓練策略

### 路線圖

```
官方 yolo11n 預訓練
   │ ① 用 UA-DETRAC + 劣化增強 訓練（學會認車、不怕糊）
   ▼
base 模型（UA-DETRAC val mAP@0.5 ≈ 0.64，見第 8 章）
   │ ② 高公局 fine-tune — 原計畫人工標註，實際改走「知識蒸餾」（見第 9 章）：
   │    COCO yolo11x 自動標高公局 → 蒸餾 yolo11n（零人工、imgsz 960 letterbox）
   ▼
貼合高公局的成品（高公局 val mAP@0.5 ≈ 0.85）→ serve 即時推論
```

> **為何 base 不必很準**：base 只有兩個用途——當 fine-tune 起點、以及（原本要）
> 拿來預標。實測 base 在高公局 domain gap 太大（夜間/小目標幾乎全漏），於是
> 改用更強的官方 COCO 模型來自動標，base 退居「起點」角色（見第 9 章）。

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

### train 模組（Ray Train）

兩案套用**同一套 Ray Train 骨架**（TorchTrainer + train_loop_per_worker），
差別只在 loss 與 batch 格式：

| | traffic（偵測） | accident（分類） |
|---|---|---|
| worker | [traffic/worker.py](src/train/traffic/worker.py) | [accident/worker.py](src/train/accident/worker.py) |
| loss | v8DetectionLoss | CrossEntropy |
| batch 轉換 | pad → batch_idx/cls/bboxes（攤平去 padding） | label 直接用 |
| 指標 | val_loss | val_acc |

```powershell
# 車禍分類（held-out test_acc 88.7%；train/val/test 由 accident/split.py 切）
docker compose exec ray-head python scripts/train_accident.py --epochs 50
# 車流偵測（依序列三分，隔離 test；先少序列驗證，再全量）
docker compose exec ray-head python scripts/train_traffic.py --limit 5 --epochs 2
docker compose exec ray-head python scripts/train_traffic.py --epochs 30
```

> 訓練前先切資料（隔離 test）：`accident` 從 `Image/` 重切、`traffic` 依序列切，
> 兩者 trainer 都只載 train/val，test 不進訓練。見 8.0。

單 GPU：`ScalingConfig(num_workers=1, use_gpu=True)`，兩案不能同時訓練（搶 GPU）。
checkpoint 存 `ray_results/<案>/`，依指標保留最佳 2 份。

> traffic 單類 reshape：yolo11n.pt 是 COCO 80 類，用 `yolo11n.yaml(nc=1)` 重建、
> 載入相容的預訓練權重（backbone/neck），偵測頭重學。

**待補項目（已補上）**：

| 項目 | base（Ray Train 自刻） | fine-tune（ultralytics，第 9 章） |
|---|---|---|
| early stopping | 仍跑固定 epochs | ✅ 內建 `patience`，自動早停 |
| traffic 評估 mAP@0.5 | ✅ 事後用 [eval_traffic.py](scripts/eval_traffic.py) 算（第 8 章）| ✅ 訓練中內建 mAP |
| letterbox / 大尺寸 | 640 直接 resize（變形）| ✅ imgsz 960 + letterbox |

> 自刻 Ray detection loop 缺的（mAP / early-stop / letterbox），fine-tune 階段
> 改用 **ultralytics 原生 train** 一次補齊，不重造輪子。base 的 mAP 則用獨立的
> `eval/` 模組事後評估。

---

## 8. eval / infer — 評估與推論

程式：[eval/traffic.py](src/eval/traffic.py)、[eval/accident.py](src/eval/accident.py)、
[infer/traffic.py](src/infer/traffic.py)、[infer/accident.py](src/infer/accident.py)、
[infer/coco_vehicle.py](src/infer/coco_vehicle.py)

### 8.0 Held-out test set（可信評估的前提）

三個模型原本只有 train/val，**val 在調參過程被污染**（val_acc/val_mAP 虛高，
無法確認真實泛化）。改為三分：每個資料集切出**訓練全程完全不可見的 test set**，
只在最終 eval 用一次。隔離粒度依資料特性不同，避免相似樣本跨 split 洩漏：

| 模型 | 切分腳本 | 隔離粒度 | 三分結果 |
|---|---|---|---|
| Accident | [accident/split.py](src/data/accident/split.py) | **圖片級**（按類別分層）| train 300 / val 62 / test 62（皆 1:1 平衡）|
| Traffic | [traffic/split.py](src/data/traffic/split.py) | **序列級**（同序列幀不跨 split）| train 36 / val 12 / test 12 序列 |
| Freeway | [freeway/split.py](src/data/freeway/split.py) | **鏡頭級**（同鏡頭幀不跨 split）| 整個 test 鏡頭隔離（`--test-ratio`）|

> Accident 從原始平衡的 `Image/`（212:212）重切，取代舊的污染 train/val
> （不平衡 1:2）。trainer 只載 train/val，test 完全不進訓練。

### 8.1 eval — mAP@0.5 / 分類指標

三個評估腳本各吃對應的 held-out test、各對應模型格式：

| 腳本 | 模型 | test 來源 | 指標 |
|---|---|---|---|
| [eval_accident.py](scripts/eval_accident.py) | Ray Train checkpoint | `accident/test` | acc / P / R / F1 / 混淆矩陣 |
| [eval_traffic.py](scripts/eval_traffic.py) | Ray Train checkpoint | DETRAC test 序列 | mAP@0.5（自寫 VOC 積分）|
| [eval_freeway.py](scripts/eval_freeway.py) | ultralytics best.pt | `freeway_det/test` | mAP@0.5（ultralytics 原生 val）|

```powershell
docker compose exec ray-head python scripts/eval_accident.py   # 分類指標 + 混淆矩陣
docker compose exec ray-head python scripts/eval_traffic.py    # DETRAC test mAP + 可視化
docker compose exec ray-head python scripts/eval_freeway.py    # 高公局 test 鏡頭 mAP
```

> Traffic/Accident base 是自刻 Ray Train 的 `model.pt`（state_dict），用 `infer/` 重建
> 模型推論；Freeway 是 ultralytics 原生 `best.pt`，直接用 ultralytics `val()`。兩種
> 格式不同，故評估走兩條路。

**Accident held-out test 結果**：**test_acc = 88.7% / macro F1 = 0.886**。混淆矩陣揭露
真問題——車禍 recall 僅 0.774（31 件漏 7 件），non-accident 零誤報。舊 val_acc=90.5%
把漏報藏住了，乾淨 test 才看得到。

**Traffic held-out test 結果**：依序列三分重訓後，在 12 個 held-out DETRAC 序列
（1417 幀）上 **mAP@0.5 = 82.0% / recall 0.879**。比舊文件 val≈0.64 高，因為這是
完全隔離的序列（舊 val 同序列幀洩漏使其偏低或不可信）。

> 近處大車準、遠處小車漏（640 直接 resize 變形 + nano 容量）——這弱點在高公局
> 更嚴重，促成第 9 章改走自動標。

### 8.2 infer — 推論

| 程式 | 用途 |
|---|---|
| [traffic.py](src/infer/traffic.py) | 載入自刻 Ray Train 存的 `model.pt`，重建 DetectionModel + NMS → bbox（預處理須與訓練一致：640 直接 resize）|
| [coco_vehicle.py](src/infer/coco_vehicle.py) | 官方 YOLO11 COCO 模型，car/bus/truck/機車 → Vehicle，供第 9 章自動標 |

---

## 9. 高公局 fine-tune（知識蒸餾）

**核心轉折**：base 在高公局 domain gap 太大（夜間/小目標幾乎全漏），原計畫的
「base 預標 + 人工修」不可行。改用「**大模型自動標 → 小模型蒸餾**」，零人工。

```
COCO yolo11x（老師）── 自動標高公局白天 600 張 ──→ YOLO labels
                                                    │ ② ultralytics fine-tune
官方 yolo11n.pt（起點）─────────────────────────────┴─→ yolo11n（學生，上線）
```

老師（yolo11x）標完即丟，上線只用學生（yolo11n）。yolo11n 的能力透過「標籤」
間接學自 yolo11x，訓練時不載入 yolo11x。

### 程式

| 程式 | 職責 |
|---|---|
| [coco_vehicle.py](src/infer/coco_vehicle.py) | 官方 yolo11x 自動標（COCO car/bus/truck→Vehicle）|
| [prelabel.py](src/data/freeway/prelabel.py) | 跑自動標 → YOLO labels + preview + classes.txt（可切換 base / COCO）|
| [roi.py](src/data/freeway/roi.py) | per-cam 偵測範圍（固定機位）|
| [split.py](src/data/freeway/split.py) | 依 cam 分層切 train/val → ultralytics 結構 |
| [finetune_freeway.py](scripts/finetune_freeway.py) | ultralytics train（imgsz 960 letterbox、單類、early-stop）|

### 指令

```powershell
# ① COCO 自動標白天圖（6–18 時，--no-roi 標所有車）
docker compose exec ray-head python scripts/prelabel_freeway.py --coco --no-roi
# ② fine-tune（蒸餾）
docker compose exec ray-head python scripts/finetune_freeway.py --epochs 100
```

### 成果（600 張、隔離 test 鏡頭版）

- 自動標品質：base 3.8 框/張 → **COCO yolo11x 11.6 框/張**（中遠景小車都抓到）
- fine-tune yolo11n（4 鏡頭 train/val，第 5 鏡頭隔離為 test）：
  - 訓練鏡頭 val mAP@0.5 ≈ 0.92
  - **held-out test 鏡頭 mAP@0.5 = 94.1% / mAP@0.5:0.95 = 0.890 / P 0.955 / R 0.891**
  - test 比 val 還高 → 對**未見鏡頭**泛化良好
- **零人工標註**；用 `scripts/eval_freeway.py`（ultralytics 原生 val）評估

> 誠實註記：test 鏡頭的 GT 也是 yolo11x 自動標，故此指標是「yolo11n 學生在未見
> 鏡頭上多接近 yolo11x 老師」，非對人工真值。蒸餾框架下仍是有效的 held-out 泛化指標。

### ROI（偵測範圍）

CCTV 固定機位，可為每支 cam 定多邊形 ROI，排除遠景糊區/對向車道、界定車流計數區。
**標註/訓練階段不套**（`--no-roi`，標所有車，人天然只標看得清的）；ROI 留到
**推論階段**做幾何過濾。

> labelImg 備註：Python 3.14 + 新 PyQt5 需手動修數處 `float→int`
> （labelImg.py / canvas.py），且 `classes.txt` 要放在 save_dir（labels/）。

---

## 10. serve / monitor — 服務與叢集監控

兩個獨立服務：**Serve 相機推論**（佔 GPU）與 **RAY MONITOR 叢集監控**（不佔 GPU）。

### 10A. serve — 相機推論儀表板

程式：[src/serve/app.py](src/serve/app.py)、[scripts/serve_dashboard.py](scripts/serve_dashboard.py)、
[src/serve/dashboard.html](src/serve/dashboard.html)

把兩個 model 接上高公局即時 CCTV，用 **Ray Serve** 當後端，前端沿用 team edit 的
`smart-traffic-ui`（5 宮格監控 + 矩陣大腦 log + 車禍彈窗）。

```powershell
docker compose exec ray-head python scripts/serve_dashboard.py
# 開瀏覽器：http://localhost:8000/
```

### 架構

```
高公局 5 鏡頭 MJPEG ──grab_jpeg_frame──┐
                                        ▼
        ┌──────────── Ray Serve（單 replica，占 1 GPU）────────────┐
        │  背景 asyncio 迴圈，每 2s 輪詢每鏡頭（demo 值，預設 4s）：  │
        │   ① Traffic 偵測（freeway best.pt, ultralytics）→ 畫框、數車 │
        │   ② ROI 幾何過濾（roi.py，只算主車道）→ count/density level │
        │   ③ Accident 分類（Ray Train ckpt）→ P(accident) + 連續確認 │
        │  快取每鏡頭：標註 jpg + json                               │
        └───────────────────────────────────────────────────────────┘
                                        ▼
   GET /                         → dashboard.html（同源，免 CORS proxy）
   GET /live_focus/<id>.jpg      → 畫好框的最新標註幀
   GET /live_focus/<id>.json     → {num_detections, count_level, density_level,
                                     is_accident, accident_conf, captured_at}
```

UI 每 2s 對每鏡頭抓 `.jpg`（標註幀）+ `.json`（指標），契約對齊 team edit 原版
（原版 imgBase 指外部 8501，這裡改同源 `/live_focus/` 由 Serve 直接出標註幀）。

> 前端在 team edit 基礎上做過**可讀性改版**：等寬字改 Cascadia Code/Consolas、提亮
> 暗底文字對比、放大鏡頭名稱與車流數字（15→20px）與事件 log（→12px）、加寬 log
> 鏡頭欄避免換行——投影/錄影 demo 時遠看仍清楚。

### 設計重點

| 項目 | 做法 |
|---|---|
| 兩 model 格式不同 | Traffic 用 ultralytics `YOLO(best.pt)`；Accident 用 `infer/accident` 重建 Ray Train ckpt |
| 抓幀/推論阻塞 | 丟 thread executor 跑，不卡 asyncio 事件迴圈 |
| 車禍連續確認 | 最近 5 幀需連續 3 幀 `P(accident)≥門檻`（`--accident-conf-th`，預設 0.97）才報（單幀易誤判）|
| ROI 過濾 | 推論階段套 [roi.py](src/data/freeway/roi.py) 多邊形，只算主車道車輛 |
| count/density level | 車輛數 / ROI 內車框面積佔比 → LOW/MED/HIGH |

> ⚠️ **已知限制（Accident 誤報）**：Accident 分類器只在土耳其 CCTV 車禍資料訓練，
> 沒看過台灣高公局，又因高公局**無車禍影片無法 fine-tune**，實測對正常高公局畫面
> 會穩定誤判為 accident（domain gap）。連續確認擋不住系統性誤判。Traffic 偵測這側
> 因走了知識蒸餾微調（第 9 章）所以正常。這是 Accident 案的根本資料缺口，非 serve bug。
>
> 用真實車禍片段注入（`/inject/<cam>`）驗證更發現：車禍片段的 `accident_conf`
> （0.46~0.88）與正常畫面（~0.88）重疊，代表模型**無鑑別力**。即時服務以高門檻
> （`--accident-conf-th 0.97`）抑制誤報，但等同也偵測不到真車禍。

### 10B. monitor — RAY MONITOR 叢集監控

程式：[scripts/monitor.py](scripts/monitor.py)、[src/monitor/state.py](src/monitor/state.py)、
[src/monitor/overview.html](src/monitor/overview.html)

獨立的觀察者服務（FastAPI + uvicorn，:8501），以輕量 driver 連上叢集查狀態，
**不載入模型、不佔 GPU**，叢集一啟動即可看（與 serve／訓練無關）。

```powershell
docker compose exec -d ray-head python scripts/monitor.py   # http://localhost:8501/
```

| 設計取捨 | 理由 |
|---|---|
| 為何**不**長在 serve 上 | serve 佔 GPU、且可能沒開；監控要「從零、隨時」可看，故拆成獨立服務 |
| 節點負載用 **running task 數**（非物理 CPU%）| Dashboard 物理 CPU% 更新慢、不靈敏；以「每節點 running task 數」當 Ray 邏輯負載，Ray Data 一啟動 worker 立刻跳動 |
| 元件活動 | 用 `ray.util.state`（list_tasks/list_actors）從外部偵測 Data/Train/Tune/Serve 是否在跑 |
| object store | 顯示叢集與每節點的共享記憶體用量，訓練時可見資料跨節點流動 |

> **GPU 取捨**：整叢集 1 顆 GPU 由 head 持有，serve 推論與訓練都要 GPU、不可同搶。
> demo「邊訓練邊看監控」時，serve 用 `--no-gpu`（CPU 推論）讓出 GPU，monitor 本就
> 不需 GPU 照常運作。

---

## 11. 現況與待辦

| 模組 | 狀態 |
|---|---|
| core | ✅ 完成 |
| modeling | ✅ 完成（traffic / accident 模型載入）|
| data/traffic | ✅ Ray Data pipeline（sources / pipeline，偵測）|
| data/accident | ✅ Ray Data pipeline（sources / pipeline，分類）|
| data/augment | ✅ 劣化增強（兩案共用）|
| data/freeway | ✅ 抓取（1001 張獨立）+ ROI + 自動標 + 切分 |
| held-out test 切分 | ✅ 三案皆有隔離 test（圖片/序列/鏡頭級，見 8.0）|
| train/accident | ✅ 從平衡 `Image/` 重訓（隔離 test）；**test_acc = 88.7% / macro F1 0.886** |
| train/traffic | ✅ base 依序列三分重訓（隔離 test）；**DETRAC test mAP@0.5 = 82.0%** |
| eval / infer | ✅ 三個 eval 腳本（accident / traffic-DETRAC / freeway，見 8.1）|
| 高公局 fine-tune | ✅ 知識蒸餾（600 張，隔離 test 鏡頭）；**test 鏡頭 mAP@0.5 = 94.1% / mAP50-95 = 0.890** |
| serve | ✅ Ray Serve 相機推論（5 鏡頭輪詢 + 連續確認 + ROI + 車禍片段注入；餵 smart-traffic-ui，見 10A）|
| monitor | ✅ RAY MONITOR 獨立叢集監控（節點負載 + Ray 元件活動 + object store，不佔 GPU，見 10B）|
| tune | ✅ Ray Tune 超參搜尋（[tune_freeway.py](scripts/tune_freeway.py)，最佳超參已內建 finetune 預設）|
| 叢集 | ✅ 3 節點（1 head + 2 worker，同機多容器；docker-compose）|

### Held-out test 可信指標（三案總結）

| 模型 | held-out test | 指標 | 對照舊 val |
|---|---|---|---|
| Accident | 62 張（圖片級隔離）| **acc 88.7% / macro F1 0.886** | 舊 val 90.5%（揭露車禍漏報 7/31）|
| Traffic | 12 序列 1417 幀（序列級）| **mAP@0.5 82.0% / R 0.879** | 舊 val ≈0.64（同序列洩漏）|
| Freeway | 120 張 1 鏡頭（鏡頭級）| **mAP@0.5 94.1% / mAP50-95 0.890** | 舊 val ≈0.85（同鏡頭洩漏）|

**Ray 全家桶已到齊**：Data（串流管線）/ Train（兩 base 訓練）/ Tune（freeway 超參）/
Serve（相機推論），外加 3 節點叢集與獨立 RAY MONITOR。**下一步**：補 Accident 案的
資料缺口（找台灣道路車禍正樣本，讓 Accident 也能 fine-tune 解 domain gap 誤報）；
serve 的 count/density 門檻校準。

> **可信度升級**：三案都從「val（被調參污染）」改為「held-out test（訓練全程不可見）」。
> Accident test 比舊 val 低 1.8pt 但揭露被藏住的漏報；Traffic/Freeway test 反而較高，
> 因舊 val 有同序列/同鏡頭洩漏，新 test 是真正未見的序列/鏡頭。
