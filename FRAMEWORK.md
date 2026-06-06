# 整套框架釐清

## 一、兩個 Base 模型 + 一個 Fine-tune

| 任務 | 型態 | 模型 | 資料集 | 訓練方式 | 角色 |
|------|------|------|--------|--------|------|
| **Accident** | 分類 | YOLO11n-cls | Roboflow 車禍圖 | Ray Train | **Base 模型**（獨立，無法 fine-tune：高公局無車禍影片）|
| **Traffic** | 偵測 | YOLO11n | UA-DETRAC | Ray Train | **Base 模型**（偵測來源）|
| **Freeway** | 偵測 | YOLO11n | 高公局 CCTV（yolo11x 自動標）| ultralytics | **不是獨立模型**：Traffic base 的蒸餾微調版 |

> **Freeway 不是第三個 base**：它是 yolo11x（teacher）自動標高公局照片後，
> 拿來 fine-tune Traffic base（student）的產物。知識蒸餾，零人工標註。

---

## 二、資料來源結構

```
datasets/
├── accident/                          ← Roboflow CCTV（分類）
│   ├── Image/                         （原始，未切）
│   │   ├── accident/       (212 張)
│   │   └── non-accident/   (212 張)
│   ├── train/              (目前 510 張，需重切)
│   ├── val/                (目前 92 張，需重切)
│   └── test/               (空，需切出)
│
├── freeway_raw/                       ← 高公局 CCTV 抓取（原始）
│   ├── CCTV-N1-N-37.050-M/  (200 張)
│   ├── CCTV-N1-S-34.018-M/  (201 張)  ← 最多
│   ├── CCTV-N1-S-93.080-M/  (200 張)
│   ├── CCTV-N1H-S-17.450-M/ (200 張)
│   └── CCTV-N3-S-40.980-M/  (200 張)
│   = 5 個鏡頭，1001 張獨立資料
│
├── freeway_yolo/                      ← COCO 自動標（已標註）
│   ├── images/              (600 張，來自上面白天部分)
│   ├── labels/              (600 個 .txt，YOLO 格式)
│   └── data.yaml
│
├── freeway_det/                       ← 切好的訓練結構（ultralytics）
│   ├── images/
│   │   ├── train/           (需重切 + test)
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
│
└── /data/detrac/                      ← UA-DETRAC（偵測，只讀）
    ├── DETRAC-Images/       (100 序列，含標註的 60 個)
    └── DETRAC-Train-Annotations-XML/  (60 個 XML)
```

---

## 三、訓練路線圖

```
【第一階段：訓練 Base 模型】

官方 yolo11n-cls.pt ──→ 用 Roboflow Accident 資料訓練 ──→ Accident Base（分類）
                       Ray Train + 劣化增強              val_acc ~90%（舊 val 指標）
                       
官方 yolo11n.pt ──────→ 用 UA-DETRAC 資料訓練 ────────→ Traffic Base（偵測）
                       Ray Train + 劣化增強             UA-DETRAC val mAP@0.5 ≈ 0.64

【第二階段：Fine-tune】

官方 yolo11n.pt ──────→ 用 Freeway COCO 自動標訓練 ────→ Freeway Model（偵測）
                       ultralytics + 劣化增強           高公局 val mAP@0.5 ≈ 0.85
                       
           （或改為：Traffic Base → Fine-tune）
```

---

## 四、資料切分策略

### Accident（分類）
**目前問題**：train/val 不平衡（1:2），且被訓練污染（調參用）

**新策略**：
```
Image/（原始 424 張，平衡）
  └─ split.py 按類別分層抽樣
       ├─ train/ 70% (297 張)
       ├─ val/   15% (64 張)
       └─ test/  15% (63 張)  ← 新增，完全隔離
```
- 確保三份的 accident:non-accident 比例一致
- test set 在訓練全程不載入

### Traffic（DETRAC 偵測）
**目前問題**：60 個序列無 test 隔離（全部混在 train/val）

**新策略**：
```
60 個序列
  └─ split.py 按序列分層分割
       ├─ train/ 60% 的序列
       ├─ val/   20% 的序列
       └─ test/  20% 的序列  ← 新增，完全隔離
       
每序列內幀數不變，只是序列級別隔離（避免同序列的相鄰幀洩漏）
```

### Freeway（高公局 CCTV）
**目前問題**：freeway_yolo 無 test 隔離（全 600 張混在 train/val）

**新策略**：
```
freeway_yolo/ 的 5 個鏡頭
  └─ make_det_split(test_ratio=0.1)
       ├─ train/ 4 個鏡頭 (480 張)
       ├─ val/   同 4 個鏡頭 (120 張)
       └─ test/  1 個鏡頭 (60 張)  ← 新增，完全隔離
       
整鏡頭級別隔離（避免同鏡頭的相同視角幀洩漏）
```

---

## 五、評估指標

### Accident（分類）
- **指標**：accuracy / per-class precision / recall / F1 / macro F1 / confusion matrix
- **評估腳本**：`scripts/eval_accident.py`（吃 Ray Train checkpoint）
- **Test set 結果**：✅ **test_acc = 88.7% / macro F1 = 0.886**（held-out 62 張）
  - accident recall 0.774（31 件漏 7 件）、non-accident recall 1.0（零誤報）
  - 舊 val_acc=90.5% 把「車禍漏報」藏住了，乾淨 test 才揭露

### Traffic（偵測）
- **指標**：mAP@0.5 / precision / recall / GT 框數 / 預測框數
- **評估腳本**：`scripts/eval_traffic.py`（吃 Ray Train checkpoint，自寫 mAP）
- **Test set 結果**：✅ **mAP@0.5 = 82.0% / recall 0.879**（12 個 held-out 序列、1417 幀）

### Freeway（偵測）
- **指標**：mAP@0.5 / mAP@0.5:0.95 / precision / recall
- **評估腳本**：`scripts/eval_freeway.py`（吃 ultralytics best.pt，用原生 val）
- **Test set 結果**：✅ **mAP@0.5 = 94.1% / mAP50-95 = 0.890**（held-out 鏡頭 120 張）
  - test 比 val(0.92) 還高 → 對未見鏡頭泛化良好

---

## 六、執行順序（優先級）

### 🔴 Phase 1：Accident Base（高優先）
```
1. python src/data/accident/split.py
   └─ 從 Image/ 重切 → accident/{train,val,test}/{accident,non-accident}/

2. python scripts/train_accident.py --epochs 50
   └─ Ray Train，載 train/val，隔離 test

3. python scripts/eval_accident.py
   └─ 在 test set 上跑，看真實指標（vs 舊 val_acc=90.5%）
```

### 🔴 Phase 2：Traffic Base（高優先）
```
1. 建立 src/data/traffic/split.py（還未實現）
   └─ 從 DETRAC 60 序列，依序列切 train/val/test

2. python scripts/train_traffic.py --epochs 30
   └─ Ray Train，載 train/val，隔離 test

3. python scripts/eval_traffic.py
   └─ 在 test set 上跑（目前跑的是 val）
```

### 🟡 Phase 3：Freeway Fine-tune（中優先）
```
1. python scripts/finetune_freeway.py --epochs 100 --test-ratio 0.1
   └─ ultralytics train，自動切 test，載 train/val，隔離 test

2. python scripts/eval_freeway.py
   └─ 在 test set 上跑（需新建）
```

---

## 七、檔案與腳本映射

| 任務 | 切分 | 訓練 | 評估 | 狀態 |
|------|------|------|------|------|
| Accident | `src/data/accident/split.py` ✅ | `scripts/train_accident.py` ✅ | `scripts/eval_accident.py` ✅ | ✅ test_acc 88.7% |
| Traffic | `src/data/traffic/split.py` ✅ | `scripts/train_traffic.py` ✅ | `scripts/eval_traffic.py` ✅（改寫吃 DETRAC test）| ✅ test mAP@0.5 82.0% |
| Freeway | `src/data/freeway/split.py` ✅ | `scripts/finetune_freeway.py` ✅ | `scripts/eval_freeway.py` ✅（新建，ultralytics val）| ✅ test mAP@0.5 94.1% |

---

## 八、預期產出

```
ray_results/
├── accident/TorchTrainer_*/
│   └── checkpoint_00NNN/
│       └── model.pt  ← 新訓練的 checkpoint
│
├── traffic/TorchTrainer_*/
│   └── checkpoint_00NNN/
│       └── model.pt  ← 新訓練的 checkpoint
│
└── freeway_final/weights/
    ├── best.pt  ← fine-tune 的最佳權重
    └── last.pt

評估結果（log）：
├── Accident: test_accuracy=?,  macro_f1=?
├── Traffic: test_mAP@0.5=?
└── Freeway: test_mAP@0.5=?
```

---

## 九、關鍵概念

### Held-out Test Set
- **定義**：訓練全程完全不見的資料
- **目的**：無偏評估模型泛化能力
- **實踐**：
  - Accident：按類別分層，保留 15% 為 test
  - Traffic：按序列分層，保留 20% 的序列為 test
  - Freeway：按鏡頭分層，保留 1 個鏡頭為 test

### 劣化增強（兩案共用）
- **降解析度**：模擬高公局 352×240
- **JPEG 壓縮**、**模糊**、**噪點** ：模擬低畫質
- **只在 train 做增強**，val/test 不做

### 資料隔離級別
- **Accident**：圖片級（同圖不跨 split）
- **Traffic**：序列級（同序列的幀不跨 split）
- **Freeway**：鏡頭級（同鏡頭的幀不跨 split）

---

## 十、與舊流程的差異

| 項目 | 舊流程 | 新流程 |
|------|--------|--------|
| **Accident 資料** | train/val（team edit，不平衡 1:2） | train/val/test（重切自 Image/，平衡） |
| **Accident 指標** | val_acc=90.5%（污染） | test_acc=88.7%（乾淨） |
| **Traffic 指標** | 無（只有訓練 loss） | test_mAP@0.5=82.0%（DETRAC 隔離 test） |
| **Freeway 指標** | val_mAP@0.5（train/val 無隔離） | test_mAP@0.5=94.1%（隔離 test 鏡頭） |
| **可信度** | ❌ 低（val 被污染） | ✅ 高（test 隔離） |

---

## 十一、部署與監控（叢集 + 服務）

### 3 節點叢集
- 1 head（8 CPU + 1 GPU）+ 2 worker（各 4 CPU），同一台機器多容器。
- Ray Data 前處理會分散到 3 節點，object store 跨節點傳資料。
- 啟動：`docker compose up -d ray-head ray-worker-1 ray-worker-2`

### 兩個服務（獨立）
| 服務 | 程式 | 埠 | GPU | 角色 |
|------|------|----|-----|------|
| **Serve 相機推論** | `scripts/serve_dashboard.py` + `src/serve/` | 8000 | ✅ 佔用 | 5 鏡頭即時偵測畫面、車禍片段注入驗證 |
| **RAY MONITOR** | `scripts/monitor.py` + `src/monitor/` | 8501 | ❌ 不需 | 叢集節點負載 + Ray 元件活動（觀察者）|

- 只有 1 顆 GPU，serve 推論與訓練不可同搶；demo 邊訓練邊看監控時 serve 用 `--no-gpu`。
- monitor 不依賴 serve，叢集一啟動即可看（從零可見）。

### Serve 相機儀表板（:8000）
- 5 鏡頭背景輪詢（demo `--poll-interval 2.0`，預設 4s），車禍連續確認門檻
  `--accident-conf-th 0.97`（最近 5 幀連 3 幀超過才報），可 `--no-roi` 偵測全幅車輛。
- 前端沿用 team edit `smart-traffic-ui` 並做可讀性改版（Cascadia Code 等寬字、提亮
  對比、放大車流數字與事件 log），投影／錄影遠看清晰。
- `/inject/<cam>` 可注入真實車禍片段驗證——實測證實 Accident 模型在此資料分佈無
  鑑別力（見 §十 與 README §10）。

