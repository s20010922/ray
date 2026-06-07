# 整套框架釐清

> 本檔說明系統各部分如何組合。完整操作見 [README.md](README.md)，演進與除錯見 [NOTE.md](NOTE.md)。

## 一、兩個模型、兩條流程

| 流程 | 模型 | 型態 | 資料 | 訓練方式 | 角色 |
|------|------|------|------|----------|------|
| **車流（Freeway）** | YOLO11s | 物件偵測 | 高公局 CCTV（yolo11x+SAHI 自動標）| ultralytics + Ray 四階段 | 偵測前端骨幹 |
| **車禍（Accident）** | LSTM/GRU/1D-CNN | 軌跡時序分類 | AccidentBench 真實高速公路事故 | Ray 四階段 | 事故判斷後端 |

兩個模型**不是平行獨立**，而是**前後串接**：YOLO 偵測 + ByteTrack 追蹤是共用前端，
車流走「計數／密度」、車禍走「軌跡→運動特徵→時序模型」。

```
            YOLO 偵測 + ByteTrack 追蹤（共用前端）
                        │
            ┌───────────┴───────────┐
          車流計數                事故時序判斷
        （用框數量）        （軌跡→運動特徵→LSTM）
```

## 二、為什麼車禍模型「跨資料集」可行

時序模型**吃抽象運動數字、不吃像素**。像素差異（畫質／角度／解析度）在偵測+追蹤
那關被吸收，流到時序模型的只剩速度、加速度、車距、航向變化等物理量 —— 這些量在
AccidentBench 與高公局之間意義一致。

```
AccidentBench（訓練）──yolo11x──┐
                                ├─►[速度,加速度,車距,…]─► 同一個時序模型
高公局 CCTV（部署）──yolo11s────┘
```

→ 換資料集 / 換部署場景**只換前端偵測器**，時序模型與特徵公式不動。fps 也要對齊
（訓練降採樣到部署等效 ~10fps）。

## 三、資料來源（現行）

```
/data/accident/                 ← AccidentBench（唯讀掛載 ./ACCIDENT）
├── real_videos/                  2027 支真實事故影片（YouTube 來源）
└── metadata-real.csv             事故幀/事故框/型態/場景/晝夜/畫質
     → 篩 highway+day+畫質OK = 271 支對齊高公局子集

datasets/                        （可寫）
├── weights/yolo11x.pt            追蹤用 teacher（離線預先下載）
├── freeway_yolo/                 高公局自動標註結果（車流訓練輸入）
└── accident_seq/                 ① Ray Data 產出：train/val/test.npz + scaler + clips.json
```

> 舊系統的 UA-DETRAC／Roboflow／合成台灣車禍／DoTA／CADP／UCF-Crime 已**不再使用**
> （資料集評選結論見 NOTE.md：唯 AccidentBench 同時滿足「真實事故 + 可偵測 + domain 對 + 可下載」）。

## 四、Ray 四階段對應

| 階段 | 車流（Freeway） | 車禍（Accident） | 用到哪些節點 |
|------|------|------|------|
| ① Data | 鏡頭級切分、CPU 前處理 | yolo11x+ByteTrack 追蹤抽軌跡 | 車流：CPU 多節點／車禍：GPU(head) |
| ② Tune | ultralytics ASHA（GPU） | ASHA 搜時序超參（**CPU 三節點平行**）| 車禍 Tune 才會吃滿 2 個 worker |
| ③ Train | TorchTrainer + yolo11s（GPU） | TorchTrainer + 時序模型 | head(GPU) |
| ④ Serve | 多鏡頭儀表板（:8000） | 接前端後的時序判斷（規劃中）| head(GPU) |

> **節點配置邏輯**：GPU 只有 head 一顆，GPU 綁定工作（追蹤／偵測訓練／推論）只在 head；
> 兩個 CPU-only worker 唯一真正吃滿的場景是**車禍模型的 CPU 平行 Ray Tune**（時序模型小，
> CPU 訓練快，ASHA 多試驗同時跑）。這也是「三節點叢集」名副其實之處。

## 五、關鍵設計取捨

| 議題 | 決定 | 理由 |
|------|------|------|
| 事故資料 | AccidentBench 真實片（非合成）| 唯一同時可偵測+domain 對+真實事故的可下載資料 |
| 前端偵測 | 訓練 yolo11x / 部署 yolo11s | 訓練要強跨域偵測；部署用高公局自身微調模型 |
| 正樣本標註 | 肇事車（IoU 最高軌跡）±1s | 排除正常過路車的標籤雜訊 |
| 不平衡 | pos_weight + AP 指標 + 分層切分 | 事故正樣本僅約 1%，accuracy 無意義 |
| 追蹤器 | ByteTrack（ultralytics 內建）| 純運動、輕量，足以串軌跡算運動特徵 |
