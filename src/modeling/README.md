# `src/modeling` — 模型載入

兩個任務各一個 YOLO11 模型。這個模組只負責「**載入模型 + 集中模型設定**」，
不含訓練、推論邏輯（那些在 `train` / `serve`）。

## 由來：參考 team edit，升級到 YOLO11

team edit 用的是 YOLOv8n；我們**沿用它的任務設定**（類別、imgsz），
但把模型**升級到最新世代 YOLO11n**（同尺寸下精度更好、一樣輕量）。

| 檔案 | 任務 | 模型 | imgsz | 類別 |
|---|---|---|---|---|
| `traffic.py` | 物件偵測 (detect) | `yolo11n.pt` | 640 | `Vehicle`（1 類）|
| `accident.py` | 影像分類 (classify) | `yolo11n-cls.pt` | 224 | `accident` / `non-accident` |

## 用法

```python
from src.modeling.traffic  import load_traffic_model
from src.modeling.accident import load_accident_model

det = load_traffic_model()    # 預設官方預訓練 yolo11n.pt
cls = load_accident_model()   # 預設官方預訓練 yolo11n-cls.pt

# 訓練出自己的權重後，傳路徑即載入微調模型：
det = load_traffic_model("/workspace/ray_results/.../best.pt")
```

常數也可直接取用，讓 data 預處理 / serve 推論共用同一份設定：

```python
from src.modeling.traffic import IMG_SIZE, CLASSES, CONF_THRES
```

## ⚠️ 重要觀念：現在是「起點」，不是「成品」

預設載入的是**官方預訓練權重**，還沒學過你的任務：

| 模型 | 預訓練資料 | 現狀 |
|---|---|---|
| `yolo11n.pt` | COCO（含車輛類） | ⚠️ 半能用——本來就認得車，微調後更準 |
| `yolo11n-cls.pt` | ImageNet（1000 類日常物） | ❌ **完全不認得車禍**，非訓練不可 |

所以 `CLASSES` 寫的是「**目標類別**」，要等用對應資料集訓練後才真的成立。
- traffic → 用 UA-DETRAC 訓練（見 [`../data/traffic`](../data/traffic)）
- accident → 需有標註的車禍分類資料（team edit 那份 / UCF-Crime 等）

## 設計取捨

| 決定 | 原因 |
|---|---|
| 預設官方權重、可傳路徑換自訓練 | 訓練前後都能用同一個 `load_*`，不用改程式 |
| 常數（imgsz/classes）抽到模組層 | data / serve 共用，避免散落、不一致 |
| 只搬 team edit 的「模型設定」 | 它的 Colab 流程寫死 `/content/` 路徑，不適用容器 |
