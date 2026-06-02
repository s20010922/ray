# `src/data/traffic` — 車流偵測訓練資料

把 **UA-DETRAC** 資料集轉成 YOLO 偵測格式（單類 Vehicle），供 traffic 模型訓練。

## 為什麼要轉換

UA-DETRAC 的標註是它**自訂的 XML 格式**，YOLO 看不懂。`detrac_to_yolo.py`
負責把它轉成 YOLO 的 `0 xc yc w h`（正規化）格式。

```
UA-DETRAC（F:/dataset → 容器 /data/detrac）
  ├─ DETRAC-Images/...            14 萬張圖、100 序列、監控攝影機視角
  └─ *.xml（每序列一個）          box: left/top/width/height（像素）+ 車種
        │  detrac_to_yolo.py
        ▼
datasets/detrac_yolo/（可寫）
  ├─ images/{train,val}/          symlink 鏡像原圖（不複製，省空間）
  ├─ labels/{train,val}/          YOLO txt：每行 `0 xc yc w h`
  └─ data.yaml                    nc=1, names=[Vehicle]
```

## 設計重點

| 決定 | 說明 |
|---|---|
| **單類 Vehicle** | car/bus/van/others 全歸為 0；只要數車算車流密度就夠 |
| **symlink 鏡像圖片** | 不複製 14 萬張原圖，labels 用真實 txt；靠 ultralytics 的 images→labels 路徑對應 |
| **依序列切 train/val** | 同一段影片的 frame 不會分散到兩邊，避免資料洩漏 |
| **檔名加序列前綴** | `MVI_20011__img00001` 避免不同序列同名撞檔 |

## 用法

```powershell
# 先轉幾個序列驗證（推薦）
docker compose exec ray-head python -m src.data.traffic.detrac_to_yolo --limit 20

# 全量轉換（100 序列、14 萬張，較久）
docker compose exec ray-head python -m src.data.traffic.detrac_to_yolo
```

參數：`--detrac-root`（預設 `/data/detrac`）、`--out-root`、`--val-ratio`（預設 0.2）、`--limit`。

## bbox 轉換公式

DETRAC 像素 box → YOLO 正規化中心點：

```
xc = (left + width/2)  / 圖寬
yc = (top  + height/2) / 圖高
w  =  width  / 圖寬
h  =  height / 圖高
```

圖寬高每序列讀一張取得（同序列尺寸一致），DETRAC 約 960×540。

## 已知待辦

- **`ignored_region` 未處理**：DETRAC 標註不全的區域目前沒遮蔽。嚴謹做法是
  把那些區域塗黑，否則模型會把「沒標到的車」當背景學。先驗證階段影響不大，
  追求品質時再補。
- **14 萬張對單卡偏多**：訓練時建議先抽樣序列，確認模型有在學再上全量。

## 之後接到哪

轉好的 `datasets/detrac_yolo/data.yaml` 直接餵給 `train`（YOLO11n 偵測）。
訓練出基礎模型後，再用標註過的高公局畫面（見 [`../freeway`](../freeway)）做 fine-tune。
