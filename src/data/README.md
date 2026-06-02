# `src/data` — 資料來源與管線

這個模組處理「資料怎麼進來」。底下分三個子模組，**前兩個是任務、第三個是來源**：

```
src/data/
├── traffic/    任務：車流偵測的訓練資料（UA-DETRAC → YOLO）
├── accident/   任務：車禍分類的訓練資料（待建）
└── freeway/    來源：高公局 CCTV 即時影像（抓取器）
```

## 三者的關係

`traffic` / `accident` 是**任務**（要訓練什麼模型）；`freeway` 是**來源**
（即時畫面從哪來）。一段高公局畫面可同時餵給兩個任務：

```
高公局 CCTV (freeway/)
      │ 抓即時畫面（無標註）
      ├──→ traffic  車流偵測
      └──→ accident 車禍判斷
```

## 子模組現況

| 子模組 | 內容 | 狀態 | 說明 |
|---|---|---|---|
| `traffic/` | `detrac_to_yolo.py` | ✅ 可用 | UA-DETRAC XML → YOLO 格式（單類 Vehicle）。見 [traffic/README](traffic/) |
| `freeway/` | `grabber.py` | ✅ 可用 | 高公局 CCTV MJPEG 抓取。見 [freeway/README](freeway/) |
| `accident/` | （空） | ⏳ 待建 | 車禍分類資料管線，需有標註的車禍資料集 |

## 資料的兩種角色：訓練 vs 推論

| | 要標註? | 用途 |
|---|---|---|
| UA-DETRAC（traffic） | ✅ 已有 bbox 標註 | **訓練** |
| 車禍資料集（accident） | ✅ 需要 | **訓練** |
| 高公局 CCTV（freeway） | ❌ 沒有 | **推論**；或人工標註後做 fine-tune |

高公局即時串流**沒有標註**，所以：
- 直接拿來「即時推論」（餵給訓練好的模型）✅
- 要拿來「微調」，得先人工標註一批（traffic 框車、accident 標類別）

## 資料存放位置（重要）

程式碼與資料**分開**，因為 `src/` 在容器是唯讀掛載（`:ro`）：

| 放什麼 | 位置 | 掛載 |
|---|---|---|
| 程式（管線/抓取器） | `src/data/...` | 唯讀 |
| 轉檔後的 YOLO 資料 | `datasets/detrac_yolo/` | 可寫 |
| 抓到的 CCTV 原圖 | `datasets/freeway_raw/` | 可寫 |
| 原始 UA-DETRAC | `/data/detrac`（= `F:/dataset`） | 唯讀 |

> ⚠️ 圖片**不能**存進 `src/data/`（唯讀），一律存 `datasets/`。
