# `src/data/freeway` — 高公局 CCTV 即時影像

從高公局 CCTV 的 MJPEG 串流抽幀，作為 fine-tune 原始資料（或即時推論輸入）。

## 核心：`grab_jpeg_frame(stream_url)`

最底層的共用單位——從 MJPEG 串流抽出「一張」JPEG，回傳 bytes：

```python
from src.data.freeway.grabber import FOCUS_CAMERAS, grab_jpeg_frame

jpg = grab_jpeg_frame(FOCUS_CAMERAS[0].stream_url)   # -> bytes
```

兩種用途都靠它：
- **收集 fine-tune 資料** → 定時呼叫、把 bytes 存檔
- **即時推論**（focus_poller）→ 呼叫、把 bytes 直接 POST 給 Ray Serve

MJPEG 是連續的 JPEG，函式邊讀邊找 JPEG 起始 `FFD8` / 結束 `FFD9`，湊滿一張就回傳。

## focus 鏡頭清單（catalog）

5 支使用者指定的鏡頭，`Camera(cctv_id, label, stream_url)`：

| cctv_id | 位置 | camera |
|---|---|---|
| CCTV-N1-S-34.018-M | 國1南 34K+020 高公局交流道 | 13400 |
| CCTV-N3-S-40.980-M | 國3南 40K+980 土城路段 | 34090 |
| CCTV-N1-N-37.050-M | 國1北 37K+050 泰山路段 | 13700 |
| CCTV-N1H-S-17.450-M | 國1高架南 17K+450 內湖交流道 | 11740 |
| CCTV-N1-S-93.080-M | 國1南 93K+080 新竹路段 | 19300 |

串流網址：`https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=<id>`

> ⚠️ 是 **`abs2mjpg`**（MJPEG）不是 `abs2jpg`。路徑打錯會回 403，
> 容易誤判成被 WAF 擋——其實只是網址錯。

## 定時收集

由 [`scripts/collect_freeway.py`](../../../scripts/collect_freeway.py) 長駐執行，
跨時段累積多樣化資料（多樣性靠「分時段多輪」，不是一次連抓一大批）：

```powershell
# 背景啟動：每鏡頭目標 200 張，30 分一輪，達標自動停
docker compose exec -d ray-head python scripts/collect_freeway.py --target-per-camera 200

# 查進度
docker compose exec ray-head sh -c "for d in datasets/freeway_raw/*/; do echo \"$(ls $d|wc -l) 張  $d\"; done"

# 停止
docker compose exec ray-head pkill -f collect_freeway
```

抓到的圖存 `datasets/freeway_raw/<cctv_id>/`（可寫掛載，不是 `src/`）。

## ⚠️ 重要限制：沒有標註

即時串流抓到的畫面**沒有標註**，且：
- **traffic**：要人工框車（bbox）才能 fine-tune
- **accident**：即時幾乎抓不到「車禍」正樣本（車禍罕見）；這裡主要得到
  正常車流，可當 accident 的 **non-accident 負樣本**，正樣本需另尋來源

## fine-tune 資料量參考

domain adaptation 不用很多，**多樣性 > 數量**：

| 任務 | 建議量 | 備註 |
|---|---|---|
| traffic 偵測 | 每鏡頭 100~200 張（共 500~1000） | 要人工框車 |
| accident 負樣本 | 每鏡頭 150~300 張 | 即時容易取得 |
| accident 正樣本 | — | 即時抓不到，需歷史事故影像 |

關鍵是橫跨**尖峰/離峰、日/夜、晴/雨**——同時段連抓再多也是同一種畫面。
