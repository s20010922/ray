"""劣化增強：模擬高公局 352×240 低畫質（cv2 + numpy，零額外依賴）。

traffic（偵測）與 accident（分類）**共用同一套**——兩案最終都部署在高公局
低畫質 CCTV，需要同樣的低畫質適應，所以放在 data/ 共用層。

這些增強只改變「畫質」、不改變座標（pixel-level），套用後 bbox/label 都
不需變換。幾何增強（水平翻轉）由各案 pipeline 自行處理。

設計成 cv2/numpy 而非 albumentations：API 穩定、容器不用重 build、
劣化邏輯完全可控可讀。每個手段獨立、依機率套用。
"""

from typing import Optional

import cv2
import numpy as np

# 各劣化手段的套用機率
P_DOWNSCALE = 0.4    # 降解析度再放大（模擬 352×240 放大的糊）
P_JPEG = 0.5         # JPEG 壓縮 artifact（MJPEG 串流）
P_BLUR = 0.3         # 失焦 / 動態模糊
P_NOISE = 0.3        # 夜間感光雜訊
P_BRIGHTNESS = 0.4   # 亮度 / 對比（日夜、逆光）


def _downscale(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """縮小再放大回原尺寸，產生低解析度的糊感。"""
    h, w = img.shape[:2]
    scale = float(rng.uniform(0.4, 0.7))
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _jpeg(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """重新以低品質 JPEG 編解碼，產生壓縮 artifact。"""
    q = int(rng.integers(30, 61))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def _motion_kernel(size: int, rng: np.random.Generator) -> np.ndarray:
    """產生隨機方向的動態模糊核（水平/垂直/兩條對角）。"""
    k = np.zeros((size, size), dtype=np.float32)
    direction = int(rng.integers(0, 4))
    mid = size // 2
    if direction == 0:                      # 水平
        k[mid, :] = 1.0
    elif direction == 1:                    # 垂直
        k[:, mid] = 1.0
    elif direction == 2:                    # 主對角
        np.fill_diagonal(k, 1.0)
    else:                                   # 副對角
        np.fill_diagonal(np.fliplr(k), 1.0)
    return k / k.sum()


def _blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """隨機選一種模糊：高斯 / 中值 / 平均 / 動態。"""
    kind = int(rng.integers(0, 4))
    ksize = int(rng.choice([3, 5]))
    if kind == 0:
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    if kind == 1:
        return cv2.medianBlur(img, ksize)
    if kind == 2:
        return cv2.blur(img, (ksize, ksize))
    return cv2.filter2D(img, -1, _motion_kernel(int(rng.choice([5, 7, 9])), rng))


def _noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """加高斯雜訊，模擬夜間感光噪點。"""
    sigma = float(rng.uniform(10.0, 50.0))
    noise = rng.normal(0.0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _brightness_contrast(img: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
    """調整亮度與對比（含偏暗，模擬夜間 CCTV）。"""
    alpha = float(rng.uniform(0.7, 1.3))    # 對比
    beta = float(rng.uniform(-30.0, 30.0))  # 亮度
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def degrade(img: np.ndarray,
            rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """對一張 BGR uint8 圖套用劣化增強管線（依機率）。

    bbox 不受影響（全是 pixel-level 操作），呼叫端不需變換座標。
    """
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < P_DOWNSCALE:
        img = _downscale(img, rng)
    if rng.random() < P_JPEG:
        img = _jpeg(img, rng)
    if rng.random() < P_BLUR:
        img = _blur(img, rng)
    if rng.random() < P_NOISE:
        img = _noise(img, rng)
    if rng.random() < P_BRIGHTNESS:
        img = _brightness_contrast(img, rng)
    return img
