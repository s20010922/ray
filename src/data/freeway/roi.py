"""每支 CCTV 鏡頭的偵測範圍（ROI）。

CCTV 是固定機位，畫面構圖永久不變，所以每支鏡頭定一個多邊形 ROI，
只在範圍內偵測/標註/推論，可以：
  - 排除極遠景糊區（base 漏抓的小車「不算漏標」）
  - 排除對向車道、邊緣建築/護欄（減少誤框）
  - 人力與模型都聚焦在「車流密度真正要算」的主車道區

座標為正規化多邊形 [(x,y), ...]，0~1，左上原點，與解析度無關。
順時針或逆時針皆可。初版為目測，靠 preview 疊邊界後迭代微調。
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

Polygon = List[Tuple[float, float]]

# 目測初版（白天有車畫面）：上窄下寬的梯形，貼合車道往消失點收斂的透視。
ROIS: dict = {
    # 國1南 34K：高架在左上，主車流中央偏右下
    "CCTV-N1-S-34.018-M": [(0.33, 0.43), (0.72, 0.43), (1.0, 1.0), (0.04, 1.0)],
    # 國1北 37K 泰山：左側近景大貨車，車道由右上往左下
    "CCTV-N1-N-37.050-M": [(0.20, 0.45), (0.80, 0.42), (1.0, 1.0), (0.0, 1.0)],
    # 國3南 40K 土城：開闊多車道，遠山在上方
    "CCTV-N3-S-40.980-M": [(0.28, 0.42), (0.72, 0.42), (1.0, 1.0), (0.0, 1.0)],
    # 國1高架南 17K 內湖：左高架、中主車道、右匝道
    "CCTV-N1H-S-17.450-M": [(0.22, 0.43), (0.75, 0.41), (1.0, 1.0), (0.0, 1.0)],
    # 國1南 93K 新竹：中央綠籬分隔南北，左邊界收到綠籬右側，只圈南下車道
    "CCTV-N1-S-93.080-M": [(0.50, 0.43), (0.78, 0.42), (1.0, 1.0), (0.40, 1.0)],
}


def cam_id_from_filename(name: str) -> str:
    """檔名 <cctv_id>_<date>_<time>_<idx>.jpg → cctv_id（cctv_id 內無底線）。"""
    return Path(name).stem.split("_")[0]


def get_roi(cctv_id: str) -> Optional[Polygon]:
    return ROIS.get(cctv_id)


def _poly_px(roi: Polygon, w: int, h: int) -> np.ndarray:
    return np.array([(x * w, y * h) for x, y in roi], dtype=np.int32)


def filter_by_roi(boxes: np.ndarray, scores: np.ndarray,
                  roi: Polygon, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """只保留「框中心」落在 ROI 多邊形內的偵測。"""
    if len(boxes) == 0:
        return boxes, scores
    poly = _poly_px(roi, w, h)
    keep = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
            keep.append(i)
    return boxes[keep], scores[keep]


def draw_roi(img: np.ndarray, roi: Polygon,
             color=(0, 0, 255)) -> np.ndarray:
    """在影像上疊出 ROI 邊界（紅線），給人眼檢視/微調。"""
    h, w = img.shape[:2]
    out = img.copy()
    cv2.polylines(out, [_poly_px(roi, w, h)], isClosed=True,
                  color=color, thickness=2)
    return out
