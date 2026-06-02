"""車流偵測模型：YOLO11 物件偵測 (detection)。

任務沿用 team edit 的 `traffic/vehicles_detection_and_counting.py`，
但模型升級到最新世代 YOLO11n（team edit 原本用 yolov8n）：
  - 基礎模型 yolo11n.pt（COCO 預訓練）
  - 微調後偵測單一類別 Vehicle，用框出的車輛數估計車流密度
  - 輸入尺寸 640
"""

from ultralytics import YOLO

# ---- 模型設定 ----
BASE_WEIGHTS = "yolo11n.pt"   # 官方預訓練：物件偵測（最新 YOLO11 nano）
IMG_SIZE = 640
CLASSES = ["Vehicle"]         # 微調資料集只有一類
CONF_THRES = 0.4              # team edit 即時車流分析用的信心門檻


def load_traffic_model(weights: str = BASE_WEIGHTS) -> YOLO:
    """載入車流偵測模型。

    Args:
        weights: 權重路徑。預設用官方 ``yolov8n.pt``；
                 傳入自己訓練好的 ``best.pt`` 即載入微調模型。

    Returns:
        Ultralytics ``YOLO`` 偵測模型（呼叫 ``.predict()`` 會回傳 bbox）。
    """
    return YOLO(weights)
