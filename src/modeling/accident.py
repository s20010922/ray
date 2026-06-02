"""車禍偵測模型：YOLO11 影像分類 (classification)。

任務沿用 team edit 的 accident 訓練設定（runs/classify/train/args.yaml），
但模型升級到最新世代 YOLO11n-cls（team edit 原本用 yolov8n-cls）：
  - 基礎模型 yolo11n-cls.pt
  - 二元分類 accident / non-accident
  - 輸入尺寸 224、epochs 50、batch 16
"""

from ultralytics import YOLO

# ---- 模型設定（任務沿用 team edit args.yaml）----
BASE_WEIGHTS = "yolo11n-cls.pt"          # 官方預訓練：影像分類（最新 YOLO11 nano）
IMG_SIZE = 224
CLASSES = ["accident", "non-accident"]   # 二元分類


def load_accident_model(weights: str = BASE_WEIGHTS) -> YOLO:
    """載入車禍分類模型。

    Args:
        weights: 權重路徑。預設用官方 ``yolov8n-cls.pt``；
                 傳入自己訓練好的 ``best.pt`` 即載入微調模型。

    Returns:
        Ultralytics ``YOLO`` 分類模型。推論時用
        ``results[0].probs.top1`` 取最高信心類別。
    """
    return YOLO(weights)
