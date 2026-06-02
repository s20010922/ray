"""車禍分類資料 → records（給 Ray Data pipeline）。

datasets/accident 已是分好的分類資料夾（team edit 的 organize.py 切過）：
  {train,val}/{accident,non-accident}/*.jpg
直接列檔成 records：{"image_path": str, "label": int}

label 對齊 modeling.accident.CLASSES：0=accident, 1=non-accident。

與 traffic 的 sources 差別：分類資料已切好 train/val、無 bbox，所以這支只是
「列檔 + 標 label」，比 traffic 解析 XML 簡單得多——也不需要轉換器。
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from src.modeling.accident import CLASSES

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png")


def list_accident_records(data_root: str = "/workspace/datasets/accident",
                          split: str = "train") -> List[Dict[str, Any]]:
    """列出某個 split 的分類 records。

    Args:
        data_root: accident 資料根（含 train/、val/）。
        split: "train" 或 "val"。

    Returns:
        [{"image_path": str, "label": int}, ...]
        label：0=accident, 1=non-accident（對齊 CLASSES 順序）。
    """
    root = Path(data_root) / split
    records: List[Dict[str, Any]] = []
    for label_idx, cls_name in enumerate(CLASSES):   # 0=accident, 1=non-accident
        cls_dir = root / cls_name
        if not cls_dir.exists():
            continue
        for ext in IMG_EXTS:
            for img in cls_dir.glob(ext):
                records.append({"image_path": str(img), "label": label_idx})
    return records


if __name__ == "__main__":
    for split in ("train", "val"):
        recs = list_accident_records(split=split)
        c = Counter(r["label"] for r in recs)
        print(f"{split}: {len(recs)} 張  "
              f"accident={c[0]}  non-accident={c[1]}")
