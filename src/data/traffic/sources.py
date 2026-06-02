"""UA-DETRAC XML → records（給 Ray Data pipeline 用）。

解析每序列的 XML，抽幀後產生 records，每筆：
  {"image_path": str,
   "boxes_xyxy": np.ndarray(N,4) float32,   # 像素座標 [x1,y1,x2,y2]
   "labels":     np.ndarray(N,)  int64}     # 單類，全為 0

與 detrac_to_yolo.py 的差別：那支「寫出 YOLO 檔案」（給 ultralytics 用）；
這支「回傳記憶體裡的 records」（給 Ray Data 串流用），不落地、含抽幀。

抽幀：相鄰幀幾乎相同，每 frame_stride 幀取 1，降冗餘。
切分：依「序列」切 train/val，避免同一段影片的幀洩漏到兩邊。
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

VEHICLE_CLASS = 0  # 單類：car/bus/van/others 全歸為 Vehicle


def _parse_sequence(xml_path: Path, img_seq_dir: Path,
                    frame_stride: int) -> List[Dict[str, Any]]:
    """解析單一序列的 XML，回傳抽幀後的 records。無車的幀會跳過。"""
    root = ET.parse(xml_path).getroot()
    records: List[Dict[str, Any]] = []

    for fi, frame in enumerate(root.findall("frame")):
        if fi % frame_stride != 0:          # 抽幀
            continue
        num = int(frame.get("num"))
        img_path = img_seq_dir / f"img{num:05d}.jpg"
        if not img_path.exists():
            continue

        boxes: List[List[float]] = []
        target_list = frame.find("target_list")
        if target_list is not None:
            for target in target_list.findall("target"):
                b = target.find("box")
                left = float(b.get("left"))
                top = float(b.get("top"))
                w = float(b.get("width"))
                h = float(b.get("height"))
                boxes.append([left, top, left + w, top + h])  # xyxy 像素

        # 無車的幀先跳過（偵測訓練主要靠有目標的幀；保留無車幀當 background
        # 負樣本可降誤報，但先簡化，之後要再加）。
        if not boxes:
            continue

        records.append({
            "image_path": str(img_path),
            "boxes_xyxy": np.asarray(boxes, dtype=np.float32),
            "labels": np.full(len(boxes), VEHICLE_CLASS, dtype=np.int64),
        })
    return records


def list_detrac_records(
        detrac_root: str = "/data/detrac",
        frame_stride: int = 10,
        val_ratio: float = 0.2,
        limit_sequences: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """產生 UA-DETRAC 的 train / val records。

    Args:
        detrac_root: DETRAC 掛載根（含 DETRAC-Images/、DETRAC-Train-Annotations-XML/）。
        frame_stride: 每幾幀取 1（降冗餘）。預設 10 → 約 1.4 萬張。
        val_ratio: 驗證集比例（依序列切，不是依幀）。
        limit_sequences: 只用前 N 個序列（先驗證用）；None=全部。

    Returns:
        (train_records, val_records)
    """
    root = Path(detrac_root)
    img_base = root / "DETRAC-Images" / "DETRAC-Images"
    xml_base = (root / "DETRAC-Train-Annotations-XML"
                / "DETRAC-Train-Annotations-XML")

    seqs = sorted(p.stem for p in xml_base.glob("*.xml"))
    if limit_sequences:
        seqs = seqs[:limit_sequences]

    n_val = max(1, int(len(seqs) * val_ratio)) if len(seqs) > 1 else 0
    val_seqs = set(seqs[:n_val])

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for seq in seqs:
        recs = _parse_sequence(xml_base / f"{seq}.xml", img_base / seq,
                               frame_stride)
        (val if seq in val_seqs else train).extend(recs)

    return train, val


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="UA-DETRAC → records（統計）")
    ap.add_argument("--detrac-root", default="/data/detrac")
    ap.add_argument("--frame-stride", type=int, default=10)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tr, va = list_detrac_records(args.detrac_root, args.frame_stride,
                                 args.val_ratio, args.limit)
    n_box_tr = sum(len(r["boxes_xyxy"]) for r in tr)
    print(f"train: {len(tr)} 幀 / {n_box_tr} boxes")
    print(f"val:   {len(va)} 幀 / {sum(len(r['boxes_xyxy']) for r in va)} boxes")
