"""高公局影像「模型輔助預標」：用自訓 base model 產生 YOLO 標籤草稿。

流程：base model 偵測 → 寫 YOLO 格式 .txt（人工再用 labelImg/Label Studio 修正）。
預標只是「省人力的草稿」，不是最終標註——base 有 domain gap，遠處小車會漏，
人工務必補漏、刪錯。

輸出結構（與 detrac_yolo 一致，方便之後 fine-tune 直接吃）：
  out_root/
    images/<原圖檔名>.jpg      （複製，保留原檔名含鏡頭+時間戳）
    labels/<原圖檔名>.txt      （YOLO：class xc yc w h，正規化）
    preview/<原圖檔名>.jpg     （畫框預覽，給人眼快速掃過用，可選）
    data.yaml
"""

import shutil
from pathlib import Path
from typing import List

import cv2

from src.data.freeway.roi import (cam_id_from_filename, draw_roi,
                                  filter_by_roi, get_roi)
from src.infer.traffic import detect as detect_traffic, draw
from src.modeling.traffic import CLASSES


def _to_yolo_lines(boxes_xyxy, w0: int, h0: int) -> List[str]:
    """xyxy 像素 → YOLO 行（class xc yc w h，正規化，單類 class=0）。"""
    lines = []
    for x1, y1, x2, y2 in boxes_xyxy:
        xc = ((x1 + x2) * 0.5) / w0
        yc = ((y1 + y2) * 0.5) / h0
        bw = (x2 - x1) / w0
        bh = (y2 - y1) / h0
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def prelabel(model, image_paths: List[str], out_root: str,
             conf: float = 0.25, preview: bool = True,
             use_roi: bool = True, detect_fn=None) -> dict:
    """對一批高公局影像產生 YOLO 預標。

    Args:
        use_roi: 是否套用 per-cam ROI 過濾。先做物件標記時設 False（標所有車，
                 ROI 留到推論階段再用）。
        detect_fn: 偵測函數 (model, img, conf) -> (boxes_xyxy, scores)。
                   None=用自訓 base；可傳 coco_vehicle.detect 改用官方 COCO 模型。
    Returns:
        {"n_images", "n_boxes", "n_empty"}：處理張數、總框數、零框張數。
    """
    _detect = detect_fn or detect_traffic
    out = Path(out_root)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    if preview:
        (out / "preview").mkdir(parents=True, exist_ok=True)
    # labelImg 的 YOLO 模式需要 classes.txt（類別順序）在 save_dir（labels/）。
    (out / "labels" / "classes.txt").write_text("\n".join(CLASSES) + "\n")

    n_boxes, n_empty = 0, 0
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        boxes, scores = _detect(model, img, conf=conf)

        roi = get_roi(cam_id_from_filename(p)) if use_roi else None
        if roi is not None:
            boxes, scores = filter_by_roi(boxes, scores, roi, w0, h0)

        stem = Path(p).stem
        shutil.copy2(p, out / "images" / f"{stem}.jpg")
        lines = _to_yolo_lines(boxes, w0, h0)
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines))

        if preview:
            prev = draw(img, boxes, scores)
            if roi is not None:
                prev = draw_roi(prev, roi)
            cv2.imwrite(str(out / "preview" / f"{stem}.jpg"), prev)

        n_boxes += len(boxes)
        n_empty += int(len(boxes) == 0)

    _write_data_yaml(out)
    return {"n_images": len(image_paths), "n_boxes": n_boxes, "n_empty": n_empty}


def _write_data_yaml(out: Path) -> None:
    """寫 ultralytics fine-tune 用的 data.yaml。"""
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images\nval: images\n\n"
        f"nc: {len(CLASSES)}\nnames:\n{names}\n")
