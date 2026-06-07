"""高公局 CCTV 影像自動標籤（teacher：YOLO11x COCO + SAHI 切片）。

高公局影像為 352x240 低畫質固定機位，遠處車小且糊。整幅 YOLO 會漏遠車，
故以 SAHI 切片（192x192）放大小目標召回；切片結果預設已與整幅偵測 NMS 合併。
COCO 的 car/bus/truck/motorcycle 一律併為單類 Vehicle，輸出 YOLO 偵測格式。
"""

import re
import shutil
from pathlib import Path

import cv2

# COCO 車輛類別 → 單類 Vehicle
VEHICLE_COCO_IDS = {2, 3, 5, 7}        # car, motorcycle, bus, truck
_TS = re.compile(r"_\d{8}_(\d{2})\d{4}_")   # 檔名 _YYYYMMDD_HHMMSS_


def _hour(path: Path):
    m = _TS.search(path.name)
    return int(m.group(1)) if m else None


def collect_images(raw_root: str, sample: int = 0,
                   hour_lo: int = 6, hour_hi: int = 18) -> list:
    """掃 freeway_raw 各鏡頭子夾的 jpg。

    只取白天 [hour_lo, hour_hi] 時段（濾掉凌晨空路）；跳過非 CCTV 彙整夾。
    sample>0 則每鏡頭等間隔抽 N 張（涵蓋整段時間）。
    """
    root = Path(raw_root)
    paths = []
    for cam_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not cam_dir.name.startswith("CCTV"):
            continue                       # 跳過 "all pic" 彙整夾
        imgs = sorted(p for p in cam_dir.glob("*.jpg")
                      if (h := _hour(p)) is not None and hour_lo <= h <= hour_hi)
        if sample > 0 and len(imgs) > sample:
            step = len(imgs) / sample
            imgs = [imgs[int(i * step)] for i in range(sample)]
        paths.extend(imgs)
    return [str(p) for p in paths]


def _load_model(weights: str, conf: float, device: str):
    from sahi import AutoDetectionModel
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=weights,
        confidence_threshold=conf, device=device)


def prelabel(raw_root: str, out_root: str, weights: str = "yolo11x.pt",
             conf: float = 0.25, slice_size: int = 192, overlap: float = 0.25,
             hour_lo: int = 6, hour_hi: int = 18, sample: int = 0,
             preview: bool = True, device: str = "cuda:0") -> dict:
    """自動標籤主流程，輸出 YOLO 偵測資料集到 out_root。"""
    from sahi.predict import get_sliced_prediction

    out = Path(out_root)
    img_dir = out / "images"
    lbl_dir = out / "labels"
    pv_dir = out / "preview"
    for d in (img_dir, lbl_dir, pv_dir if preview else img_dir):
        d.mkdir(parents=True, exist_ok=True)

    model = _load_model(weights, conf, device)
    paths = collect_images(raw_root, sample, hour_lo, hour_hi)
    print(f"[資料] 待標 {len(paths)} 張（時段 {hour_lo}-{hour_hi} 時，"
          f"切片 {slice_size}px／重疊 {overlap}／conf {conf}）")

    n_boxes = n_empty = 0
    for i, p in enumerate(paths, 1):
        im = cv2.imread(p)
        if im is None:
            continue
        h, w = im.shape[:2]
        res = get_sliced_prediction(
            p, model, slice_height=slice_size, slice_width=slice_size,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            verbose=0)
        lines, pv = [], (im.copy() if preview else None)
        for o in res.object_prediction_list:
            if o.category.id not in VEHICLE_COCO_IDS:
                continue
            x1, y1, x2, y2 = o.bbox.to_xyxy()
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if preview:
                cv2.rectangle(pv, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 128, 255), 1)
        name = Path(p).name
        shutil.copy(p, img_dir / name)
        (lbl_dir / (Path(p).stem + ".txt")).write_text("\n".join(lines))
        if preview:
            cv2.imwrite(str(pv_dir / name), pv)
        n_boxes += len(lines)
        n_empty += (len(lines) == 0)
        if i % 50 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)}  累計框 {n_boxes}")

    yaml = (f"path: {out_root}\ntrain: images\nval: images\n"
            f"nc: 1\nnames: ['Vehicle']\n")
    (out / "data.yaml").write_text(yaml)

    stats = {"n_images": len(paths), "n_boxes": n_boxes, "n_empty": n_empty}
    print(f"\n=== 自動標籤完成 ===\n  影像 {stats['n_images']} 張"
          f"／總框 {stats['n_boxes']}（平均 {n_boxes/max(1,len(paths)):.1f}/張）"
          f"／零框 {stats['n_empty']} 張\n  輸出 {out_root}")
    return stats
