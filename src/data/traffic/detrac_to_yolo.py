"""UA-DETRAC → YOLO 格式轉換（單類 Vehicle）。

UA-DETRAC 的標註是「每個序列一個 XML」（自訂格式），YOLO 看不懂。
這支程式把它轉成 YOLO 偵測訓練格式：

  - 每張有標註的圖產生一個 .txt：每行 ``0 xc yc w h``
    （正規化 0~1；單類，所以 class 一律 0）
  - 圖片用 symlink 鏡像到 images/（不複製，省空間），labels/ 放 txt
    （ultralytics 靠把路徑中的 /images/ 換成 /labels/ 找標註）
  - 依「序列」切 train/val（同一段影片的 frame 不會同時落在兩邊，避免資料洩漏）
  - 產生 data.yaml

DETRAC box 是像素 left/top/width/height → 轉成 YOLO 的正規化中心點+寬高。
ignored_region（標註不完整的區域）目前不處理，僅在統計時提示。
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import cv2

VEHICLE_CLASS = 0  # 單類：car/bus/van/others 全部歸為 Vehicle


def _image_size(img_path: Path) -> tuple[int, int]:
    """讀一張圖取得 (寬, 高)。同序列尺寸一致，每序列只需呼叫一次。"""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"讀不到圖片：{img_path}")
    h, w = img.shape[:2]
    return w, h


def convert_sequence(xml_path: Path, img_seq_dir: Path,
                     out_images: Path, out_labels: Path) -> tuple[int, int]:
    """轉換單一序列。回傳 (處理的frame數, 寫出的box數)。"""
    root = ET.parse(xml_path).getroot()
    seq_name = root.get("name")

    # 取該序列圖片尺寸（讀第一張存在的圖）
    sample = next((img_seq_dir / f"img{int(f.get('num')):05d}.jpg"
                   for f in root.findall("frame")
                   if (img_seq_dir / f"img{int(f.get('num')):05d}.jpg").exists()),
                  None)
    if sample is None:
        return 0, 0
    img_w, img_h = _image_size(sample)

    n_frames = n_boxes = 0
    for frame in root.findall("frame"):
        num = int(frame.get("num"))
        img_name = f"img{num:05d}.jpg"
        src_img = img_seq_dir / img_name
        if not src_img.exists():
            continue

        lines = []
        target_list = frame.find("target_list")
        if target_list is not None:
            for target in target_list.findall("target"):
                box = target.find("box")
                left = float(box.get("left"))
                top = float(box.get("top"))
                bw = float(box.get("width"))
                bh = float(box.get("height"))
                xc = (left + bw / 2) / img_w
                yc = (top + bh / 2) / img_h
                lines.append(
                    f"{VEHICLE_CLASS} {xc:.6f} {yc:.6f} "
                    f"{bw / img_w:.6f} {bh / img_h:.6f}"
                )
                n_boxes += 1

        # 用「序列名__幀名」當檔名，避免不同序列的 imgNNNNN 撞名
        stem = f"{seq_name}__{img_name[:-4]}"
        link = out_images / f"{stem}.jpg"
        if not link.exists():
            link.symlink_to(src_img)
        (out_labels / f"{stem}.txt").write_text("\n".join(lines))
        n_frames += 1

    return n_frames, n_boxes


def convert_detrac(detrac_root: str = "/data/detrac",
                   out_root: str = "/workspace/datasets/detrac_yolo",
                   val_ratio: float = 0.2,
                   limit_sequences: Optional[int] = None) -> dict:
    """把 UA-DETRAC 訓練集轉成 YOLO 格式。

    Args:
        detrac_root: DETRAC 掛載根目錄（含 DETRAC-Images/、DETRAC-Train-Annotations-XML/）。
        out_root: 輸出根目錄（會建 images/{train,val}、labels/{train,val}、data.yaml）。
        val_ratio: 驗證集比例（依序列切，不是依圖片）。
        limit_sequences: 只轉前 N 個序列（先驗證用）；None=全部。
    """
    root = Path(detrac_root)
    img_base = root / "DETRAC-Images" / "DETRAC-Images"
    xml_base = root / "DETRAC-Train-Annotations-XML" / "DETRAC-Train-Annotations-XML"

    out = Path(out_root)
    dirs = {split: {"img": out / "images" / split, "lbl": out / "labels" / split}
            for split in ("train", "val")}
    for d in dirs.values():
        d["img"].mkdir(parents=True, exist_ok=True)
        d["lbl"].mkdir(parents=True, exist_ok=True)

    seqs = sorted(p.stem for p in xml_base.glob("*.xml"))
    if limit_sequences:
        seqs = seqs[:limit_sequences]

    n_val = max(1, int(len(seqs) * val_ratio)) if len(seqs) > 1 else 0
    val_seqs = set(seqs[:n_val])

    stats = {"sequences": len(seqs), "train_frames": 0, "val_frames": 0, "boxes": 0}
    for seq in seqs:
        split = "val" if seq in val_seqs else "train"
        nf, nb = convert_sequence(
            xml_base / f"{seq}.xml", img_base / seq,
            dirs[split]["img"], dirs[split]["lbl"],
        )
        stats[f"{split}_frames"] += nf
        stats["boxes"] += nb

    # data.yaml
    (out / "data.yaml").write_text(
        f"path: {out_root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: [Vehicle]\n"
    )
    return stats


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="UA-DETRAC → YOLO（單類）")
    ap.add_argument("--detrac-root", default="/data/detrac")
    ap.add_argument("--out-root", default="/workspace/datasets/detrac_yolo")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None,
                    help="只轉前 N 個序列（先驗證用）")
    args = ap.parse_args()

    s = convert_detrac(args.detrac_root, args.out_root,
                       args.val_ratio, args.limit)
    print(f"✅ 轉換完成：{s}")
