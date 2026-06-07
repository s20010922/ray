"""Stage 1a — YOLO 偵測 + ByteTrack 追蹤：影片 → 每台車的正規化軌跡。

吃像素的只有這一關(domain-specific 的偵測器)。輸出之後全是抽象數字。
座標一律用「畫面尺寸正規化」(cx/W, cy/H, w/W, h/H)，讓不同解析度的影片
(UCF 雜解析度 vs 高公局 352×240)產生相同尺度的運動數字 → 利於跨域遷移。

vid_stride 把 UCF 的 30fps 降到部署等效 fps(高公局實測 ~10fps)，
讓訓練與部署的速度/加速度數值尺度一致。
"""

VEHICLE_COCO_IDS = [2, 3, 5, 7]   # car, motorcycle, bus, truck


def track_video(video_path: str, model, stride: int = 3,
                conf: float = 0.25, classes=VEHICLE_COCO_IDS) -> dict:
    """追蹤單支影片，回傳 {track_id: [(t, cxn, cyn, wn, hn), ...]}。

    t 為「等效幀索引」(降採樣後，0,1,2,…)，對應原生幀 t*stride。
    座標已用畫面寬高正規化到 0~1。
    """
    tracks = {}
    for t, res in enumerate(model.track(
            video_path, stream=True, vid_stride=stride,
            tracker="bytetrack.yaml", classes=classes, conf=conf,
            persist=True, verbose=False)):
        if res.boxes is None or res.boxes.id is None:
            continue
        h, w = res.orig_shape
        xywh = res.boxes.xywh.cpu().numpy()
        ids = res.boxes.id.cpu().numpy().astype(int)
        for (cx, cy, bw, bh), tid in zip(xywh, ids):
            tracks.setdefault(int(tid), []).append(
                (t, cx / w, cy / h, bw / w, bh / h))
    return tracks
