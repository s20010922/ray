"""輕量 IOU 追蹤器 + 車道內靜止車輛偵測（高速公路事故/拋錨的物理徵兆）。

設計動機（方案 A）：整幅影像分類在高公局 domain gap 太大、無鑑別力。改用
「建在偵測器上的可觀測徵兆」判斷事故——其中最強、最不需正樣本的就是
**車道內出現持續靜止的車輛**。

為何單靠 IOU 匹配就能浮現靜止車：serve 約每 2 秒輪詢一幀，高速公路上正常
車輛兩幀間位移很大、IOU 低，無法持續匹配（每幀都被當成新 track）；唯有
停住的車兩幀間高度重疊、會被持續匹配且中心位移趨零。據此累計每條 track
的「連續靜止次數」，達門檻即視為事故徵兆。

不依賴外部追蹤套件（無 scipy / lap），貪婪 IOU 匹配即可，每鏡頭一個實例。

用法：
    tracker = VehicleTracker(stall_frames=3, move_frac=0.15)
    stalled = tracker.update(boxes_xyxy)   # 回傳達門檻的靜止 Track 清單
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Track:
    tid: int
    box: np.ndarray                       # xyxy
    center: np.ndarray                    # (cx, cy)
    stationary: int = 0                   # 連續靜止幀數
    misses: int = 0                       # 連續未匹配幀數（用於老化淘汰）
    hits: int = 0                         # 累計被匹配次數
    history: List[np.ndarray] = field(default_factory=list)  # 近幾幀中心


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a:(M,4) b:(N,4) xyxy → IOU (M,N)。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


class VehicleTracker:
    """每鏡頭一個：貪婪 IOU 匹配 + 靜止車輛偵測。

    參數：
      iou_thr     匹配門檻（停住的車兩幀重疊高，0.3 容忍鏡頭抖動/微移）
      move_frac   中心位移 < move_frac × box 對角線 → 視為「未移動」
      stall_frames 連續靜止幀數達此值 → 列為事故徵兆（poll 2s 時 3≈6 秒）
      max_misses  連續未匹配超過此值 → 淘汰 track
    """

    def __init__(self, iou_thr: float = 0.3, move_frac: float = 0.15,
                 stall_frames: int = 3, max_misses: int = 2):
        self.iou_thr = iou_thr
        self.move_frac = move_frac
        self.stall_frames = stall_frames
        self.max_misses = max_misses
        self.tracks: List[Track] = []
        self._next_id = 0

    def update(self, boxes: np.ndarray) -> List[Track]:
        """餵入本幀偵測框（xyxy, (N,4)），更新 track 並回傳靜止達門檻者。"""
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        centers = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                                   (boxes[:, 1] + boxes[:, 3]) / 2]) \
            if len(boxes) else np.zeros((0, 2), np.float32)

        track_boxes = np.array([t.box for t in self.tracks], np.float32) \
            if self.tracks else np.zeros((0, 4), np.float32)
        iou = _iou_matrix(track_boxes, boxes)

        matched_t, matched_d = set(), set()
        # 貪婪：每次取全域最高 IOU 配對，超過門檻才接受
        while iou.size and iou.max() >= self.iou_thr:
            ti, di = np.unravel_index(int(iou.argmax()), iou.shape)
            if ti in matched_t or di in matched_d:
                iou[ti, di] = -1
                continue
            self._update_track(self.tracks[ti], boxes[di], centers[di])
            matched_t.add(ti)
            matched_d.add(di)
            iou[ti, :] = -1
            iou[:, di] = -1

        # 未匹配 track → 累計 miss，淘汰過期者
        survivors: List[Track] = []
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.misses += 1
                if t.misses > self.max_misses:
                    continue
            survivors.append(t)
        self.tracks = survivors

        # 未匹配偵測 → 新 track
        for di in range(len(boxes)):
            if di not in matched_d:
                self.tracks.append(Track(
                    tid=self._next_id, box=boxes[di], center=centers[di],
                    hits=1, history=[centers[di]]))
                self._next_id += 1

        return [t for t in self.tracks if t.stationary >= self.stall_frames]

    def _update_track(self, t: Track, box: np.ndarray, center: np.ndarray):
        diag = float(np.hypot(box[2] - box[0], box[3] - box[1])) or 1.0
        moved = float(np.hypot(*(center - t.center)))
        if moved < self.move_frac * diag:
            t.stationary += 1
        else:
            t.stationary = 0
        t.box = box
        t.center = center
        t.misses = 0
        t.hits += 1
        t.history.append(center)
        if len(t.history) > 10:
            t.history.pop(0)
