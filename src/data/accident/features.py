"""Stage 1b — 軌跡 → 運動特徵序列(純數學，無模型，跨域共用)。

把每台車的正規化座標序列，逐幀算成物理運動量。這層之後完全不碰像素，
所以 UCF 訓練出的時序模型能直接吃高公局的軌跡(只要前端 YOLO 換成 11s)。

每幀每台車輸出 F=10 維特徵；再以滑動視窗切成 (T, F) 的時序樣本。
"""

import math

import numpy as np

FEATURE_NAMES = [
    "speed",        # 速率大小 |v|
    "vx", "vy",     # 速度分量
    "accel",        # 加速度大小 |Δv|
    "heading",      # 航向角變化 |Δθ|
    "area",         # 框面積 (w*h)
    "area_rate",    # 面積變化率
    "nearest",      # 最近他車距離
    "near_delta",   # 最近車距變化(逼近=負)
    "stall",        # 停滯旗標(低速持續)
]
F = len(FEATURE_NAMES)

STALL_SPEED = 0.003     # 正規化單位/幀，低於此視為近乎靜止
NO_NEIGHBOR = 1.0       # 畫面內只有自己時的「最近車距」預設值


def _nearest_dist(t, cx, cy, frame_pos, self_tid):
    """同一幀內，到最近他車中心的歐氏距離(正規化)。沒有他車回 NO_NEIGHBOR。"""
    best = NO_NEIGHBOR
    for tid, ox, oy in frame_pos.get(t, ()):
        if tid == self_tid:
            continue
        d = math.hypot(cx - ox, cy - oy)
        if d < best:
            best = d
    return best


def build_features(tracks: dict) -> dict:
    """{tid: 軌跡} → {tid: [(t, cx, cy, featvec[F]), ...]}。

    保留 t(連續性判斷)與 cx,cy(供打標的空間定位；不進特徵矩陣，維持 domain 不變)。
    """
    # 先建每幀的車輛中心索引(算最近車距用)
    frame_pos = {}
    for tid, seq in tracks.items():
        for (t, cx, cy, w, h) in seq:
            frame_pos.setdefault(t, []).append((tid, cx, cy))

    feats = {}
    for tid, seq in tracks.items():
        seq = sorted(seq)
        prev = None          # (t, cx, cy)
        prev_v = (0.0, 0.0)
        prev_theta = None
        prev_area = None
        prev_near = None
        out = []
        for (t, cx, cy, w, h) in seq:
            cont = prev is not None and prev[0] == t - 1   # 與前一幀連續?
            if cont:
                vx, vy = cx - prev[1], cy - prev[2]
            else:
                vx = vy = 0.0
            speed = math.hypot(vx, vy)
            accel = math.hypot(vx - prev_v[0], vy - prev_v[1]) if cont else 0.0
            theta = math.atan2(vy, vx) if speed > 1e-6 else prev_theta
            if cont and prev_theta is not None and theta is not None:
                dth = abs(math.atan2(math.sin(theta - prev_theta),
                                     math.cos(theta - prev_theta)))
            else:
                dth = 0.0
            area = w * h
            area_rate = (area - prev_area) if (cont and prev_area is not None) else 0.0
            near = _nearest_dist(t, cx, cy, frame_pos, tid)
            near_delta = (near - prev_near) if (cont and prev_near is not None) else 0.0
            stall = 1.0 if speed < STALL_SPEED else 0.0

            out.append((t, cx, cy, np.array(
                [speed, vx, vy, accel, dth, area, area_rate,
                 near, near_delta, stall], dtype=np.float32)))

            prev = (t, cx, cy)
            prev_v = (vx, vy)
            prev_theta = theta
            prev_area = area
            prev_near = near
        feats[tid] = out
    return feats


def make_windows(feats: dict, positives: dict, T: int = 20, win_stride: int = 5):
    """滑動視窗切時序樣本。回傳 (X[N,T,F], y[N], centers[N])。

    只在「t 連續」的軌跡片段上切窗(追蹤中斷處斷開)。正樣本條件：該軌跡是
    肇事車(tid 在 positives)且視窗中心幀落在其肇事範圍 (lo,hi) 內。
    centers 為各窗中心的等效幀索引(供事件級評估對齊事故時刻)。
    """
    Xs, ys, cs = [], [], []
    for tid, seq in feats.items():
        if len(seq) < T:
            continue
        rng = positives.get(tid)
        # 依 t 連續性切成多段
        runs, cur = [], [seq[0]]
        for p in seq[1:]:
            if p[0] == cur[-1][0] + 1:
                cur.append(p)
            else:
                runs.append(cur)
                cur = [p]
        runs.append(cur)

        for run in runs:
            if len(run) < T:
                continue
            arr = np.stack([v for (_, _, _, v) in run])      # (L, F)
            for s in range(0, len(run) - T + 1, win_stride):
                ct = run[s + T // 2][0]
                y = 1 if (rng and rng[0] <= ct <= rng[1]) else 0
                Xs.append(arr[s:s + T])
                ys.append(y)
                cs.append(ct)

    if not Xs:
        return (np.zeros((0, T, F), np.float32),
                np.zeros((0,), np.int64), np.zeros((0,), np.int64))
    return (np.stack(Xs).astype(np.float32),
            np.array(ys, np.int64), np.array(cs, np.int64))
