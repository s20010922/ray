"""診斷：新 Accident 模型在台灣域上分不分得開（合成訓練是否有效）。

比較三組的 P(accident)：
  A. 合成事故正樣本（synth/accident）── 訓練分佈內
  B. 真實高公局正常（synth/non-accident）── 訓練分佈內
  C. 101 支真車禍片段抽幀 ── 真實泛化（最關鍵）
若 A 高、B 低 → 學到合成事故；再看 C 能否跟 B 拉開 → 對真事故是否有鑑別力。
"""

import glob
import random

import cv2
import numpy as np

from src.infer.accident import (classify, find_best_accident_checkpoint,
                                 load_classifier)
from src.modeling.accident import CLASSES


def _p_accident(model, dev, img):
    pred, conf = classify(model, img, dev)
    return conf if CLASSES[pred] == "accident" else 1.0 - conf


def _stats(name, vals):
    a = np.array(vals)
    print(f"{name:28s} n={len(a):3d}  mean={a.mean():.3f}  "
          f"std={a.std():.3f}  min={a.min():.3f}  max={a.max():.3f}")
    return a


def main():
    ckpt = find_best_accident_checkpoint()
    print("checkpoint:", ckpt)
    model, dev = load_classifier(ckpt, device="cuda")
    rng = random.Random(0)

    # A. 合成事故
    synth_acc = glob.glob("/workspace/datasets/accident/synth/accident/*.jpg")
    rng.shuffle(synth_acc)
    a = [_p_accident(model, dev, cv2.imread(p)) for p in synth_acc[:80]]

    # B. 真實正常
    synth_non = glob.glob("/workspace/datasets/accident/synth/non-accident/*.jpg")
    rng.shuffle(synth_non)
    b = [_p_accident(model, dev, cv2.imread(p)) for p in synth_non[:80]]

    # C. 真車禍片段抽幀（每支取中間幀）
    clips = glob.glob("/workspace/datasets/accident/video/accident/*.mp4")
    rng.shuffle(clips)
    c = []
    for p in clips[:80]:
        cap = cv2.VideoCapture(p)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, frame = cap.read()
        cap.release()
        if ok:
            c.append(_p_accident(model, dev, frame))

    print("\n=== P(accident) 分佈 ===")
    A = _stats("A 合成事故(train域)", a)
    B = _stats("B 真實正常(train域)", b)
    C = _stats("C 真車禍片段(泛化)", c)
    print("\n判讀：")
    print(f"  A-B 分離度（訓練域內）= {A.mean()-B.mean():+.3f}（越大越好，>0.3 算學到）")
    print(f"  C-B 分離度（真事故 vs 正常）= {C.mean()-B.mean():+.3f}（>0 才有真鑑別力）")


if __name__ == "__main__":
    main()
