"""Cross-cutting constants shared by Ray Data / Train / Serve / inference."""

GB = 1024 ** 3

# Image / target shapes
IMG_SIZE = 640
MAX_BOXES = 200       # padding cap per image for Ray Data fixed-length columns
PAD_LABEL = -1        # label sentinel for padded slots; unmasked in train loop

# Smaller input for the binary accident classifier (YOLOv8n-cls native size).
# 640 is wasteful for whole-image classification and ~8x slower.
CLS_IMG_SIZE = 224
CLS_NUM_CLASSES = 2   # normal vs accident

# Pretrained yolov8n.pt detection head outputs 80 COCO classes.
NUM_CLASSES = 80

# UA-DETRAC vehicle types -> COCO class ids the pretrained head already knows.
# Keeps us on pretrained weights without rebuilding the detection head.
UADETRAC_TO_COCO = {
    "car":    2,
    "bus":    5,
    "van":    7,   # COCO has no "van"; truck is closest
    "others": 7,
}

# Friendly names for the COCO ids we actually emit (used by viz + serve).
COCO_NAMES = {2: "car", 5: "bus", 7: "truck"}

# Ray cluster caps (apply when starting a fresh local cluster from Python).
# If we're attached to an external Ray cluster (RAY_ADDRESS=auto), these are
# ignored and the head's `ray start --head` flags rule instead.
RAY_NUM_CPUS = 16
RAY_NUM_GPUS = 1
RAY_OBJECT_STORE_BYTES = 5 * GB
RAY_HEAP_BYTES = 6 * GB

# Traffic-level classification (used by Ray Serve /detect).
# Both are (low_upper, mid_upper) pairs:
#   value < low_upper   -> "low"
#   value < mid_upper   -> "medium"
#   value >= mid_upper  -> "high"
# Tuned against UA-DETRAC's ~7.5 boxes/frame average; adjust per camera angle.
TRAFFIC_COUNT_THRESHOLDS = (5, 15)        # vehicles in frame
TRAFFIC_DENSITY_THRESHOLDS = (0.05, 0.15)  # total bbox area / image area


def classify_level(value: float, thresholds: tuple) -> str:
    low_upper, mid_upper = thresholds
    if value < low_upper:
        return "low"
    if value < mid_upper:
        return "medium"
    return "high"
