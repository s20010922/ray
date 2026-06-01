# Ray + YOLOv8 framework template

Clean copy of the Ray scaffolding from the Ray0527 traffic-detection project.
No datasets, no trained checkpoints, no experimental scripts — just the
generic Ray Data / Train / Tune / Serve wiring around an Ultralytics YOLOv8
detection + classification pipeline.

## What's in here

```
new ray/
├── Dockerfile               # CUDA + Python + ray + ultralytics base image
├── docker-compose.yml       # two services: `train` (batch) and `ray-head` (long-lived)
├── requirements.txt         # pinned Python deps
├── .dockerignore
├── scripts/                 # (empty — drop your entry points here)
├── ray_results/             # (empty — Ray Train + Tune write run artifacts here)
└── src/
    ├── config.py            # IMG_SIZE, NUM_CLASSES, MAX_BOXES, UADETRAC_TO_COCO, ...
    ├── viz.py               # bbox drawing helper
    ├── core/
    │   └── cluster.py       # init_ray() — auto-attach to RAY_ADDRESS or start local
    ├── data/
    │   ├── pipeline.py      # build_ray_dataset() / build_ray_dataset_cls() — Ray Data
    │   │                    # streaming preprocess with detection + classification augs
    │   └── targets.py       # pad_to_max, unpack_to_v8_targets, xywhn_to_xyxy_px
    ├── modeling/
    │   ├── yolo.py          # load_yolo_for_training, load_yolo_cls_for_training
    │   ├── loss.py          # build_loss (wraps ultralytics v8DetectionLoss)
    │   └── metrics.py       # match_and_score, precision_recall_f1 (IoU>=0.5 greedy)
    ├── train/
    │   ├── trainer.py       # build_trainer / build_trainer_cls (TorchTrainer factory)
    │   └── worker.py        # train_loop_per_worker(_cls) — mixup, cosine LR, val
    ├── serve/
    │   └── api.py           # build_traffic_app, build_accident_app (FastAPI + Serve)
    └── tune/
        ├── sweep.py         # Ray Tune wrapper for traffic detection
        └── sweep_cls.py     # Ray Tune wrapper for accident classification
```

## What was intentionally NOT copied

| | Why |
|---|---|
| `src/data/sources.py`      | Project-specific dataset loaders (UA-DETRAC, UCF Crime, DoTA). Write your own `list_*` returning `[{image_path, ...}]` records. |
| `src/data/freeway_cctv.py` | Project-specific Taiwan freeway CCTV MJPEG client. |
| `scripts/*.py`             | All entry points (train, serve, eval, dashboards). Copy from Ray0527 as needed, or write fresh. |
| `ray_results/*`            | Trained checkpoints (gigabytes). Retrain when you have data. |
| `cctv_snapshots/`, `mock_data/`, `reference/`, `eval/`, `Li/` | Runtime data dirs. |
| `docs/`                    | Original-project docs. |

## How to bring it up

```powershell
cd "C:\Users\s9663\Desktop\new ray"
docker compose up -d ray-head
# dashboard: http://localhost:8265
docker compose exec ray-head python -c "from src.core.cluster import init_ray; init_ray(); print('ray ok')"
docker compose down
```

## How to add a dataset

1. Drop your data on the host (e.g. `D:/my_dataset/`).
2. Add a bind in `docker-compose.yml` under `x-bind-mounts`:
   ```yaml
   - D:/my_dataset:/data/my_dataset:ro
   ```
3. Write `src/data/sources.py` with a function returning records the pipeline
   expects:
   - detection: `[{"image_path": str, "boxes_xyxy": np.ndarray(N,4), "labels": np.ndarray(N,)}]`
   - classification: `[{"image_path": str, "label": int}]`
4. Restart: `docker compose down && docker compose up -d ray-head`.

## How to write a training script

Mirror this skeleton in `scripts/train_*.py`:

```python
# BLAS thread caps before numpy/torch/cv2 import
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")

from pathlib import Path
from src.core.cluster import init_ray
from src.data.pipeline import build_ray_dataset       # or build_ray_dataset_cls
from src.train.trainer  import build_trainer          # or build_trainer_cls

init_ray()
train_records = ...   # your records (see "How to add a dataset" above)
val_records   = ...
trainer = build_trainer(
    train_ds=build_ray_dataset(train_records, augment=True),
    val_ds  =build_ray_dataset(val_records,   augment=False),
    train_loop_config={
        "epochs":        10,
        "batch_size":    8,
        "lr":            1e-4,
        "model_weights": "yolov8n.pt",
        "use_cosine_lr": True,
    },
    experiment_name="my_run",
    storage_path=Path("/workspace/ray_results"),
)
result = trainer.fit()
print(result.metrics)
print(result.checkpoint)
```

## How to deploy with Ray Serve

```python
from ray import serve
from src.core.cluster import init_ray
from src.serve.api    import build_traffic_app, build_accident_app

init_ray()
serve.start(http_options={"host": "0.0.0.0", "port": 8000})
serve.run(build_traffic_app(
    checkpoint_path="/workspace/ray_results/.../checkpoint_NNN/model.pt",
    model_weights  ="yolov8n.pt",
    conf_thres=0.15, nms_iou=0.45,
), name="traffic", route_prefix="/traffic")
```

Then `curl -X POST -F 'file=@img.jpg' http://localhost:8000/traffic/detect`.

## Dependency notes

- Container Python is **3.10**. Host Python 3.14 is not compatible with the
  pinned `ray` / `torch` / `ultralytics` versions in `requirements.txt`.
- GPU: a single CUDA device is exclusively used. `Train`, `Tune`, and `Serve`
  fight over it — shut Serve down before launching training:
  `docker compose exec ray-head serve shutdown -y`.
- `train.report(metrics, checkpoint=...)` saves to `ray_results/<experiment>/.../checkpoint_NNNNNN/model.pt`.
  Top-K kept via `CheckpointConfig(num_to_keep=N, checkpoint_score_attribute=..., ...)`.
