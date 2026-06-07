"""Stage 3 — Ray Train：用 TorchTrainer 編排 ultralytics 訓練 yolo11s。

單 GPU 環境 → 1 個 GPU worker（head 持有）。訓練本體仍由 ultralytics 負責
（內建偵測 loss/letterbox/mAP/early-stop），Ray Train 負責資源排程與結果彙整，
讓 freeway 訓練納入 Ray Train 階段。最佳超參由 Stage 2（tune_freeway）搜得。
"""

from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer


def _train_loop(config: dict):
    import ray.train
    from ultralytics import YOLO

    model = YOLO(config["weights"])
    model.train(
        data=config["data"],
        imgsz=config["imgsz"],
        epochs=config["epochs"],
        batch=config["batch"],
        patience=config["patience"],
        single_cls=True,
        optimizer="AdamW",              # 固定 optimizer，讓搜到的 lr0 生效
        device=0,                       # Ray 已把指派的 GPU 設為可見
        project=config["project"],
        name=config["name"],
        exist_ok=True,
        **config["hp"],
    )
    m = getattr(model.trainer, "metrics", None) or {}
    ray.train.report({
        "mAP50": float(m.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(m.get("metrics/mAP50-95(B)", 0.0)),
    })


def run_train(data: str, weights: str = "yolo11s.pt", imgsz: int = 640,
              epochs: int = 100, batch: int = 16, patience: int = 30,
              hp: dict | None = None,
              project: str = "/workspace/ray_results",
              name: str = "freeway_final"):
    """啟動 Ray Train 訓練，回傳 ultralytics best.pt 路徑與最終指標。"""
    cfg = {
        "data": data, "weights": weights, "imgsz": imgsz, "epochs": epochs,
        "batch": batch, "patience": patience, "project": project,
        "name": name, "hp": hp or {},
    }
    trainer = TorchTrainer(
        _train_loop,
        train_loop_config=cfg,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=True),
        run_config=RunConfig(storage_path=project, name=f"{name}_raytrain"),
    )
    result = trainer.fit()
    best = f"{project}/{name}/weights/best.pt"
    print(f"[Ray Train] 完成 | metrics={result.metrics} | best={best}")
    return best, result.metrics
