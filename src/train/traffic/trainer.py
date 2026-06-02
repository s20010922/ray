"""組裝車流偵測的 Ray Train TorchTrainer。

吃 traffic 的 Ray Data pipeline（UA-DETRAC，含劣化增強），單 GPU 訓練
YOLO11n 偵測（單類 Vehicle）。checkpoint 依 val_loss 取最佳。
"""

from typing import Optional

from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from src.data.traffic.pipeline import build_ray_dataset
from src.data.traffic.sources import list_detrac_records
from src.train.traffic.worker import train_loop_per_worker


def build_trainer(epochs: int = 30,
                  lr: float = 1e-3,
                  batch_size: int = 16,
                  frame_stride: int = 10,
                  limit_sequences: Optional[int] = None,
                  detrac_root: str = "/data/detrac",
                  storage_path: str = "/workspace/ray_results",
                  experiment_name: str = "traffic") -> TorchTrainer:
    """建立車流偵測 TorchTrainer。

    Args:
        frame_stride: 抽幀間隔（10≈1.4萬張）。
        limit_sequences: 只用前 N 序列（先驗證用）；None=全部 100 序列。
    """
    train_records, val_records = list_detrac_records(
        detrac_root=detrac_root, frame_stride=frame_stride,
        limit_sequences=limit_sequences)

    train_ds = build_ray_dataset(train_records, augment=True, batch_size=batch_size)
    val_ds = build_ray_dataset(val_records, augment=False, batch_size=batch_size)

    return TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={"epochs": epochs, "lr": lr, "batch_size": batch_size},
        scaling_config=ScalingConfig(num_workers=1, use_gpu=True),
        datasets={"train": train_ds, "val": val_ds},
        run_config=RunConfig(
            storage_path=storage_path,
            name=experiment_name,
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
            ),
        ),
    )
