"""組裝車禍分類的 Ray Train TorchTrainer。

build_trainer() 把 accident 的 Ray Data pipeline（train/val）餵給 TorchTrainer，
單 GPU（num_workers=1, use_gpu=True）訓練 YOLO11n-cls。

data_root 預期結構（由 src/data/accident/split.py 切好）：
  data_root/{train,val,test}/{accident,non-accident}/*.jpg
train/val 用於訓練，test 完全不進來。
"""

from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from src.data.accident.pipeline import build_ray_dataset_cls
from src.data.accident.sources import list_accident_records
from src.train.accident.worker import train_loop_per_worker


def build_trainer(epochs: int = 20,
                  lr: float = 1e-3,
                  batch_size: int = 32,
                  weight_decay: float = 0.0,
                  data_root: str = "/workspace/datasets/accident",
                  storage_path: str = "/workspace/ray_results",
                  experiment_name: str = "accident") -> TorchTrainer:
    """建立車禍分類 TorchTrainer。

    train split 帶劣化增強（模擬高公局低畫質），val 不增強。
    test split 不載入（隔離，僅 eval 用）。
    checkpoint 依 val_acc 取最佳，保留 2 份。
    """
    train_records = list_accident_records(data_root, split="train")
    val_records = list_accident_records(data_root, split="val")

    train_ds = build_ray_dataset_cls(train_records, augment=True)
    val_ds = build_ray_dataset_cls(val_records, augment=False)

    return TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={"epochs": epochs, "lr": lr, "batch_size": batch_size,
                           "weight_decay": weight_decay},
        scaling_config=ScalingConfig(num_workers=1, use_gpu=True),
        datasets={"train": train_ds, "val": val_ds},
        run_config=RunConfig(
            storage_path=storage_path,
            name=experiment_name,
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="val_acc",
                checkpoint_score_order="max",
            ),
        ),
    )
