"""Ray Train: build a TorchTrainer wired up with our worker function +
datasets + checkpoint config. Exposed as a factory so Ray Tune can reuse it.
"""

from pathlib import Path
from typing import Any, Dict

from ray.data import Dataset
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from src.train.worker import train_loop_per_worker, train_loop_per_worker_cls


def build_trainer(train_ds: Dataset, val_ds: Dataset,
                  train_loop_config: Dict[str, Any],
                  experiment_name: str,
                  storage_path: Path,
                  num_workers: int = 1,
                  use_gpu: bool = True,
                  checkpoints_to_keep: int = 3) -> TorchTrainer:
    return TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        datasets={"train": train_ds, "val": val_ds},
        run_config=RunConfig(
            name=experiment_name,
            storage_path=str(storage_path),
            checkpoint_config=CheckpointConfig(
                num_to_keep=checkpoints_to_keep,
                checkpoint_score_attribute="val_f1@0.5",
                checkpoint_score_order="max",
            ),
        ),
    )


def build_trainer_cls(train_ds: Dataset, val_ds: Dataset,
                      train_loop_config: Dict[str, Any],
                      experiment_name: str,
                      storage_path: Path,
                      num_workers: int = 1,
                      use_gpu: bool = True,
                      checkpoints_to_keep: int = 3) -> TorchTrainer:
    """Classification variant. Same scaffolding, different worker and the
    checkpoint selector tracks `val_f1` (no @0.5 suffix -- no IoU threshold)."""
    return TorchTrainer(
        train_loop_per_worker=train_loop_per_worker_cls,
        train_loop_config=train_loop_config,
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        datasets={"train": train_ds, "val": val_ds},
        run_config=RunConfig(
            name=experiment_name,
            storage_path=str(storage_path),
            checkpoint_config=CheckpointConfig(
                num_to_keep=checkpoints_to_keep,
                checkpoint_score_attribute="val_f1",
                checkpoint_score_order="max",
            ),
        ),
    )
