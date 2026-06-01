"""Ray Tune: hyperparameter sweep for the accident classifier.

Same single-GPU caveat as sweep.py -- trials run sequentially because each
needs num_gpus=1. ASHA still helps by stopping bad trials early.
"""

from pathlib import Path
from typing import Any, Dict

from ray import tune
from ray.data import Dataset
from ray.tune import Tuner
from ray.tune.schedulers import ASHAScheduler
from ray.tune.tune_config import TuneConfig

from src.train.trainer import build_trainer_cls

DEFAULT_CLS_PARAM_SPACE = {
    "train_loop_config": {
        "lr":           tune.grid_search([5e-5, 2e-5, 1e-5]),
        "weight_decay": tune.grid_search([1e-4, 1e-3]),
        "batch_size":   tune.choice([32]),
    },
}


def build_tuner_cls(train_ds: Dataset, val_ds: Dataset,
                    base_loop_config: Dict[str, Any],
                    experiment_name: str,
                    storage_path: Path,
                    param_space: Dict[str, Any] = None,
                    num_samples: int = 1,
                    grace_period: int = 2) -> Tuner:
    """Mirrors build_tuner() in sweep.py. metric/mode go on TuneConfig only --
    setting them on ASHAScheduler too triggers Ray's duplicate-config error."""
    base_trainer = build_trainer_cls(
        train_ds, val_ds,
        train_loop_config=base_loop_config,
        experiment_name=experiment_name,
        storage_path=storage_path,
    )

    space = param_space or DEFAULT_CLS_PARAM_SPACE
    return Tuner(
        base_trainer,
        param_space=space,
        tune_config=TuneConfig(
            num_samples=num_samples,
            metric="val_f1",
            mode="max",
            scheduler=ASHAScheduler(
                grace_period=grace_period,
                reduction_factor=2,
            ),
        ),
    )
