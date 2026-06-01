"""Ray Tune: hyperparameter sweep wrapped around our Ray Train trainer.

Single-GPU caveat: trials run *sequentially* because each needs `num_gpus=1`.
ASHA scheduler still helps -- it stops bad trials early so the next one
starts sooner. With more GPUs you'd get real concurrency.
"""

from pathlib import Path
from typing import Any, Dict

from ray import tune
from ray.data import Dataset
from ray.tune import Tuner
from ray.tune.schedulers import ASHAScheduler
from ray.tune.tune_config import TuneConfig

from src.train.trainer import build_trainer

DEFAULT_PARAM_SPACE = {
    "train_loop_config": {
        "lr":         tune.grid_search([1e-3, 1e-4, 5e-5]),
        "batch_size": tune.choice([8, 16]),
    },
}


def build_tuner(train_ds: Dataset, val_ds: Dataset,
                base_loop_config: Dict[str, Any],
                experiment_name: str,
                storage_path: Path,
                param_space: Dict[str, Any] = None,
                num_samples: int = 1,
                grace_period: int = 3) -> Tuner:
    """`base_loop_config` is the static part (epochs, model_weights); the
    sweep replaces only what's in `param_space["train_loop_config"]`.
    """
    base_trainer = build_trainer(
        train_ds, val_ds,
        train_loop_config=base_loop_config,
        experiment_name=experiment_name,
        storage_path=storage_path,
    )

    space = param_space or DEFAULT_PARAM_SPACE
    return Tuner(
        base_trainer,
        param_space=space,
        tune_config=TuneConfig(
            num_samples=num_samples,
            metric="val_f1@0.5",
            mode="max",
            # ASHA inherits metric/mode from TuneConfig; don't set them here
            # too or Ray raises "you passed metric/mode to both" ValueError.
            scheduler=ASHAScheduler(
                grace_period=grace_period,
                reduction_factor=2,
            ),
        ),
    )
