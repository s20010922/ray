"""Ray Core: cluster lifecycle.

Two modes:
  * Standalone — `ray.init(num_cpus=..., num_gpus=..., ...)` starts a fresh
    local cluster with our project's resource caps.
  * Attach — if `RAY_ADDRESS` env var is set (the ray-head compose service
    does this), `ray.init(address=RAY_ADDRESS)` joins the running cluster.
"""

import os

import ray

from src.config import (
    RAY_HEAP_BYTES,
    RAY_NUM_CPUS,
    RAY_NUM_GPUS,
    RAY_OBJECT_STORE_BYTES,
)


def init_ray() -> None:
    if ray.is_initialized():
        return
    addr = os.environ.get("RAY_ADDRESS")
    if addr:
        ray.init(address=addr, ignore_reinit_error=True)
    else:
        ray.init(
            num_cpus=RAY_NUM_CPUS,
            num_gpus=RAY_NUM_GPUS,
            object_store_memory=RAY_OBJECT_STORE_BYTES,
            _memory=RAY_HEAP_BYTES,
            include_dashboard=True,
            dashboard_host="0.0.0.0",
            ignore_reinit_error=True,
        )
