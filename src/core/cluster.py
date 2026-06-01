"""Ray Core：叢集連線。

本專案所有任務都在 Docker 容器內執行（`docker compose exec ray-head python ...`）。
容器啟動時已用 `ray start --head --num-cpus=16 --num-gpus=1` 把資源池建好，
並在環境變數設了 RAY_ADDRESS=auto。

所以這裡只需要做一件事：**接上那個現成的叢集**。
不需要自己決定要配多少 CPU/GPU——資源池由容器的 ray-head 保證（16 核 + 1 GPU）。
"""

import os

import ray


def init_ray(verbose: bool = True) -> None:
    """接上正在執行的 Ray 叢集（容器內的 ray-head）。

    重複呼叫安全：若已連線則直接返回，不會重複初始化。

    Args:
        verbose: 連上後印出叢集的資源池（CPU/GPU 總量），方便確認接到的是
                 預期的 16 核 + 1 GPU。
    """
    if ray.is_initialized():
        return

    # 容器內 RAY_ADDRESS=auto；"auto" 會讓 Ray 自動找到本機正在跑的 head。
    address = os.environ.get("RAY_ADDRESS", "auto")
    ray.init(address=address, ignore_reinit_error=True)

    if verbose:
        res = ray.cluster_resources()
        print(f"[init_ray] 已接上叢集 | "
              f"CPU={res.get('CPU')} GPU={res.get('GPU')}")
