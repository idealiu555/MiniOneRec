import math
import os
from typing import Any

import numpy as np
import swanlab
from transformers import TrainerCallback


def init_swanlab_run(
    *,
    project: str,
    run_name: str,
    requested_mode: str,
    default_log_dir: str,
    config: dict,
    is_main_process: bool,
):
    env_mode = os.environ.get("SWANLAB_MODE")
    if requested_mode == "cloud" and env_mode:
        requested_mode = env_mode
    log_dir = os.environ.get("SWANLAB_LOG_DIR", default_log_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.environ["SWANLAB_MODE"] = requested_mode
    os.environ["SWANLAB_LOG_DIR"] = log_dir
    os.environ["SWANLAB_ACTIVE_RUN"] = "0"

    if not is_main_process:
        return requested_mode, log_dir, None

    try:
        run = swanlab.init(
            project=project,
            experiment_name=run_name,
            mode=requested_mode,
            logdir=log_dir,
            config=config,
        )
        os.environ["SWANLAB_ACTIVE_RUN"] = "1"
        print(f"SwanLab initialized: project={project}, run_name={run_name}, mode={requested_mode}, logdir={log_dir}")
        return requested_mode, log_dir, run
    except Exception as exc:
        if requested_mode != "cloud":
            raise
        fallback_mode = "local"
        os.environ["SWANLAB_MODE"] = fallback_mode
        print(f"SwanLab cloud init failed ({exc}). Falling back to local mode with logdir={log_dir}.")
        run = swanlab.init(
            project=project,
            experiment_name=run_name,
            mode=fallback_mode,
            logdir=log_dir,
            config=config,
        )
        os.environ["SWANLAB_ACTIVE_RUN"] = "1"
        print(f"SwanLab initialized: project={project}, run_name={run_name}, mode={fallback_mode}, logdir={log_dir}")
        return fallback_mode, log_dir, run


def _normalize_log_value(value: Any):
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return None


class SwanLabCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if os.environ.get("SWANLAB_ACTIVE_RUN") != "1" or not getattr(state, "is_world_process_zero", True):
            return control

        payload = {}
        for key, value in (logs or {}).items():
            normalized = _normalize_log_value(value)
            if normalized is not None:
                payload[key] = normalized

        if payload:
            swanlab.log(payload, step=state.global_step)

        return control

    def on_train_end(self, args, state, control, **kwargs):
        if os.environ.get("SWANLAB_ACTIVE_RUN") != "1" or not getattr(state, "is_world_process_zero", True):
            return control

        finish = getattr(swanlab, "finish", None)
        if callable(finish):
            finish()
        return control
