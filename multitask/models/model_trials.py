from typing import Any

import optuna
import torch

from multitask.models.models import (
    MultiTaskHardShareMLP,
    MultiTaskNaiveMLP,
    MultiTaskResidualNetwork,
    MultiTaskTTSoftShareMLP,
)


def naive_mlp_params(trial: optuna.trial.Trial) -> dict[str, list[int]]:
    # Optimize number of layers and hidden size
    num_layers = trial.suggest_int("num_layers", 1, 5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    return {"hidden_sizes": [hidden_dim] * num_layers}


def hard_share_params(trial: optuna.trial.Trial) -> dict[str, list[int]]:
    # Shared layers and task-specific layers
    num_shared = trial.suggest_int("num_shared_layers", 1, 5)
    num_task = trial.suggest_int("num_task_layers", 1, 5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    return {
        "shared_hidden_sizes": [hidden_dim] * num_shared,
        "task_hidden_sizes": [hidden_dim] * num_task,
    }


def tt_soft_params(trial: optuna.trial.Trial) -> dict[str, list[int] | int]:
    num_layers = trial.suggest_int("num_layers", 1, 5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    tt_rank = trial.suggest_int("tt_rank", 2, 32)
    return {"hidden_sizes": [hidden_dim] * num_layers, "tt_rank": tt_rank}


def residual_network_params(trial: optuna.trial.Trial) -> dict[str, list[int]]:
    num_shared = trial.suggest_int("num_shared_layers", 1, 5)
    num_task = trial.suggest_int("num_task_layers", 1, 2)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    return {
        "shared_hidden_sizes": [hidden_dim] * num_shared,
        "task_hidden_sizes": [hidden_dim] * num_task,
    }


MODEL_LIST: list[tuple[type, Any]] = [
    (MultiTaskResidualNetwork, residual_network_params),
    (MultiTaskTTSoftShareMLP, tt_soft_params),
    (MultiTaskNaiveMLP, naive_mlp_params),
    (MultiTaskHardShareMLP, hard_share_params),
]
