import re
from typing import Any

import optuna
import torch

from multitask.models.models import (
    MultiTaskHardShareMLP,
    MultiTaskNaiveMLP,
    MultiTaskResidualNetwork,
    MultiTaskTTSoftShareMLP,
    MultiTaskTuckerSoftShareMLP,
)


def num_layers_to_layers_list(num_layers: int, hidden_dim: int) -> list[int]:
    return [hidden_dim] * num_layers


def naive_mlp_params(trial: optuna.trial.Trial) -> dict[str, list[int]]:
    # Optimize number of layers and hidden size
    num_layers = trial.suggest_int("num_layers", 1, 5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    return naive_mlp_optuna_params_to_model_params(num_layers, hidden_dim)


def naive_mlp_optuna_params_to_model_params(
    num_layers: int, hidden_dim: int
) -> dict[str, list[int]]:
    return {"hidden_sizes": num_layers_to_layers_list(num_layers, hidden_dim)}


def hard_share_params(trial: optuna.trial.Trial) -> dict[str, list[int]]:
    # Shared layers and task-specific layers
    num_shared = trial.suggest_int("num_shared_layers", 1, 5)
    num_task = trial.suggest_int("num_task_layers", 1, 5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    return hard_share_optuna_params_to_model_params(num_shared, num_task, hidden_dim)


def hard_share_optuna_params_to_model_params(
    num_shared_layers: int, num_task_layers: int, hidden_dim: int
) -> dict[str, list[int]]:
    return {
        "shared_hidden_sizes": num_layers_to_layers_list(num_shared_layers, hidden_dim),
        "task_hidden_sizes": num_layers_to_layers_list(num_task_layers, hidden_dim),
    }


def tucker_soft_params(trial: optuna.trial.Trial) -> dict[str, list[int] | int]:
    num_layers = trial.suggest_int("num_layers", 1, 5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    tucker_rank = trial.suggest_int("tucker_rank", 2, 32)
    return tucker_soft_share_optuna_params_to_model_params(
        num_layers, hidden_dim, tucker_rank
    )


def tucker_soft_share_optuna_params_to_model_params(
    num_layers: int, hidden_dim: int, tucker_rank: int
) -> dict[str, list[int] | int]:
    return {
        "hidden_sizes": num_layers_to_layers_list(num_layers, hidden_dim),
        "tucker_rank": tucker_rank,
    }


def tt_soft_params(trial: optuna.trial.Trial) -> dict[str, list[int] | int]:
    num_layers = trial.suggest_int("num_layers", 1, 5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    tt_rank = trial.suggest_int("tt_rank", 2, 32)
    return tt_soft_share_optuna_params_to_model_params(num_layers, hidden_dim, tt_rank)


def tt_soft_share_optuna_params_to_model_params(
    num_layers: int, hidden_dim: int, tt_rank: int
) -> dict[str, list[int] | int]:
    return {
        "hidden_sizes": num_layers_to_layers_list(num_layers, hidden_dim),
        "tt_rank": tt_rank,
    }


def residual_network_params(trial: optuna.trial.Trial) -> dict[str, list[int]]:
    num_shared = trial.suggest_int("num_shared_layers", 1, 5)
    num_task = trial.suggest_int("num_task_layers", 1, 2)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    return residual_network_optuna_params_to_model_params(
        num_shared, num_task, hidden_dim
    )


def residual_network_optuna_params_to_model_params(
    num_shared_layers: int, num_task_layers: int, hidden_dim: int
) -> dict[str, list[int]]:
    return {
        "shared_hidden_sizes": num_layers_to_layers_list(num_shared_layers, hidden_dim),
        "task_hidden_sizes": num_layers_to_layers_list(num_task_layers, hidden_dim),
    }


MODEL_LIST: list[tuple[type, Any, Any]] = [
    (
        MultiTaskTuckerSoftShareMLP,
        tucker_soft_params,
        tucker_soft_share_optuna_params_to_model_params,
    ),
    (
        MultiTaskResidualNetwork,
        residual_network_params,
        residual_network_optuna_params_to_model_params,
    ),
    (
        MultiTaskTTSoftShareMLP,
        tt_soft_params,
        tt_soft_share_optuna_params_to_model_params,
    ),
    (MultiTaskNaiveMLP, naive_mlp_params, naive_mlp_optuna_params_to_model_params),
    (
        MultiTaskHardShareMLP,
        hard_share_params,
        hard_share_optuna_params_to_model_params,
    ),
]
