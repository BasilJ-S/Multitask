from abc import ABC, abstractmethod

import torch
import xgboost as xgb

from multitask.utils.logger import logger


class BaselineModel(ABC):
    @abstractmethod
    def fit_and_predict(
        self,
        X_train: torch.Tensor,
        y_train: list[torch.Tensor],
        X_eval: torch.Tensor,
        y_eval: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class GlobalMeanBaseline(BaselineModel):
    """Global mean predictor for multitask learning. Predicts the mean value of the target variable from the training set for all inputs."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return "GlobalMeanBaseline"

    def fit_and_predict(
        self,
        X_train: torch.Tensor,
        y_train: list[torch.Tensor],
        X_eval: torch.Tensor,
        y_eval: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        logger.info("Running Global Mean Predictor baseline...")
        val_pred: list[torch.Tensor] = []
        train_pred: list[torch.Tensor] = []
        for task_idx in range(len(y_train)):
            task_y_train = y_train[task_idx]
            task_y_train_mean = torch.mean(task_y_train, dim=0)  # mean over batch
            task_y_train_mean = torch.mean(
                task_y_train_mean, dim=0
            )  # mean over time/targets if needed

            logger.info(
                f"Task {task_idx} mean value over training set: {task_y_train_mean}"
            )

            task_y_mean_train_size = task_y_train_mean.repeat(
                task_y_train.shape[0], task_y_train.shape[1], 1
            )

            train_pred.append(task_y_mean_train_size)

            task_y_mean_validation_size = task_y_train_mean.repeat(
                y_eval[task_idx].shape[0],
                y_eval[task_idx].shape[1],
                1,
            )
            logger.info(
                f"Mean Predictor predictions for task {task_idx} shape: {task_y_mean_validation_size.shape}"
            )
            val_pred.append(task_y_mean_validation_size)

        return train_pred, val_pred


class LinearPredictor(BaselineModel):
    """Linear predictor for multitask learning. Trains a separate linear model for each task, and for each forecast step."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return "LinearPredictor"

    def fit_and_predict(
        self,
        X_train: torch.Tensor,
        y_train: list[torch.Tensor],
        X_eval: torch.Tensor,
        y_eval: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        val_pred = []
        train_pred = []
        logger.info("Running Linear Predictor baseline...")
        for task_idx in range(len(y_train)):
            y_train_task = y_train[task_idx]  # (B, T_f, D_out)
            X_train_task = X_train  # (B, T_c, D_in)
            B, T_c, D_in = X_train_task.shape
            _, T_f, D_out = y_train_task.shape

            B_val, _, _ = X_eval.shape

            y_t = y_train_task.reshape(B, -1)  # flatten (B, T_f*D_out)
            X_flat = X_train_task.reshape(B, -1)  # flatten context: (B, T_c*D_in)
            X_b = torch.cat([X_flat, torch.ones(B, 1)], dim=1)  # (B, T_c*D_in+1)

            # Least squares
            w = torch.linalg.lstsq(X_b, y_t).solution  # (T_c*D_in+1, D_out)

            # Predictions
            pred_train = (X_b @ w).reshape(B, T_f, D_out)

            X_val_flat = X_eval.reshape(B_val, -1)
            X_val_b = torch.cat([X_val_flat, torch.ones(B_val, 1)], dim=1)
            pred_val = (X_val_b @ w).reshape(B_val, T_f, D_out)

            # Concatenate across forecast horizon
            train_pred.append(pred_train)  # (B, T_f, D_out)
            val_pred.append(pred_val)  # (B, T_f, D_out)

        return train_pred, val_pred


class XGBoostPredictor(BaselineModel):
    """XGBoost predictor for multitask learning. Trains a separate XGBoost model for each task."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return "XGBoostPredictor"

    def fit_and_predict(
        self,
        X_train: torch.Tensor,
        y_train: list[torch.Tensor],
        X_eval: torch.Tensor,
        y_eval: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        val_pred = []
        train_pred = []
        for task_idx in range(len(y_train)):
            logger.info(f"Training XGBoost for task {task_idx}...")
            task_y_train = y_train[task_idx].reshape(
                y_train[task_idx].shape[0], -1
            )  # flatten (B, T_f*D_out)

            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=10,
                max_depth=3,
                subsample=0.5,
                colsample_bytree=0.5,
                gamma=1,
                eta=0.1,
            )
            model.fit(
                X_train.reshape(X_train.shape[0], -1).numpy(),
                task_y_train.numpy(),
            )

            train_pred_task = model.predict(
                X_train.reshape(X_train.shape[0], -1).numpy()
            )
            val_pred_task = model.predict(X_eval.reshape(X_eval.shape[0], -1).numpy())

            train_pred.append(
                torch.tensor(train_pred_task, dtype=torch.float32).reshape_as(
                    y_train[task_idx]
                )
            )
            val_pred.append(
                torch.tensor(val_pred_task, dtype=torch.float32).reshape_as(
                    y_eval[task_idx]
                )
            )

        return train_pred, val_pred


ALL_BASELINES: list[BaselineModel] = [
    GlobalMeanBaseline(),
    LinearPredictor(),
    XGBoostPredictor(),
]

BASELINE_NAMES = [model.name for model in ALL_BASELINES]
