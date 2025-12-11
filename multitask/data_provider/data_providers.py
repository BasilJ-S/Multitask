from typing import Callable, Protocol

import pandas as pd
import torch
from datasets import Dataset as hfDataset
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler

from multitask.data.preprocess_gridstatus_dataset import path_to_file
from multitask.data_provider.multi_datasets import (
    PREDICTION_NODES,
    PreScaledHFDataset,
    PreScaledTimeseriesDataset,
)
from multitask.models.models import (
    MultiTaskHardShareMLP,
    MultiTaskNaiveMLP,
    MultiTaskResidualNetwork,
    MultiTaskTTSoftShareMLP,
    NaiveMultiTaskTimeseriesWrapper,
)
from multitask.utils.logger import logger


class DataProvider(Protocol):
    def __call__(self, device=torch.device("cpu")) -> tuple[
        torch.utils.data.Dataset,  # train dataset
        torch.utils.data.Dataset,  # validation dataset
        list[str],  # feature names
        list[list[str]],  # target names per task
        list[float],  # task weights
        list[StandardScaler],  # target scalers
        int,  # input size
        list[int],  # output sizes per task
        int,  # context length
        int,  # forecast horizon
    ]: ...


def prepare_dataset(data_preparer: DataProvider, device=torch.device("cpu")):
    return data_preparer(device=device)


def prepare_housing_dataset(device=torch.device("cpu")):
    ds = load_dataset("gvlassis/california_housing").with_format("torch")

    hf_train: hfDataset = ds["train"]  # type: ignore
    hf_validation: hfDataset = ds["validation"]  # type: ignore

    targets = [["MedHouseVal", "AveRooms"], ["Longitude"]]
    task_weights = [1.0, 0.5]  # Weight for each task in the loss computation
    all_targets = [t for sublist in targets for t in sublist]
    features = [col for col in hf_train.column_names if col not in all_targets]

    for i, target_group in enumerate(targets):
        logger.info(
            f"Targets for Task {i} with weight: {task_weights[i]}: {target_group}"
        )

    # ---- CREATE PRE-SCALED DATASETS ----
    train_dataset = PreScaledHFDataset(
        hf_train, features, targets, scaler_X=None, scaler_y=None, create_scalers=True
    )
    feature_scaler = train_dataset.scaler_X
    target_scalers = train_dataset.scaler_y
    validation_dataset = PreScaledHFDataset(
        hf_validation,
        features,
        targets,
        scaler_X=feature_scaler,
        scaler_y=target_scalers,
    )
    input_size = len(features)
    output_sizes = [len(t) for t in targets]

    # use MultiTaskMLP for two targets (MedHouseVal and AveRooms)
    return (
        train_dataset,
        validation_dataset,
        features,
        targets,
        task_weights,
        target_scalers,
        input_size,
        output_sizes,
        1,  # context length not applicable
        1,  # forecast horizon not applicable
    )


def prepare_weather_data():
    train = pd.read_csv(path_to_file("weather_train_set"), index_col="Date Time")
    train_inference_times = pd.read_csv(path_to_file("weather_train_inference_times"))
    validation = pd.read_csv(
        path_to_file("weather_validation_set"),
        index_col="Date Time",
    )
    validation_inference_times = pd.read_csv(
        path_to_file("weather_validation_inference_times")
    )

    target_cols = [
        ["T (degC)", "Tpot (K)", "Tdew (degC)"],
        ["rh (%)", "sh (g/kg)", "H2OC (mmol/mol)"],
        ["p (mbar)", "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)"],
        ["rho (g/m**3)", "wv (m/s)", "max. wv (m/s)", "wd (deg)"],
    ]

    are_features_available_through_prediction_time = False

    # Forecasting all targets together
    features = [col for col in train.columns]
    logger.info(f"All features ({len(features)}): {features}")
    task_weights = [1.0 for _ in target_cols]  # Weight for each task equally

    context_length = 24 * 2  # 2 days
    forecast_horizon = 24  # 1 day
    for i, target_group in enumerate(target_cols):
        logger.info(
            f"Targets for Task {i} with weight: {task_weights[i]}: {target_group}"
        )
    return (
        train,
        train_inference_times,
        validation,
        validation_inference_times,
        features,
        target_cols,
        task_weights,
        context_length,
        forecast_horizon,
        are_features_available_through_prediction_time,
    )


def prepare_ercot_data():
    train = pd.read_csv(
        path_to_file("gridstatus_train_set"),
        index_col="interval_end_utc",
    )
    train_inference_times = pd.read_csv(
        path_to_file("gridstatus_train_inference_times")
    )
    validation = pd.read_csv(
        path_to_file("gridstatus_validation_set"),
        index_col="interval_end_utc",
    )
    validation_inference_times = pd.read_csv(
        path_to_file("gridstatus_validation_inference_times")
    )

    target_cols = [[f"spp_{prediction_node}"] for prediction_node in PREDICTION_NODES]
    all_targets = [t for sublist in target_cols for t in sublist]
    features = [col for col in train.columns if col not in all_targets]
    task_weights = [1.0 for _ in target_cols]  # Weight for each task equally

    are_features_available_through_prediction_time = True

    context_length = 24 * 2  # 2 days
    forecast_horizon = 24  # 1 day

    for i, target_group in enumerate(target_cols):
        logger.info(
            f"Targets for Task {i} with weight: {task_weights[i]}: {target_group}"
        )
    return (
        train,
        train_inference_times,
        validation,
        validation_inference_times,
        features,
        target_cols,
        task_weights,
        context_length,
        forecast_horizon,
        are_features_available_through_prediction_time,
    )


def prepare_ercot_full(device=torch.device("cpu")):
    return prepare_timeseries_dataset(prepare_ercot_data, device=device)


def prepare_weather_full(device=torch.device("cpu")):
    return prepare_timeseries_dataset(prepare_weather_data, device=device)


def prepare_timeseries_dataset(data_preparer: Callable, device=torch.device("cpu")):

    (
        train,
        train_inference_times,
        validation,
        validation_inference_times,
        features,
        target_cols,
        task_weights,
        context_length,
        forecast_horizon,
        are_features_available_through_prediction_time,
    ) = data_preparer()

    # ---- CREATE PRE-SCALED DATASETS ----
    train_dataset = PreScaledTimeseriesDataset(
        timeseries=train,
        inference_and_prediction_intervals=train_inference_times,
        feature_cols=features,
        target_cols=target_cols,
        create_scalers=True,
        context_window_hours=context_length,
        prediction_horizon_hours=forecast_horizon,
        are_features_available_through_prediction_time=are_features_available_through_prediction_time,
    )
    feature_scaler = train_dataset.scaler_X
    target_scalers = train_dataset.scaler_y
    validation_dataset = PreScaledTimeseriesDataset(
        timeseries=validation,
        inference_and_prediction_intervals=validation_inference_times,
        feature_cols=features,
        target_cols=target_cols,
        scaler_X=feature_scaler,
        scaler_y=target_scalers,
        context_window_hours=context_length,
        prediction_horizon_hours=forecast_horizon,
        are_features_available_through_prediction_time=are_features_available_through_prediction_time,
    )
    input_size = len(features)
    output_sizes = [len(t) for t in target_cols]

    # use MultiTaskMLP for two targets (MedHouseVal and AveRooms)
    return (
        train_dataset,
        validation_dataset,
        features,
        target_cols,
        task_weights,
        target_scalers,
        input_size,
        output_sizes,
        context_length,
        forecast_horizon,
    )


def model_factory(
    input_size: int,
    output_sizes: list[int],
    context_length: int,
    forecast_horizon: int,
    device=torch.device("cpu"),
) -> list[NaiveMultiTaskTimeseriesWrapper]:
    modelList = [
        NaiveMultiTaskTimeseriesWrapper(
            model_class=MultiTaskResidualNetwork,
            input_size=input_size,
            task_hidden_sizes=[16, 16],
            shared_hidden_sizes=[16, 16],
            output_sizes=output_sizes,
            context_length=context_length,
            forecast_horizon=forecast_horizon,
        ).to(device),
        NaiveMultiTaskTimeseriesWrapper(
            model_class=MultiTaskTTSoftShareMLP,
            input_size=input_size,
            hidden_sizes=[16, 16, 16],
            output_sizes=output_sizes,
            tt_rank=4,
            context_length=context_length,
            forecast_horizon=forecast_horizon,
        ).to(device),
        NaiveMultiTaskTimeseriesWrapper(
            model_class=MultiTaskNaiveMLP,
            input_size=input_size,
            output_sizes=output_sizes,
            hidden_sizes=[16, 16, 16],
            context_length=context_length,
            forecast_horizon=forecast_horizon,
        ).to(device),
        NaiveMultiTaskTimeseriesWrapper(
            model_class=MultiTaskHardShareMLP,
            input_size=input_size,
            shared_hidden_sizes=[18, 18, 18],
            task_hidden_sizes=[],
            output_sizes=output_sizes,
            context_length=context_length,
            forecast_horizon=forecast_horizon,
        ).to(device),
    ]
    return modelList
