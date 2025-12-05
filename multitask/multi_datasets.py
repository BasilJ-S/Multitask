import datetime as dt

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as hfDataset
from datasets import load_dataset
from gridstatus_api import PREDICTION_NODES
from logger import logger
from sklearn.preprocessing import StandardScaler


def assert_contiguous_indices(df: pd.DataFrame):
    """
    Assert that the dataframe has contiguous datetime indices without gaps.
    Raises an AssertionError if gaps are found.
    """
    df = df.sort_index()
    logger.info(
        f"Checking for contiguous indices from {df.index.min()} to {df.index.max()}"
    )
    expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
    if not df.index.equals(expected_index):
        missing_indices = expected_index.difference(df.index)
        raise AssertionError(
            f"DataFrame index is not contiguous. Missing {len(missing_indices)} indices: {missing_indices}"
        )
    logger.info("DataFrame index is contiguous.")


class PreScaledHFDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for Hugging Face datasets that applies pre-scaling to features and targets.
    """

    def __init__(
        self,
        hf_dataset: hfDataset,
        feature_cols: list[str],
        target_cols: list[list[str]],
        scaler_X: StandardScaler | None = None,
        scaler_y: list[StandardScaler] | None = None,
        create_scalers: bool = False,
    ):

        self.feature_cols = feature_cols
        self.target_cols = target_cols

        # Convert to NumPy arrays
        X = np.stack([hf_dataset[col] for col in self.feature_cols], axis=1)

        if create_scalers and (scaler_X is not None or scaler_y is not None):
            raise ValueError(
                "Cannot create scalers and use provided scalers at the same time."
            )

        # Per task
        y_overall = []
        y_created_scalers = []
        for i, task_targets in enumerate(self.target_cols):
            y = np.stack([hf_dataset[col] for col in task_targets], axis=1)
            if create_scalers:
                scaler = StandardScaler()
                scaler.fit(y)
                y_created_scalers.append(scaler)
                y = scaler.transform(y)
            elif scaler_y is not None:
                y = scaler_y[i].transform(y)

            y_overall.append(y)
        # Apply scaling if provided
        created_scaler_x = None
        if create_scalers:
            created_scaler_x = StandardScaler()
            created_scaler_x.fit(X)
            X = created_scaler_x.transform(X)
        elif scaler_X is not None:
            X = scaler_X.transform(X)

        # Store as tensors directly
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = [torch.tensor(y_task, dtype=torch.float32) for y_task in y_overall]

        self.scaler_X = created_scaler_x if create_scalers else scaler_X
        self.scaler_y = y_created_scalers if create_scalers else scaler_y

    def __len__(self) -> int:
        return len(self.y[0])

    def __getitem__(
        self, idx: int
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        y = {f"task_{i}": y_task[idx] for i, y_task in enumerate(self.y)}
        batch = {
            "X": self.X[idx],
            "y": y,
        }
        return batch


class PreScaledTimeseriesDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for Hugging Face datasets that applies pre-scaling to features and targets.
    Assumes hourly data with contiguous datetime index.
    """

    def __init__(
        self,
        timeseries: pd.DataFrame,
        inference_and_prediction_intervals: pd.DataFrame,
        feature_cols: list[str],
        target_cols: list[list[str]],
        scaler_X: StandardScaler | None = None,
        scaler_y: list[StandardScaler] | None = None,
        create_scalers: bool = False,
        context_window_hours: int = 7 * 24,
        prediction_horizon_hours: int = 24,
    ):

        # Initialize basic parameters
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.context_window_hours = context_window_hours
        self.prediction_horizon_hours = prediction_horizon_hours
        logger.info(
            f"Target columns: {self.target_cols}, feature columns: {self.feature_cols}"
        )

        # Convert to pandas datetime if not already
        timeseries = timeseries.copy()
        timeseries.index = pd.to_datetime(timeseries.index, utc=True)
        self.min_date = timeseries.index.min()
        self.max_date = timeseries.index.max()
        logger.info(f"Timeseries date range from {self.min_date} to {self.max_date}")

        inference_and_prediction_intervals = inference_and_prediction_intervals.copy()
        inference_and_prediction_intervals["inference_time_utc"] = pd.to_datetime(
            inference_and_prediction_intervals["inference_time_utc"], utc=True
        )
        inference_and_prediction_intervals["prediction_day_start_utc"] = pd.to_datetime(
            inference_and_prediction_intervals["prediction_day_start_utc"], utc=True
        )
        self.inference_and_prediction_intervals = self.select_valid_inference_times(
            inference_and_prediction_intervals
        )  # Filter to inference times with sufficient context and prediction data

        # Validate inputs
        assert (
            timeseries.index.is_monotonic_increasing
        ), "Timeseries index must be sorted."
        assert (
            inference_and_prediction_intervals.index.is_monotonic_increasing
        ), "Inference and prediction intervals index must be sorted."
        if create_scalers and (scaler_X is not None or scaler_y is not None):
            raise ValueError(
                "Cannot create scalers and use provided scalers at the same time."
            )
        assert_contiguous_indices(timeseries)

        # Map datetime to index for quick lookup
        self.timeseries_date_to_index = {
            date: idx for idx, date in enumerate(timeseries.index)
        }

        # Convert to NumPy arrays
        X = timeseries[self.feature_cols].to_numpy()
        ys = [
            timeseries[target_col_set].to_numpy() for target_col_set in self.target_cols
        ]
        logger.info(f"Feature shape: {X.shape}")
        logger.info(f"Target shapes: {[y.shape for y in ys]}")

        # Per task scaling
        y_overall = []
        y_created_scalers = []
        for i, task_targets in enumerate(self.target_cols):
            y = ys[i]
            if create_scalers:
                scaler = StandardScaler()
                scaler.fit(y)
                y_created_scalers.append(scaler)
                y = scaler.transform(y)
            elif scaler_y is not None:
                y = scaler_y[i].transform(y)
            y_overall.append(y)

        created_scaler_x = None
        if create_scalers:
            created_scaler_x = StandardScaler()
            created_scaler_x.fit(X)
            X = created_scaler_x.transform(X)
        elif scaler_X is not None:
            X = scaler_X.transform(X)

        # Store as tensors directly
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = [torch.tensor(y_task, dtype=torch.float32) for y_task in y_overall]

        self.scaler_X = created_scaler_x if create_scalers else scaler_X
        self.scaler_y = y_created_scalers if create_scalers else scaler_y
        self.validate_indices()

    def select_valid_inference_times(
        self, inference_times: pd.DataFrame
    ) -> pd.DataFrame:
        inference_times = inference_times.copy()
        new_intervals = inference_times[
            inference_times["inference_time_utc"]
            >= self.min_date + self.context_window_hours * dt.timedelta(hours=1)
        ]

        new_intervals = new_intervals[
            new_intervals["prediction_day_start_utc"]
            + self.prediction_horizon_hours * dt.timedelta(hours=1)
            <= self.max_date
        ]
        new_intervals = new_intervals.reset_index(drop=True)
        logger.info(
            f"Filtered inference intervals from {len(inference_times)} to {len(new_intervals)} based on available context and prediction data."
        )
        return new_intervals

    def validate_indices(self):
        for idx, row in self.inference_and_prediction_intervals.iterrows():
            inference_date = row["inference_time_utc"]
            prediction_date = row["prediction_day_start_utc"]
            if inference_date not in self.timeseries_date_to_index:
                raise ValueError(f"Inference date {inference_date} not in timeseries.")
            if prediction_date not in self.timeseries_date_to_index:
                raise ValueError(
                    f"Prediction date {prediction_date} not in timeseries."
                )

        if (
            inference_date - self.context_window_hours * dt.timedelta(hours=1)
            < self.min_date
        ):
            raise ValueError(
                f"Not enough context data for inference date {inference_date}."
            )
        if (
            prediction_date + self.prediction_horizon_hours * dt.timedelta(hours=1)
            > self.max_date
        ):
            raise ValueError(
                f"Not enough prediction data for prediction date {prediction_date}."
            )

    def __len__(self) -> int:
        return len(self.inference_and_prediction_intervals)

    def __getitem__(  # TODO: Fetch from the list, find a valid start and stuff
        self, idx: int
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        inference_date = self.inference_and_prediction_intervals.loc[
            idx, "inference_time_utc"
        ]
        prediction_date = self.inference_and_prediction_intervals.loc[
            idx, "prediction_day_start_utc"
        ]
        inference_idx = self.timeseries_date_to_index[inference_date]
        prediction_idx = self.timeseries_date_to_index[prediction_date]
        logger.debug(
            f"Inference date: {inference_date}, index: {inference_idx}\nPrediction date: {prediction_date}, index: {prediction_idx}"
        )

        y = {
            f"task_{i}": y_task[
                prediction_idx : prediction_idx + self.prediction_horizon_hours
            ]
            for i, y_task in enumerate(self.y)
        }

        batch = {
            "X": self.X[inference_idx - self.context_window_hours : inference_idx],
            "y": y,
        }
        return batch


if __name__ == "__main__":
    # Read in dataframe and set index to datetime
    dataframe = pd.read_csv(
        "multitask/data/gridstatus_train_set.csv",
        index_col="interval_end_utc",
    )
    inference_prediction_intervals = pd.read_csv(
        "multitask/data/gridstatus_train_inference_times.csv",
    )

    target_cols = [[f"spp_{prediction_node}"] for prediction_node in PREDICTION_NODES]
    all_targets = [t for sublist in target_cols for t in sublist]

    dataset = PreScaledTimeseriesDataset(
        timeseries=dataframe,
        inference_and_prediction_intervals=inference_prediction_intervals,
        feature_cols=[
            col for col in dataframe.columns if col not in all_targets
        ],  # Replace with actual feature columns
        target_cols=target_cols,
        create_scalers=False,
    )
    logger.info(
        f"Scalers created: X scaler: {dataset.scaler_X is not None}, y scalers: {[s is not None for s in dataset.scaler_y] if dataset.scaler_y else None}"
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    for batch in dataloader:
        logger.info(f"X shape: {batch['X'].shape}")
        shapes_y = [v.shape for k, v in batch["y"].items()]
        logger.info(f"y shape: {shapes_y}")
        break

    hf_ds = load_dataset("gvlassis/california_housing").with_format("torch")

    hf_train: hfDataset = hf_ds["train"]  # type: ignore
    train_dataset = PreScaledHFDataset(
        hf_train,
        feature_cols=[
            col for col in hf_train.column_names if col not in ["MedHouseVal"]
        ],
        target_cols=[["MedHouseVal"], ["AveRooms"]],
        create_scalers=True,
    )
    logger.info(
        f"Scalers created: X scaler: {train_dataset.scaler_X is not None}, y scalers: {[s is not None for s in train_dataset.scaler_y] if train_dataset.scaler_y else None}"
    )
# %%
