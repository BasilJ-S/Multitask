# %%
import datetime as dt
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from os import path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

from multitask.gridstatus_api import PREDICTION_NODES
from multitask.logger import logger

PATH = "src/multitask/data/"


def path_to_file(filename: str) -> str:
    return path.join(PATH, filename + ".csv")


ACTUALS_STANDARD = [
    "net_load",
    "renewables",
    "renewables_to_load_ratio",
    "load.load",
    "fuel_mix.coal_and_lignite",
    "fuel_mix.hydro",
    "fuel_mix.nuclear",
    "fuel_mix.power_storage",
    "fuel_mix.solar",
    "fuel_mix.wind",
    "fuel_mix.natural_gas",
    "fuel_mix.other",
]


@dataclass
class GridstatusTransformation:
    function: Callable
    parameters: dict = field(default_factory=dict)


@dataclass
class GridStatusDatasetConfig:
    name: str
    transformations: list[GridstatusTransformation]


def index_by_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["interval_end_utc"] = pd.to_datetime(df["interval_end_utc"])
    df = df.set_index("interval_end_utc").sort_index()
    df = df.drop(columns=["interval_start_utc"])
    return df


def apply_transformations(
    df: pd.DataFrame, transformations: list[GridstatusTransformation]
) -> pd.DataFrame:
    for transformation in transformations:
        df = transformation.function(df, **transformation.parameters)
    return df


def apply_all_transformations(
    configs: list[GridStatusDatasetConfig],
) -> list[pd.DataFrame]:
    transformed_tables = []
    for config in configs:
        read_path = path_to_file(config.name)
        logger.info(f"Loading dataset from {read_path}")
        df = pd.read_csv(read_path)
        transformed = apply_transformations(df, config.transformations)
        write_path = path_to_file(config.name + "_transformed")
        transformed.to_csv(write_path)
        logger.info(f"Transformed dataset saved to {write_path}")
        transformed_tables.append(transformed)

    return transformed_tables


def remove_columns(df: pd.DataFrame, columns_to_remove: list[str]) -> pd.DataFrame:
    """Remove specified columns from the dataframe."""
    df = df.copy()
    df = df.drop(columns=columns_to_remove)
    return df


def merge_by_location(
    df: pd.DataFrame, location_column: str, locations: list[str] | None = None
) -> pd.DataFrame:
    """Merge rows by location, averaging numerical columns."""
    df = df.copy()
    columns = [col for col in df.columns if col != location_column]

    if locations is not None:
        df = df[df[location_column].isin(locations)]

    df = df.pivot_table(
        index=df.index,  # type: ignore
        columns=location_column,
        values=columns,
        aggfunc="mean",
    )

    df.columns = [f"{val}_{loc}" for val, loc in df.columns]
    logger.info(f"\n{df}")
    return df


def downsample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample the dataframe to hourly frequency by averaging."""
    df = df.copy()
    df_resampled = df.resample("h").mean()
    return df_resampled


# %%
def add_lags_to_actuals(
    df: pd.DataFrame, columns: list[str], lags: list[pd.Timedelta]
) -> pd.DataFrame:
    """Add lags of specified columns to the dataframe."""
    df = df.copy()

    logger.info(f"Original data interval range: {df.index.max() - df.index.min()}")

    for col in columns:
        for lag in lags:
            logger.info(f"Adding lag of {lag} to column {col}")
            df[f"{col}_lag_{lag}"] = df[col].shift(freq=lag)

    df = df.sort_index()

    logger.info(f"New data interval range: {df.index.max() - df.index.min()}")

    return df


CONFIG = [
    GridStatusDatasetConfig(
        name="ercot_standardized_hourly",
        transformations=[
            GridstatusTransformation(
                function=index_by_datetime,
            ),
            GridstatusTransformation(
                function=add_lags_to_actuals,
                parameters={
                    "columns": ACTUALS_STANDARD,
                    "lags": [pd.Timedelta(days=1)],
                },
            ),
            GridstatusTransformation(
                function=remove_columns,
                parameters={
                    "columns_to_remove": [
                        "load_forecast.load_forecast",
                        "net_load",
                        "renewables",
                        "renewables_to_load_ratio",
                        "load.load",
                        "fuel_mix.coal_and_lignite",
                        "fuel_mix.hydro",
                        "fuel_mix.nuclear",
                        "fuel_mix.power_storage",
                        "fuel_mix.solar",
                        "fuel_mix.wind",
                        "fuel_mix.natural_gas",
                        "fuel_mix.other",
                    ],
                },
            ),
        ],
    ),  # ,interval_start_utc,interval_end_utc,location,spp
    GridStatusDatasetConfig(
        name="ercot_spp_dart_15_min",
        transformations=[
            GridstatusTransformation(
                function=index_by_datetime,
            ),
            GridstatusTransformation(
                function=merge_by_location,
                parameters={
                    "location_column": "location",
                    "locations": PREDICTION_NODES,
                },
            ),
            GridstatusTransformation(
                function=downsample_to_hourly,
            ),
            GridstatusTransformation(
                function=add_lags_to_actuals,
                parameters={
                    "columns": [f"spp_{loc}" for loc in PREDICTION_NODES],
                    "lags": [pd.Timedelta(days=1)],
                },
            ),
        ],
    ),
    GridStatusDatasetConfig(
        name="ercot_load_forecast_dam",
        transformations=[
            GridstatusTransformation(
                function=index_by_datetime,
            ),
            GridstatusTransformation(
                function=remove_columns,
                parameters={
                    "columns_to_remove": [
                        "publish_time_utc",
                    ],
                },
            ),
        ],
    ),
]


def get_inference_and_prediction_intervals(start: dt.datetime, end: dt.datetime):
    """Get inference and prediction intervals for the dataset."""

    inference_times = (
        []
    )  # List of times at which point we need to make an inference about the next day (10:00 AM CT)
    prediction_times = (
        []
    )  # List of times corresponding to the prediction targets (entire next day)
    delta = timedelta(days=1)

    current = start
    while current <= end:
        # ---- Get Inference Time ----
        # 10:00 AM in CT on this date
        ct_time = datetime.combine(current, time(10, 0), tzinfo=ct)
        # Convert to UTC
        utc_time = ct_time.astimezone(ZoneInfo("UTC"))
        inference_times.append(utc_time)

        # ---- Get Prediction Times ----
        current += delta
        ct_time_next_day = datetime.combine(current, time(0), tzinfo=ct)
        utc_time_next_day = ct_time_next_day.astimezone(ZoneInfo("UTC"))
        prediction_times.append(utc_time_next_day)
        logger.info(
            f"INFERENCE: {ct_time} 10:00 AM CT = {utc_time} UTC. PREDICTION DAY STARTS AT {ct_time_next_day} CT = {utc_time_next_day} UTC"
        )

    return pd.DataFrame(
        {
            "inference_time_utc": inference_times,
            "prediction_day_start_utc": prediction_times,
        }
    )


if __name__ == "__main__":
    # %%
    dfs = apply_all_transformations(CONFIG)

    overall_df = dfs[0]
    for df in dfs[1:]:
        overall_df = overall_df.join(df, how="outer")

    overall_path = path_to_file("gridstatus_overall_transformed")
    overall_df.to_csv(overall_path)
    logger.info(f"Overall transformed dataset saved to {overall_path}")

    start, end = overall_df.index.min(), overall_df.index.max()

    # Define your CT timezone
    ct = ZoneInfo("America/Chicago")

    inf_times = get_inference_and_prediction_intervals(start, end)
    inf_path = path_to_file("gridstatus_inference_prediction_intervals")
    inf_times.to_csv(inf_path, index=False)
    logger.info(f"Inference and prediction intervals saved to {inf_path}")


# %%
