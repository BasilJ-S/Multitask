# %%
import datetime as dt
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

from multitask.data.gridstatus_api import PREDICTION_NODES
from multitask.data.nrl_api import LOCATION_NAMES
from multitask.utils.logger import logger
from multitask.utils.utils import path_to_file

PATH = "multitask/data_store/"
ERCOT_TRAIN_END_DATE = dt.datetime(2023, 1, 1, tzinfo=ZoneInfo("UTC"))
ERCOT_VAL_END_DATE = dt.datetime(2024, 1, 1, tzinfo=ZoneInfo("UTC"))

WEATHER_TRAIN_END_DATE = dt.datetime(2014, 1, 1, tzinfo=ZoneInfo("UTC"))
WEATHER_VAL_END_DATE = dt.datetime(2015, 1, 1, tzinfo=ZoneInfo("UTC"))

WEATHER_MULTILOC_TRAIN_END_DATE = dt.datetime(
    2021, 1, 1, tzinfo=ZoneInfo("UTC")
)  # 2018-2020 test
WEATHER_MULTILOC_VAL_END_DATE = dt.datetime(
    2022, 1, 1, tzinfo=ZoneInfo("UTC")
)  # 2021 val, 2022-2023 test


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


def index_by_datetime(
    df: pd.DataFrame,
    datetime_column: str = "interval_end_utc",
    format: str | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df[datetime_column] = pd.to_datetime(df[datetime_column], format=format, utc=True)
    df = df.set_index(datetime_column).sort_index()
    df = df.drop(columns=["interval_start_utc"], errors="ignore")
    return df


def reindex_by_datetime(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = df.copy()
    duplicates = df.index.duplicated(keep=False)
    index_name = df.index.name
    if duplicates.any():
        logger.warning(
            f"Found {duplicates.sum()} duplicate datetime indices. Averaging duplicates."
        )
        logger.warning(f"\n{df[duplicates]}")
    min = df.index.min()
    max = df.index.max()
    full_index = pd.date_range(start=min, end=max, freq=freq, tz="UTC")
    df = df.reindex(full_index)
    df.index.name = index_name
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


def remove_negative_values(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Remove negative values from specified columns in the dataframe."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            num_negatives = (df[col] < 0).sum()
            if num_negatives > 0:
                logger.info(
                    f"Removing {num_negatives} negative values from column {col}"
                )
                df.loc[df[col] < 0, col] = pd.NA
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

    df.dropna(inplace=True)
    df = df.sort_index()

    logger.info(f"New data interval range: {df.index.max() - df.index.min()}")

    return df


def ffill(df: pd.DataFrame) -> pd.DataFrame:
    """Forward fill missing values in the dataframe."""
    df = df.copy()
    logger.info(f"Missing values before ffill:\n{df.isna().sum()}")
    df = df.ffill()
    logger.info(f"Missing values after ffill:\n{df.isna().sum()}")
    return df


def build_date_from_components(
    df: pd.DataFrame,
    year_col: str = "Year",
    month_col: str = "Month",
    day_col: str = "Day",
    hour_col: str = "Hour",
    minute_col: str = "Minute",
) -> pd.DataFrame:
    df = df.copy()
    df["constructed_datetime"] = pd.to_datetime(
        dict(  # type: ignore
            year=df[year_col],
            month=df[month_col],
            day=df[day_col],
            hour=df[hour_col],
            minute=df[minute_col],
        ),
        utc=True,
    )
    df = df.drop(
        columns=[year_col, month_col, day_col, hour_col, minute_col], errors="ignore"
    )

    df = df.set_index("constructed_datetime").sort_index()
    return df


CONFIG_ERCOT = [
    GridStatusDatasetConfig(
        name="ercot_standardized_hourly",
        transformations=[
            GridstatusTransformation(
                function=index_by_datetime,
            ),
            GridstatusTransformation(
                function=reindex_by_datetime,
                parameters={
                    "freq": "h",
                },
            ),
            GridstatusTransformation(
                function=ffill,
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
                function=reindex_by_datetime,
                parameters={
                    "freq": "15min",
                },
            ),
            GridstatusTransformation(
                function=ffill,
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
                function=reindex_by_datetime,
                parameters={
                    "freq": "h",
                },
            ),
            GridstatusTransformation(
                function=ffill,
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

CONFIG_WEATHER = [
    GridStatusDatasetConfig(
        name="max_planck_weather_ts",
        transformations=[
            GridstatusTransformation(
                function=index_by_datetime,
                parameters={
                    "datetime_column": "Date Time",
                    "format": "%d.%m.%Y %H:%M:%S",
                },
            ),
            GridstatusTransformation(
                function=downsample_to_hourly,
            ),
            GridstatusTransformation(
                function=remove_negative_values,
                parameters={  # Remove negative values in wind speed columns (sentinel values
                    "columns": ["wv (m/s)", "max. wv (m/s)"],
                },
            ),
            GridstatusTransformation(
                function=ffill,
            ),
        ],
    ),
]

CONFIG_WEATHER_MULTILOC = [
    GridStatusDatasetConfig(
        name=loc,
        transformations=[
            GridstatusTransformation(
                function=build_date_from_components,
            ),
            GridstatusTransformation(
                function=reindex_by_datetime,
                parameters={
                    "freq": "h",
                },
            ),
            GridstatusTransformation(
                function=ffill,
            ),
        ],
    )
    for loc in LOCATION_NAMES
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
        ct_time = datetime.combine(
            current, time(10, 0), tzinfo=ZoneInfo("America/Chicago")
        )
        # Convert to UTC
        utc_time = ct_time.astimezone(ZoneInfo("UTC"))
        inference_times.append(utc_time)

        # ---- Get Prediction Times ----
        current += delta
        ct_time_next_day = datetime.combine(
            current, time(0), tzinfo=ZoneInfo("America/Chicago")
        )
        utc_time_next_day = ct_time_next_day.astimezone(ZoneInfo("UTC"))
        prediction_times.append(utc_time_next_day)
        logger.debug(
            f"INFERENCE: {ct_time} 10:00 AM CT = {utc_time} UTC. PREDICTION DAY STARTS AT {ct_time_next_day} CT = {utc_time_next_day} UTC"
        )

    return pd.DataFrame(
        {
            "inference_time_utc": inference_times,
            "prediction_day_start_utc": prediction_times,
        }
    )


def write_splits_with_inference_times(
    full_dataset: pd.DataFrame,
    train_end_date: dt.datetime,
    val_end_date: dt.datetime,
    file_name_prefix: str,
) -> None:
    """Write train/val/test splits along with inference times to CSV files."""

    start, end = full_dataset.index.min(), full_dataset.index.max()
    logger.info(f"Dataset covers from {start} to {end}")

    train_set = full_dataset[full_dataset.index < train_end_date]
    val_set = full_dataset[
        (full_dataset.index >= train_end_date) & (full_dataset.index < val_end_date)
    ]
    test_set = full_dataset[full_dataset.index >= val_end_date]

    for subset, name in [
        (train_set, "train"),
        (val_set, "validation"),
        (test_set, "test"),
    ]:
        start, end = subset.index.min(), subset.index.max()

        inf_times = get_inference_and_prediction_intervals(start, end)

        subset_path = path_to_file(f"{file_name_prefix}_{name}_set")
        subset.to_csv(subset_path)
        logger.info(f"{name.capitalize()} set saved to {subset_path}")
        inf_times_path = path_to_file(f"{file_name_prefix}_{name}_inference_times")
        inf_times.to_csv(inf_times_path, index=False)
        logger.info(f"{name.capitalize()} inference times saved to {inf_times_path}")


if __name__ == "__main__":

    dfs = apply_all_transformations(CONFIG_WEATHER_MULTILOC)
    # Add prefix to columns
    for i, loc in enumerate(["kingston", "ottawa", "montreal"]):
        dfs[i].columns = [f"{col}_{loc}" for col in dfs[i].columns]

    overall_df = dfs[0]
    for df in dfs[1:]:
        overall_df = overall_df.join(df, how="inner")

    overall_path = path_to_file("weather_multiloc_transformed")
    overall_df.to_csv(overall_path)

    write_splits_with_inference_times(
        overall_df,
        train_end_date=WEATHER_MULTILOC_TRAIN_END_DATE,
        val_end_date=WEATHER_MULTILOC_VAL_END_DATE,
        file_name_prefix="weather_multiloc",
    )

    dfs = apply_all_transformations(CONFIG_ERCOT)

    overall_df = dfs[0]
    for df in dfs[1:]:
        overall_df = overall_df.join(df, how="inner")

    overall_path = path_to_file("gridstatus_overall_transformed")
    overall_df.to_csv(overall_path)
    logger.info(f"Overall transformed dataset saved to {overall_path}")

    start, end = overall_df.index.min(), overall_df.index.max()

    logger.info(f"Dataset covers from {start} to {end}")

    write_splits_with_inference_times(
        overall_df,
        train_end_date=ERCOT_TRAIN_END_DATE,
        val_end_date=ERCOT_VAL_END_DATE,
        file_name_prefix="gridstatus",
    )

    # Prepare inference times for etth datasets

    dfs_weather = apply_all_transformations(CONFIG_WEATHER)
    weather_df = dfs_weather[0]
    weather_oath = path_to_file("weather_transformed")
    weather_df.to_csv(weather_oath)

    write_splits_with_inference_times(
        weather_df,
        train_end_date=WEATHER_TRAIN_END_DATE,
        val_end_date=WEATHER_VAL_END_DATE,
        file_name_prefix="weather",
    )
