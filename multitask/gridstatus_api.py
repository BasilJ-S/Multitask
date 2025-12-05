import datetime as dt
import os

import gridstatusio as gf
import keyring
import pandas as pd
from datasets import Dataset as hfDataset

"""
Note: This module requires access to the GridStatusIO API, and as is will pull ~700000 rows of data.
"""

START_DATE = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
END_DATE = dt.datetime(2025, 1, 1, 1, tzinfo=dt.timezone.utc)

# Small set of ERCOT nodes for prediction, all in or near Austin
# we expect there is shared information among these nodes
ERCOT_NODES = [
    "LZ_AEN",
    "AUSTPL_ALL",
    "DECKER_GT",
    "GIGA_ESS_RN",
    "SANDHSYD_CC1",
]

EXCLUDED_NODES = [
    "GIGA_ESS_RN"
]  # This has less data available, so not included in first round of tests
# This will eventually be a good example of a node with sparse data for testing

PREDICTION_NODES = [node for node in ERCOT_NODES if node not in EXCLUDED_NODES]

TABLES_TO_PULL = {
    "ercot_standardized_hourly": {},
    "ercot_spp_dart_15_min": {
        "filter_column": "location",
        "filter_value": ERCOT_NODES,
        "filter_operator": "in",
    },
    "ercot_load_forecast_dam": {},
}


def load_gridstatusio_data(
    api_key: str,
    table: str,
    start_date: dt.datetime,
    end_date: dt.datetime,
    columns: list[str] | None = None,
    filter_column: str | None = None,
    filter_value: str | int | list[str] | None = None,
    filter_operator: str = "=",
) -> pd.DataFrame:
    """
    Thin wrapper on GridStatusIO client to load data from GridStatusIO.
    """

    # Initialize GridStatusIO client
    gf_client = gf.GridStatusClient(api_key=api_key)

    # Fetch data
    data = gf_client.get_dataset(
        dataset=table,
        start=start_date,  # type: ignore
        end=end_date,  # type: ignore
        columns=columns,
        filter_column=filter_column,
        filter_value=filter_value,
        filter_operator=filter_operator,
        sleep_time=1,
    )

    return data


if __name__ == "__main__":
    gs_key = keyring.get_password("gridstatusio", "gridstatus")
    if gs_key is None:
        gs_key = os.getenv("GRIDSTATUSIO_API_KEY")
    if gs_key is None:
        raise ValueError(
            "GridStatusIO API key not found in keyring or environment variables."
        )
    print(f"GridStatusIO API key retrieved successfully: {gs_key is not None}")
    cont = input(
        "Proceed to load data from GridStatusIO? This may take a while and use significant bandwidth. (y/n): "
    )
    if cont.lower() != "y":
        print("Aborting data load.")
        exit(0)
    length = 0
    for table_name, table_params in TABLES_TO_PULL.items():
        print(f"Loading data from table: {table_name}")
        df = load_gridstatusio_data(
            api_key=gs_key,
            table=table_name,
            start_date=START_DATE,
            end_date=END_DATE,
            **table_params,
        )
        length += len(df)
        print(f"Data loaded with shape: {df.shape}")
        print(df.head())
        df.to_csv(
            f"{table_name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            index=False,
        )
        df.to_csv(
            f"{table_name}.csv",
            index=False,
        )
        print(f"Data saved to {table_name}.csv")

    print(f"Total records loaded: {length}")
