# Read all csvs in folder, and concatenate them into a single dataframe
# NRL API provides CSV per year. This merges into single file per location.
import os

import pandas as pd

from multitask.data.nrl_api import LOCATION_NAMES

for city in LOCATION_NAMES:
    dataframes = []
    for filename in os.listdir(f"multitask/data_store/{city}"):
        print(f"Processing file: {filename}")
        df = pd.read_csv(
            os.path.join(f"multitask/data_store/{city}", filename), skiprows=2
        )
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv(f"multitask/data_store/{city}.csv", index=False)
