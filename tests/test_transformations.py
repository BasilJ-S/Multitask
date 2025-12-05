import datetime as dt
import logging

import pandas as pd

from multitask.preprocess_gridstatus_dataset import (
    add_lags_to_actuals,
    downsample_to_hourly,
    merge_by_location,
)

logger = logging.getLogger(__name__)


class TestMergeByLocation:
    def test_one_value(self):
        data = {
            "interval_end_utc": [
                dt.datetime(2023, 1, 1, 0),
                dt.datetime(2023, 1, 1, 0),
                dt.datetime(2023, 1, 1, 1),
                dt.datetime(2023, 1, 1, 2),
            ],
            "location": ["A", "B", "B", "C"],
            "value": [1, 2, 3, 4],
        }
        df = pd.DataFrame(data)
        df.set_index("interval_end_utc", inplace=True)
        merged_df = merge_by_location(df, "location")

        for location in df["location"].unique():
            assert "value_" + location in merged_df.columns

        expected_data = {
            "interval_end_utc": [
                dt.datetime(2023, 1, 1, 0),
                dt.datetime(2023, 1, 1, 1),
                dt.datetime(2023, 1, 1, 2),
            ],
            "value_A": [1, None, None],
            "value_B": [2, 3, None],
            "value_C": [None, None, 4],
        }
        pd.testing.assert_frame_equal(
            merged_df,
            pd.DataFrame(expected_data).set_index("interval_end_utc"),
        )

    def test_multiple_values(self):
        data = {
            "interval_end_utc": [
                dt.datetime(2023, 1, 1, 0),
                dt.datetime(2023, 1, 1, 0),
                dt.datetime(2023, 1, 1, 1),
                dt.datetime(2023, 1, 1, 2),
            ],
            "location": ["A", "B", "B", "C"],
            "value1": [1, 2, 3, 4],
            "value2": [10, 20, 30, 40],
        }
        df = pd.DataFrame(data)
        df.set_index("interval_end_utc", inplace=True)
        merged_df = merge_by_location(df, "location")

        for location in df["location"].unique():
            assert "value1_" + location in merged_df.columns
            assert "value2_" + location in merged_df.columns

        expected_data = pd.DataFrame(
            {
                "interval_end_utc": [
                    dt.datetime(2023, 1, 1, 0),
                    dt.datetime(2023, 1, 1, 1),
                    dt.datetime(2023, 1, 1, 2),
                ],
                "value1_A": [1, None, None],
                "value1_B": [2, 3, None],
                "value1_C": [None, None, 4],
                "value2_A": [10, None, None],
                "value2_B": [20, 30, None],
                "value2_C": [None, None, 40],
            }
        ).set_index("interval_end_utc")
        pd.testing.assert_frame_equal(
            merged_df,
            expected_data,
        )

    def test_filters_locations(self):
        data = {
            "interval_end_utc": [
                dt.datetime(2023, 1, 1, 0),
                dt.datetime(2023, 1, 1, 0),
                dt.datetime(2023, 1, 1, 1),
                dt.datetime(2023, 1, 1, 2),
            ],
            "location": ["A", "B", "B", "C"],
            "value": [1, 2, 3, 4],
        }
        df = pd.DataFrame(data)
        df.set_index("interval_end_utc", inplace=True)
        merged_df = merge_by_location(
            df, location_column="location", locations=["A", "C"]
        )

        for location in ["A", "C"]:
            assert "value_" + location in merged_df.columns
        assert "value_B" not in merged_df.columns

        expected_data = {
            "interval_end_utc": [
                dt.datetime(2023, 1, 1, 0),
                dt.datetime(2023, 1, 1, 2),
            ],
            "value_A": [1, None],
            "value_C": [None, 4],
        }
        pd.testing.assert_frame_equal(
            merged_df,
            pd.DataFrame(expected_data).set_index("interval_end_utc"),
        )


class TestAddLagsToActuals:

    def test_add_lags_to_actuals(self):
        data = {
            "interval_end_utc": [
                dt.datetime(2023, 1, 1, 0),
                dt.datetime(2023, 1, 1, 1),
                dt.datetime(2023, 1, 1, 2),
                dt.datetime(2023, 1, 1, 3),
            ],
            "actuals": [10, 20, 30, 40],
        }
        df = pd.DataFrame(data).set_index("interval_end_utc")
        lag_hours = [pd.Timedelta(hours=1), pd.Timedelta(hours=2)]

        df_with_lags = add_lags_to_actuals(df, columns=["actuals"], lags=lag_hours)
        logger.info(f"\n{df_with_lags}")

        expected_actuals = [30, 40]
        expected_lag_1h = [20, 30]
        expected_lag_2h = [10, 20]

        pd.testing.assert_series_equal(
            df_with_lags["actuals"],
            pd.Series(expected_actuals, index=df.index[-2:]),
            check_names=False,
            check_dtype=False,
        )
        pd.testing.assert_series_equal(
            df_with_lags["actuals_lag_0 days 01:00:00"],
            pd.Series(expected_lag_1h, index=df.index[-2:]),
            check_names=False,
            check_dtype=False,
        )
        pd.testing.assert_series_equal(
            df_with_lags["actuals_lag_0 days 02:00:00"],
            pd.Series(expected_lag_2h, index=df.index[-2:]),
            check_names=False,
            check_dtype=False,
        )

    def test_off_hours(self):
        data = {
            "interval_end_utc": [
                dt.datetime(2023, 1, 1, 0),
                dt.datetime(2023, 1, 1, 0, 15),
                dt.datetime(2023, 1, 1, 0, 30),
                dt.datetime(2023, 1, 1, 0, 45),
            ],
            "actuals": [10, 20, 30, 40],
        }
        df = pd.DataFrame(data).set_index("interval_end_utc")
        lag_hours = [pd.Timedelta(minutes=30), pd.Timedelta(minutes=45)]

        df_with_lags = add_lags_to_actuals(df, columns=["actuals"], lags=lag_hours)
        logger.info(f"\n{df_with_lags}")

        expected_actuals = [40]
        expected_lag_30m = [20]
        expected_lag_45m = [10]

        assert len(df_with_lags) == 1
        for col, expected in zip(
            [
                "actuals",
                "actuals_lag_0 days 00:30:00",
                "actuals_lag_0 days 00:45:00",
            ],
            [expected_actuals, expected_lag_30m, expected_lag_45m],
        ):
            assert df_with_lags[col].tolist() == expected

    def test_nonoverlap(self):
        data = {
            "interval_end_utc": [
                dt.datetime(2023, 1, 1, 0),
                dt.datetime(2023, 1, 1, 2),
                dt.datetime(2023, 1, 1, 4),
                dt.datetime(2023, 1, 1, 6),
            ],
            "actuals": [10, 20, 30, 40],
        }
        df = pd.DataFrame(data).set_index("interval_end_utc")
        lag_hours = [pd.Timedelta(hours=2), pd.Timedelta(hours=16)]
        df_with_lags = add_lags_to_actuals(df, columns=["actuals"], lags=lag_hours)

        assert len(df_with_lags) == 0


class TestResample:

    def test_15_min(self):
        data = {
            "interval_end_utc": [
                dt.datetime(2023, 1, 1, 0),
                dt.datetime(2023, 1, 1, 0, 15),
                dt.datetime(2023, 1, 1, 0, 30),
                dt.datetime(2023, 1, 1, 0, 45),
                dt.datetime(2023, 1, 1, 1),
            ],
            "actuals": [10, 20, 30, 40, 50],
        }
        df = pd.DataFrame(data).set_index("interval_end_utc")

        df_resampled = downsample_to_hourly(df)
        logger.info(f"\n{df_resampled}")

        expected_data = {
            "interval_end_utc": [
                dt.datetime(2023, 1, 1, 0),
                dt.datetime(2023, 1, 1, 1),
            ],
            "actuals": [25.0, 50.0],
        }
        expected_df = pd.DataFrame(expected_data).set_index("interval_end_utc")
        logger.info(f"\n{expected_df}\n{df_resampled}")

        pd.testing.assert_frame_equal(df_resampled, expected_df, check_freq=False)
