import time
import urllib.parse

import keyring
import pandas as pd
import requests
from torchgen import api

BASE_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-full-disc-v4-0-0-download.json?"
POINTS = ["4666571", "4775164", "5103270"]  # Kingston, Ottawa, Montreal
LOCATION_NAMES = ["kingston", "ottawa", "montreal"]


def main(api_key: str, email: str) -> None:
    input_data = {
        "attributes": "air_temperature,alpha,aod,asymmetry,clearsky_dhi,clearsky_dni,clearsky_ghi,cloud_fill_flag,cloud_type,dew_point,dhi,dni,fill_flag,ghi,ozone,relative_humidity,solar_zenith_angle,ssa,surface_albedo,surface_pressure,total_precipitable_water,wind_direction,wind_speed",
        "interval": "60",
        "api_key": api_key,
        "email": email,
    }
    for name in ["2018", "2019", "2020", "2021", "2022", "2023", "2024"]:
        print(f"Processing name: {name}")
        for id, location_ids in enumerate(POINTS):
            input_data["names"] = [name]
            input_data["location_ids"] = location_ids
            print(f"Making request for point group {id + 1} of {len(POINTS)}...")

            if ".csv" in BASE_URL:
                url = BASE_URL + urllib.parse.urlencode(data, True)
                # Note: CSV format is only supported for single point requests
                # Suggest that you might append to a larger data frame
                data = pd.read_csv(url)
                print(
                    f"Response data (you should replace this print statement with your processing): {data}"
                )
                # You can use the following code to write it to a file
                # data.to_csv('SingleBigDataPoint.csv')
            else:
                headers = {"x-api-key": api_key}
                data = get_response_json_and_handle_errors(
                    requests.post(BASE_URL, input_data, headers=headers)
                )
                download_url = data["outputs"]["downloadUrl"]
                # You can do with what you will the download url
                print(data["outputs"]["message"])
                print(
                    f"Data can be downloaded from this url when ready: {download_url}"
                )

                # Delay for 1 second to prevent rate limiting
                time.sleep(1)
            print(f"Processed")


def get_response_json_and_handle_errors(response: requests.Response) -> dict:
    """Takes the given response and handles any errors, along with providing
    the resulting json

    Parameters
    ----------
    response : requests.Response
        The response object

    Returns
    -------
    dict
        The resulting json
    """
    if response.status_code != 200:
        print(
            f"An error has occurred with the server or the request. The request response code/status: {response.status_code} {response.reason}"
        )
        print(f"The response body: {response.text}")
        exit(1)

    try:
        response_json = response.json()
    except:
        print(
            f"The response couldn't be parsed as JSON, likely an issue with the server, here is the text: {response.text}"
        )
        exit(1)

    if len(response_json["errors"]) > 0:
        errors = "\n".join(response_json["errors"])
        print(f"The request errored out, here are the errors: {errors}")
        exit(1)
    return response_json


if __name__ == "__main__":

    api_key = keyring.get_password("nlRockies", "nlRockiesAPI")
    email = keyring.get_password("nlRockies", "nlRockiesEmail")

    main(api_key, email)
