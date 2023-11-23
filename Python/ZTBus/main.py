import pandas as pd
from datetime import datetime

pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", 100)


def parse_name(name):
    parts = name.split("_")
    timestamp_start = datetime.strptime(parts[1] + '_' + parts[2], "%Y-%m-%d_%H-%M-%S")
    timestamp_end = datetime.strptime(parts[3] + '_' + parts[4], "%Y-%m-%d_%H-%M-%S")
    return pd.Series({
        "startTime": timestamp_start,
        "endTime": timestamp_end
    })


data = pd.read_csv("metaData.csv")
parsed_data = data["name"].apply(parse_name)

columns_to_drop = [
    "name",
    "startTime_iso",
    "startTime_unix",
    "endTime_iso",
    "endTime_unix",
    "busRoute"
]

new_data = pd.concat([data.drop(columns=columns_to_drop), parsed_data], axis=1)
new_data["startDate"] = new_data["startTime"].dt.date
new_data["endDate"] = new_data["endTime"].dt.date

grouped = new_data.groupby([
    "busNumber",
    "startDate"
])

aggregation_rules = {
    'startTime': 'min',
    'endTime': 'max',
    'drivenDistance': 'mean',
    'energyConsumption': 'mean',
    'itcs_numberOfPassengers_mean': 'mean',
    'itcs_numberOfPassengers_min': 'min',
    'itcs_numberOfPassengers_max': 'max',
    'temperature_ambient_mean': 'mean',
    'temperature_ambient_min': 'min',
    'temperature_ambient_max': 'max',
}

columns_to_drop = [
    "startTime",
    "endTime",
]

columns_to_rename = {
    "startDate": "date",
    "itcs_numberOfPassengers_mean": "numberOfPassengers_mean",
    "itcs_numberOfPassengers_min": "numberOfPassengers_min",
    "itcs_numberOfPassengers_max": "numberOfPassengers_max",
}

merged_data = grouped.agg(aggregation_rules).reset_index()
merged_data = merged_data.drop(columns=columns_to_drop)
merged_data = merged_data.rename(columns=columns_to_rename)

print(merged_data)
