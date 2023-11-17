import os
import warnings
import numpy as np
import keras
from keras import layers
from pmdarima.datasets import load_airpassengers
from tabulate import tabulate


def load_and_process_data():
    data = load_airpassengers().data
    months = np.arange(1, len(data) + 1) % 12  # Months: 1-12
    data_with_month = np.column_stack((data, months))
    return data_with_month


def prepare_data_with_month(data, time_steps):
    features, labels = [], []
    for i in range(len(data) - time_steps):
        features.append(data[i: i + time_steps])
        labels.append(data[i + time_steps])
    return np.array(features), np.array(labels)


def build_model(time_steps):
    model = keras.Sequential([
        layers.LSTM(units=100, return_sequences=True, input_shape=(time_steps, 2)),
        layers.LSTM(units=100, return_sequences=True),
        layers.LSTM(units=100),
        layers.Dense(units=1),
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(model, features, labels):
    model.fit(features, labels, epochs=10, batch_size=1)
    return model


def forecast(data_with_month, time_steps, model, num_periods):
    last_sequence = data_with_month[-time_steps:].reshape(1, time_steps, 2)
    forecast_ranges = []

    for _ in range(num_periods):
        predictions = []
        for _ in range(3):
            next_month = (last_sequence[0, -1, 1] + 1) % 12
            next_pred = model.predict(last_sequence)[0, 0]
            predictions.append(next_pred)
            next_sequence = np.concatenate(
                (last_sequence[:, 1:, :], [[[next_pred, next_month]]]), axis=1
            )
            last_sequence = next_sequence

        highest_pred = max(predictions)
        lowest_pred = min(predictions)
        middle_pred = sum(predictions) - highest_pred - lowest_pred

        forecast_ranges.append((highest_pred, middle_pred, lowest_pred))

    return forecast_ranges


def display_forecast_table(forecast_ranges, num_periods):
    periods = [f"Period {i + 1}" for i in range(num_periods)]
    new_column = ["Max", "Mid", "Min"]

    rounded_forecasts = np.around(np.array(forecast_ranges), 5).tolist()
    forecasts_transposed = list(map(list, zip(*rounded_forecasts)))
    forecasts_transposed = [
        [new_column[i]] + row for i, row in enumerate(forecasts_transposed)
    ]
    table = tabulate(forecasts_transposed, headers=[''] + periods, tablefmt='pretty')
    print("\n", table)


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings("ignore", module='tensorflow')
    data_with_month = load_and_process_data()
    time_steps = 3
    features, labels = prepare_data_with_month(data_with_month, time_steps)
    features = features.reshape(features.shape[0], features.shape[1], 2)
    model = build_model(time_steps)
    trained_model = train_model(model, features, labels)
    num_periods = 4
    forecast_ranges = forecast(data_with_month, time_steps, trained_model, num_periods)
    display_forecast_table(forecast_ranges, num_periods)


if __name__ == "__main__":
    main()
