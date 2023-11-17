import os
import warnings
import numpy as np
import keras
from keras import layers
from pmdarima.datasets import load_airpassengers
from tabulate import tabulate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", module='tensorflow')

data = load_airpassengers()

time_series_data = data.data

months = np.arange(1, len(time_series_data) + 1) % 12  # Months 1-12

time_series_data_with_month = np.column_stack((time_series_data, months))

time_steps = 3


def prepare_data_with_month(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i: i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)


X, y = prepare_data_with_month(time_series_data_with_month, time_steps)
X = X.reshape(X.shape[0], X.shape[1], 2)

model = keras.Sequential([
    layers.LSTM(units=100, return_sequences=True, input_shape=(time_steps, 2)),
    layers.LSTM(units=100, return_sequences=True),
    layers.LSTM(units=100),
    layers.Dense(units=1),
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=1)
print("")

last_sequence = time_series_data_with_month[-time_steps:].reshape(1, time_steps, 2)
forecast_values, forecast_ranges = [], []

num_periods = 4

for _ in range(num_periods):
    predictions = []
    for _ in range(3):
        next_pred = model.predict(last_sequence)[0, 0]
        predictions.append(next_pred)
        next_sequence = np.append(last_sequence[:, 1:, :],
                                  [[[next_pred, (last_sequence[0, -1, 1] + 1) % 12]]],
                                  axis=1)
        last_sequence = next_sequence

    highest_pred = max(predictions)
    lowest_pred = min(predictions)
    middle_pred = sum(predictions) - highest_pred - lowest_pred

    forecast_ranges.append((highest_pred, middle_pred, lowest_pred))


periods = [f"Period {i + 1}" for i in range(num_periods)]
new_column = ["Highest", "Middle", "Lowest"]

rounded_forecast_ranges = np.around(np.array(forecast_ranges), 5).tolist()
forecast_ranges_transposed = list(map(list, zip(*rounded_forecast_ranges)))
forecast_ranges_transposed = [[new_column[i]] + row for i, row in enumerate(forecast_ranges_transposed)]
table = tabulate(forecast_ranges_transposed, headers=[''] + periods, tablefmt='pretty')
print("\n", table)
