import matplotlib.pyplot as plt
import numpy as np
from sktime.datasets import load_airline
from sktime.forecasting.theta import ThetaForecaster
from sktime.utils import plotting


# https://www.sktime.net/en/latest/examples/01_forecasting.html

def main():
    y = load_airline()
    fh = np.arange(1, 13)
    forecaster = ThetaForecaster(sp=12)
    forecaster.fit(y, fh=fh)
    coverage = 0.9
    y_pred_ints = forecaster.predict_interval(coverage=coverage)
    y_pred = forecaster.predict()
    _, _ = plotting.plot_series(
        y, y_pred, labels=["y", "y_pred"], pred_interval=y_pred_ints
    )
    plt.show()


if __name__ == "__main__":
    main()
