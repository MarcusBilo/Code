import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from scipy.stats import norm

data = np.random.normal(loc=0.0, scale=1.0, size=1000)

kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
kde.fit(data[:, None])

x = np.linspace(min(data), max(data), 1000)
log_density = kde.score_samples(x[:, None])
density = np.exp(log_density)
mean_data = np.mean(data)
std_data = np.std(data)
sem = std_data / np.sqrt(len(data))
conf_interval = 1.96 * sem

print("95% Confidence Interval for the Mean:", mean_data - conf_interval, "to", mean_data + conf_interval)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=50, density=True, alpha=0.5, label='Data Histogram')
plt.plot(x, density, color='red', label='KDE Estimate')
plt.plot(x, norm.pdf(x, 0, 1), color='blue', label='Standard Normal Curve')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Kernel Density Estimation')
plt.axvline(mean_data, color='black', linestyle='--', label='Mean')
plt.axvline(mean_data - conf_interval, color='green', linestyle='--', label='95% Confidence Interval around Mean')
plt.axvline(mean_data + conf_interval, color='green', linestyle='--')
plt.legend()
plt.show()
