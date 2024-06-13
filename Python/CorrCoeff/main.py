import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, kendalltau, rankdata, norm


# https://github.com/jlbloesch/miscellaneous/blob/main/xicor.py
def xicor(x, y, ties=False):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    n = len(x)

    if len(y) != n:
        raise IndexError(f'X & Y variables array size mismatch: {len(x)}, {len(y)}')

    y = y[np.argsort(x)]
    r = rankdata(y, method='ordinal')
    nominator = np.sum(np.abs(np.diff(r)))

    if ties:

        l = rankdata(y, method='max')
        denominator = 2 * np.sum(l * (n - 1))
        nominator *= n

    else:

        denominator = np.power(n, 2) - 1
        nominator *= 3

    xi = 1 - nominator / denominator
    p_value = norm.sf(xi, scale=2 / 5 / np.sqrt(n))

    return xi, p_value


# Generate x values
x = np.linspace(-10, 10, 250)

# Define distributions
y_cubic = x ** 3
y_sinusoidal = np.sin(x)
y_parabolic = x ** 2


# Scale distributions to range [-1, 1]
def scale_to_range(y):
    return 2 * (y - np.min(y)) / (np.max(y) - np.min(y)) - 1


y_cubic_scaled = scale_to_range(y_cubic)
y_sinusoidal_scaled = scale_to_range(y_sinusoidal)
y_parabolic_scaled = scale_to_range(y_parabolic)

# Generate noise
noise_10 = np.random.normal(0, 0.10, x.shape)
noise_25 = np.random.normal(0, 0.25, x.shape)
noise_50 = np.random.normal(0, 0.50, x.shape)

# Define distributions with the same noise
y_cubic_noisy_10 = y_cubic_scaled + noise_10
y_sinusoidal_noisy_10 = y_sinusoidal_scaled + noise_10
y_parabolic_noisy_10 = y_parabolic_scaled + noise_10

# Define distributions with the same noise
y_cubic_noisy_25 = y_cubic_scaled + noise_25
y_sinusoidal_noisy_25 = y_sinusoidal_scaled + noise_25
y_parabolic_noisy_25 = y_parabolic_scaled + noise_25

# Define distributions with the same noise
y_cubic_noisy_50 = y_cubic_scaled + noise_50
y_sinusoidal_noisy_50 = y_sinusoidal_scaled + noise_50
y_parabolic_noisy_50 = y_parabolic_scaled + noise_50


# Calculate correlation coefficients
def calculate_correlations(x, y):
    pearson_corr, _ = pearsonr(x, y)
    kendall_corr, _ = kendalltau(x, y)
    chatterjee_corr, _ = xicor(x, y)
    return pearson_corr, kendall_corr, chatterjee_corr


distributions = {
    'Cubic': y_cubic_scaled,
    'Sinusoidal': y_sinusoidal_scaled,
    'Parabolic': y_parabolic_scaled
}

noisy_distributions_10 = {
    'Cubic Noisy': y_cubic_noisy_10,
    'Sinusoidal Noisy': y_sinusoidal_noisy_10,
    'Parabolic Noisy': y_parabolic_noisy_10
}

noisy_distributions_25 = {
    'Cubic Noisy': y_cubic_noisy_25,
    'Sinusoidal Noisy': y_sinusoidal_noisy_25,
    'Parabolic Noisy': y_parabolic_noisy_25
}

noisy_distributions_50 = {
    'Cubic Noisy': y_cubic_noisy_50,
    'Sinusoidal Noisy': y_sinusoidal_noisy_50,
    'Parabolic Noisy': y_parabolic_noisy_50
}

correlations = {name: calculate_correlations(x, y) for name, y in distributions.items()}
noisy_correlations_10 = {name: calculate_correlations(x, y) for name, y in noisy_distributions_10.items()}
noisy_correlations_25 = {name: calculate_correlations(x, y) for name, y in noisy_distributions_25.items()}
noisy_correlations_50 = {name: calculate_correlations(x, y) for name, y in noisy_distributions_50.items()}

# Plot distributions and correlation coefficients
fig, axs = plt.subplots(4, 3, figsize=(15, 10))

for ax, (name, y) in zip(axs[0], distributions.items()):
    ax.plot(x, y)
    pearson_corr, kendall_corr, chatterjee_corr = correlations[name]
    ax.set_title(f'{name}\nPearson: {pearson_corr:.2f}, Kendall: {kendall_corr:.2f}, Chatterjee: {chatterjee_corr:.2f}')

for ax, (name, y) in zip(axs[1], noisy_distributions_10.items()):
    ax.plot(x, y)
    pearson_corr, kendall_corr, chatterjee_corr = noisy_correlations_10[name]
    ax.set_title(f'{name}\nPearson: {pearson_corr:.2f}, Kendall: {kendall_corr:.2f}, Chatterjee: {chatterjee_corr:.2f}')

for ax, (name, y) in zip(axs[2], noisy_distributions_25.items()):
    ax.plot(x, y)
    pearson_corr, kendall_corr, chatterjee_corr = noisy_correlations_25[name]
    ax.set_title(f'{name}\nPearson: {pearson_corr:.2f}, Kendall: {kendall_corr:.2f}, Chatterjee: {chatterjee_corr:.2f}')

for ax, (name, y) in zip(axs[3], noisy_distributions_50.items()):
    ax.plot(x, y)
    pearson_corr, kendall_corr, chatterjee_corr = noisy_correlations_50[name]
    ax.set_title(f'{name}\nPearson: {pearson_corr:.2f}, Kendall: {kendall_corr:.2f}, Chatterjee: {chatterjee_corr:.2f}')

plt.tight_layout()
plt.show()
