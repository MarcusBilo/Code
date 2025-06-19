import itertools
import numpy

LEFT, ROPE, RIGHT = range(3)


def heaviside(X):
    Y = numpy.zeros(X.shape)
    Y[numpy.where(X > 0)] = 1
    Y[numpy.where(X == 0)] = 0.5
    return Y  # 1 * (x > 0)


def signrank_MC(x, rope, prior_strength, prior_place, nsamples):
    if x.ndim == 2:
        zm = x[:, 1] - x[:, 0]
    else:
        zm = x
    nm = len(zm)
    z0 = 0
    if prior_place == "ROPE":
        z0 = [0]
    elif prior_place == "LEFT":
        z0 = [-float('inf')]
    elif prior_place == "RIGHT":
        z0 = [float('inf')]
    z = numpy.concatenate((zm, z0))
    n = len(z)
    z = numpy.transpose(numpy.asmatrix(z))
    X = numpy.tile(z, (1, n))
    Y = numpy.tile(-z.T + 2 * rope, (n, 1))
    Aright = numpy.heaviside(X - Y, 0.5)
    X = numpy.tile(-z, (1, n))
    Y = numpy.tile(z.T + 2 * rope, (n, 1))
    Aleft = numpy.heaviside(X - Y, 0.5)
    alpha = numpy.concatenate((numpy.ones(nm), [prior_strength]), axis=0)
    samples = numpy.zeros((nsamples, 3), dtype=float)
    for i in range(nsamples):
        data = numpy.random.dirichlet(alpha, 1)
        Aright_result = numpy.dot(data, Aright)
        Aleft_result = numpy.dot(data, Aleft)
        samples[i, 2] = numpy.inner(Aright_result, data).item()
        samples[i, 0] = numpy.inner(Aleft_result, data).item()
        samples[i, 1] = 1 - samples[i, 0] - samples[i, 2]
    return samples


def signrank(x, rope):
    samples = signrank_MC(x, rope, prior_strength=0.5, prior_place="ROPE", nsamples=100000)
    winners = numpy.argmax(samples, axis=1)
    pl, pe, pr = numpy.bincount(winners, minlength=3) / len(winners)
    return pl, pe, pr


# FSC RF
scores = {
    "AP": [0.6000000000000001, 0.6213333400000001, 0.6106666700000001, 0.632, 0.584, 0.59466667],
    "Llncosh": [0.6106666700000001, 0.65333334, 0.6453333299999999, 0.6213333400000001, 0.624, 0.6693333300000001],
    "LMS": [0.624, 0.6000000000000001, 0.6106666700000001, 0.60266667, 0.6106666700000001, 0.6373333299999999],
    "NLMS": [0.68266666, 0.67733333, 0.65333334, 0.65333334, 0.65066666, 0.632],
    "Noisereduce": [0.632, 0.624, 0.6453333299999999, 0.63466667, 0.7040000000000001, 0.68266666],
    "RLS": [0.58133334, 0.5413333300000001, 0.57866666, 0.6213333400000001, 0.5760000000000001, 0.59466667],
    "Vanilla": [0.6693333300000001, 0.69066666, 0.66133333, 0.6693333300000001, 0.69066666, 0.66666667],
    "Wiener": [0.632, 0.68, 0.664, 0.6693333300000001, 0.6373333299999999, 0.688],
}

rope = 0.01  # Region of Practical Equivalence
results = {}

for (name1, scores1), (name2, scores2) in itertools.combinations(scores.items(), 2):
    mock_scores = numpy.column_stack((scores1, scores2))
    left, within, right = signrank(mock_scores, rope=rope)
    results[(name1, name2)] = (left, within, right)

systems = list(scores)
max_len = max(len(s) for s in systems)

# Individual Comparison Table:
# The second cell from the left represents the chance that the method on the left is better than the one on the right
# The third cell from the left represents the chance that both are practically equivalent
# The fourth cell from the left represents the chance that the method on the left is worse than the one on the right
print(max_len*" " + "     >      =      <")
for sys1 in systems:
    for sys2 in systems:
        if sys1 == sys2:
            continue
        if (sys1, sys2) in results:
            left, within, right = results[(sys1, sys2)]
        elif (sys2, sys1) in results:
            right, within, left = results[(sys2, sys1)]
        else:
            left, within, right = (None, None, None)
        sys1_padded = sys1.ljust(max_len)
        print(f"{sys1_padded}, {left:.3f}, {within:.3f}, {right:.3f}, {sys2}")

"""
# Full Comparison Matrix:
# Each cell in a row represents the chance that the method of said row is better than the method of the column
# Each cell in a column represents the chance that the method of said column is worse than the method of the row
padded_sys = [s.rjust(max_len) for s in systems]
grid = {}
for row_system in systems:
    row = {}
    for col_system in systems:
        if row_system == col_system:
            row[col_system] = "-----"
        elif (row_system, col_system) in results:
            row[col_system] = f"{results[(row_system, col_system)][0]:.3f}"  # Left value
        elif (col_system, row_system) in results:
            row[col_system] = f"{results[(col_system, row_system)][2]:.3f}"  # Right value
        else:
            row[col_system] = "N/A"  # Shouldn't happen if all pairs exist
    grid[row_system] = row
for row_system in systems:
    row_str = f"{row_system.ljust(max_len)}: " + ", ".join(
        f"{col}: {grid[row_system][col.strip()]}" for col in padded_sys
    )
    print(row_str)
"""


# ~60s Runtime


# Time for a Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis:
# https://www.jmlr.org/papers/volume18/16-305/16-305.pdf

# A Bayesian Wilcoxon signed-rank test based on the Dirichlet process:
# https://proceedings.mlr.press/v32/benavoli14.pdf
