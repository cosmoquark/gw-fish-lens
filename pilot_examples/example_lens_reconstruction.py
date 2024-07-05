import corner
import numpy as np
import pylab as plt
from scipy.stats import multivariate_normal


# PLOT STEP
# # Sample from the multivariate normal distribution
samples = multivariate_normal.rvs(mean=true_values, cov=fim_matrix, size=10000)
# Plot the corner plot
fig = corner.corner(samples, labels=labels, truths=true_values)