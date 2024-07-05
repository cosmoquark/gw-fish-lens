import corner
import numpy as np
import jax
import jax.numpy as np
import pylab as plt
from scipy.stats import multivariate_normal

def calculate_fim_matrix_from_image_data(theta_l_true):
    # Compute the Hessian matrix
    hessian = jax.hessian(chi_sq)(theta_l_true)
    # Compute the Fisher Information Matrix
    fim_matrix = jnp.linalg.inv(hessian)
    

# FIM STEP
fim_matrix = calculate_fim_matrix_from_image_data(true_values)

# PLOT STEP
# # Sample from the multivariate normal distribution
samples = multivariate_normal.rvs(mean=true_values, cov=fim_matrix, size=10000)
# Plot the corner plot
fig = corner.corner(samples, labels=labels, truths=true_values)