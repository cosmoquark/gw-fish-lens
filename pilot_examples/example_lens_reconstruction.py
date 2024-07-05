import corner
import numpy as np
import jax
import jax.numpy as jnp
import pylab as plt
from scipy.stats import multivariate_normal
from herculens.LensImage.lens_image import LensImage
from herculens.Instrument.noise import Noise
from herculens.Instrument.psf import PSF
from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.MassModel.mass_model import MassModel
from herculens.LightModel.light_model import LightModel


# DEFINE TELESCOPE
npix = 80 # Number of pixels on a side
pixel_size = 0.08 # Pixel size in arcseconds
half_size = npix*pixel_size/2.
ra_at_xy_0 = dec_at_xy_0 = -half_size +  pixel_size / 2.
transform_pix2angle = pixel_size * np.eye(2)
kwargs_pixel = {'nx': npix, 
                'ny': npix, 
                'ra_at_xy_0': ra_at_xy_0, 
                'dec_at_xy_0': dec_at_xy_0, 
                'transform_pix2angle': transform_pix2angle}
pixel_grid = PixelGrid(**kwargs_pixel)
psf = PSF(psf_type='GAUSSIAN', fwhm=0.3, pixel_size=pixel_size)
noise = Noise(npix, npix, background_rms=1e-2, exposure_time=1000)
kwargs_numerics_fit = {'supersampling_factor': 3}
# DEFINE MODEL FOR IMAGE
lens_mass_model_input = MassModel(['SIE'])
source_model_input = LightModel(['SERSIC_ELLIPSE'])
lens_light_model_input = LightModel(['SERSIC_ELLIPSE'])
lens_image = LensImage(
    pixel_grid,
    psf,
    noise_class = noise,
    lens_mass_model_class = lens_mass_model_input,
    source_model_class = source_model_input, 
    lens_light_model_class = lens_light_model_input,
    kwargs_numerics=kwargs_numerics_fit
)

def log_likelihood(theta_l, theta_l_true):
    kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps = convert_theta_to_model_params(theta_l)
    kwargs_lens_true, kwargs_source_true, kwargs_lens_light_true, kwargs_ps_true = convert_theta_to_model_params(theta_l_true)

    model_params = dict(kwargs_lens = kwargs_lens, 
                        kwargs_source = kwargs_source, 
                        kwargs_lens_light = kwargs_lens_light, 
                        kwargs_ps = kwargs_ps)
    true_params = dict(kwargs_lens = kwargs_lens_true,
                          kwargs_source = kwargs_source_true,
                          kwargs_lens_light = kwargs_lens_light_true,
                          kwargs_ps = kwargs_ps_true)
    d_true = 
    d_model = lens_image.model(**model_params)


def log_posterior(theta_l, theta_l_true):
    return log_likelihood( theta_l, theta_l_true ) + log_prior( theta_l )

def calculate_fim_matrix_from_image_data(theta_l_true):
    # Compute the Hessian matrix
    log_posterior_fixed = lambda theta_l: log_posterior(theta_l, theta_l_true)
    hessian = jax.hessian(log_posterior)(log_posterior_fixed)
    # Compute the Fisher Information Matrix
    fim_matrix = jnp.linalg.inv(hessian)
    return fim_matrix
    

# FIM STEP
fim_matrix = calculate_fim_matrix_from_image_data(true_values)

# PLOT STEP
# # Sample from the multivariate normal distribution
samples = multivariate_normal.rvs(mean=true_values, cov=fim_matrix, size=10000)
# Plot the corner plot
fig = corner.corner(samples, labels=labels, truths=true_values)