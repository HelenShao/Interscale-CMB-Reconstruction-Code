from __future__ import annotations
import numpy as np

def apply_ilc_to_single_patch(freq_patches, cmb_patch, get_residuals=False):
    (n_freq, height, width) = freq_patches.shape
    observed_patches = freq_patches + cmb_patch[np.newaxis, :, :]
    observed_flat = observed_patches.reshape(n_freq, -1)
    covariance = np.cov(observed_flat)
    regularization = 1e-10 * np.trace(covariance) / n_freq
    covariance += regularization * np.eye(n_freq)
    a_matrix = np.ones(n_freq)
    try:
        inv_cov = np.linalg.inv(covariance)
        denominator = np.dot(a_matrix.T, np.dot(inv_cov, a_matrix))
        weights = np.dot(a_matrix.T, inv_cov) / denominator
    except np.linalg.LinAlgError:
        weights = np.ones(n_freq) / n_freq
    ilc_cmb = np.zeros((height, width))
    for i in range(n_freq):
        ilc_cmb += weights[i] * observed_patches[i]
    ilc_foregrounds = np.zeros((n_freq, height, width))
    for i in range(n_freq):
        ilc_foregrounds[i] = observed_patches[i] - ilc_cmb
    if get_residuals:
        ilc_residuals = ilc_cmb - cmb_patch
        return (ilc_cmb, ilc_foregrounds, weights, ilc_residuals)
    return (ilc_cmb, ilc_foregrounds, weights)


def apply_ilc_to_single_patch_with_unet(freq_patches, cmb_patch, unet_cmb_prediction):
    """ILC with an extra synthetic channel from UNet CMB prediction (Appendix enhanced ILC)."""
    n_freq, height, width = freq_patches.shape
    observed_patches = freq_patches + cmb_patch[np.newaxis, :, :]
    observed_patches_with_unet = np.vstack(
        [observed_patches, unet_cmb_prediction[np.newaxis, :, :]]
    )
    n_channels = n_freq + 1
    observed_flat = observed_patches_with_unet.reshape(n_channels, -1)
    covariance = np.cov(observed_flat)
    regularization = 1e-10 * np.trace(covariance) / n_channels
    covariance += regularization * np.eye(n_channels)
    a_matrix = np.ones(n_channels)
    try:
        inv_cov = np.linalg.inv(covariance)
        denominator = np.dot(a_matrix.T, np.dot(inv_cov, a_matrix))
        weights = np.dot(a_matrix.T, inv_cov) / denominator
    except np.linalg.LinAlgError:
        weights = np.ones(n_channels) / n_channels
    ilc_cmb = np.zeros((height, width))
    for i in range(n_freq):
        ilc_cmb += weights[i] * observed_patches[i]
    ilc_cmb += weights[n_freq] * unet_cmb_prediction
    ilc_foregrounds = np.zeros((n_freq, height, width))
    for i in range(n_freq):
        ilc_foregrounds[i] = observed_patches[i] - ilc_cmb
    ilc_unet = unet_cmb_prediction - ilc_cmb
    return ilc_cmb, ilc_foregrounds, ilc_unet, weights
