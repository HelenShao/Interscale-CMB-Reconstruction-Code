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
