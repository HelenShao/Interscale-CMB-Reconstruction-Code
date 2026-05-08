from __future__ import annotations
import numpy as np

def calculate_2d_spectrum(Map1, Map2, delta_ell, ell_max, pix_size, N, lib_mode: str='NumPy'):
    if lib_mode != 'NumPy':
        raise ValueError("This release supports only lib_mode='NumPy'.")
    N = int(N)
    ell_scale_factor = 2.0 * np.pi
    Map1 = np.asarray(Map1)
    Map2 = np.asarray(Map2)
    ones = np.ones(N)
    inds = (np.arange(N) + 0.5 - N / 2.0) / (N - 1.0)
    kX = np.outer(ones, inds) / (pix_size / 60.0 * np.pi / 180.0)
    kY = np.transpose(kX)
    K = np.sqrt(kX ** 2.0 + kY ** 2.0)
    ell2d = K * ell_scale_factor
    N_bins = int(ell_max / delta_ell)
    ell_array = np.arange(N_bins, dtype=float)
    CL_array = np.zeros(N_bins)
    FMap1 = np.fft.ifft2(np.fft.fftshift(Map1))
    FMap2 = np.fft.ifft2(np.fft.fftshift(Map2))
    PSMap = np.fft.fftshift(np.real(np.conj(FMap1) * FMap2))
    for i in range(N_bins):
        ell_array[i] = (i + 0.5) * delta_ell
        inds_in_bin = np.where((ell2d >= i * delta_ell) & (ell2d < (i + 1) * delta_ell))
        CL_array[i] = np.mean(PSMap[inds_in_bin])
    scaling_factor = np.sqrt(pix_size / 60.0 * np.pi / 180.0) * 2.0
    return (ell_array, CL_array * scaling_factor)
