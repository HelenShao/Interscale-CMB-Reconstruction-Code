from __future__ import annotations
import os
_CODE_RELEASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(_CODE_RELEASE_DIR, 'DATA')
_DATA_ROOT = os.environ.get('ILC_ML_DATA_ROOT', os.environ.get('ILC_DATA_ROOT', _DEFAULT_DATA_DIR)).rstrip('/')
CAMB_CL_FILE = os.path.join(_DEFAULT_DATA_DIR, 'CAMB_fiducial_cosmo_scalCls.dat')
B_SMALL_DIR = os.path.join(_DATA_ROOT, 'new_DF_patches_B')
T_E_DIR = os.path.join(_DATA_ROOT, 'new_DF_patches_T_E')
E_T_DIR = T_E_DIR
ILC_DIR = os.path.join(_DATA_ROOT, 'ILC_products')
STATS_FILE_SINGLEFREQ = os.path.join(B_SMALL_DIR, 'singlefreq_normalization_stats.npz')
STATS_FILE_8CH = os.path.join(ILC_DIR, 'interscale_multifreq_normalization_stats.npz')
STATS_FILE_16CH = os.path.join(ILC_DIR, 'interscale_multifreq_normalization_stats_16ch.npz')
OUTPUT_DIR_SINGLEFREQ = os.environ.get('ILC_ML_OUTPUT_SINGLEFREQ', os.path.join(_DATA_ROOT, 'singlefreq_unet_1024_0.05_scaled'))
OUTPUT_DIR_INTERSCALE = os.environ.get('ILC_ML_OUTPUT_INTERSCALE', os.path.join(_DATA_ROOT, 'ILC_interscale_unet_1024'))
BACKUP_DIR_SINGLEFREQ = os.environ.get('ILC_ML_BACKUP_SINGLEFREQ', '/path/to/backup/singlefreq_unet_1024_0.05_scaled')
BACKUP_DIR_INTERSCALE = os.environ.get('ILC_ML_BACKUP_INTERSCALE', '/path/to/backup/ILC_interscale_unet_1024')
