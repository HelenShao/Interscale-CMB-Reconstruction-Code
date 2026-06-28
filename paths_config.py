from __future__ import annotations
import os

_CODE_RELEASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(_CODE_RELEASE_DIR, 'DATA')
DATA_ROOT = os.environ.get(
    'ILC_ML_DATA_ROOT', os.environ.get('ILC_DATA_ROOT', _DEFAULT_DATA_DIR)
).rstrip('/')

CAMB_CL_FILE = os.path.join(_DEFAULT_DATA_DIR, 'CAMB_fiducial_cosmo_scalCls.dat')
B_SMALL_DIR = os.path.join(DATA_ROOT, 'new_DF_patches_B')
T_E_DIR = os.path.join(DATA_ROOT, 'new_DF_patches_T_E')
E_T_DIR = T_E_DIR
ILC_DIR = os.path.join(DATA_ROOT, 'ILC_products')
STATS_FILE_SINGLEFREQ = os.path.join(B_SMALL_DIR, 'singlefreq_normalization_stats.npz')
STATS_FILE_8CH = os.path.join(ILC_DIR, 'interscale_multifreq_normalization_stats.npz')
STATS_FILE_16CH = os.path.join(ILC_DIR, 'interscale_multifreq_normalization_stats_16ch.npz')

OUTPUT_DIR_SINGLEFREQ = os.environ.get(
    'ILC_ML_OUTPUT_SINGLEFREQ', os.path.join(DATA_ROOT, 'singlefreq_unet_1024_0.05_scaled')
)
OUTPUT_DIR_INTERSCALE = os.environ.get(
    'ILC_ML_OUTPUT_INTERSCALE', os.path.join(DATA_ROOT, 'ILC_interscale_unet_1024')
)
DEFAULT_TEST_RESULTS_NPZ = os.path.join(
    OUTPUT_DIR_SINGLEFREQ, 'unnormalized_teb', 'test_results.npz'
)

# ILC baseline products on the UNet test split (compute_ilc_12ch_test_patches.py)
ILC_TEST_OUTPUT_ROOT = os.environ.get('ILC_ML_ILC_TEST_ROOT', DATA_ROOT)
ILC_PRODUCTS_12CH_TEST = os.path.join(ILC_TEST_OUTPUT_ROOT, 'ILC_products_12ch_teb_test')
ILC_PRODUCTS_4CH_B_TEST = os.path.join(ILC_TEST_OUTPUT_ROOT, 'ILC_products_4ch_b_test')
ILC_PRODUCTS_8CH_BT_TEST = os.path.join(ILC_TEST_OUTPUT_ROOT, 'ILC_products_8ch_bt_test')
ILC_PRODUCTS_8CH_BE_TEST = os.path.join(ILC_TEST_OUTPUT_ROOT, 'ILC_products_8ch_be_test')
ILC_PRODUCTS_3CH_BTE_220_TEST = os.path.join(
    ILC_TEST_OUTPUT_ROOT, 'ILC_products_3ch_bte_220_test'
)

BACKUP_DIR_SINGLEFREQ = os.environ.get(
    'ILC_ML_BACKUP_SINGLEFREQ', '/path/to/backup/singlefreq_unet_1024_0.05_scaled'
)
BACKUP_DIR_INTERSCALE = os.environ.get(
    'ILC_ML_BACKUP_INTERSCALE', '/path/to/backup/ILC_interscale_unet_1024'
)
