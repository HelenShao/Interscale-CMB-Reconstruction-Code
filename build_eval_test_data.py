#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import shutil
import numpy as np
FREQUENCIES = [95, 145, 220, 270]

def rewrite_stats(src_path: str, dst_path: str) -> None:
    with np.load(src_path, allow_pickle=False) as z:
        kw = {k: np.array(z[k]) for k in z.files}
    n_test = kw['test_indices'].shape[0]
    kw['test_indices'] = np.arange(n_test, dtype=np.int64)
    np.savez(dst_path, **kw)

def slice_npy(src_path: str, dst_path: str, test_idx: np.ndarray) -> None:
    mm = np.load(src_path, mmap_mode='r')
    sub = np.asarray(mm[test_idx])
    np.save(dst_path, sub)

def main() -> None:
    parser = argparse.ArgumentParser(description='Copy test-split patch arrays into code_release/DATA for evaluation scripts.')
    parser.add_argument('--source', default=os.environ.get('ILC_ML_SOURCE_DATA', '/scratch/gpfs/JDUNKLEY/hshao/old_data'), help='Root with new_DF_patches_B, new_DF_patches_T_E, ILC_products')
    parser.add_argument('--dest', default=None, help='Output root (default: DATA next to this script)')
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dest = args.dest or os.path.join(script_dir, 'DATA')
    src = os.path.abspath(args.source).rstrip('/')
    b_in = os.path.join(src, 'new_DF_patches_B')
    te_in = os.path.join(src, 'new_DF_patches_T_E')
    ilc_in = os.path.join(src, 'ILC_products')
    b_out = os.path.join(dest, 'new_DF_patches_B')
    te_out = os.path.join(dest, 'new_DF_patches_T_E')
    ilc_out = os.path.join(dest, 'ILC_products')
    os.makedirs(b_out, exist_ok=True)
    os.makedirs(te_out, exist_ok=True)
    os.makedirs(ilc_out, exist_ok=True)
    stats_src = os.path.join(b_in, 'singlefreq_normalization_stats.npz')
    if not os.path.isfile(stats_src):
        raise FileNotFoundError(stats_src)
    test_idx = np.load(stats_src)['test_indices']
    n_test = len(test_idx)
    print(f'Test split: {n_test} patches (full-array index range {int(test_idx.min())}-{int(test_idx.max())})')
    jobs = []
    for name in ['sim1-150_freq220_B_patches_small.npy', 'sim1-150_freq220_B_patches_large.npy', 'sim1-150_freq220_cmb_draw_b_all_scales.npy', 'sim1-150_freq220_B_patches.npy']:
        jobs.append((os.path.join(b_in, name), os.path.join(b_out, name)))
    for name in ['sim1-150_freq220_T_patches.npy', 'sim1-150_freq220_E_patches.npy']:
        jobs.append((os.path.join(te_in, name), os.path.join(te_out, name)))
    for freq in FREQUENCIES:
        jobs.append((os.path.join(b_in, f'sim1-150_freq{freq}_B_patches_small.npy'), os.path.join(b_out, f'sim1-150_freq{freq}_B_patches_small.npy')))
    for freq in FREQUENCIES:
        jobs.append((os.path.join(te_in, f'sim1-150_freq{freq}_E_patches.npy'), os.path.join(te_out, f'sim1-150_freq{freq}_E_patches.npy')))
        jobs.append((os.path.join(te_in, f'sim1-150_freq{freq}_T_patches.npy'), os.path.join(te_out, f'sim1-150_freq{freq}_T_patches.npy')))
    for name in ['ILC_foregrounds_b_95.npy', 'ILC_foregrounds_b_145.npy', 'ILC_foregrounds_b_220.npy', 'ILC_foregrounds_b_270.npy', 'ilc_residuals_b.npy', 'ILC_cmb_b.npy']:
        jobs.append((os.path.join(ilc_in, name), os.path.join(ilc_out, name)))
    for (i, (inp, outp)) in enumerate(jobs):
        if not os.path.isfile(inp):
            raise FileNotFoundError(inp)
        print(f'[{i + 1}/{len(jobs)}] {os.path.basename(outp)}', flush=True)
        slice_npy(inp, outp, test_idx)
    rewrite_stats(os.path.join(b_in, 'singlefreq_normalization_stats.npz'), os.path.join(b_out, 'singlefreq_normalization_stats.npz'))
    rewrite_stats(os.path.join(ilc_in, 'interscale_multifreq_normalization_stats.npz'), os.path.join(ilc_out, 'interscale_multifreq_normalization_stats.npz'))
    rewrite_stats(os.path.join(ilc_in, 'interscale_multifreq_normalization_stats_16ch.npz'), os.path.join(ilc_out, 'interscale_multifreq_normalization_stats_16ch.npz'))
    camb_candidates = [os.path.normpath(os.path.join(script_dir, '..', '..', '..', '..', '..', 'CAMB_fiducial_cosmo_scalCls.dat')), os.environ.get('CAMB_CLS_PATH', '')]
    camb_src = next((c for c in camb_candidates if c and os.path.isfile(c)), '')
    if os.path.isfile(camb_src):
        shutil.copy2(camb_src, os.path.join(dest, 'CAMB_fiducial_cosmo_scalCls.dat'))
        print('Copied CAMB_fiducial_cosmo_scalCls.dat', flush=True)
    else:
        print('WARNING: CAMB file not found; set path manually', flush=True)
    print('Done. Arrays are test-split only (first axis = %d).' % n_test, flush=True)

if __name__ == '__main__':
    main()
