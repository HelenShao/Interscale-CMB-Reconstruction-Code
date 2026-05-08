import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import argparse
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
import plot_params
plot_params.setup_latex_path()
plot_params.patch_texmanager()
plot_params.setup_latex_preamble()
matplotlib.rcParams.update(plot_params.params)
import eval_test_results_singlefreq as eval_module
visualize_cmb_reconstructions = eval_module.visualize_cmb_reconstructions
visualize_fg_reconstructions = eval_module.visualize_fg_reconstructions

def load_and_reproduce_visualizations(data_path, output_dir=None):
    print('=' * 80)
    print('LOADING DATA FROM first_sample_data.npz')
    print('=' * 80)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Data file not found: {data_path}')
    data = np.load(data_path, allow_pickle=True)
    print(f'\nLoaded data from: {data_path}')
    print(f'\nAvailable keys in data file:')
    for key in sorted(data.keys()):
        arr = data[key]
        if isinstance(arr, np.ndarray):
            print(f'  {key}: shape {arr.shape}, dtype {arr.dtype}')
        else:
            print(f'  {key}: {type(arr)} = {arr}')
    sample_idx = int(data['sample_idx'])
    array_idx = int(data['array_idx'])
    print(f'\nSample information:')
    print(f'  Original dataset index: {sample_idx}')
    print(f'  Array position: {array_idx}')
    cmb_draw = data['cmb_draw'][np.newaxis, :, :]
    observed_b_all_scales = data['observed_b_all_scales'][np.newaxis, :, :]
    unet_cmb_reconstructed = data['unet_cmb_reconstructed'][np.newaxis, :, :]
    b_fg_all_scales = data['b_fg_all_scales'][np.newaxis, :, :]
    target = data['target'][np.newaxis, :, :]
    prediction = data['prediction'][np.newaxis, :, :]
    print(f'\nPatch shapes (after adding batch dimension):')
    print(f'  cmb_draw: {cmb_draw.shape}')
    print(f'  observed_b_all_scales: {observed_b_all_scales.shape}')
    print(f'  unet_cmb_reconstructed: {unet_cmb_reconstructed.shape}')
    print(f'  b_fg_all_scales: {b_fg_all_scales.shape}')
    print(f'  target: {target.shape}')
    print(f'  prediction: {prediction.shape}')
    unet_spatial_correlation = np.array([data['unet_spatial_correlation']])
    pred_vs_target_correlation = np.array([data['pred_vs_target_correlation']])
    observed_b_mse_array = data['observed_b_mse_array']
    unet_recon_mse_array = data['unet_recon_mse_array']
    pred_vs_target_mse_array = data['pred_vs_target_mse_array']
    target_vs_zero_mse_array = data['target_vs_zero_mse_array']
    print(f'\nMSE arrays (for all patches):')
    print(f'  observed_b_mse_array: {observed_b_mse_array.shape}')
    print(f'  unet_recon_mse_array: {unet_recon_mse_array.shape}')
    print(f'  pred_vs_target_mse_array: {pred_vs_target_mse_array.shape}')
    print(f'  target_vs_zero_mse_array: {target_vs_zero_mse_array.shape}')
    ell_array = data['ell_array']
    unet_cross_spectrum = None
    mean_unet_cross_ps = None
    std_unet_cross_ps = None
    if 'unet_cross_spectrum' in data:
        unet_cross_spectrum = data['unet_cross_spectrum'][np.newaxis, :]
        mean_unet_cross_ps = data['mean_unet_cross_ps']
        std_unet_cross_ps = data['std_unet_cross_ps']
        print(f'\nUNet cross-spectrum:')
        print(f'  unet_cross_spectrum: {unet_cross_spectrum.shape}')
        print(f'  mean_unet_cross_ps: {mean_unet_cross_ps.shape}')
        print(f'  std_unet_cross_ps: {std_unet_cross_ps.shape}')
    pred_vs_target_cross_spectrum = None
    mean_pred_vs_target_cross_ps = None
    std_pred_vs_target_cross_ps = None
    if 'pred_vs_target_cross_spectrum' in data:
        pred_vs_target_cross_spectrum = data['pred_vs_target_cross_spectrum'][np.newaxis, :]
        mean_pred_vs_target_cross_ps = data['mean_pred_vs_target_cross_ps']
        std_pred_vs_target_cross_ps = data['std_pred_vs_target_cross_ps']
        print(f'\nPrediction vs Target cross-spectrum:')
        print(f'  pred_vs_target_cross_spectrum: {pred_vs_target_cross_spectrum.shape}')
        print(f'  mean_pred_vs_target_cross_ps: {mean_pred_vs_target_cross_ps.shape}')
        print(f'  std_pred_vs_target_cross_ps: {std_pred_vs_target_cross_ps.shape}')
    null_cross_spectra_array = None
    mean_null_cross_ps_overall = None
    if 'null_cross_spectra_array' in data:
        null_cross_spectra_array = data['null_cross_spectra_array']
        mean_null_cross_ps_overall = data['mean_null_cross_ps_overall']
        print(f'\nNull cross-spectra:')
        print(f'  null_cross_spectra_array: {null_cross_spectra_array.shape}')
        print(f'  mean_null_cross_ps_overall: {mean_null_cross_ps_overall.shape}')
    null_correlations_array = None
    mean_null_correlation_overall = None
    if 'null_correlations_array' in data:
        null_correlations_array = data['null_correlations_array']
        mean_null_correlation_overall = float(data['mean_null_correlation_overall'])
        print(f'\nNull correlations (CMB):')
        print(f'  null_correlations_array: {null_correlations_array.shape}')
        print(f'  mean_null_correlation_overall: {mean_null_correlation_overall}')
    fg_null_correlations_array = None
    fg_mean_null_correlation_overall = None
    if 'fg_null_correlations_array' in data:
        fg_null_correlations_array = data['fg_null_correlations_array']
        fg_mean_null_correlation_overall = float(data['fg_mean_null_correlation_overall'])
        print(f'\nNull correlations (Foreground):')
        print(f'  fg_null_correlations_array: {fg_null_correlations_array.shape}')
        print(f'  fg_mean_null_correlation_overall: {fg_mean_null_correlation_overall}')
    if output_dir is None:
        output_dir = os.path.dirname(data_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f'\nOutput directory: {output_dir}')
    sample_indices = [sample_idx]
    index_mapping = {sample_idx: 0}
    print('\n' + '=' * 80)
    print('REPRODUCING CMB RECONSTRUCTION VISUALIZATIONS')
    print('=' * 80)
    visualize_cmb_reconstructions(cmb_draw, observed_b_all_scales, unet_cmb_reconstructed, sample_indices, output_dir=output_dir, ell_array=ell_array, unet_cross_spectra_array=unet_cross_spectrum, observed_cross_spectra_array=None, mean_unet_cross_ps=mean_unet_cross_ps, std_unet_cross_ps=std_unet_cross_ps, mean_observed_cross_ps=None, std_observed_cross_ps=None, null_cross_spectra_array=null_cross_spectra_array, mean_null_cross_ps_overall=mean_null_cross_ps_overall, unet_spatial_correlations=unet_spatial_correlation, null_correlations_array=null_correlations_array, mean_null_correlation_overall=mean_null_correlation_overall, observed_b_mse_array=observed_b_mse_array, unet_recon_mse_array=unet_recon_mse_array, index_mapping=index_mapping)
    print('\n' + '=' * 80)
    print('REPRODUCING FOREGROUND RECONSTRUCTION VISUALIZATIONS')
    print('=' * 80)
    visualize_fg_reconstructions(b_fg_all_scales, target, prediction, sample_indices, output_dir=output_dir, ell_array=ell_array, pred_vs_target_cross_spectra_array=pred_vs_target_cross_spectrum, mean_pred_vs_target_cross_ps=mean_pred_vs_target_cross_ps, std_pred_vs_target_cross_ps=std_pred_vs_target_cross_ps, null_cross_spectra_array=None, mean_null_cross_ps_overall=None, pred_vs_target_correlations=pred_vs_target_correlation, null_correlations_array=fg_null_correlations_array, mean_null_correlation_overall=fg_mean_null_correlation_overall, target_vs_zero_mse_array=target_vs_zero_mse_array, pred_vs_target_mse_array=pred_vs_target_mse_array, index_mapping=index_mapping)
    print('\n' + '=' * 80)
    print('VISUALIZATION COMPLETE')
    print('=' * 80)
    print(f'\nPlots saved to:')
    print(f"  CMB reconstructions: {os.path.join(output_dir, 'cmb_reconstructions')}")
    print(f"  Foreground reconstructions: {os.path.join(output_dir, 'fg_reconstructions')}")

def main():
    parser = argparse.ArgumentParser(description='Reproduce visualization plots from first_sample_data.npz')
    parser.add_argument('--data-path', type=str, default=None, help='Path to first_sample_data.npz file (default: search in current directory)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for plots (default: same as data file directory)')
    args = parser.parse_args()
    if args.data_path is None:
        current_dir = os.getcwd()
        data_path = os.path.join(current_dir, 'first_sample_data.npz')
        if not os.path.exists(data_path):
            parent_dir = os.path.dirname(current_dir)
            data_path = os.path.join(parent_dir, 'first_sample_data.npz')
            if not os.path.exists(data_path):
                raise FileNotFoundError(f'Could not find first_sample_data.npz. Please specify --data-path.\n  Searched in: {current_dir}\n  Searched in: {parent_dir}')
    else:
        data_path = args.data_path
    load_and_reproduce_visualizations(data_path, args.output_dir)
if __name__ == '__main__':
    main()
