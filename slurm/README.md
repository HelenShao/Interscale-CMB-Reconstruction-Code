# Slurm batch scripts

Run from the repo root. Set data paths before submitting:

```bash
export ILC_ML_DATA_ROOT=/scratch/gpfs/JDUNKLEY/hshao/old_data
sbatch slurm/run_ilc_12ch_test.sbatch
```

| Script | ILC variant |
|--------|-------------|
| `run_ilc_12ch_test.sbatch` | 12ch B+T+E @ 4 frequencies |
| `run_ilc_4ch_b_test.sbatch` | 4ch B-only |
| `run_ilc_8ch_bt_test.sbatch` | 8ch B+T |
| `run_ilc_8ch_be_test.sbatch` | 8ch B+E |
| `run_ilc_3ch_bte_220_test.sbatch` | 3ch B+T+E @ 220 GHz |
| `run_ilc_220ghz_all_test.sbatch` | All single-freq 220 GHz variants |
| `eval_unnormalized_full_et.sh` | 16ch hybrid eval example |

`run_ilc_12ch_test.sbatch` uses portable env vars. Other sbatch files may still contain cluster-specific paths—edit `WORKDIR` and data paths or set the same `ILC_ML_*` variables.
