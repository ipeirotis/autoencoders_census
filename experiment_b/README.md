# Experiment B pipeline

Per-dataset AE hyperparameter tuning, Linear AE / Chow-Liu scoring, and the
psychometric-baseline comparison. Run everything from the repo root.

| Step | Command | Output |
|---|---|---|
| AE search + p=80 score | `bash experiment_b/run_tuning.sh` | `cache/_tuned_<ds>_ae/` |
| AE finalize p=100/p=85 | `bash experiment_b/run_finalize.sh` | `cache/_tuned_<ds>_ae_p{100,85}/` |
| Linear AE train + MSE score | `bash experiment_b/run_linear.sh` | `cache/_lin_<ds>/` |
| Chow-Liu (binning fix) | `python main.py chow_liu_outliers --data <ds> --output cache/_fix9_<ds>_cl/` | `cache/_fix9_<ds>_cl/` |
| Full comparison table | `python -m experiment_b.final_table` | `experiment_b_full.csv`, `experiment_b_latex.txt` |
| Draft detailed tables | `python -m experiment_b.generate_detailed_tables_draft` | `../inattentiveness_paper/sections/_regenerated_detailed_tables_draft.tex` |

Notes:
- The Linear AE is the trained `LinearAutoencoder` (`model_name=PCA`, MSE loss); it
  is scored by `experiment_b.score_linear` (mean squared reconstruction error),
  not `find_outliers` (categorical cross-entropy, wrong objective for a linear
  reconstruction).
- Scoring/labels go through `evaluate/detection.py`; baselines through
  `evaluate/baselines.py`.
