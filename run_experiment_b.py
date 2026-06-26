"""
Experiment B: psychometric careless-responding baselines vs the unsupervised
methods (autoencoder / linear AE / Chow-Liu), scored by the identical
attention-check detection harness.

Produces:
  - experiment_b_results.csv  : long format (dataset, check, method, n, n_pos, auc)
  - printed per-dataset tables

Run:  python run_experiment_b.py
"""
import os
import numpy as np
import pandas as pd

from evaluate.detection import (
    aligned_battery, evaluate_scores, evaluate_detection, CHECKS,
)
from evaluate import baselines as bl

# Unsupervised methods come from cached/fresh errors.csv. AE p100/p85 and the
# linear (PCA) baseline are the paper's cached outputs; CL is regenerated with
# the current (correct-sign) implementation.
UNSUP = [
    ("AE_p100", "cache/{ds}_100perc_newloss/errors.csv"),
    ("AE_p85",  "cache/{ds}_85perc_newloss/errors.csv"),
    ("LinearAE", "cache/{ds}_pca_/errors.csv"),
    ("ChowLiu", "cache/_repro_{ds}_cl/errors.csv"),
]
DATASETS = ["attention_check", "inattentive", "racial_data", "moral_data",
            "mturk_ethics", "bot_bot_mturk", "public_opinion", "pennycook_1"]


def main():
    rows = []
    for ds in DATASETS:
        B = aligned_battery(ds)
        print(f"\n### {ds}  (battery items={B.shape[1]}, n={len(B)})")

        # psychometric baselines
        for name, scores in bl.compute_all(B).items():
            if not np.isfinite(scores).any():
                continue  # index undefined for this dataset (no battery)
            rows += evaluate_scores(ds, scores, name)

        # unsupervised methods
        for name, tmpl in UNSUP:
            p = tmpl.format(ds=ds)
            if os.path.exists(p):
                try:
                    rows += evaluate_detection(ds, p, name)
                except Exception as e:
                    print(f"   {name}: skip ({str(e)[:50]})")

    df = pd.DataFrame(rows)
    df.to_csv("experiment_b_results.csv", index=False)

    # Pretty per-dataset wide table (rows = method, cols = check)
    method_order = ["longstring", "irv", "person_total_r", "mahalanobis",
                    "even_odd", "lz", "LinearAE", "ChowLiu", "AE_p100", "AE_p85"]
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        if sub.empty:
            continue
        wide = sub.pivot_table(index="method", columns="check", values="auc")
        wide = wide.reindex([m for m in method_order if m in wide.index])
        print(f"\n{'='*70}\n{ds}\n{'='*70}")
        print(wide.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\nSaved experiment_b_results.csv "
          f"({df.dataset.nunique()} datasets, {len(df)} rows)")


if __name__ == "__main__":
    main()
