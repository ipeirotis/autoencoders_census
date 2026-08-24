"""
M6 supervised fusion -- WITHIN-dataset out-of-fold logistic combiner.

Verifies the manuscript claim (sections/analysis.tex): "a supervised fusion
(out-of-fold logistic regression on the detector and index ranks) beats the best
single method on six of nine datasets, with substantial gains on the well-powered
instruments (+0.08 ivanov2021, +0.07 alvarez2019, +0.06 ogrady2019 and
buchanan2018; mean +0.026), losing only on the smallest-sample datasets."

The committed r2_m6_complementarity.py only implements the LEAVE-ONE-DATASET-OUT
supervised upper bound (section 5b), which trains on 8 datasets and tests on the
9th. That is a different estimator from the WITHIN-dataset out-of-fold fusion the
manuscript sentence describes. This script implements the within-dataset version.

Specification (as named by the author, pinned before running):
  For each of the 9 datasets INDEPENDENTLY:
    features X = per-respondent rank-percentile of each available scorer column
                 (AE, CL, LIN + baselines LS, IRV, PT, MD, EO, lz), standardized,
                 NaN -> 0.
    target  y = lib_r2.primary_label(ds).
    5-fold StratifiedKFold (shuffle, fixed random_state) via cross_val_predict to
    get out-of-fold predicted probabilities; fused_OOF_AUC = auc_native(y, p_oof).
    best_single(ds) = max over individual scorer columns of native AUC on that ds.
    delta(ds) = fused_OOF_AUC - best_single.
  Report per-dataset fused AUC, best-single AUC (+ which scorer), delta; then the
  number of datasets with delta>0 (of 9) and the mean delta.

All scores come from the REAL cached scores via lib_r2 / the r2_m6 primitives.
Higher score = more inattentive for every column (native-orientation convention).
This script AUTHORS a new file only; it neither edits nor overwrites any tracked
analysis code, data, figure, or manuscript.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from revision_experiments import lib_r2 as L
# Reuse the exact primitives from the committed complementarity script so
# score-loading, the rank-percentile transform, and the column lists match
# the paper's conventions one and only one way.
from revision_experiments.r2_m6_complementarity import (
    DETECTORS, BASELINES, ALL_COLS, get_all_scores, rank_pct,
)

RANDOM_STATE = 0
N_SPLITS = 5
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "r2_m6_supervised_fusion_results.csv")


def standardize_fill(X):
    """Column-standardize; fill NaN with 0 (= col mean after standardization).
    Identical convention to r2_m6_complementarity.standardize_fill."""
    Xs = X.astype(float).copy()
    mu = np.nanmean(Xs, axis=0)
    sd = np.nanstd(Xs, axis=0)
    sd[sd == 0] = 1.0
    Xs = (Xs - mu) / sd
    Xs[~np.isfinite(Xs)] = 0.0
    return Xs


def available_cols(sc, y):
    """Scorer columns with usable finite, non-degenerate support on this dataset."""
    cols = []
    for k in ALL_COLS:
        v = sc[k]
        m = np.isfinite(v)
        if m.sum() >= 5 and np.nanstd(v[m]) > 0:
            cols.append(k)
    return cols


rows = []
print("=" * 88, flush=True)
print("M6 WITHIN-dataset out-of-fold supervised fusion "
      f"({N_SPLITS}-fold StratifiedKFold, shuffle, random_state={RANDOM_STATE})", flush=True)
print("features = standardized rank-percentiles of {AE,CL,LIN + LS,IRV,PT,MD,EO,lz}; NaN->0",
      flush=True)
print("=" * 88, flush=True)

for ds in L.DS_ALL:
    sc, y = get_all_scores(ds)
    y = np.asarray(y, dtype=int)

    cols = available_cols(sc, y)

    # ---- best single scorer by native AUC on its own finite support ----
    singles = {}
    for k in cols:
        v = sc[k]
        m = np.isfinite(v)
        singles[k] = L.auc_native(y[m], v[m])
    best_single_k = max(singles, key=singles.get)
    best_single = singles[best_single_k]

    # ---- features: per-respondent rank-percentiles, standardized, NaN->0 ----
    X = np.vstack([rank_pct(sc[k]) for k in cols]).T  # N x len(cols), NaN preserved
    Xs = standardize_fill(X)

    # ---- 5-fold out-of-fold predicted probabilities ----
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    p_oof = cross_val_predict(clf, Xs, y, cv=skf, method="predict_proba")[:, 1]
    fused_auc = L.auc_native(y, p_oof)

    delta = fused_auc - best_single
    rows.append({
        "dataset": ds,
        "cite": L.CITE_ALL[ds],
        "N": int(len(y)),
        "pos": int(y.sum()),
        "n_scorers": len(cols),
        "scorers": "+".join(cols),
        "fused_oof_auc": round(fused_auc, 4),
        "best_single_auc": round(best_single, 4),
        "best_single_scorer": best_single_k,
        "delta": round(delta, 4),
        "win": bool(delta > 0),
    })
    print(f"  {L.CITE_ALL[ds]:16s} N={len(y):5d} pos={int(y.sum()):5d}  "
          f"fused={fused_auc:.3f}  best_single={best_single:.3f}({best_single_k})  "
          f"delta={delta:+.3f}  {'WIN' if delta > 0 else 'loss'}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

n_win = int(df["win"].sum())
mean_delta = float(df["delta"].mean())

print("\n" + "=" * 88, flush=True)
print("SUMMARY", flush=True)
print("=" * 88, flush=True)
print(df[["cite", "N", "fused_oof_auc", "best_single_auc",
          "best_single_scorer", "delta", "win"]].to_string(index=False), flush=True)
print(f"\n  datasets with delta>0 : {n_win}/9", flush=True)
print(f"  mean delta            : {mean_delta:+.4f}", flush=True)
print(f"\n  results CSV: {OUT_CSV}", flush=True)
print("DONE.", flush=True)
