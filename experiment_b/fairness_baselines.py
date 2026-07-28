"""
Experiment A, baseline comparison: do the psychometric indices practitioners
already use exhibit the SAME fairness behavior as our unsupervised detectors?

Runs the three fairness tests (differential fairness, opinion-minority
over-flagging, high-variance penalty) for the six baselines (longstring, IRV,
person-total, Mahalanobis, even-odd, l_z) alongside the autoencoder and
Chow-Liu, on the same passers, clusters, and demographic axes. If the baselines
are no less biased, the caveat is a property of careless-response detection in
general, not of our method.

Baselines are scored on the ordinal battery (evaluate.baselines.INDICES on
detection.aligned_battery, sadc via the fairness helper), aligned to scored rows.

Usage:  python -m experiment_b.fairness_baselines
Out: fairness_baselines_opinion.csv, fairness_baselines_variance.csv,
     fairness_baselines_diff.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

from evaluate import baselines as bl
from experiment_b.fairness import (AXES, CITE, _demographics, _check_fail,
                                    _battery, _responses, _z, _eta2, _flags)

BASELINES = [("longstring", "LS"), ("irv", "IRV"), ("person_total_r", "PT"),
             ("mahalanobis", "MD"), ("even_odd", "EO"), ("lz", "lz")]
METHODS = ["AE", "CL", "LS", "IRV", "PT", "MD", "EO", "lz"]


def _scores(ds):
    """{method: per-respondent atypicality score} for AE, CL and the 6 baselines."""
    ae, cl, n = _flags(ds)
    bat = _battery(ds)
    M = {"AE": ae, "CL": cl}
    for key, lab in BASELINES:
        try:
            M[lab] = np.asarray(bl.INDICES[key](bat), dtype=float)
        except Exception:  # noqa: BLE001
            M[lab] = np.full(n, np.nan)
    return M, n


def main():
    opin, hivar, diff = [], [], []
    for ds, axes in AXES.items():
        M, n = _scores(ds)
        C = _check_fail(ds, n)
        passer = C == 0
        demo = _demographics(ds)
        # shared structure computed once
        Xr = _responses(ds).to_numpy(dtype=float)[passer]
        lab = KMeans(n_clusters=2, n_init=10, random_state=1).fit(Xr).labels_
        O = lab == (0 if (lab == 0).sum() <= (lab == 1).sum() else 1)
        V = _battery(ds).std(axis=1, skipna=True).to_numpy()[passer]
        DC = {label: (_eta2(demo[col].to_numpy(), C) if col in demo.columns else None)
              for label, col in axes.items()}

        for name in METHODS:
            s = M.get(name)
            if s is None or not np.isfinite(s[passer]).any():
                continue
            sz = _z(s[passer])
            # Test 2: opinion-minority over-flagging
            a, b = sz[O], sz[~O]
            d = (np.nanmean(a) - np.nanmean(b)) / np.nanstd(sz)
            _, pop = stats.ttest_ind(a[np.isfinite(a)], b[np.isfinite(b)], equal_var=False)
            opin.append({"dataset": CITE[ds], "method": name, "d": round(float(d), 3),
                         "p": round(float(pop), 3)})
            # Test 3: high-variance penalty
            m = np.isfinite(V) & np.isfinite(sz)
            if m.sum() >= 10:
                r, prr = stats.pearsonr(V[m], sz[m])
                hivar.append({"dataset": CITE[ds], "method": name, "r": round(float(r), 3),
                              "p": round(float(prr), 3)})
            # Test 1: differential fairness per axis
            for label, col in axes.items():
                if col not in demo.columns or DC[label] is None:
                    continue
                eta = _eta2(demo[col].to_numpy()[passer], sz)
                if eta is None:
                    continue
                diff.append({"dataset": CITE[ds], "method": name, "axis": label,
                             "D_M": round(float(np.sqrt(eta)), 3),
                             "D_C": round(float(np.sqrt(DC[label])), 3),
                             "fair": bool(np.sqrt(eta) <= np.sqrt(DC[label]))})

    od = pd.DataFrame(opin); od.to_csv("fairness_baselines_opinion.csv", index=False)
    hd = pd.DataFrame(hivar); hd.to_csv("fairness_baselines_variance.csv", index=False)
    dd = pd.DataFrame(diff); dd.to_csv("fairness_baselines_diff.csv", index=False)
    pd.set_option("display.width", 200)

    print("===== TEST 2: opinion-minority d, by method (mean over datasets; d>0 = over-flag) =====")
    print(od.groupby("method")["d"].agg(["mean", "min", "max"]).round(3).reindex(METHODS).to_string())
    print("\n  per-dataset d (rows=method, cols=dataset):")
    print(od.pivot(index="method", columns="dataset", values="d").reindex(METHODS).to_string())

    print("\n===== TEST 3: high-variance r, by method (r>0 = penalize high variance) =====")
    print(hd.groupby("method")["r"].agg(["mean", "min", "max"]).round(3).reindex([m for m in METHODS if m in hd.method.values]).to_string())
    print("\n  per-dataset r:")
    print(hd.pivot(index="method", columns="dataset", values="r").reindex([m for m in METHODS if m in hd.method.values]).to_string())

    print("\n===== TEST 1: differential fairness, share of axes with D_M<=D_C per method =====")
    frac = dd.groupby("method")["fair"].mean().round(3).reindex(METHODS)
    dm = dd.groupby("method")["D_M"].mean().round(3).reindex(METHODS)
    print(pd.DataFrame({"frac_axes_fair": frac, "mean_D_M": dm}).to_string())


if __name__ == "__main__":
    main()
