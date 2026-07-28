"""
Experiment A -- full fairness / heterogeneity audit (the three tests the theory
of Section 4.2 commits to).

Test 1 (demographic subgroups + differential fairness). Among attention-check
PASSERS, regress each detector's standardized flag score on each demographic
axis (one-way ANOVA) and report the disparity D_M(Z)=sqrt(eta^2). Alongside it
report the attention check's OWN disparity D_C(Z)=sqrt(eta^2 of the fail
indicator on Z, all respondents). The detector is *differentially fair* on Z
(Definition, differential-fair) when D_M(Z) <= D_C(Z): it adds no subgroup
disparity beyond the practice it would replace.

Test 2 (opinion minorities). Cluster passers on their substantive responses
(k=2); the smaller cluster is the opinion minority O. Test whether the flag is
higher for O (standardized mean difference d, Welch t-test). d>0 means the
screener over-flags an atypical-but-attentive subpopulation.

Test 3 (high-variance-but-coherent responders). Among passers, correlate the
flag with within-person response variability V (SD across the ordinal battery).
A positive correlation means high-variance (but attentive) responders are
penalized.

Flag scores: AE (PL p=85) reconstruction error, Chow-Liu anomaly (1-pct), from
the cached errors.csv. Demographics / responses aligned to scored rows via
_survivor_index (sadc via the full loader). ivanov excluded (no codebook).

Usage:  python -m experiment_b.fairness
Out: fairness_diff.csv, fairness_opinion.csv, fairness_variance.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

from utils import define_necessary_elements
from dataset.loader import DataLoader
from evaluate.detection import (aligned_labels, aligned_battery, _survivor_index,
                                _read_raw, _to_ordinal, CHECKS, _relevant)

AXES = {
    "inattentive":    {"gender": "f.gender", "age": "f.agecat", "education": "f.educ",
                       "party": "party.senate.choice", "region": "f.region"},
    "pennycook_1":    {"party": "Party", "gender": "gender", "age": "age",
                       "ethnicity": "ethnicity", "education": "education",
                       "ideology": "Social_Conserv"},
    "public_opinion": {"gender": "gender", "age": "age", "race": "race",
                       "education": "education", "politics": "politics_self",
                       "religion": "religion_self"},
    "sadc_2017":      {"sex": "sex", "age": "age", "grade": "grade",
                       "race": "race", "sexual_identity": "sexual_identity"},
}
CITE = {"inattentive": "alvarez2019", "pennycook_1": "pennycook2020",
        "public_opinion": "mastroianni2022", "sadc_2017": "robinson2014"}
# datasets whose checks leave enough passers for a reliable passer-restricted test
WELL_POWERED = {"inattentive", "sadc_2017"}


def _ae_path(ds):
    return ("cache/sadc_2017_85perc_newloss/errors.csv" if ds == "sadc_2017"
            else f"cache/_tuned_{ds}_ae_p85/errors.csv")
def _cl_path(ds):
    return f"cache/_fix9_{ds}_cl/errors.csv"


def _loader(ds):
    dc, rc, ic, adc, arc, aic = define_necessary_elements(ds, None, None, None)
    return DataLoader(dc, rc, ic, additional_drop_columns=adc,
                      additional_rename_columns=arc, additional_columns_of_interest=aic)


def _demographics(ds):
    if ds == "sadc_2017":
        L = _loader(ds); L.COLUMNS_OF_INTEREST = []
        df, _ = L.load_data(ds)
        return df.reset_index(drop=True)
    return _read_raw(ds).loc[_survivor_index(ds)].reset_index(drop=True)


def _check_fail(ds, n):
    """C: 1 = failed a check / mischievous, 0 = passer/attentive."""
    if ds == "sadc_2017":
        return _loader(ds).find_outlier_data_sadc_2017(ds, ["outlier"])["outlier"].values.astype(int)
    labels = aligned_labels(ds)
    failed = np.zeros(n, dtype=bool)
    for ch in CHECKS[ds]:
        failed |= _relevant(labels, ch).astype(bool)
    return failed.astype(int)


def _battery(ds):
    if ds != "sadc_2017":
        return aligned_battery(ds)
    L = _loader(ds); data, _ = L.load_data(ds)
    cols = {}
    for c in data.columns:
        enc = _to_ordinal(data[c])
        if enc is not None and 2 <= int(pd.Series(enc).dropna().nunique()) <= 9:
            cols[c] = np.asarray(enc, dtype=float)
    return pd.DataFrame(cols)


def _responses(ds):
    L = _loader(ds); data, _ = L.load_data(ds)
    return pd.get_dummies(data.astype(str), dummy_na=False).reset_index(drop=True)


def _z(x):
    x = np.asarray(x, dtype=float); sd = np.nanstd(x)
    return (x - np.nanmean(x)) / sd if sd > 0 else x * np.nan


def _eta2(levels, y):
    """Share of variance in y explained by categorical/continuous axis `levels`."""
    s = pd.Series(levels)
    num = pd.to_numeric(s, errors="coerce")
    lvl = (pd.qcut(num, 4, duplicates="drop").astype(str)
           if num.notna().mean() > 0.9 and num.nunique() > 8 else s.astype(str))
    df = pd.DataFrame({"lvl": lvl.values, "y": np.asarray(y, dtype=float)}).dropna()
    counts = df["lvl"].value_counts()
    df = df[df["lvl"].isin(counts[counts >= 10].index)]
    groups = [g["y"].values for _, g in df.groupby("lvl")]
    if len(groups) < 2:
        return None
    grand = df["y"].mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    sst = ((df["y"] - grand) ** 2).sum()
    return float(ssb / sst) if sst > 0 else None


def _flags(ds):
    ae = pd.read_csv(_ae_path(ds)); cl = pd.read_csv(_cl_path(ds)); n = len(ae)
    ae_flag = ae["error"].to_numpy()
    cl_flag = (1.0 - cl["pct"]).to_numpy() if "pct" in cl else cl["error"].to_numpy()
    return ae_flag, cl_flag, n


def main():
    diff, opin, hivar = [], [], []
    for ds, axes in AXES.items():
        ae_flag, cl_flag, n = _flags(ds)
        C = _check_fail(ds, n)
        passer = C == 0
        demo = _demographics(ds)
        assert len(demo) == n, f"{ds}: demo {len(demo)} vs {n}"
        aez_p, clz_p = _z(ae_flag[passer]), _z(cl_flag[passer])

        # ---- Test 1: differential fairness D_M(Z) vs D_C(Z) ----
        for label, col in axes.items():
            if col not in demo.columns:
                continue
            Zall = demo[col].to_numpy()
            eta_ae = _eta2(Zall[passer], aez_p)
            eta_cl = _eta2(Zall[passer], clz_p)
            eta_C = _eta2(Zall, C)                       # check disparity, all respondents
            if eta_C is None:
                continue
            DC = np.sqrt(eta_C)
            DM_ae = np.sqrt(eta_ae) if eta_ae is not None else np.nan
            DM_cl = np.sqrt(eta_cl) if eta_cl is not None else np.nan
            diff.append({"dataset": CITE[ds], "axis": label, "n_pass": int(passer.sum()),
                         "well_powered": ds in WELL_POWERED,
                         "D_M_AE": round(DM_ae, 3), "D_M_CL": round(DM_cl, 3),
                         "D_C": round(DC, 3),
                         "AE_diff_fair": bool(DM_ae <= DC), "CL_diff_fair": bool(DM_cl <= DC)})

        # ---- Test 2: opinion minorities ----
        Xr = _responses(ds).to_numpy(dtype=float)[passer]
        km = KMeans(n_clusters=2, n_init=10, random_state=1).fit(Xr)
        lab = km.labels_
        O = lab == (0 if (lab == 0).sum() <= (lab == 1).sum() else 1)   # smaller cluster
        def _d(M):
            a, b = M[O], M[~O]
            d = (np.nanmean(a) - np.nanmean(b)) / np.nanstd(M)
            _, p = stats.ttest_ind(a[np.isfinite(a)], b[np.isfinite(b)], equal_var=False)
            return round(float(d), 3), round(float(p), 3)
        d_ae, p_ae = _d(aez_p); d_cl, p_cl = _d(clz_p)
        opin.append({"dataset": CITE[ds], "n_pass": int(passer.sum()),
                     "frac_minority": round(float(O.mean()), 3),
                     "AE_d": d_ae, "AE_p": p_ae, "CL_d": d_cl, "CL_p": p_cl})

        # ---- Test 3: high-variance-but-coherent responders ----
        V = _battery(ds).std(axis=1, skipna=True).to_numpy()[passer]
        def _corr(M):
            m = np.isfinite(V) & np.isfinite(M)
            if m.sum() < 10:
                return (np.nan, np.nan)
            r, p = stats.pearsonr(V[m], M[m])
            return round(float(r), 3), round(float(p), 3)
        r_ae, pr_ae = _corr(aez_p); r_cl, pr_cl = _corr(clz_p)
        hivar.append({"dataset": CITE[ds], "n_pass": int(passer.sum()),
                      "AE_r": r_ae, "AE_p": pr_ae, "CL_r": r_cl, "CL_p": pr_cl})

    pd.set_option("display.width", 170)
    dd = pd.DataFrame(diff); pd.DataFrame(dd).to_csv("fairness_diff.csv", index=False)
    oo = pd.DataFrame(opin); oo.to_csv("fairness_opinion.csv", index=False)
    hh = pd.DataFrame(hivar); hh.to_csv("fairness_variance.csv", index=False)
    print("===== TEST 1: differential fairness (D_M vs D_C; fair if D_M<=D_C) =====")
    print(dd.to_string(index=False))
    print(f"  AE differentially fair on {dd.AE_diff_fair.sum()}/{len(dd)} axes; "
          f"CL on {dd.CL_diff_fair.sum()}/{len(dd)}")
    print("\n===== TEST 2: opinion minorities (d>0 => minority over-flagged) =====")
    print(oo.to_string(index=False))
    print("\n===== TEST 3: high-variance responders (r>0 => penalized) =====")
    print(hh.to_string(index=False))


if __name__ == "__main__":
    main()
