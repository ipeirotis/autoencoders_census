"""
Experiment A -- heterogeneity / fairness audit.

Among respondents who PASSED the attention checks (the genuinely attentive), do
our detectors assign systematically higher flag scores to identifiable
subgroups? For each dataset x method x demographic axis we regress the
standardized flag score on the subgroup (one-way ANOVA), reporting the variance
the subgroup explains (eta^2) and the F-test p-value. Small, non-significant
eta^2 means the detector is not biased along that axis; a large one is a
deployment caveat.

Flag scores: AE (Percentile-Loss p=85) reconstruction error, and Chow-Liu
anomaly (1 - typicality percentile), read from the cached errors.csv (scored-row
order). Demographics come from the raw survey aligned to the same scored rows
via evaluate.detection._survivor_index (sadc_2017 via the full loader, since it
is outside the attention-check harness).

Usage:  python -m experiment_b.fairness
Out: fairness_results.csv  (+ a paste-ready LaTeX table on stdout)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats

from utils import define_necessary_elements
from dataset.loader import DataLoader
from evaluate.detection import (aligned_labels, _survivor_index, _read_raw,
                                CHECKS, _relevant)

# dataset -> {axis label: raw column}.  Numeric columns are binned into quartiles.
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

def _ae_path(ds):
    return ("cache/sadc_2017_85perc_newloss/errors.csv" if ds == "sadc_2017"
            else f"cache/_tuned_{ds}_ae_p85/errors.csv")
def _cl_path(ds):
    return f"cache/_fix9_{ds}_cl/errors.csv"


def _demographics(ds):
    """Raw demographic frame aligned to the scored rows (0..N-1)."""
    if ds == "sadc_2017":
        dc, rc, ic, adc, arc, aic = define_necessary_elements(ds, None, None, None)
        L = DataLoader(dc, rc, ic, additional_drop_columns=adc,
                       additional_rename_columns=arc, additional_columns_of_interest=aic)
        L.COLUMNS_OF_INTEREST = []          # full frame so renamed demographics exist
        df, _ = L.load_data(ds)
        return df.reset_index(drop=True)
    raw = _read_raw(ds)
    return raw.loc[_survivor_index(ds)].reset_index(drop=True)


def _passer_mask(ds, n):
    """True for respondents who passed every attention check (attentive)."""
    if ds == "sadc_2017":
        dc, rc, ic, adc, arc, aic = define_necessary_elements(ds, None, None, None)
        L = DataLoader(dc, rc, ic, additional_drop_columns=adc,
                       additional_rename_columns=arc, additional_columns_of_interest=aic)
        y = L.find_outlier_data_sadc_2017(ds, ["outlier"])["outlier"].values.astype(int)
        return y == 0                        # non-mischievous = attentive
    labels = aligned_labels(ds)
    failed = np.zeros(n, dtype=bool)
    for ch in CHECKS[ds]:
        failed |= _relevant(labels, ch).astype(bool)
    return ~failed


def _as_groups(axis_vals, score):
    """Bin a subgroup axis and return the list of score arrays per level
    (levels with >=10 respondents), plus eta^2 and the ANOVA p-value."""
    s = pd.Series(axis_vals)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9 and num.nunique() > 8:        # continuous -> quartiles
        lvl = pd.qcut(num, 4, duplicates="drop").astype(str)
    else:
        lvl = s.astype(str)
    lvl = lvl.where(pd.Series(score).notna().values, other=np.nan)
    df = pd.DataFrame({"lvl": lvl.values, "y": np.asarray(score, dtype=float)}).dropna()
    counts = df["lvl"].value_counts()
    keep = counts[counts >= 10].index
    df = df[df["lvl"].isin(keep)]
    groups = [g["y"].values for _, g in df.groupby("lvl")]
    if len(groups) < 2:
        return None
    grand = df["y"].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((df["y"] - grand) ** 2).sum()
    eta2 = float(ss_between / ss_total) if ss_total > 0 else float("nan")
    F, p = stats.f_oneway(*groups)
    return {"eta2": round(eta2, 3), "p": round(float(p), 3),
            "levels": len(groups), "n": len(df)}


def main():
    rows = []
    for ds, axes in AXES.items():
        ae = pd.read_csv(_ae_path(ds))
        cl = pd.read_csv(_cl_path(ds))
        n = len(ae)
        assert len(cl) == n, f"{ds}: AE {n} vs CL {len(cl)} rows"
        demo = _demographics(ds)
        assert len(demo) == n, f"{ds}: demo {len(demo)} vs scores {n} rows"
        ae_flag = ae["error"].to_numpy()
        cl_flag = (1.0 - cl["pct"]).to_numpy() if "pct" in cl else cl["error"].to_numpy()
        # "passers" = the reviewer's requested restriction to attentive respondents;
        # "all" = full-sample robustness (the two datasets whose checks flag 80-94%
        # leave too few passers for a reliable passer-restricted test).
        for sample, mask in (("passers", _passer_mask(ds, n)),
                             ("all", np.ones(n, dtype=bool))):
            def z(x):
                x = x[mask]; sd = np.nanstd(x)
                return (x - np.nanmean(x)) / sd if sd > 0 else x * np.nan
            aez, clz = z(ae_flag), z(cl_flag)
            for label, col in axes.items():
                if col not in demo.columns:
                    continue
                av = demo[col].to_numpy()[mask]
                a = _as_groups(av, aez)
                c = _as_groups(av, clz)
                if a is None or c is None:
                    continue
                rows.append({"dataset": CITE[ds], "axis": label, "sample": sample,
                             "n": int(mask.sum()), "levels": a["levels"],
                             "AE_eta2": a["eta2"], "AE_p": a["p"],
                             "CL_eta2": c["eta2"], "CL_p": c["p"]})
    out = pd.DataFrame(rows)
    out.to_csv("fairness_results.csv", index=False)
    pd.set_option("display.width", 170)
    for sample in ("passers", "all"):
        sub = out[out["sample"] == sample]
        print(f"\n===== {sample} =====")
        print(sub.drop(columns=["sample"]).to_string(index=False))
        print(f"  max eta^2 AE={sub.AE_eta2.max():.3f} CL={sub.CL_eta2.max():.3f}; "
              f"median AE={sub.AE_eta2.median():.3f} CL={sub.CL_eta2.median():.3f}")


if __name__ == "__main__":
    main()
