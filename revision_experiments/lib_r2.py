"""
lib_r2 -- validated primitives for the MISQ round-1 referee reanalyses.

Thin layer over the battle-tested harness (evaluate.detection) and the existing
fairness primitives (experiment_b.fairness). Every reanalysis script for the
referee report should import from HERE so score-loading, label alignment,
native orientation, quartile binning, and the passer mask are done ONE correct
way. Nothing here retrains a model; it reads the cached per-respondent and
per-feature scores the paper already committed.

Canonical cached scores (per scored row, in scoring-row order):
  AE p85 : cache/_tuned_{ds}_ae_p85/errors.csv  (col 'error'; robinson: cache/sadc_2017_85perc_newloss)
  AE p100: cache/_tuned_{ds}_ae_p100/errors.csv (robinson: cache/sadc_2017_100perc)
  LIN    : cache/_lin0_{ds}/errors.csv
  CL     : cache/_fix9_{ds}_cl/errors.csv        (anomaly = 1 - pct)
  seeds  : cache/_ms_{ds}_p{85,100}_s{1..5}/errors.csv
Per-feature AE errors live in the same errors.csv as columns 'col_error__<feat>'.

Dataset key <-> paper citation (all nine):
  racial_data=ivanov2021  pennycook_1=pennycook2020  public_opinion=mastroianni2022
  moral_data=ogrady2019   inattentive=alvarez2019    bot_bot_mturk=moss2023
  mturk_ethics=buchanan2018  attention_check=uhalt2020  sadc_2017=robinson2014
"""
from __future__ import annotations
import os, sys, types
from unittest.mock import MagicMock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The harness imports utils.py, which imports tensorflow/seaborn/yaml at module
# level. None are exercised on cache-only code paths (no retraining, no plotting,
# hardcoded column specs), and TF has no wheel on this box's Python. Stub any that
# are absent so the (pure-numpy) scoring/label/fairness primitives import cleanly.
def _ensure(mod):
    try:
        __import__(mod)
    except Exception:
        m = types.ModuleType(mod)
        m.__getattr__ = lambda name: MagicMock()
        sys.modules[mod] = m
for _m in ("tensorflow", "seaborn", "yaml"):
    _ensure(_m)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# re-export the validated harness + fairness primitives
from evaluate.detection import (aligned_labels, aligned_battery, CHECKS, _relevant,
                                 _survivor_index, _read_raw, _interest_columns,
                                 evaluate_scores)
from evaluate import baselines as bl
from experiment_b.fairness import (_ae_path, _cl_path, _battery, _responses,
                                   _demographics, _check_fail, _eta2, _z, _flags,
                                   AXES, CITE, WELL_POWERED)

# ---- all nine datasets ----
DS_ALL = ["racial_data", "pennycook_1", "public_opinion", "moral_data",
          "inattentive", "bot_bot_mturk", "mturk_ethics", "attention_check",
          "sadc_2017"]
# NOTE: mturk_ethics=moss2023 (no ordinal battery, baselines dashed) and
# bot_bot_mturk=buchanan2018 (has battery), verified against Table 4 native AUC.
CITE_ALL = {"racial_data": "ivanov2021", "pennycook_1": "pennycook2020",
            "public_opinion": "mastroianni2022", "moral_data": "ogrady2019",
            "inattentive": "alvarez2019", "bot_bot_mturk": "buchanan2018",
            "mturk_ethics": "moss2023", "attention_check": "uhalt2020",
            "sadc_2017": "robinson2014"}
PRIMARY_CHECK = {  # index into CHECKS[ds] of the headline check (matches Table 4)
    "racial_data": 0, "pennycook_1": 0, "public_opinion": 0, "moral_data": 0,
    "inattentive": 0, "bot_bot_mturk": 0, "mturk_ethics": 0, "attention_check": 0}
BASELINE_KEYS = [("longstring", "LS"), ("irv", "IRV"), ("person_total_r", "PT"),
                 ("mahalanobis", "MD"), ("even_odd", "EO"), ("lz", "lz")]


def _p100_path(ds):
    return ("cache/sadc_2017_100perc/errors.csv" if ds == "sadc_2017"
            else f"cache/_tuned_{ds}_ae_p100/errors.csv")
def _lin_path(ds):
    return f"cache/_lin0_{ds}/errors.csv"


def load_scores(ds):
    """dict of per-respondent anomaly scores (higher = more inattentive), all
    aligned to scoring rows: AE (p85), AE_p100, LIN, CL. Missing caches -> key absent."""
    out = {}
    ae, cl, n = _flags(ds)
    out["AE"], out["CL"], out["N"] = ae, cl, n
    for key, path in (("AE_p100", _p100_path(ds)), ("LIN", _lin_path(ds))):
        if os.path.exists(path):
            df = pd.read_csv(path)
            s = (1.0 - df["pct"]).to_numpy() if "pct" in df.columns else df["error"].to_numpy()
            if len(s) == n:
                out[key] = s
    return out


def load_colerrors(ds):
    """per-feature AE reconstruction errors (DataFrame, cols = feature names) for
    scoring-time missingness masking. Reads the canonical p85 errors.csv."""
    df = pd.read_csv(_ae_path(ds))
    ce = [c for c in df.columns if c.startswith("col_error__")]
    ren = {c: c[len("col_error__"):] for c in ce}
    return df[ce].rename(columns=ren)


def load_multiseed(ds, p=85):
    """list of per-respondent AE error arrays across cached seeds (s1..s5)."""
    arrs = []
    for s in range(1, 6):
        path = f"cache/_ms_{ds}_p{p}_s{s}/errors.csv"
        if os.path.exists(path):
            arrs.append(pd.read_csv(path)["error"].to_numpy())
    return arrs


def baseline_scores(ds):
    """dict {LS,IRV,PT,MD,EO,lz: per-respondent index} on the ordinal battery.
    These carry a designed orientation (high = careless); score native (no flip)."""
    bat = _battery(ds)
    n = len(bat)
    out = {}
    for key, lab in BASELINE_KEYS:
        try:
            # pandas 3.0 copy-on-write hands out read-only arrays; some indices
            # (IRV) mutate in place, so pass a fresh writable copy.
            out[lab] = np.asarray(bl.INDICES[key](bat.copy()), dtype=float)
        except Exception:
            out[lab] = np.full(n, np.nan)
    return out


def primary_label(ds):
    """per-scored-row inattentive label {0,1} for the headline check."""
    if ds == "sadc_2017":
        return _check_fail(ds, load_scores(ds)["N"])
    return _relevant(aligned_labels(ds), CHECKS[ds][PRIMARY_CHECK[ds]]).astype(int)


def missingness_count(ds):
    """per-scored-row count of model-input variables the respondent left blank
    (the 'No Answer' driver of atypicality flagged in referee M10)."""
    if ds == "sadc_2017":
        from experiment_b.fairness import _loader
        L = _loader(ds); data, _ = L.load_data(ds)
        return data.isna().sum(axis=1).to_numpy()
    raw = _read_raw(ds).loc[_survivor_index(ds)].reset_index(drop=True)
    cols = [c for c in _interest_columns(ds) if c in raw.columns]
    return raw[cols].isna().sum(axis=1).to_numpy()


def auc_native(y, s):
    """raw ROC AUC, no orientation flip (the paper's native-orientation convention)."""
    y = np.asarray(y); s = np.nan_to_num(np.asarray(s, float))
    if not (0 < y.sum() < len(y)):
        return np.nan
    return float(roc_auc_score(y, s))


def bootstrap_ci(stat_fn, n, B=2000, seed=0, alpha=0.05):
    """percentile bootstrap CI for a statistic that takes a row-index array.
    stat_fn(idx) -> float. Returns (lo, hi). Deterministic given seed."""
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for b in range(B):
        vals[b] = stat_fn(rng.integers(0, n, n))
    vals = vals[np.isfinite(vals)]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def auc_ci(y, s, B=2000, seed=0, orient=False):
    """bootstrap CI for AUC (native orientation by default)."""
    y = np.asarray(y); s = np.nan_to_num(np.asarray(s, float))
    def stat(idx):
        yy, ss = y[idx], s[idx]
        if not (0 < yy.sum() < len(yy)):
            return np.nan
        a = roc_auc_score(yy, ss)
        return max(a, 1 - a) if orient else a
    return bootstrap_ci(stat, len(y), B=B, seed=seed)


def eta(levels, y, bins=4):
    """correlation-ratio eta (sqrt of eta^2); numeric axes binned into `bins`
    quantiles. bins is the knob for the M9 binning-sensitivity check."""
    s = pd.Series(levels)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9 and num.nunique() > 8:
        lvl = pd.qcut(num, bins, duplicates="drop").astype(str)
    else:
        lvl = s.astype(str)
    df = pd.DataFrame({"lvl": lvl.values, "y": np.asarray(y, float)}).dropna()
    counts = df["lvl"].value_counts()
    df = df[df["lvl"].isin(counts[counts >= 10].index)]
    groups = [g["y"].values for _, g in df.groupby("lvl")]
    if len(groups) < 2:
        return None
    grand = df["y"].mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    sst = ((df["y"] - grand) ** 2).sum()
    return float(np.sqrt(ssb / sst)) if sst > 0 else None


def flag_disparity(score, Z, flag_rate, bins=4):
    """COMMENSURATE M-flag disparity (referee M9): threshold `score` at the top
    `flag_rate` fraction to a binary flag, then eta of that flag vs Z -- so it is
    built exactly like D_C (binary indicator), on whatever population `score`/`Z`
    span. Higher score = more inattentive."""
    score = np.asarray(score, float)
    k = max(1, int(round(flag_rate * len(score))))
    thr = np.sort(score)[::-1][k - 1]
    flag = (score >= thr).astype(int)
    return eta(Z, flag, bins=bins)
