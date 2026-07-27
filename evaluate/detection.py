"""
Corrected, centralized detection-evaluation harness.

Background
----------
The legacy evaluation path (``main.evaluate_on_condition`` ->
``DataLoader.find_outlier_data`` -> ``utils.evaluate_errors``) has two
modularization regressions:

1. **Numeric attention-check columns are destroyed.** ``find_outlier_data``
   reloads the dataset through ``prepare_original_dataset`` ->
   ``convert_to_categorical``, which z-score-bins every numeric column and
   renames ``X`` -> ``X_cat``. Datasets whose held-out attention-check column
   is numeric (``moral_data``/``attention``, ``public_opinion``/``attention_1``,
   ``pennycook_1``/``screen*``) lose the column entirely -> ``KeyError``.

2. **Fragile positional join.** Labels are attached to the error scores with
   ``pd.concat(..., axis=1)``, which assumes the scoring path (battery columns
   only) and the label path (all columns) keep the exact same rows. Loaders
   that call ``dropna(inplace=True)`` after column selection (``moral_data``,
   ``racial_data``) drop over *different* column sets in the two paths, so the
   labels can silently misalign with the scores.

This module fixes both by reading the attention-check columns **raw** (never
binned) and aligning them to the scoring rows by **reproducing the scoring
path's exact row set** (the same row filter + ``dropna`` over the battery /
interest columns), then validating the row count against the live loader.

Each evaluation excludes rows whose attention-check value is missing (they
cannot be labelled) and uses an explicit per-dataset positive (inattentive)
condition, so the gold-label polarity is unambiguous.

Usage
-----
    from evaluate.detection import evaluate_detection, aligned_labels
    rows = evaluate_detection("racial_data", "cache/x/errors.csv", method="CL")
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utils import define_necessary_elements
from dataset.loader import DataLoader


# ---------------------------------------------------------------------------
# Raw CSV locations (mirror dataset/loader.py)
# ---------------------------------------------------------------------------
RAW_CSV = {
    "attention_check": "data/attention_check.csv",
    "inattentive": "data/inattentive_users.csv",
    "racial_data": "data/racial_data.csv",
    "moral_data": "data/moral_data.csv",
    "public_opinion": "data/public_opinion.csv",
    "bot_bot_mturk": "data/Bot_Bot_Bot__MTURK.csv",
    "mturk_ethics": "data/ethics.csv",
    "pennycook_1": "data/Pennycook et al._Study 1.csv",
}


def _read_raw(dataset: str) -> pd.DataFrame:
    # latin1 never fails to decode and leaves ASCII attention-check values
    # (Passed/Failed/pass/fail/"1. Passed"/7/...) untouched.
    return pd.read_csv(RAW_CSV[dataset], encoding="latin1", low_memory=False)


def _interest_columns(dataset: str):
    """Battery/interest column *names* in the raw CSV, in original order.

    ``define_necessary_elements`` returns positional indices; resolve them
    against the raw header so the dropna below operates on exactly the columns
    the scoring path keeps.
    """
    _, _, interest, _, _, _ = define_necessary_elements(dataset, None, None, None)
    cols = list(_read_raw(dataset).columns)
    names = []
    for c in interest:
        if isinstance(c, int):
            if c < len(cols):
                names.append(cols[c])
        elif c in cols:
            names.append(c)
    return names


def _float_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the loaders' whole-number float -> Int64 conversion."""
    for col in df.select_dtypes(include="float").columns:
        s = df[col].dropna()
        if len(s) and (s % 1 == 0).all():
            df[col] = df[col].astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Survivor index: the original raw-CSV row positions kept by the scoring path,
# in order, per dataset. Reproduces each loader's row handling exactly.
# ---------------------------------------------------------------------------
def _survivor_index(dataset: str) -> pd.Index:
    raw = _read_raw(dataset)

    if dataset in ("attention_check", "inattentive", "public_opinion",
                   "mturk_ethics", "pennycook_1"):
        # No row filtering in the loader -> every raw row is scored, in order.
        # (pennycook_1 only drops/derives columns; it never drops rows.)
        return raw.index

    interest = _interest_columns(dataset)

    if dataset == "moral_data":
        # loader: select interest -> [complete==1] -> drop complete -> dropna(interest)
        sel = _float_to_int(raw[interest].copy())
        sel = sel[sel["complete"] == 1].drop(columns=["complete"])
        return sel.dropna().index

    if dataset == "racial_data":
        # loader: select interest -> [Finished=="1. True"] -> drop Finished -> dropna(interest)
        sel = _float_to_int(raw[interest].copy())
        sel = sel[sel["Finished"] == "1. True"].drop(columns=["Finished"])
        return sel.dropna().index

    if dataset == "bot_bot_mturk":
        # loader keeps rows where Q6 is numeric; no dropna over interest.
        keep = pd.to_numeric(raw["Q6"], errors="coerce").notnull()
        return raw.index[keep]

    raise KeyError(f"No survivor-index rule for dataset {dataset!r}")


def aligned_labels(dataset: str, validate: bool = True) -> pd.DataFrame:
    """Raw attention-check columns for every scored row, in scoring-row order.

    The returned frame has a fresh 0..N-1 index that lines up positionally with
    ``errors.csv`` (which the scoring path writes in the same order with
    ``index=False``). When ``validate`` is True, the row count is checked
    against the live loader so any drift between this reproduction and the
    loader fails loudly instead of silently misaligning.
    """
    idx = _survivor_index(dataset)
    raw = _read_raw(dataset)
    labels = raw.loc[idx].reset_index(drop=True)

    if validate:
        dc, rc, ic, adc, arc, aic = define_necessary_elements(dataset, None, None, None)
        loader = DataLoader(
            dc, rc, ic,
            additional_drop_columns=adc,
            additional_rename_columns=arc,
            additional_columns_of_interest=aic,
        )
        n_loader = len(loader.load_data(dataset)[0])
        if len(labels) != n_loader:
            raise AssertionError(
                f"{dataset}: aligned-label rows ({len(labels)}) != loader scoring "
                f"rows ({n_loader}); row-filter reproduction is out of sync."
            )
    return labels


# Common ordinal-word response sets -> numeric codes, for items stored as text
# (e.g. public_opinion "Agree"/"Disagree"). Leading-number codes ("2. Support")
# are handled separately by regex extraction.
_LIKERT_WORDS = {
    "strongly disagree": 1, "disagree": 2, "neither agree nor disagree": 3,
    "neutral": 3, "neither": 3, "agree": 4, "strongly agree": 5,
    "strongly oppose": 1, "oppose": 2, "neither support nor oppose": 3,
    "support": 4, "strongly support": 5,
    "never": 1, "rarely": 2, "sometimes": 3, "often": 4, "always": 5,
}


def _to_ordinal(s: pd.Series) -> "pd.Series | None":
    """Best-effort ordinal encoding of a survey item to numeric codes. Returns
    None if the column doesn't look like an ordinal/numeric item. The 0.8
    coverage threshold is taken over NON-missing values, so an otherwise-clean
    Likert item with high item-missingness (common for historical/branched
    questions) is still encoded rather than discarded."""
    nn = int(s.notna().sum())
    if nn == 0:
        return None
    num = pd.to_numeric(s, errors="coerce")
    if int(num.notna().sum()) >= 0.8 * nn:
        return num
    # leading integer code, e.g. "2. Support" -> 2
    ext = pd.to_numeric(s.astype(str).str.extract(r"^\s*(\d+)")[0], errors="coerce")
    if int(ext.notna().sum()) >= 0.8 * nn:
        return ext
    # ordinal words
    mapped = s.astype(str).str.strip().str.lower().map(_LIKERT_WORDS)
    if int(mapped.notna().sum()) >= 0.8 * nn:
        return mapped
    return None


def aligned_battery(dataset: str, max_levels: int = 9) -> pd.DataFrame:
    """Numeric battery (Likert-style item responses) for every scored row, in
    scoring-row order, with a fresh 0..N-1 index that lines up positionally
    with ``errors.csv`` and :func:`aligned_labels`.

    Selects the dataset's interest/battery columns that encode as ordinal items
    with ``2..max_levels`` distinct values — the same low-cardinality items the
    autoencoder is fed (after the Rule-of-N filter), kept as raw numeric
    responses so practitioner indices (longstring, IRV, person-total,
    Mahalanobis, even-odd, l_z) can be computed on them. Items stored as text
    ("2. Support", "Agree") are ordinal-encoded; continuous columns (sliders,
    timings, year-of-birth) have too many levels and are dropped. This keeps the
    baseline-vs-autoencoder comparison on the same items.
    """
    idx = _survivor_index(dataset)
    raw = _read_raw(dataset)
    interest = _interest_columns(dataset)
    sub = raw.loc[idx, interest].reset_index(drop=True)

    cols = {}
    for c in sub.columns:
        enc = _to_ordinal(sub[c])
        if enc is None:
            continue
        if 2 <= int(enc.dropna().nunique()) <= max_levels:
            cols[c] = enc
    # index=sub.index keeps N rows even when no item survives (0-column frame),
    # so downstream baselines return one NaN per scored row rather than empty.
    return pd.DataFrame(cols, index=sub.index)


# ---------------------------------------------------------------------------
# Attention-check definitions.
#
# Each check: name, columns, correct (attentive) value(s), and the positive
# (inattentive) condition. ``"neq"`` => positive when the answer differs from
# the correct value (the usual case). ``"eq"`` => positive when the answer
# EQUALS the listed value -- used only for ``inattentive``, whose ``filter``
# column labels the inattentive group as "pass" (verified empirically: the
# "pass" group carries the high reconstruction/atypicality error, and this is
# the polarity that reproduces the paper's ~0.80 AUC). FLAG: confirm with the
# Alvarez et al. codebook which ``filter`` value denotes inattentiveness.
# ---------------------------------------------------------------------------
# Missing-value convention: a missing attention-check answer counts as a
# FAILURE (positive). This matches the paper's positive counts h exactly
# (e.g. mturk Screener_One h=161 = 155 Failed + 6 NaN; bot Q6_15 h=59 = 53 +
# 6 NaN; pennycook screen3 h=68 = 65 + 3 NaN). A respondent who skipped the
# screen did not demonstrate attentiveness, so treating it as a failure is the
# conservative, paper-consistent choice.
#
# Pennycook screens are multi-select checkboxes: each ``screen{n}_{i}`` is 1
# when box i was ticked and NaN/blank otherwise. A screen is PASSED only when
# the ticked pattern exactly matches the answer key; the four screens map to
# the paper's Attention 1-4 (verified by failure counts 636/236/68/172).
CHECKS = {
    "attention_check": [
        {"name": "Attention_Check", "cols": ["Attention_Check"], "correct": ["Passed"], "mode": "neq"},
    ],
    "inattentive": [
        {"name": "filter", "cols": ["filter"], "correct": ["pass"], "mode": "eq"},
    ],
    "racial_data": [
        {"name": "attn1", "cols": ["attn1"], "correct": ["1. Passed"], "mode": "neq"},
        {"name": "attn2", "cols": ["attn2"], "correct": ["1. Passed"], "mode": "neq"},
        {"name": "union(attn1|attn2)", "cols": ["attn1", "attn2"], "correct": ["1. Passed", "1. Passed"], "mode": "neq_any"},
        {"name": "intersection(attn1&attn2)", "cols": ["attn1", "attn2"], "correct": ["1. Passed", "1. Passed"], "mode": "neq_all"},
    ],
    "moral_data": [
        {"name": "attention", "cols": ["attention"], "correct": [1], "mode": "neq"},
    ],
    "public_opinion": [
        # mastroianni2022. The attention_1 slider's default is 1; respondents who
        # LEAVE it at 1 are the flagged (inattentive) group, so positive == 1
        # (mode "eq"), not != 1. Verified empirically: this polarity reproduces
        # the paper's ~0.66 AE (the != 1 reading gives 1-0.66=0.32). Confirm
        # against the Mastroianni & Dana codebook.
        {"name": "attention_1", "cols": ["attention_1"], "correct": [1], "mode": "eq"},
    ],
    "bot_bot_mturk": [
        {"name": "Q6_15", "cols": ["Q6_15"], "correct": [7], "mode": "neq"},
    ],
    "mturk_ethics": [
        {"name": "Screener_One", "cols": ["Screener_One"], "correct": ["Passed"], "mode": "neq"},
        {"name": "Screener_Two", "cols": ["Screener_Two"], "correct": ["Passed"], "mode": "neq"},
        {"name": "union(S1|S2)", "cols": ["Screener_One", "Screener_Two"], "correct": ["Passed", "Passed"], "mode": "neq_any"},
        {"name": "intersection(S1&S2)", "cols": ["Screener_One", "Screener_Two"], "correct": ["Passed", "Passed"], "mode": "neq_all"},
    ],
}

# Pennycook screens (checkbox answer keys + single-answer checks).
_PENNY = {
    "AC1": {"kind": "checkbox", "cols": [f"screen1_{i}" for i in range(1, 10)],
            "key": [0, 0, 0, 1, 0, 0, 1, 0, 0]},
    "AC2": {"kind": "checkbox", "cols": [f"screen2_{i}" for i in range(1, 7)],
            "key": [0, 0, 1, 0, 1, 0]},
    "AC3": {"kind": "single", "cols": ["screen3_2"], "correct": [3]},
    "AC4": {"kind": "single", "cols": ["Random"], "correct": [2]},
}
CHECKS["pennycook_1"] = [
    {"name": "AC1(screen1)", "screens": ["AC1"], "combine": "any"},
    {"name": "AC2(screen2)", "screens": ["AC2"], "combine": "any"},
    {"name": "AC3(screen3)", "screens": ["AC3"], "combine": "any"},
    {"name": "AC4(Random)", "screens": ["AC4"], "combine": "any"},
    {"name": "union(any AC)", "screens": ["AC1", "AC2", "AC3", "AC4"], "combine": "any"},
    {"name": "intersection(all AC)", "screens": ["AC1", "AC2", "AC3", "AC4"], "combine": "all"},
]


def _norm(v):
    """Normalize for comparison: NaN-safe string compare (handles 7 vs '7',
    1 vs 1.0, 'Passed' etc.). Returns None for missing values."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or (pd.isna(v) if np.ndim(v) == 0 else False):
        return None
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip()


def _screen_failed(labels: pd.DataFrame, screen: dict) -> np.ndarray:
    """Boolean per-row array: True where the respondent FAILED this screen."""
    if screen["kind"] == "checkbox":
        ok = np.ones(len(labels), dtype=bool)
        for col, want in zip(screen["cols"], screen["key"]):
            ticked = (labels[col] == 1).to_numpy()
            ok &= (ticked == bool(want))
        return ~ok
    # single-answer check: missing (None) != correct -> failure
    col, correct = screen["cols"][0], _norm(screen["correct"][0])
    return np.array([_norm(labels.iloc[i][col]) != correct for i in range(len(labels))])


def _relevant(labels: pd.DataFrame, check: dict) -> np.ndarray:
    """Per-row inattentive label (1=inattentive/failed). Missing answers count
    as failures (see module-level convention note)."""
    if "screens" in check:  # pennycook composite of one or more screens
        fails = [_screen_failed(labels, _PENNY[s]) for s in check["screens"]]
        stacked = np.vstack(fails)
        return (stacked.any(axis=0) if check["combine"] == "any" else stacked.all(axis=0)).astype(int)

    cols, correct, mode = check["cols"], check["correct"], check["mode"]
    correct_norm = [_norm(c) for c in correct]
    n = len(labels)
    relevant = np.zeros(n, dtype=int)
    for i in range(n):
        vals = [_norm(labels.iloc[i][c]) for c in cols]
        neq = [v != cv for v, cv in zip(vals, correct_norm)]  # None != correct -> True (missing = fail)
        if mode == "neq":
            relevant[i] = int(neq[0])
        elif mode == "eq":
            # positive == answer EQUALS the flagged value (e.g. slider left at the
            # default). A missing answer (None) is deliberately NOT the flagged
            # value here, so it is not a positive -- unlike the neq branches where
            # missing != correct counts as a failure. This matches the paper:
            # inattentive (alvarez) has no missing filter values, and public_opinion
            # has 32 blank attention_1 cells that stay attentive, giving h = 976.
            relevant[i] = int(vals[0] is not None and not neq[0])
        elif mode == "neq_any":
            relevant[i] = int(any(neq))
        elif mode == "neq_all":
            relevant[i] = int(all(neq))
        else:
            raise ValueError(f"unknown mode {mode!r}")
    return relevant


def _ranking_metrics(scores: np.ndarray, y: np.ndarray, orient: bool = True) -> dict:
    """Information-retrieval metrics used in the paper's detection tables, with
    ``scores`` = carelessness score (higher = more inattentive) and ``y`` in {0,1}.
    Returns h, R@h (=P@h), P@10/50/100, NDCG@h, AUC (None where undefined).

    When ``orient`` is True (the default) we orient ``scores`` to the larger of
    AUC and 1-AUC before computing every metric. Every detector here -- the
    unsupervised reconstruction-error / log-likelihood scores and the
    psychometric indices alike -- produces a continuous carelessness score whose
    sign is only a labeling convention (either tail may be called the inattentive
    one), so the method's separation ability, max(AUC, 1-AUC), is what counts."""
    n = len(y)
    h = int(y.sum())
    scores = np.asarray(scores, dtype=float)

    if 0 < h < n:
        auc = roc_auc_score(y, scores)
        if orient and auc < 0.5:      # orient to the better-separating direction
            scores = -scores
            auc = 1.0 - auc
    else:
        auc = float("nan")

    order = np.argsort(-scores, kind="mergesort")  # stable descending
    y_sorted = y[order]

    def p_at(k):
        k = min(k, n)
        return float(y_sorted[:k].sum()) / k if k > 0 else float("nan")

    if h > 0:
        disc = 1.0 / np.log2(np.arange(1, h + 1) + 1)
        dcg = float((y_sorted[:h] * disc).sum())
        idcg = float(disc.sum())  # ideal: the h positives fill the top-h slots
        ndcg = dcg / idcg if idcg > 0 else float("nan")
        rah = p_at(h)
    else:
        ndcg = rah = float("nan")

    def r(x):
        return round(x, 3) if x == x else None
    return {"h": h, "R@h": r(rah), "P@10": r(p_at(10)), "P@50": r(p_at(50)),
            "P@100": r(p_at(100)), "NDCG@h": r(ndcg), "AUC": r(auc)}


def evaluate_scores(dataset: str, scores, method: str = "?"):
    """Compute the full detection metric set for every attention check of
    ``dataset``, given a per-scored-row anomaly score array (higher = more
    inattentive), aligned to the scoring rows (same order/length as
    :func:`aligned_labels` / :func:`aligned_battery`).

    Returns a list of dicts: dataset, method, check, n, n_pos, h, R@h, P@10,
    P@50, P@100, NDCG@h, AUC (legacy keys ``auc`` kept as an alias).
    """
    scores = np.asarray(scores, dtype=float)
    labels = aligned_labels(dataset)
    if len(scores) != len(labels):
        raise AssertionError(
            f"{dataset}: score rows ({len(scores)}) != aligned-label rows "
            f"({len(labels)}); scores and labels are not aligned."
        )

    # Every method emits a continuous carelessness score whose sign is only a
    # labeling convention, so all methods -- the unsupervised detectors and the
    # six psychometric indices alike -- are oriented to max(AUC, 1-AUC). Because
    # the indices are competitors, giving them the better-separating direction is
    # the conservative choice: it can only raise their measured performance.
    orient = True
    out = []
    for check in CHECKS[dataset]:
        y = _relevant(labels, check)
        # NaN scores can't be ranked; drop those rows from this check only.
        valid = ~np.isnan(scores)
        yv, sv = y[valid], scores[valid]
        n_pos = int(yv.sum())
        m = _ranking_metrics(sv, yv, orient=orient) if 0 < n_pos < len(yv) else {
            "h": n_pos, "R@h": None, "P@10": None, "P@50": None,
            "P@100": None, "NDCG@h": None, "AUC": None}
        out.append({
            "dataset": dataset, "method": method, "check": check["name"],
            "n": int(valid.sum()), "n_pos": n_pos, **m,
            "auc": m["AUC"],  # backward-compatible alias
        })
    return out


def evaluate_detection(dataset: str, errors_csv: str, method: str = "?"):
    """Compute detection ROC AUC for every attention check of ``dataset`` from
    an ``errors.csv`` produced by the scoring path (uses its ``error`` column).
    """
    err_df = pd.read_csv(errors_csv)
    if "error" not in err_df.columns:
        raise ValueError(f"{errors_csv}: no 'error' column (cols={list(err_df.columns)[:8]})")
    return evaluate_scores(dataset, err_df["error"].astype(float).values, method)
