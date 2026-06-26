"""
Practitioner psychometric careless-responding baselines (Experiment B).

Each index maps a respondent's item-response vector to a single score, oriented
so that **higher = more careless / inattentive** (the same orientation the
autoencoder reconstruction error and Chow-Liu anomaly score use), so all
methods are scored by the identical detection harness
(:func:`evaluate.detection.evaluate_scores`) against attention-check labels.

Indices implemented (see Curran 2016; Meade & Craig 2012; Johnson 2005;
Drasgow et al. 1985 for l_z):

- ``longstring``        : longest run of identical consecutive responses
                          (straight-lining). High = careless.
- ``irv``               : inter-item response variability (within-person SD).
                          Straight-liners have LOW variability, so the score is
                          ``-IRV``. High score = careless.
- ``person_total_r``    : person-total correlation -- how well a respondent
                          tracks the item means. Careless responders track
                          weakly, so the score is ``-r``.
- ``mahalanobis``       : Mahalanobis distance from the sample mean in
                          standardized item space (multivariate outlyingness).
                          High = careless.
- ``even_odd``          : even-odd consistency -- within-person agreement
                          between even- and odd-indexed items. Low consistency
                          = careless, so the score is ``-consistency``.
- ``lz``                : standardized IRT person-fit l_z on dichotomized items
                          (2PL). Misfit is negative l_z, so the score is
                          ``-l_z``. Computed only when IRT calibration is
                          feasible (enough items and respondents); otherwise the
                          score array is all-NaN and the index is skipped.

All functions take a numeric battery ``DataFrame`` ``B`` (rows = respondents in
scoring-row order, columns = items) that may contain NaN, and return a float
array of length ``len(B)`` (NaN where the index is undefined for that row).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def longstring(B: pd.DataFrame) -> np.ndarray:
    X = B.to_numpy(dtype=float)
    n, p = X.shape
    out = np.full(n, np.nan)
    for i in range(n):
        row = X[i]
        best = cur = 0
        prev = np.nan
        for v in row:
            if np.isnan(v):
                cur = 0
                prev = np.nan
                continue
            if v == prev:
                cur += 1
            else:
                cur = 1
                prev = v
            best = max(best, cur)
        if best > 0:
            out[i] = best
    return out


def irv(B: pd.DataFrame) -> np.ndarray:
    # within-person SD; low SD = straight-lining -> negate so high = careless
    sd = B.std(axis=1, ddof=0, skipna=True).to_numpy(dtype=float)
    sd[B.notna().sum(axis=1).to_numpy() < 2] = np.nan
    return -sd


def person_total_r(B: pd.DataFrame) -> np.ndarray:
    X = B.to_numpy(dtype=float)
    item_mean = np.nanmean(X, axis=0)  # normative profile
    n = X.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        m = ~np.isnan(X[i])
        if m.sum() < 3:
            continue
        a, b = X[i][m], item_mean[m]
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            # no within-person variance (e.g. perfect straight-line): treat as
            # maximally inconsistent with the normative profile.
            out[i] = -1.0
            continue
        out[i] = np.corrcoef(a, b)[0, 1]
    return -out


def mahalanobis(B: pd.DataFrame) -> np.ndarray:
    X = B.to_numpy(dtype=float)
    if X.shape[1] < 2 or X.shape[0] < 3:
        return np.full(X.shape[0], np.nan)
    # impute item means, then standardize so no single item dominates
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    Ximp = X.copy()
    Ximp[inds] = np.take(col_mean, inds[1])
    sd = Ximp.std(axis=0, ddof=0)
    keep = sd > 1e-9
    if int(keep.sum()) < 2:
        return np.full(X.shape[0], np.nan)
    Z = (Ximp[:, keep] - Ximp[:, keep].mean(axis=0)) / sd[keep]
    cov = np.cov(Z, rowvar=False)
    inv = np.linalg.pinv(cov)  # pseudo-inverse: robust to collinear/singular
    d2 = np.einsum("ij,jk,ik->i", Z, inv, Z)
    return np.sqrt(np.clip(d2, 0, None))


def even_odd(B: pd.DataFrame) -> np.ndarray:
    X = B.to_numpy(dtype=float)
    even, odd = X[:, 0::2], X[:, 1::2]
    w = min(even.shape[1], odd.shape[1])
    even, odd = even[:, :w], odd[:, :w]
    n = X.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        m = ~np.isnan(even[i]) & ~np.isnan(odd[i])
        if m.sum() < 3:
            continue
        a, b = even[i][m], odd[i][m]
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            out[i] = -1.0
            continue
        out[i] = np.corrcoef(a, b)[0, 1]
    return -out


def lz(B: pd.DataFrame, min_items: int = 8, min_n: int = 200) -> np.ndarray:
    """Standardized IRT person-fit l_z on median-dichotomized items under a 2PL
    fitted by marginal/joint MLE-lite (alternating logistic regressions).

    Returns ``-l_z`` (high = misfit = careless). All-NaN when calibration is
    not feasible (too few items or respondents), so the caller can skip it.
    """
    X = B.to_numpy(dtype=float)
    n, p = X.shape
    if n < min_n or p < min_items:
        return np.full(n, np.nan)

    # dichotomize each item at its median (>= median -> 1)
    med = np.nanmedian(X, axis=0)
    U = (X >= med).astype(float)
    U[np.isnan(X)] = np.nan
    # drop items with no variance after dichotomizing
    keep = [j for j in range(p) if np.nanstd(U[:, j]) > 1e-6]
    U = U[:, keep]
    p = U.shape[1]
    if p < min_items:
        return np.full(n, np.nan)

    mask = ~np.isnan(U)
    Uf = np.where(mask, U, 0.0)

    # Alternating estimation of 2PL params (a_j, b_j) and abilities theta_i.
    theta = (np.nanmean(np.where(mask, U, np.nan), axis=1) - 0.5) * 2.0
    a = np.ones(p)
    b = np.zeros(p)

    def sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    for _ in range(60):
        # update items: logistic regression of U_.j on theta
        for j in range(p):
            mj = mask[:, j]
            t = theta[mj]
            y = U[mj, j]
            aj, bj = a[j], b[j]
            for _ in range(8):
                z = aj * (t - bj)
                pr = sig(z)
                w = np.clip(pr * (1 - pr), 1e-6, None)
                # gradient wrt (aj, c=-aj*bj): logit = aj*t + c
                c = -aj * bj
                g_a = np.sum((y - pr) * t)
                g_c = np.sum(y - pr)
                h_aa = -np.sum(w * t * t)
                h_ac = -np.sum(w * t)
                h_cc = -np.sum(w)
                H = np.array([[h_aa, h_ac], [h_ac, h_cc]])
                grad = np.array([g_a, g_c])
                try:
                    step = np.linalg.solve(H, grad)
                except np.linalg.LinAlgError:
                    break
                aj -= step[0]
                c -= step[1]
                aj = float(np.clip(aj, 0.1, 4.0))
                bj = float(np.clip(-c / aj if aj != 0 else 0.0, -4, 4))
            a[j], b[j] = aj, bj
        # update abilities: Newton on theta_i
        for i in range(n):
            mi = mask[i]
            ai, bi, yi = a[mi], b[mi], U[i, mi]
            th = theta[i]
            for _ in range(10):
                pr = sig(ai * (th - bi))
                w = np.clip(pr * (1 - pr), 1e-6, None)
                g = np.sum(ai * (yi - pr))
                h = -np.sum((ai ** 2) * w)
                if abs(h) < 1e-9:
                    break
                th -= g / h
                th = float(np.clip(th, -5, 5))
            theta[i] = th

    # l_z: standardized log-likelihood (Drasgow, Levine & Williams 1985)
    out = np.full(n, np.nan)
    for i in range(n):
        mi = mask[i]
        if mi.sum() < min_items:
            continue
        ai, bi, yi = a[mi], b[mi], U[i, mi]
        pr = np.clip(sig(ai * (theta[i] - bi)), 1e-6, 1 - 1e-6)
        ll = np.sum(yi * np.log(pr) + (1 - yi) * np.log(1 - pr))
        ell = np.sum(pr * np.log(pr) + (1 - pr) * np.log(1 - pr))
        var = np.sum(pr * (1 - pr) * (np.log(pr / (1 - pr)) ** 2))
        if var <= 1e-9:
            continue
        out[i] = (ll - ell) / np.sqrt(var)
    return -out  # high = misfit = careless


# Registry: name -> function. ``lz`` last (heaviest / may be skipped).
INDICES = {
    "longstring": longstring,
    "irv": irv,
    "person_total_r": person_total_r,
    "mahalanobis": mahalanobis,
    "even_odd": even_odd,
    "lz": lz,
}


def compute_all(B: pd.DataFrame) -> dict:
    """Return ``{index_name: score_array}`` for every baseline (NaN-skipping)."""
    return {name: fn(B) for name, fn in INDICES.items()}
