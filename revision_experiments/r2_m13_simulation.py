"""
r2_m13_simulation -- Psychometric-ML Alignment as a CONTROLLED simulation
(referee M13). The paper's claim "instrument coherence drives detectability" rests
on n=9 observational correlations. Here we vary battery coherence while HOLDING
sample size and contamination fixed, and show detection AUC rises with coherence
at essentially zero data cost. Detectors: the paper's Chow-Liu (chow_liu_rank) and
a PCA-linear reconstruction detector (stands in for the linear AE; no TensorFlow).

Data-generating model: attentive respondents share a latent trait theta~N(0,1);
item j response = quantile-binned (loading*sign_j*theta + sqrt(1-loading^2)*noise)
into n_cats categories, so `loading` tunes inter-item correlation (coherence) and
`n_items` tunes battery breadth. Inattentive respondents (fixed fraction) answer
uniformly at random (random-style invalidity). Higher coherence => attentive rows
compress better and random rows stand out more.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from chow_liu_rank import CLTree


def make_data(n, n_items, n_cats, loading, contamination, seed, reverse_frac=0.0):
    rng = np.random.default_rng(seed)
    n_in = int(round(contamination * n)); n_at = n - n_in
    theta = rng.standard_normal(n_at)
    signs = np.ones(n_items)
    if reverse_frac > 0:
        k = int(round(reverse_frac * n_items))
        signs[rng.choice(n_items, k, replace=False)] = -1.0
    att = np.zeros((n_at, n_items), int)
    for j in range(n_items):
        lat = loading * signs[j] * theta + np.sqrt(1 - loading ** 2) * rng.standard_normal(n_at)
        att[:, j] = np.clip((rankdata(lat) - 1) / n_at * n_cats, 0, n_cats - 1).astype(int)
    inatt = rng.integers(0, n_cats, size=(n_in, n_items))
    X = np.vstack([att, inatt]); y = np.r_[np.zeros(n_at, int), np.ones(n_in, int)]
    perm = rng.permutation(n); X, y = X[perm], y[perm]
    df = pd.DataFrame({f"q{j}": [f"c{v}" for v in X[:, j]] for j in range(n_items)})
    return df, y


def auc_cl(df, y):
    cl = CLTree(alpha=1.0).fit(df, random_state=0)
    logp = cl.score_dataframe(df)["logp"].to_numpy()
    return roc_auc_score(y, -logp)  # low logp = atypical = inattentive


def auc_pca(df, y, k=5):
    X = pd.get_dummies(df).to_numpy(float); Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vk = Vt[:min(k, Vt.shape[0])]
    err = ((Xc - Xc @ Vk.T @ Vk) ** 2).sum(1)
    return roc_auc_score(y, err)


def mean_ci(vals):
    v = np.asarray(vals); return v.mean(), v.std(ddof=1)


def sweep(param, values, base, seeds=12):
    rows = []
    for val in values:
        kw = dict(base); kw[param] = val
        cl, pca = [], []
        for s in range(seeds):
            df, y = make_data(seed=100 + s, **kw)
            cl.append(auc_cl(df, y)); pca.append(auc_pca(df, y))
        cm, cs = mean_ci(cl); pm, ps = mean_ci(pca)
        rows.append((val, cm, cs, pm, ps))
        print(f"  {param}={val!s:>5}  CL AUC={cm:.3f}±{cs:.3f}   PCA-linear AUC={pm:.3f}±{ps:.3f}")
    return rows


if __name__ == "__main__":
    base = dict(n=800, n_items=12, n_cats=5, loading=0.6, contamination=0.15)
    print("Fixed: N=800, contamination=15%, n_cats=5, 12 seeds each.\n")
    print("=== SWEEP 1: coherence = factor loading (n_items=12 fixed) ===")
    r1 = sweep("loading", [0.15, 0.30, 0.45, 0.60, 0.75, 0.90], base)
    print("\n=== SWEEP 2: coherence = battery breadth (loading=0.6 fixed) ===")
    r2 = sweep("n_items", [4, 8, 12, 20, 32], base)
    print("\n=== SWEEP 3: reverse-keyed fraction (loading=0.6, n_items=12) ===")
    r3 = sweep("reverse_frac", [0.0, 0.25, 0.5], base)
    lo, hi = r1[0][1], r1[-1][1]
    print(f"\nHEADLINE: at fixed N=800 & 15% contamination, Chow-Liu detection AUC rises "
          f"from {lo:.2f} (loading 0.15) to {hi:.2f} (loading 0.90); "
          f"PCA-linear from {r1[0][3]:.2f} to {r1[-1][3]:.2f}. "
          f"Battery breadth: CL {r2[0][1]:.2f} (4 items) -> {r2[-1][1]:.2f} (32 items). "
          f"Coherence drives detectability with N and contamination held fixed.")
