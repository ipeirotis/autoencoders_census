"""
Build the full Experiment-B detection table: tuned base AE (p=100), tuned PL AE
(p=85), Linear AE (PCA, fixed preprocessing), Chow-Liu (fixed), and the six
psychometric baselines, each scored by the identical attention-check harness
with the paper's full metric set (h, R@h, P@10/50/100, NDCG@h, AUC).

Outputs:
  experiment_b_full.csv         long format, every method x dataset x check
  experiment_b_latex.txt        per-dataset LaTeX rows ready to paste

Run after run_tuning.sh + (per dataset) finalize_models.py.
Usage: python final_table.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils import define_necessary_elements
from dataset.loader import DataLoader
from main import prepare_for_model
from evaluate.detection import aligned_battery, evaluate_scores, evaluate_detection, CHECKS
from evaluate import baselines as bl

DATASETS = ["attention_check", "inattentive", "racial_data", "moral_data",
            "mturk_ethics", "bot_bot_mturk", "public_opinion", "pennycook_1"]

# Display order / labels for the paper rows.
METHOD_ORDER = [
    ("Non-Linear Autoencoder ($p=100$)", "AE_p100"),
    ("Non-Linear Autoencoder ($p=85$)", "AE_p85"),
    ("Linear Autoencoder", "LinearAE"),
    ("Chow-Liu", "ChowLiu"),
    ("Longstring", "longstring"),
    ("IRV", "irv"),
    ("Person-Total $r$", "person_total_r"),
    ("Mahalanobis-$D$", "mahalanobis"),
    ("Even-Odd", "even_odd"),
    ("Person-Fit $l_z$", "lz"),
]
METRIC_COLS = ["h", "R@h", "P@10", "P@50", "P@100", "NDCG@h", "AUC"]


def linear_ae_scores(dataset):
    """PCA reconstruction error on the fixed-preprocessing one-hot matrix."""
    dc, rc, ic, adc, arc, aic = define_necessary_elements(dataset, None, None, None)
    loader = DataLoader(dc, rc, ic, additional_drop_columns=adc,
                        additional_rename_columns=arc, additional_columns_of_interest=aic)
    data, meta = loader.load_data(dataset)
    _, vectorized, _, _ = prepare_for_model(data, meta.get("variable_types", {}))
    X = vectorized.to_numpy(dtype=float)
    k = min(10, X.shape[1] - 1)
    pca = PCA(n_components=k).fit(X)
    recon = pca.inverse_transform(pca.transform(X))
    return ((X - recon) ** 2).mean(axis=1)


def errors_path(dataset, key):
    return {
        "AE_p100": f"cache/_tuned_{dataset}_ae_p100/errors.csv",
        "AE_p85": f"cache/_tuned_{dataset}_ae_p85/errors.csv",
        # Linear AE = the main AutoencoderModel made 0-layer/linear (softmax
        # output, CE loss), via experiment_b/linear_ae.py -- NOT the MSE
        # LinearAutoencoder and NOT sklearn PCA.
        "LinearAE": f"cache/_lin0_{dataset}/errors.csv",
        "ChowLiu": f"cache/_fix9_{dataset}_cl/errors.csv",
    }.get(key)


def rows_for(dataset, key):
    if key in ("AE_p100", "AE_p85", "LinearAE", "ChowLiu"):
        p = errors_path(dataset, key)
        return evaluate_detection(dataset, p, key) if p and os.path.exists(p) else None
    # psychometric baseline
    scores = bl.INDICES[key](aligned_battery(dataset))
    return evaluate_scores(dataset, scores, key) if np.isfinite(scores).any() else None


def main():
    long_rows = []
    latex = []
    for ds in DATASETS:
        per_method = {}
        for label, key in METHOD_ORDER:
            r = rows_for(ds, key)
            if r:
                per_method[key] = {row["check"]: row for row in r}
                long_rows.extend(r)
        # LaTeX block per check
        for check in CHECKS[ds]:
            cname = check["name"]
            latex.append(f"% {ds} / {cname}")
            for label, key in METHOD_ORDER:
                row = per_method.get(key, {}).get(cname)
                if not row:
                    continue
                cells = " & ".join(
                    ("--" if row.get(c) is None else f"{row[c]:.2f}" if c != "h" else f"{row[c]}")
                    for c in METRIC_COLS)
                latex.append(f"\\quad {label} & {cells} \\\\")
            latex.append("")

    df = pd.DataFrame(long_rows)
    df.to_csv("experiment_b_full.csv", index=False)
    open("experiment_b_latex.txt", "w").write("\n".join(latex))

    # Console summary: AUC per method x dataset (primary check)
    PRIMARY = {"attention_check": "Attention_Check", "inattentive": "filter",
               "racial_data": "attn1", "moral_data": "attention",
               "mturk_ethics": "Screener_One", "bot_bot_mturk": "Q6_15",
               "public_opinion": "attention_1", "pennycook_1": "AC1(screen1)"}
    print(f"\n{'method':24s}" + "".join(f"{d[:9]:>10}" for d in DATASETS))
    for label, key in METHOD_ORDER:
        cells = ""
        for ds in DATASETS:
            sub = df[(df.dataset == ds) & (df.method == key) & (df.check == PRIMARY[ds])]
            v = sub["AUC"].iloc[0] if len(sub) and pd.notna(sub["AUC"].iloc[0]) else None
            cells += f"{v:>10.3f}" if v is not None else f"{'--':>10}"
        print(f"{label[:24]:24s}{cells}")
    print(f"\nSaved experiment_b_full.csv ({len(df)} rows) + experiment_b_latex.txt")


if __name__ == "__main__":
    main()
