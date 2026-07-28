"""
Recompute the dataset-level and method-specific correlation tables of the
Analysis section, using the CORRECTED detection AUCs (median-of-seeds AE, union
rows; from the regenerated detection tables) and the CORRECTED reconstruction
Lift (experiment_b.recon_grid -> recon_grid.csv).

This supersedes the hard-coded rows in run_stats.py / correlation_analysis_v2.py,
which were computed before the binning fix. Run after recon_grid.py:

    python -m experiment_b.recompute_stats
Out: recompute_stats.txt  (numbers ready to transcribe into analysis.tex)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats

# Canonical dataset order (matches recon_grid.py and the paper's data_stats).
KEYS = ["robinson2014", "pennycook2020", "alvarez2019", "uhalt2020",
        "ogrady2019", "buchanan2018", "moss2023", "mastroianni2022", "ivanov2021"]
# As reported in the paper's data_stats table (original, broader loader config).
CHAR_PAPER = pd.DataFrame({
    "Samples":   [14765, 853, 2725, 308, 355, 1038, 2277, 1036, 860],
    "Variables": [98, 188, 39, 60, 72, 23, 51, 51, 67],
    "Features":  [619, 708, 196, 337, 322, 159, 332, 322, 310],
    "AFV":       [6.32, 3.75, 5.03, 5.62, 4.47, 6.91, 6.51, 6.31, 4.63],
}, index=KEYS)
# Actually modeled by the reproducing (modularized) loader; differs for 5
# datasets (pennycook/moss severe) due to narrower interest-column configs.
CHAR_NOW = pd.DataFrame({
    "Samples":   [14765, 212, 2725, 308, 355, 1038, 2277, 1036, 860],
    "Variables": [98, 92, 37, 60, 72, 20, 35, 49, 67],
    "Features":  [619, 323, 173, 337, 325, 135, 112, 222, 310],
    "AFV":       [6.32, 3.51, 4.68, 5.62, 4.51, 6.75, 3.20, 4.53, 4.63],
}, index=KEYS)
CHAR = CHAR_NOW  # used by the method-specific block; dataset-level reports both

# Corrected detection AUC, union row where multiple checks exist, from the
# regenerated Tables (AE rows = median over 5 seeds; Linear/CL single run).
# Corrected: 0-layer linear AE (not MSE class), unsupervised AUC symmetry,
# pennycook full battery. Union row where multiple checks exist.
AUC = pd.DataFrame({
    "AE_p100": [0.71, 0.51, 0.80, 0.64, 0.57, 0.57, 0.69, 0.69, 0.66],
    "AE_p85":  [0.74, 0.52, 0.78, 0.81, 0.66, 0.70, 0.71, 0.70, 0.67],
    "Linear":  [0.75, 0.62, 0.78, 0.72, 0.53, 0.60, 0.65, 0.70, 0.63],
    "ChowLiu": [0.75, 0.61, 0.83, 0.87, 0.77, 0.65, 0.73, 0.68, 0.74],
}, index=KEYS)


def _ps(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)  # drop datasets with a missing value
    x, y = x[m], y[m]
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    return (round(float(pr.statistic), 2), round(float(pr.pvalue), 3),
            round(float(sr.statistic), 2), round(float(sr.pvalue), 3))


def _fmt(t):
    r, rp, rho, rhop = t
    star_r = "*" if rp < 0.05 else ""
    star_s = "*" if rhop < 0.05 else ""
    return f"r={r:+.2f}{star_r} (p={rp:.3f}); rho={rho:+.2f}{star_s} (p={rhop:.3f})"


def main():
    recon = pd.read_csv("recon_grid.csv").set_index("dataset_key").reindex(KEYS)
    # Linear reconstruction Lift now comes from the 0-layer linear AE
    # (experiment_b.linear_ae), keyed by internal name -> citation key.
    i2c = {"sadc_2017": "robinson2014", "pennycook_1": "pennycook2020", "inattentive": "alvarez2019",
           "attention_check": "uhalt2020", "moral_data": "ogrady2019", "bot_bot_mturk": "buchanan2018",
           "mturk_ethics": "moss2023", "public_opinion": "mastroianni2022", "racial_data": "ivanov2021"}
    ll = pd.read_csv("linear_ae_lift.csv")
    ll["k"] = ll["dataset_key"].map(i2c)
    linlift = ll.set_index("k")["Lift"].reindex(KEYS)
    lift = pd.DataFrame({
        "AE_p100": recon["AE_NL"].values,
        "AE_p85":  recon["AE_PL"].values,
        "Linear":  linlift.values,
    }, index=KEYS)
    MeanAUC = AUC.mean(axis=1)
    MeanLift = lift.mean(axis=1)

    out = []
    out.append("=== per-dataset summary (MeanAUC corrected; MeanLift corrected) ===")
    summ = CHAR_NOW.copy()
    summ["MeanAUC"] = MeanAUC.round(3)
    summ["MeanLift"] = MeanLift.round(3)
    out.append(summ.to_string())

    for label, CH in (("PAPER characteristics (data_stats as reported)", CHAR_PAPER),
                      ("CORRECTED characteristics (actually modeled)", CHAR_NOW)):
        out.append(f"\n=== DATASET-LEVEL with {label} ===")
        out.append(f"{'Predictor':10s} | vs MeanAUC                              | vs MeanLift")
        for p in ["Samples", "Variables", "Features", "AFV"]:
            a = _ps(CH[p].values, MeanAUC.values)
            l = _ps(CH[p].values, MeanLift.values)
            out.append(f"{p:10s} | {_fmt(a):38s} | {_fmt(l)}")
        out.append(f"{'MeanLift':10s} | {_fmt(_ps(MeanLift.values, MeanAUC.values))}")

    out.append("\n=== METHOD-SPECIFIC (Table tab:method_correlations) ===")
    out.append("AUC vs characteristics (Pearson;Spearman) + AUC<->Lift:")
    for m in ["AE_p100", "AE_p85", "Linear", "ChowLiu"]:
        cells = []
        for p in ["Samples", "Variables", "Features", "AFV"]:
            r, rp, rho, rhop = _ps(CHAR[p].values, AUC[m].values)
            cells.append(f"{p[:4]}:{r:+.2f};{rho:+.2f}")
        if m in lift.columns:
            r, rp, rho, rhop = _ps(AUC[m].values, lift[m].values)
            cells.append(f"AUCxLift:{r:+.2f};{rho:+.2f}" + ("*" if rp < 0.05 else ""))
        else:
            cells.append("AUCxLift: n/a (no reconstruction)")
        out.append(f"  {m:9s} " + "  ".join(cells))
    out.append("Lift vs characteristics (Pearson;Spearman):")
    for m in ["AE_p100", "AE_p85", "Linear"]:
        cells = []
        for p in ["Samples", "Variables", "Features", "AFV"]:
            r, rp, rho, rhop = _ps(CHAR[p].values, lift[m].values)
            cells.append(f"{p[:4]}:{r:+.2f};{rho:+.2f}")
        out.append(f"  {m:9s} " + "  ".join(cells))

    # ---- paste-ready LaTeX rows for tab:method_correlations ----
    def cell(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
        pr = stats.pearsonr(x, y); sr = stats.spearmanr(x, y)
        r = f"{pr.statistic:.2f}" + ("^{\\ast}" if pr.pvalue < 0.05 else "")
        rho = f"{sr.statistic:.2f}" + ("^{\\ast}" if sr.pvalue < 0.05 else "")
        return f"  & ${r};\\;{rho}$"
    labels = {"AE_p100": "Non-Linear AE ($p=100$)", "AE_p85": "Non-Linear AE ($p=85$)",
              "Linear": "Linear AE", "ChowLiu": "Chow--Liu Tree"}
    out.append("\n=== LaTeX rows for tab:method_correlations ===")
    for m in ["AE_p100", "AE_p85", "Linear", "ChowLiu"]:
        row = [labels[m]]
        for p in ["Samples", "Variables", "Features", "AFV"]:
            row.append(cell(CHAR_NOW[p].values, AUC[m].values))
        if m in lift.columns:
            row.append(cell(AUC[m].values, lift[m].values))        # AUC<->Lift
            for p in ["Samples", "Variables", "Features", "AFV"]:
                row.append(cell(CHAR_NOW[p].values, lift[m].values))
        else:
            row.append("  & \\multicolumn{1}{c}{---}")
            row.append("  & \\multicolumn{4}{c}{(no reconstruction metric)}")
        out.append(row[0] + "\n" + "\n".join(row[1:]) + "\n\\\\")

    # dataset_summary.csv for the scatter-plot regeneration (make_plots.py).
    pd.DataFrame({"dataset_key": KEYS, "Samples": CHAR["Samples"].values,
                  "Variables": CHAR["Variables"].values, "Features": CHAR["Features"].values,
                  "AFV": CHAR["AFV"].values, "MeanAUC": MeanAUC.round(4).values,
                  "MeanLift": MeanLift.round(4).values}).to_csv("dataset_summary_corrected.csv", index=False)

    text = "\n".join(out)
    open("recompute_stats.txt", "w").write(text + "\n")
    print(text)
    print("\nwrote recompute_stats.txt + dataset_summary_corrected.csv")


if __name__ == "__main__":
    main()
