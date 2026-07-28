"""
Build the Percentile-Loss trade-off figures + paired-test numbers from the
regenerated sweep (experiment_b.pl_sweep -> pl_sweep_detection.csv /
pl_sweep_recon.csv), on the corrected preprocessing. Replaces the hard-coded
deltas in tradeoff_plots.py. Writes the nine boxswarm_delta_*.pdf figures
directly into the paper's figures/ directory and prints the mean delta + exact
Wilcoxon p per metric per percentile for the analysis text.

Run from the repo root after pl_sweep.py:  python -m experiment_b.pl_tradeoff_figs
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/Users/iliastriantafyllopoulos/Documents/my_projects/inattentiveness_paper/figures"
PCTS = [80, 85, 90, 95]
DET = [("AUC", "AUC"), ("NDCG@h", "NDCGath"), ("P@10", "Pat10"),
       ("P@50", "Pat50"), ("P@100", "Pat100"), ("R@h", "Rath")]
REC = ["Accuracy", "Lift", "ORA"]


def deltas(df, group_cols, metric):
    """metric(p) - metric(100), one row per group x p in {80,85,90,95}."""
    out = []
    for _, g in df.groupby(group_cols):
        base = g[g["Percentile"] == 100][metric]
        if base.empty or pd.isna(base.iloc[0]):
            continue
        b = float(base.iloc[0])
        for p in PCTS:
            v = g[g["Percentile"] == p][metric]
            if not v.empty and pd.notna(v.iloc[0]):
                out.append({"Percentile": p, "delta": float(v.iloc[0]) - b})
    return pd.DataFrame(out)


def _offsets(y, sep=0.02, step=0.02, iters=200):
    y = np.asarray(y, float); off = np.zeros_like(y); order = np.argsort(y)
    for _ in range(iters):
        moved = False
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                i, j = order[a], order[b]
                if abs(y[i] - y[j]) < sep and abs(off[i] - off[j]) < step:
                    off[i] -= step / 2; off[j] += step / 2; moved = True
        if not moved:
            break
    return off


def boxplot(dl, metric, outfile):
    data = [dl[dl["Percentile"] == p]["delta"].dropna().values for p in PCTS]
    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=[str(p) for p in PCTS], showmeans=True)
    plt.axhline(0.0, linestyle="--", color="gray")
    for i, p in enumerate(PCTS, start=1):
        ys = dl[dl["Percentile"] == p]["delta"].dropna().values
        if len(ys):
            plt.plot(np.full_like(ys, i, dtype=float) + _offsets(ys), ys,
                     marker="o", linestyle="None", alpha=0.8)
    plt.xlabel("Percentile threshold $p$"); plt.ylabel(f"$\\Delta${metric} vs $p{{=}}100$")
    plt.grid(True, axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight"); plt.close()


def wilcoxon_zero(diffs):
    d = np.asarray([x for x in diffs if x != 0.0], float); n = d.size
    if n == 0:
        return 1.0, 0
    ad = np.abs(d); order = np.argsort(ad); ranks = np.empty(n)
    i = 0; r = 1
    while i < n:
        j = i
        while j + 1 < n and ad[order[j + 1]] == ad[order[i]]:
            j += 1
        avg = (r + r + (j - i)) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        r += (j - i + 1); i = j + 1
    Wp = float(np.sum(ranks[d > 0])); S = float(np.sum(ranks))
    sums = Counter({0.0: 1})
    for rk in ranks:
        new = Counter()
        for s, c in sums.items():
            new[s + rk] += c; new[s - rk] += c
        sums = new
    dev = abs(Wp - S / 2.0)
    extreme = sum(c for s, c in sums.items() if abs((s + S) / 2.0 - S / 2.0) >= dev)
    return extreme / (2 ** n), n


def main():
    det = pd.read_csv("pl_sweep_detection.csv")
    rec = pd.read_csv("pl_sweep_recon.csv")
    summary = []
    for col, suf in DET:
        dl = deltas(det, ["Dataset", "Subgroup"], col)
        boxplot(dl, col, f"{FIG}/boxswarm_delta_random_{suf}.pdf")
        for p in PCTS:
            s = dl[dl["Percentile"] == p]["delta"].dropna().values
            pv, n = wilcoxon_zero(s)
            summary.append(("DET", col, p, round(float(np.mean(s)), 3) if len(s) else None,
                            round(pv, 4), n))
    for col in REC:
        dl = deltas(rec, ["Dataset"], col)
        boxplot(dl, col, f"{FIG}/boxswarm_delta_{col}.pdf")
        for p in PCTS:
            s = dl[dl["Percentile"] == p]["delta"].dropna().values
            pv, n = wilcoxon_zero(s)
            summary.append(("REC", col, p, round(float(np.mean(s)), 3) if len(s) else None,
                            round(pv, 4), n))
    sdf = pd.DataFrame(summary, columns=["kind", "metric", "p", "mean_delta", "wilcoxon_p", "n"])
    sdf.to_csv("pl_tradeoff_tests.csv", index=False)
    print(sdf.to_string(index=False))
    print("\nFigures written to", FIG)


if __name__ == "__main__":
    main()
