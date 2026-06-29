"""
Regenerate the paper's three detailed per-check detection tables (Parts I-III)
with the corrected preprocessing + tuning. Autoencoder rows use the MEDIAN across
the multi-seed runs (experiment_b.multiseed_ae) to remove single-run variance;
Linear AE and Chow-Liu are single runs. robinson2014 uses the SADC composite
indicator. Bolds the best method per metric column within each check, matching
the paper's formatting.

Run from the repo root: python -m experiment_b.build_detailed_tables
Out: experiment_b_detailed_tables.tex (paste over the three table environments)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from utils import define_necessary_elements
from dataset.loader import DataLoader
from evaluate.detection import evaluate_detection, _ranking_metrics

METHODS = [("AE_p100", "Non-Linear Autoencoder ($p=100$)"),
           ("AE_p85", "Non-Linear Autoencoder ($p=85$)"),
           ("Lin", "Linear Autoencoder"),
           ("CL", "Chow-Liu")]
BOLD_COLS = ["R@h", "P@10", "P@50", "P@100", "NDCG@h", "AUC"]

# Three tables, matching the paper's Parts. Each entry: (citation, dataset key,
# [(internal-check, sub-label or None)]).
PART1 = [
    ("robinson2014", "sadc_2017", [("__sadc__", None)]),
    ("pennycook2020", "pennycook_1", [("AC1(screen1)", "Attention 1"), ("AC2(screen2)", "Attention 2"),
        ("AC3(screen3)", "Attention 3"), ("AC4(Random)", "Attention 4"),
        ("union(any AC)", "Union"), ("intersection(all AC)", "Intersection")]),
    ("alvarez2019", "inattentive", [("filter", None)]),
]
PART2 = [
    ("uhalt2020", "attention_check", [("Attention_Check", None)]),
    ("ogrady2019", "moral_data", [("attention", None)]),
    ("buchanan2018", "bot_bot_mturk", [("Q6_15", None)]),
    ("moss2023", "mturk_ethics", [("Screener_One", "Attention 1"), ("Screener_Two", "Attention 2"),
        ("union(S1|S2)", "Union"), ("intersection(S1&S2)", "Intersection")]),
]
PART3 = [
    ("mastroianni2022", "public_opinion", [("attention_1", None)]),
    ("ivanov2021", "racial_data", [("attn1", "Attention 1"), ("attn2", "Attention 2"),
        ("union(attn1|attn2)", "Union"), ("intersection(attn1&attn2)", "Intersection")]),
]

_SADC_Y = None


def _sadc_labels():
    global _SADC_Y
    if _SADC_Y is None:
        dc, rc, ic, adc, arc, aic = define_necessary_elements("sadc_2017", None, None, None)
        L = DataLoader(dc, rc, ic, additional_drop_columns=adc, additional_rename_columns=arc,
                       additional_columns_of_interest=aic)
        _SADC_Y = L.find_outlier_data_sadc_2017("sadc_2017", ["outlier"])["outlier"].values.astype(int)
    return _SADC_Y


def _metrics_from_errors(path, dataset, check):
    if not os.path.exists(path):
        return None
    if dataset == "sadc_2017":
        y = _sadc_labels()
        e = pd.read_csv(path)["error"].astype(float).values
        return _ranking_metrics(e, y) if len(e) == len(y) else None
    rows = evaluate_detection(dataset, path, "x")
    return next((r for r in rows if r["check"] == check), None)


def metrics_for(dataset, key, check):
    """Median over seeds for the AE; single run for Linear/CL. robinson2014 uses
    the cached single-run AE (it reproduces) since it is not multi-seeded."""
    if key in ("AE_p100", "AE_p85"):
        p = 100 if key == "AE_p100" else 85
        if dataset == "sadc_2017":
            return _metrics_from_errors(f"cache/sadc_2017_{p}perc_newloss/errors.csv", dataset, check)
        per_seed = [_metrics_from_errors(f"cache/_ms_{dataset}_p{p}_s{s}/errors.csv", dataset, check)
                    for s in range(1, 6)]
        per_seed = [m for m in per_seed if m]
        if not per_seed:
            return None
        med = {"h": per_seed[0]["h"]}
        for c in BOLD_COLS:
            vals = [m[c] for m in per_seed if m.get(c) is not None]
            med[c] = round(float(np.median(vals)), 2) if vals else None
        return med
    if key == "Lin":
        return _metrics_from_errors(f"cache/_lin_{dataset}/errors.csv", dataset, check)
    if key == "CL":
        return _metrics_from_errors(f"cache/_fix9_{dataset}_cl/errors.csv", dataset, check)


def _row(label, m, best, indent):
    cells = [str(m.get("h", ""))]
    for c in BOLD_COLS:
        v = m.get(c)
        if v is None:
            cells.append("--")
        else:
            s = f"{v:.2f}"
            cells.append(f"\\textbf{{{s}}}" if best.get(c) is not None and abs(v - best[c]) < 1e-9 else s)
    return f"{indent} {label} & " + " & ".join(cells) + " \\\\"


def dataset_block(cite, dataset, checks):
    lines = [f"\\cite{{{cite}}} &&&& \\\\"]
    for chk, sub in checks:
        indent = "\\quad \\quad" if sub else "\\quad"
        if sub:
            lines.append(f"\\quad {sub} &&&&& \\\\")
        rows = {key: metrics_for(dataset, key, chk) for key, _ in METHODS}
        best = {}
        for c in BOLD_COLS:
            vals = [m[c] for m in rows.values() if m and m.get(c) is not None]
            best[c] = max(vals) if vals else None
        for key, label in METHODS:
            if rows[key]:
                lines.append(_row(label, rows[key], best, indent))
    lines.append("\\addlinespace")
    return lines


HEADER = (r"""\begin{table}
\TABLE
{Randomness Detection results across datasets and methods (%s) \label{%s}}
{\begin{tabular}{@{}l@{\quad}c@{\quad}c@{\quad}c@{\quad}c@{\quad}c@{\quad}c@{\quad}c@{}}
\textbf{Dataset \& Method} & \textbf{h} & \textbf{R@h} & \textbf{P@10} & \textbf{P@50} & \textbf{P@100} & \textbf{NDCG@h} & \textbf{AUC} \\
\hline\up""")
FOOTER = (r"""\end{tabular}}{Autoencoder rows report the median over five random seeds (tuned architecture, corrected preprocessing); Linear AE and Chow-Liu are single runs. Best per metric in bold. $^\dagger$\cite{robinson2014} uses the composite mischievous-responder indicator.}
\end{table}""")


def build(part, name, label):
    lines = [HEADER % (name, label)]
    for cite, ds, checks in part:
        lines += dataset_block(cite, ds, checks)
    lines.append(FOOTER)
    return "\n".join(lines)


def main():
    out = "\n\n".join([
        build(PART1, "Part I", "tab:randomness_det1"),
        build(PART2, "Part II", "tab:randomness_det2"),
        build(PART3, "Part III", "tab:randomness_det3"),
    ])
    open("experiment_b_detailed_tables.tex", "w").write(out + "\n")
    print("wrote experiment_b_detailed_tables.tex")


if __name__ == "__main__":
    main()
