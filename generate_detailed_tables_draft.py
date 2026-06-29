"""
DRAFT regeneration of the paper's three detailed per-check detection tables with
the tuned/fixed numbers (binning fix + per-dataset hyperparameter search). Does
NOT touch the live tables; writes a standalone .tex draft for review.

Covers 8/9 datasets: the 7 attention-check datasets from experiment_b_full.csv
plus robinson2014 (SADC composite indicator, computed here). mastroianni2022 is
omitted (loader uses 156 vars vs the paper's 51 -> not reproducible yet).

Run: python generate_detailed_tables_draft.py
Out: ../inattentiveness_paper/sections/_regenerated_detailed_tables_draft.tex
"""
import os
import numpy as np
import pandas as pd

from utils import define_necessary_elements
from dataset.loader import DataLoader
from main import prepare_for_model
from evaluate.detection import _ranking_metrics, _to_ordinal
from evaluate import baselines as bl
from sklearn.decomposition import PCA

METHOD_ROWS = [("AE_p100", "Non-Linear Autoencoder ($p=100$)"),
               ("AE_p85", "Non-Linear Autoencoder ($p=85$)"),
               ("LinearAE", "Linear Autoencoder"),
               ("ChowLiu", "Chow-Liu")]
MCOLS = ["R@h", "P@10", "P@50", "P@100", "NDCG@h", "AUC"]

# (dataset, internal check, paper sub-label or None for single)
LAYOUT = [
    ("robinson2014", "\\cite{robinson2014} (composite indicator)", [("__sadc__", None)]),
    ("pennycook_1", "\\cite{pennycook2020}", [("AC1(screen1)", "Attention 1"), ("AC2(screen2)", "Attention 2"),
        ("AC3(screen3)", "Attention 3"), ("AC4(Random)", "Attention 4"),
        ("union(any AC)", "Union"), ("intersection(all AC)", "Intersection")]),
    ("inattentive", "\\cite{alvarez2019}", [("filter", None)]),
    ("attention_check", "\\cite{uhalt2020}", [("Attention_Check", None)]),
    ("moral_data", "\\cite{ogrady2019}", [("attention", None)]),
    ("bot_bot_mturk", "\\cite{buchanan2018}", [("Q6_15", None)]),
    ("mturk_ethics", "\\cite{moss2023}", [("Screener_One", "Attention 1"), ("Screener_Two", "Attention 2"),
        ("union(S1|S2)", "Union"), ("intersection(S1&S2)", "Intersection")]),
    ("racial_data", "\\cite{ivanov2021}", [("attn1", "Attention 1"), ("attn2", "Attention 2"),
        ("union(attn1|attn2)", "Union"), ("intersection(attn1&attn2)", "Intersection")]),
]


def sadc_rows():
    """Full metrics for robinson2014 (SADC composite) per method."""
    dc, rc, ic, adc, arc, aic = define_necessary_elements("sadc_2017", None, None, None)
    L = DataLoader(dc, rc, ic, additional_drop_columns=adc, additional_rename_columns=arc,
                   additional_columns_of_interest=aic)
    data, meta = L.load_data("sadc_2017")
    y = L.find_outlier_data_sadc_2017("sadc_2017", ["outlier"])["outlier"].values.astype(int)
    out = {}
    for key, path in [("AE_p100", "cache/sadc_2017_100perc_newloss/errors.csv"),
                      ("AE_p85", "cache/sadc_2017_85perc_newloss/errors.csv"),
                      ("ChowLiu", "cache/_fix9_sadc_2017_cl/errors.csv")]:
        if os.path.exists(path):
            e = pd.read_csv(path)["error"].astype(float).values
            if len(e) == len(y):
                out[key] = _ranking_metrics(e, y)
    _, vec, _, _ = prepare_for_model(data, meta.get("variable_types", {}))
    X = vec.to_numpy(float); k = min(10, X.shape[1] - 1)
    p = PCA(n_components=k).fit(X)
    out["LinearAE"] = _ranking_metrics(((X - p.inverse_transform(p.transform(X))) ** 2).mean(1), y)
    return out


def fmt(m, col):
    v = m.get(col)
    return "--" if v is None else (f"{v}" if col == "h" else f"{v:.2f}")


def main():
    df = pd.read_csv("experiment_b_full.csv")
    sadc = sadc_rows()
    lines = ["% DRAFT: regenerated detailed detection tables (tuned + binning-fix numbers).",
             "% Review only -- not wired into the build. mastroianni2022 omitted (config gap).", ""]
    for ds, header, checks in LAYOUT:
        lines.append(f"{header} &&&&&& \\\\")
        for chk, sub in checks:
            if sub:
                lines.append(f"\\quad {sub} &&&&&& \\\\")
            indent = "\\quad\\quad" if sub else "\\quad"
            for key, label in METHOD_ROWS:
                if ds == "robinson2014":
                    m = sadc.get(key)
                else:
                    r = df[(df.dataset == ds) & (df.method == key) & (df.check == chk)]
                    m = r.iloc[0].to_dict() if len(r) else None
                if not m:
                    continue
                h = m.get("h", "")
                cells = " & ".join(fmt(m, c) for c in MCOLS)
                lines.append(f"{indent} {label} & {h} & {cells} \\\\")
        lines.append("\\addlinespace")
    out = "../inattentiveness_paper/sections/_regenerated_detailed_tables_draft.tex"
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
