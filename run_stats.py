# run_stats.py
"""
Example runner: builds summaries; computes per-dataset stats, correlations, and LODO robustness;
writes CSVs you can include in your paper/appendix.
"""

import pandas as pd
from stats_summary import (
    build_dataset_summary, corr_table_dataset_level, meanlift_without_PL,
    corr_meanlift_vs_auc, lodo_robustness, per_dataset_stats, method_specific_correlations
)

# 1) Load or construct your three input tables
# Replace the lines below with your existing DataFrames if they already exist in memory.
datasets = pd.DataFrame({
    "dataset_key": [
        "robinson2014", "pennycook2020", "alvarez2019", "uhalt2020",
        "ogrady2019", "buchanan2018", "moss2023", "mastroianni2022", "ivanov2021"
    ],
    "Dataset": [
        "Robinson-Cimpian (2014)", "Pennycook et al. (2020)", "Alvarez et al. (2019)", "Uhalt (2020)",
        "O’Grady et al. (2019)", "Buchanan & Scofield (2018)", "Moss et al. (2023)", "Mastroianni & Dana (2022)", "Ivanov et al. (2021)"
    ],
    "Samples": [14765, 853, 2725, 308, 355, 1038, 2277, 1036, 860],
    "Variables": [98, 188, 39, 60, 72, 23, 51, 51, 67],
    "Features": [619, 708, 196, 337, 322, 159, 332, 322, 310],
    "AFV": [6.32, 3.75, 5.03, 5.62, 4.47, 6.91, 6.51, 6.31, 4.63],
})

# ----------------------------
# 2) Reconstruction performance
# ----------------------------
recon_rows = [
    # dataset_key, method, Acc, Baseline, Lift, ORA
    ("robinson2014","AE_NL",81.37,58.27,1.53,0.73),
    ("robinson2014","AE_LIN",83.16,58.27,1.58,0.70),
    ("robinson2014","AE_PL",80.83,58.27,1.52,0.70),

    ("pennycook2020","AE_NL",86.84,66.22,1.38,0.83),
    ("pennycook2020","AE_LIN",89.18,66.22,1.44,0.86),
    ("pennycook2020","AE_PL",86.86,66.22,1.38,0.83),

    ("alvarez2019","AE_NL",90.31,52.56,2.05,0.88),
    ("alvarez2019","AE_LIN",90.07,52.56,2.07,0.85),
    ("alvarez2019","AE_PL",89.69,52.56,2.02,0.86),

    ("uhalt2020","AE_NL",85.94,35.66,2.49,0.89),
    ("uhalt2020","AE_LIN",91.44,35.66,2.64,0.89),
    ("uhalt2020","AE_PL",81.31,35.66,2.35,0.83),

    ("ogrady2019","AE_NL",86.89,58.78,1.70,0.89),
    ("ogrady2019","AE_LIN",79.80,58.78,1.68,0.86),
    ("ogrady2019","AE_PL",84.30,58.78,1.64,0.85),

    ("buchanan2018","AE_NL",94.62,50.87,2.18,0.90),
    ("buchanan2018","AE_LIN",92.80,50.87,2.19,0.84),
    ("buchanan2018","AE_PL",91.89,50.87,2.12,0.75),

    ("moss2023","AE_NL",83.46,59.78,1.51,0.76),
    ("moss2023","AE_LIN",88.59,59.78,1.66,0.82),
    ("moss2023","AE_PL",82.73,59.78,1.49,0.75),

    ("mastroianni2022","AE_NL",77.77,61.64,1.26,0.68),
    ("mastroianni2022","AE_LIN",83.11,61.64,1.34,0.72),
    ("mastroianni2022","AE_PL",77.57,61.64,1.25,0.68),

    ("ivanov2021","AE_NL",76.15,45.32,1.86,0.78),
    ("ivanov2021","AE_LIN",80.24,45.32,2.00,0.84),
    ("ivanov2021","AE_PL",74.69,45.32,1.82,0.76),
]
recon = pd.DataFrame(recon_rows, columns=["dataset_key","method","Acc","Baseline","Lift","ORA"])

# ----------------------------
# 3) Detection AUC (union rows where multiple checks exist)
# ----------------------------
det_rows = [
    # robinson2014
    ("robinson2014","AE_NL",0.71),
    ("robinson2014","AE_LIN",0.51), ("robinson2014","AE_PL",0.74), ("robinson2014","CL",0.75),

    # pennycook2020 (Union)
    ("pennycook2020","AE_NL",0.53),
    ("pennycook2020","AE_LIN",0.55), ("pennycook2020","AE_PL",0.53), ("pennycook2020","CL",0.53),

    # alvarez2019
    ("alvarez2019","AE_NL",0.80),
    ("alvarez2019","AE_LIN",0.58), ("alvarez2019","AE_PL",0.77), ("alvarez2019","CL",0.80),

    # uhalt2020
    ("uhalt2020","AE_NL",0.67),
    ("uhalt2020","AE_LIN",0.60), ("uhalt2020","AE_PL",0.82), ("uhalt2020","CL",0.88),

    # ogrady2019
    ("ogrady2019","AE_NL",0.55),
    ("ogrady2019","AE_LIN",0.63), ("ogrady2019","AE_PL",0.55), ("ogrady2019","CL",0.76),

    # buchanan2018
    ("buchanan2018","AE_NL",0.61),
    ("buchanan2018","AE_LIN",0.51), ("buchanan2018","AE_PL",0.72), ("buchanan2018","CL",0.65),

    # moss2023 (Union)
    ("moss2023","AE_NL",0.54),
    ("moss2023","AE_LIN",0.53), ("moss2023","AE_PL",0.59), ("moss2023","CL",0.57),

    # mastroianni2022
    ("mastroianni2022","AE_NL",0.66),
    ("mastroianni2022","AE_LIN",0.52), ("mastroianni2022","AE_PL",0.68), ("mastroianni2022","CL",0.61),

    # ivanov2021 (Union)
    ("ivanov2021","AE_NL",0.65),
    ("ivanov2021","AE_LIN",0.50), ("ivanov2021","AE_PL",0.70), ("ivanov2021","CL",0.74),
]
detect = pd.DataFrame(det_rows, columns=["dataset_key","method","AUC"])


# 2) Dataset-level summary (means across methods)
summary = build_dataset_summary(datasets, recon, detect,
                                avg_detect_cols=['AUC'],
                                avg_recon_cols=['Lift','ORA'])
summary.to_csv("out_dataset_summary.csv", index=False)

# 3) Fill MeanLift_noPL and compute correlations with/without PL
summary["MeanLift_noPL"] = meanlift_without_PL(summary, recon)
corr_all = corr_table_dataset_level(summary,
                                    predictors=['Samples','Variables','Features','AFV','MeanLift'],
                                    y_cols=['MeanAUC','MeanLift'])
corr_all.to_csv("out_correlations_dataset_level.csv", index=False)

corr_withPL   = corr_meanlift_vs_auc(summary, use_noPL=False)
corr_withoutPL= corr_meanlift_vs_auc(summary, use_noPL=True)

pd.DataFrame([{"setting":"with_PL", **corr_withPL},
              {"setting":"without_PL", **corr_withoutPL}]).to_csv("out_lift_auc_summary.csv", index=False)

# 4) LODO robustness for MeanLift (w/ and w/o PL) vs MeanAUC
lodo_withPL = lodo_robustness(summary, col_x="MeanLift", col_y="MeanAUC")
lodo_withPL.to_csv("out_lodo_withPL.csv", index=False)

lodo_withoutPL = lodo_robustness(summary, col_x="MeanLift_noPL", col_y="MeanAUC")
lodo_withoutPL.to_csv("out_lodo_withoutPL.csv", index=False)

# 5) Per-dataset “best method” stats (for narrative/appendix)
perds = per_dataset_stats(datasets, recon, detect)
perds.to_csv("out_per_dataset_stats.csv", index=False)

# 6) Method-specific correlation tables (Appendix X)
ms_corr = method_specific_correlations(datasets, recon, detect,
                                       predictors=['Samples','Variables','Features','AFV'])
ms_corr.to_csv("out_method_specific_correlations.csv", index=False)

print("Wrote: out_dataset_summary.csv, out_correlations_dataset_level.csv, out_lift_auc_summary.csv, out_lodo_withPL.csv, out_lodo_withoutPL.csv, out_per_dataset_stats.csv, out_method_specific_correlations.csv")
