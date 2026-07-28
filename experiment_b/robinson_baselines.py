"""
Recompute the six psychometric baseline detection AUCs for robinson2014
(sadc_2017), scored against the composite mischievous-responder indicator --
the SAME ground truth used for robinson's AE / Chow-Liu detection in
experiment_b.build_detailed_tables (via DataLoader.find_outlier_data_sadc_2017).

sadc_2017 has no embedded attention check, so it is not part of the
evaluate.detection CHECKS / aligned_labels / RAW_CSV harness. This script wires
it in for the psychometric baselines: it builds the ordinal item battery from
the modeled sadc responses (same _to_ordinal + Rule-of-N encoding as
detection.aligned_battery uses for the other datasets), runs the six indices,
and scores each with the identical _ranking_metrics(..., orient=False) the
baseline harness uses (designed direction, sub-0.5 = genuine failure).

Usage (repo root):  python -m experiment_b.robinson_baselines
Prints AUC per index + the ordinal-item count; writes robinson_baselines.csv.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from utils import define_necessary_elements
from dataset.loader import DataLoader
from evaluate.detection import _ranking_metrics, _to_ordinal
from evaluate import baselines as bl

DS = "sadc_2017"
INDEX_ORDER = ["longstring", "irv", "person_total_r", "mahalanobis", "even_odd", "lz"]


def _loader():
    dc, rc, ic, adc, arc, aic = define_necessary_elements(DS, None, None, None)
    return DataLoader(dc, rc, ic, additional_drop_columns=adc,
                      additional_rename_columns=arc, additional_columns_of_interest=aic)


def main():
    L = _loader()
    # composite mischievous-responder label, positionally aligned to the scored
    # rows (exactly the array build_detailed_tables uses for AE/CL detection).
    y = L.find_outlier_data_sadc_2017(DS, ["outlier"])["outlier"].values.astype(int)
    data, _ = L.load_data(DS)
    if len(data) != len(y):
        raise AssertionError(f"row mismatch: data={len(data)} label={len(y)}")

    # ordinal battery: modeled items that encode as 2..9-level ordinals, exactly
    # as detection.aligned_battery selects them for the other datasets.
    cols = {}
    for c in data.columns:
        enc = _to_ordinal(data[c])
        if enc is None:
            continue
        if 2 <= int(pd.Series(enc).dropna().nunique()) <= 9:
            cols[c] = np.asarray(enc, dtype=float)
    B = pd.DataFrame(cols)
    print(f"robinson2014/sadc_2017: n={len(y)}, positives(h)={int(y.sum())}, "
          f"ordinal items={B.shape[1]}")

    rows = []
    for key in INDEX_ORDER:
        s = np.asarray(bl.INDICES[key](B), dtype=float)
        valid = ~np.isnan(s)
        npos = int(y[valid].sum())
        if valid.sum() == 0 or not (0 < npos < int(valid.sum())):
            auc = None
        else:
            auc = _ranking_metrics(s[valid], y[valid], orient=False)["AUC"]
        rows.append({"index": key, "n_scored": int(valid.sum()), "AUC": auc})
        print(f"  {key:16s} n={int(valid.sum()):6d}  AUC={auc}")
    pd.DataFrame(rows).to_csv("robinson_baselines.csv", index=False)
    print("wrote robinson_baselines.csv")


if __name__ == "__main__":
    main()
