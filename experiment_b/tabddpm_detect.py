"""
Score a TabDDPM per-respondent NLL file (from the tabddpm repo's
inattentiveness/score_nll.py) through the SAME detection harness as the
autoencoder and Chow-Liu network, so its detection AUC / R@h / P@k / NDCG are
directly comparable and can drop into the detection tables as a third generative
detector.

The TabDDPM NLL is an atypicality score exactly like the AE reconstruction error
and the CL negative log-likelihood: higher = more improbable = more likely
inattentive. Row i of the NLL csv must correspond to scored respondent i (the
tabddpm exporter writes the survey in the same order aligned_labels() returns).

Usage (autoencoders_census repo root, venv_ae):
    python -m experiment_b.tabddpm_detect <survey_name> path/to/nll.csv
e.g. python -m experiment_b.tabddpm_detect attention_check ../generation_synthetic/tabddpm/exp/attention_check_score/check/nll.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from evaluate.detection import aligned_labels, _relevant, _ranking_metrics, CHECKS


def main(dataset, nll_csv):
    nll = pd.read_csv(nll_csv)["nll"].to_numpy(dtype=float)
    if dataset == "sadc_2017":
        from utils import define_necessary_elements
        from dataset.loader import DataLoader
        dc, rc, ic, adc, arc, aic = define_necessary_elements(dataset, None, None, None)
        L = DataLoader(dc, rc, ic, additional_drop_columns=adc,
                       additional_rename_columns=arc, additional_columns_of_interest=aic)
        y0 = L.find_outlier_data_sadc_2017(dataset, ["outlier"])["outlier"].values.astype(int)
        checks = [{"name": "__sadc__", "_y": y0}]
    else:
        labels = aligned_labels(dataset)
        checks = [{"name": c["name"], "_y": _relevant(labels, c).astype(int)} for c in CHECKS[dataset]]

    print(f"{'check':22s} {'h':>6} {'AUC':>6} {'R@h':>6} {'P@100':>7} {'NDCG@h':>7}")
    for ch in checks:
        y = ch["_y"]
        if len(y) != len(nll):
            print(f"{ch['name']:22s}  SKIP (nll {len(nll)} vs labels {len(y)} -- row misalignment)")
            continue
        valid = np.isfinite(nll)
        m = _ranking_metrics(nll[valid], y[valid], orient=True)   # unsupervised -> AUC symmetry
        print(f"{ch['name']:22s} {m['h']:>6} {m['AUC']:>6} {str(m['R@h']):>6} "
              f"{str(m['P@100']):>7} {str(m['NDCG@h']):>7}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
