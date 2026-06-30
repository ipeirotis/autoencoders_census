"""
Same-input robustness check: run our generative models on EXACTLY the ordinal
battery the psychometric baselines use (evaluate.detection.aligned_battery), so
the comparison is apples-to-apples. Trains the tuned AE with Percentile Loss
(p=85) and fits a Chow-Liu tree on the same ordinal items, scores each by the
standard harness, and reports detection AUC per attention check.

Datasets without an ordinal battery (moss2023) are skipped; robinson2014 has no
embedded attention check and is excluded.

Run from the repo root:  python -m experiment_b.same_input [dataset]
Out: same_input_results.csv  (dataset, check, AE_ord AUC, CL_ord AUC, full metrics)
"""
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import tensorflow as tf

from utils import set_seed
from main import (prepare_for_training, _clean_for_saved_vectorizer,
                  _compute_attr_layout)
from model.factory import get_model
from train.trainer import Trainer
from evaluate.outliers import get_outliers_list
from evaluate.detection import aligned_battery, evaluate_scores
from chow_liu_rank import rank_rows_by_chow_liu

DATASETS = [
    ("pennycook2020", "pennycook_1"), ("alvarez2019", "inattentive"),
    ("uhalt2020", "attention_check"), ("ogrady2019", "moral_data"),
    ("buchanan2018", "bot_bot_mturk"), ("mastroianni2022", "public_opinion"),
    ("ivanov2021", "racial_data"),
]
PRIMARY = {"pennycook_1": "AC1(screen1)", "inattentive": "filter",
           "attention_check": "Attention_Check", "moral_data": "attention",
           "bot_bot_mturk": "Q6_15", "public_opinion": "attention_1",
           "racial_data": "attn1"}


def ae_battery_scores(bat, best, p=85):
    """Train the tuned AE (PL p) on the one-hot of the ordinal battery; return
    per-row reconstruction anomaly score aligned to the battery rows."""
    tf.keras.backend.clear_session()  # avoid graph accumulation across datasets
    set_seed(1)
    vt = {c: "categorical" for c in bat.columns}
    _, X_train, X_test, vec, card = prepare_for_training(bat, vt, test_size=0.2)
    proj = _clean_for_saved_vectorizer(bat.copy(), vec)
    Xfull = vec.transform(proj).astype("float32")
    ac, aic2, an = _compute_attr_layout(vec, proj.columns)
    cfg = dict(best)
    cfg.update(epochs=120, batch_size=64, test_size=0.2, percentile=p)
    model, _ = Trainer(get_model("AE", card), cfg).train(
        dataset=X_train, prior="gaussian", X_train=X_train, X_test=X_test)
    err = get_outliers_list(Xfull, model, 1.0, ac, vec, "gaussian",
                            attr_is_categorical=aic2, attr_names=an)
    return err["error"].to_numpy()


def run(ds, rows):
    bat = aligned_battery(ds)
    if bat.shape[1] < 2:
        print(f"[{ds}] battery has <2 ordinal items ({bat.shape[1]}); skipped", flush=True)
        return
    # Chow-Liu on the ordinal battery
    ranked, _ = rank_rows_by_chow_liu(bat, alpha=1.0, random_state=2)
    cl_scores = (1.0 - ranked["pct"]).to_numpy()
    cl = {r["check"]: r for r in evaluate_scores(ds, cl_scores, "CL_ord")}
    # AE (PL p=85) on the ordinal battery
    best = {}
    hp = f"cache/_tuned_{ds}_ae/best_hyperparameters.yaml"
    if os.path.exists(hp):
        best = yaml.safe_load(open(hp))
    ae_scores = ae_battery_scores(bat, best, p=85)
    ae = {r["check"]: r for r in evaluate_scores(ds, ae_scores, "AE_ord")}
    for chk in ae:
        rows.append({"dataset": ds, "check": chk, "n_items": bat.shape[1],
                     "AE_ord_AUC": ae[chk].get("AUC"), "CL_ord_AUC": cl.get(chk, {}).get("AUC"),
                     "AE_ord_R@h": ae[chk].get("R@h"), "CL_ord_R@h": cl.get(chk, {}).get("R@h")})
    pr = PRIMARY[ds]
    print(f"[{ds}] battery={bat.shape[1]} items | primary '{pr}': "
          f"AE_ord AUC={ae.get(pr, {}).get('AUC')}  CL_ord AUC={cl.get(pr, {}).get('AUC')}", flush=True)


def main(only=None):
    rows = []
    for cite, ds in DATASETS:
        if only and ds != only:
            continue
        try:
            run(ds, rows)
        except Exception as e:  # noqa: BLE001
            print(f"[{ds}] FAILED: {type(e).__name__}: {e}", flush=True)
    pd.DataFrame(rows).to_csv("same_input_results.csv", index=False)
    print(f"\nwrote same_input_results.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
