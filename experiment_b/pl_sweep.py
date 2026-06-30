"""
Percentile-Loss trade-off sweep on the CORRECTED preprocessing. For each dataset
and each p in {80,85,90,95,100}, train the tuned AE with Percentile Loss at that
p and record BOTH detection metrics (per attention-check, via the standard
harness) and reconstruction metrics (Accuracy, Lift, ORA = one-vs-all ROC AUC,
averaged over modeled variables). The hard-coded deltas in tradeoff_plots.py were
computed before the binning fix; this regenerates them from real runs.

Run from the repo root:  python -m experiment_b.pl_sweep [dataset]
Out: pl_sweep_detection.csv  (dataset, percentile, check, R@h..AUC)
     pl_sweep_recon.csv       (dataset, percentile, Accuracy, Lift, ORA)
"""
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

from utils import set_seed, save_to_csv, define_necessary_elements
from dataset.loader import DataLoader
from main import (prepare_for_training, _clean_for_saved_vectorizer,
                  _compute_attr_layout)
from model.factory import get_model
from train.trainer import Trainer
from evaluate.outliers import get_outliers_list
from evaluate.detection import evaluate_detection

PERCENTILES = [80, 85, 90, 95, 100]
# citation key -> internal dataset key (robinson/sadc excluded: no tuned AE,
# and its detection uses the composite indicator, not the PL sweep).
DATASETS = [
    ("pennycook2020", "pennycook_1"), ("alvarez2019", "inattentive"),
    ("uhalt2020", "attention_check"), ("ogrady2019", "moral_data"),
    ("buchanan2018", "bot_bot_mturk"), ("moss2023", "mturk_ethics"),
    ("mastroianni2022", "public_opinion"), ("ivanov2021", "racial_data"),
]


def recon_metrics(proj, tab):
    """Mean reconstruction Accuracy(%), Lift, ORA over modeled variables."""
    accs, lifts, oras = [], [], []
    n = len(proj)
    for v in [c for c in proj.columns if c in tab.columns]:
        t = proj[v].astype(str).values
        p = tab[v].astype(str).values
        acc = float((t == p).mean()) * 100
        base = proj[v].astype(str).value_counts().max() / n * 100
        accs.append(acc)
        if base > 0:
            lifts.append(acc / base)
        classes = np.unique(t)
        if len(classes) >= 2:
            try:
                yt = label_binarize(t, classes=classes)
                yp = label_binarize(p, classes=classes)
                if yt.shape[1] > 1 and yp.sum() > 0:
                    oras.append(roc_auc_score(yt, yp, multi_class="ovr", average="macro"))
            except Exception:  # noqa: BLE001
                pass
    return (round(float(np.mean(accs)), 2),
            round(float(np.mean(lifts)), 3),
            round(float(np.mean(oras)), 3) if oras else None)


def run(cite, ds, det_rows, rec_rows):
    hp = f"cache/_tuned_{ds}_ae/best_hyperparameters.yaml"
    if not os.path.exists(hp):
        print(f"[{ds}] no tuned AE; skipped", flush=True)
        return
    best = yaml.safe_load(open(hp))
    name = {"pennycook_1": "Pennycook et al. (2020)", "inattentive": "Alvarez et al. (2019)",
            "attention_check": "Uhalt (2020)", "moral_data": "O'Grady et al. (2019)",
            "bot_bot_mturk": "Buchanan & Scofield (2018)", "mturk_ethics": "Moss et al. (2023)",
            "public_opinion": "Mastroianni & Dana (2022)", "racial_data": "Ivanov et al. (2021)"}[ds]
    set_seed(1)  # one fixed split + vectorizer shared across all p
    dc, rc, ic, adc, arc, aic = define_necessary_elements(ds, None, None, None)
    loader = DataLoader(dc, rc, ic, additional_drop_columns=adc,
                        additional_rename_columns=arc, additional_columns_of_interest=aic)
    data, meta = loader.load_data(ds)
    _, X_train, X_test, vec, card = prepare_for_training(
        data, meta.get("variable_types", {}), test_size=0.2)
    proj = _clean_for_saved_vectorizer(data.copy(), vec)
    Xfull = vec.transform(proj).astype("float32")
    ac, aic2, an = _compute_attr_layout(vec, proj.columns)
    for p in PERCENTILES:
        tf.keras.backend.clear_session()  # avoid graph/optimizer accumulation across builds
        set_seed(1)  # reproducible weight init per model
        cfg = dict(best)
        cfg.update(epochs=120, batch_size=64, test_size=0.2, percentile=p)
        model, _ = Trainer(get_model("AE", card), cfg).train(
            dataset=X_train, prior="gaussian", X_train=X_train, X_test=X_test)
        # detection
        err = get_outliers_list(Xfull, model, 1.0, ac, vec, "gaussian",
                                attr_is_categorical=aic2, attr_names=an)
        path = f"cache/_pl_{ds}_p{p}/"
        save_to_csv(err, path, "errors")
        for r in evaluate_detection(ds, f"{path}errors.csv", "x"):
            det_rows.append({"Dataset": name, "Subgroup": r["check"], "Percentile": p,
                             "R@h": r.get("R@h"), "P@10": r.get("P@10"), "P@50": r.get("P@50"),
                             "P@100": r.get("P@100"), "NDCG@h": r.get("NDCG@h"), "AUC": r.get("AUC")})
        # reconstruction
        preds = model.predict(Xfull.values, verbose=0)
        if isinstance(preds, tuple):
            preds = preds[0]
        tab = vec.tabularize_vector(pd.DataFrame(np.asarray(preds), columns=Xfull.columns, index=Xfull.index))
        acc, lift, ora = recon_metrics(proj, tab)
        rec_rows.append({"Dataset": name, "Percentile": p, "Accuracy": acc, "Lift": lift, "ORA": ora})
        print(f"[{ds}] p={p}: AUC(first check)={det_rows[-1]['AUC']} Acc={acc} Lift={lift} ORA={ora}", flush=True)


def main(only=None):
    det_rows, rec_rows = [], []
    for cite, ds in DATASETS:
        if only and ds != only:
            continue
        try:
            run(cite, ds, det_rows, rec_rows)
        except Exception as e:  # noqa: BLE001
            print(f"[{ds}] FAILED: {type(e).__name__}: {e}", flush=True)
    pd.DataFrame(det_rows).to_csv("pl_sweep_detection.csv", index=False)
    pd.DataFrame(rec_rows).to_csv("pl_sweep_recon.csv", index=False)
    print(f"\nwrote pl_sweep_detection.csv ({len(det_rows)} rows) + pl_sweep_recon.csv ({len(rec_rows)} rows)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
