"""
Experiment C -- detection AUC vs training sample size.

For each dataset we subsample N respondents (stratified on the primary
attention-check label to hold the positive rate roughly fixed), train the
non-linear autoencoder (Percentile Loss p=85) and the Chow-Liu network on that
subsample, score the same subsample, and record the detection AUC on the primary
check. Repeating over seeds and sweeping N in {100,250,500,1000,2500,full} maps
how much data each detector needs and yields a minimum recommended sample size.

Usage:  python -m experiment_b.sample_size [dataset ...]
Out: sample_size_results.csv (dataset, N, seed, AE_AUC, CL_AUC)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml
import tensorflow as tf

from utils import set_seed, define_necessary_elements, training_batch_size
from dataset.loader import DataLoader
from main import prepare_for_training, _clean_for_saved_vectorizer, _compute_attr_layout
from model.factory import get_model
from train.trainer import Trainer
from evaluate.outliers import get_outliers_list
from evaluate.detection import aligned_labels, _relevant, _ranking_metrics, CHECKS
from chow_liu_rank import CLTree

PRIMARY = {"attention_check": "Attention_Check", "inattentive": "filter",
           "racial_data": "attn1", "moral_data": "attention",
           "mturk_ethics": "Screener_One", "bot_bot_mturk": "Q6_15",
           "public_opinion": "attention_1", "pennycook_1": "AC1(screen1)"}
DATASETS = list(PRIMARY) + ["sadc_2017"]
NGRID = [100, 250, 500, 1000, 2500]
SEEDS = [1, 2, 3]


def _loader(ds):
    dc, rc, ic, adc, arc, aic = define_necessary_elements(ds, None, None, None)
    return DataLoader(dc, rc, ic, additional_drop_columns=adc,
                      additional_rename_columns=arc, additional_columns_of_interest=aic)


def _load(ds):
    """Return (categorical data frame, variable_types, per-row 0/1 label)."""
    L = _loader(ds)
    if ds == "sadc_2017":
        y = L.find_outlier_data_sadc_2017(ds, ["outlier"])["outlier"].values.astype(int)
        data, meta = L.load_data(ds)
    else:
        data, meta = L.load_data(ds)
        lab = aligned_labels(ds)
        ch = [c for c in CHECKS[ds] if c["name"] == PRIMARY[ds]][0]
        y = _relevant(lab, ch).astype(int)
    if len(data) != len(y):
        raise AssertionError(f"{ds}: data {len(data)} vs label {len(y)}")
    return data.reset_index(drop=True), meta.get("variable_types", {}), np.asarray(y)


def _subsample(y, N, seed):
    rng = np.random.RandomState(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n_pos = min(len(pos), max(1, round(N * len(pos) / len(y))))
    n_neg = min(len(neg), N - n_pos)
    idx = np.concatenate([rng.choice(pos, n_pos, replace=False),
                          rng.choice(neg, n_neg, replace=False)])
    return np.sort(idx)


def _ae_auc(ds, data_sub, data_full, vt, best, y_full):
    """Train the AE on the N-row subsample; evaluate detection AUC on the FULL
    dataset (fixed positive set, so the curve isolates training-data size)."""
    tf.keras.backend.clear_session()
    _, X_train, X_test, vec, card = prepare_for_training(data_sub, vt, test_size=0.2)
    cfg = dict(best)
    cfg.update(epochs=300, batch_size=training_batch_size(ds), test_size=0.2, percentile=85)
    model, _ = Trainer(get_model("AE", card), cfg).train(
        dataset=X_train, prior="gaussian", X_train=X_train, X_test=X_test)
    proj = _clean_for_saved_vectorizer(data_full.copy(), vec)   # score FULL data
    Xv = vec.transform(proj).astype("float32")
    ac, aic, an = _compute_attr_layout(vec, proj.columns)
    err = get_outliers_list(Xv, model, 1.0, ac, vec, "gaussian",
                            attr_is_categorical=aic, attr_names=an)
    return _ranking_metrics(err["error"].to_numpy(), y_full, orient=True)["AUC"]


def _cl_auc(data_sub, data_full, y_full):
    cl = CLTree(alpha=1.0).fit(data_sub, random_state=2)          # fit on subsample
    ranked = cl.score_dataframe(data_full)                        # score FULL data
    return _ranking_metrics((1.0 - ranked["pct"]).to_numpy(), y_full, orient=True)["AUC"]


def run(ds, rows):
    data, vt, y = _load(ds)
    Ntot = len(data)
    hp = f"cache/_tuned_{ds}_ae/best_hyperparameters.yaml"
    best = yaml.safe_load(open(hp)) if os.path.exists(hp) else {
        "encoder_layers": 1, "encoder_units_1": 128, "encoder_activation_1": "relu",
        "latent_space_dim": 20, "latent_activation": "relu",
        "decoder_layers": 1, "decoder_units_1": 128, "decoder_activation_1": "relu",
        "learning_rate": 1e-3}
    grid = [n for n in NGRID if n < Ntot] + [Ntot]
    for N in grid:
        for seed in SEEDS:
            set_seed(seed)
            idx = _subsample(y, N, seed) if N < Ntot else np.arange(Ntot)
            ds_sub = data.iloc[idx].reset_index(drop=True)
            if y[idx].sum() == 0:                       # need >=1 positive to train on
                continue
            try:
                ae = _ae_auc(ds, ds_sub, data, vt, best, y)
            except Exception as e:  # noqa: BLE001
                ae = None
                print(f"  [{ds} N={N} s{seed}] AE failed: {type(e).__name__}: {e}", flush=True)
            try:
                cl = _cl_auc(ds_sub, data, y)
            except Exception as e:  # noqa: BLE001
                cl = None
                print(f"  [{ds} N={N} s{seed}] CL failed: {type(e).__name__}: {e}", flush=True)
            rows.append({"dataset": ds, "N": int(len(idx)), "seed": seed,
                         "AE_AUC": ae, "CL_AUC": cl, "n_pos_train": int(y[idx].sum())})
            print(f"[{ds}] N={len(idx)} seed={seed} AE={ae} CL={cl}", flush=True)
            pd.DataFrame(rows).to_csv("sample_size_results.csv", index=False)


def main(only):
    rows = []
    for ds in (only or DATASETS):
        try:
            run(ds, rows)
        except Exception as e:  # noqa: BLE001
            print(f"[{ds}] SKIPPED: {type(e).__name__}: {e}", flush=True)
    print("done ->", "sample_size_results.csv")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
