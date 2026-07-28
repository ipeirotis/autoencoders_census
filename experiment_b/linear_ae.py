"""
True linear autoencoder built from the MAIN AutoencoderModel made 0-layer (no
hidden layers, linear latent activation, categorical/softmax reconstruction +
the same scoring path as the non-linear AE), NOT sklearn PCA and NOT the
separate MSE LinearAutoencoder. This is the "comment out the non-linearities"
linear AE, formalized via config: encoder_layers=0, decoder_layers=0,
latent_activation='linear'. Trained full-batch (p=100).

For each dataset it writes detection errors (cache/_lin0_<dataset>/errors.csv)
and prints the reconstruction Lift, so both the detection tables and the
reconstruction/correlation analysis use the same linear model.

Run from the repo root:  python -m experiment_b.linear_ae [dataset]
Out: cache/_lin0_<dataset>/errors.csv (+ linear_ae_lift.csv)
"""
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import tensorflow as tf

from utils import set_seed, save_to_csv, define_necessary_elements, training_batch_size
from dataset.loader import DataLoader
from main import (prepare_for_training, _clean_for_saved_vectorizer,
                  _compute_attr_layout)
from model.factory import get_model
from train.trainer import Trainer
from evaluate.outliers import get_outliers_list

DATASETS = [
    ("robinson2014", "sadc_2017"), ("pennycook2020", "pennycook_1"),
    ("alvarez2019", "inattentive"), ("uhalt2020", "attention_check"),
    ("ogrady2019", "moral_data"), ("buchanan2018", "bot_bot_mturk"),
    ("moss2023", "mturk_ethics"), ("mastroianni2022", "public_opinion"),
    ("ivanov2021", "racial_data"),
]


def mean_lift(proj, tab):
    n = len(proj)
    lifts = []
    for v in [c for c in proj.columns if c in tab.columns]:
        t = proj[v].astype(str).values
        p = tab[v].astype(str).values
        acc = float((t == p).mean())
        base = proj[v].astype(str).value_counts().max() / n
        if base > 0:
            lifts.append(acc / base)
    return round(float(np.mean(lifts)), 3) if lifts else None


def run(ds, rows):
    tf.keras.backend.clear_session()
    set_seed(1)
    dc, rc, ic, adc, arc, aic = define_necessary_elements(ds, None, None, None)
    loader = DataLoader(dc, rc, ic, additional_drop_columns=adc,
                        additional_rename_columns=arc, additional_columns_of_interest=aic)
    data, meta = loader.load_data(ds)
    _, X_train, X_test, vec, card = prepare_for_training(
        data, meta.get("variable_types", {}), test_size=0.2)
    proj = _clean_for_saved_vectorizer(data.copy(), vec)
    Xfull = vec.transform(proj).astype("float32")
    ac, aic2, an = _compute_attr_layout(vec, proj.columns)

    # latent dim from the tuned AE if available, else a PCA-like 10.
    latent = 10
    hp = f"cache/_tuned_{ds}_ae/best_hyperparameters.yaml"
    if os.path.exists(hp):
        latent = yaml.safe_load(open(hp)).get("latent_space_dim", 10)
    cfg = {"encoder_layers": 0, "decoder_layers": 0, "latent_space_dim": latent,
           "latent_activation": "linear", "learning_rate": 1e-3,
           "epochs": 120, "batch_size": training_batch_size(ds), "test_size": 0.2, "percentile": 100}
    model, _ = Trainer(get_model("AE", card), cfg).train(
        dataset=X_train, prior="gaussian", X_train=X_train, X_test=X_test)

    err = get_outliers_list(Xfull, model, 1.0, ac, vec, "gaussian",
                            attr_is_categorical=aic2, attr_names=an)
    save_to_csv(err, f"cache/_lin0_{ds}/", "errors")

    preds = model.predict(Xfull.values, verbose=0)
    if isinstance(preds, tuple):
        preds = preds[0]
    tab = vec.tabularize_vector(pd.DataFrame(np.asarray(preds), columns=Xfull.columns, index=Xfull.index))
    lift = mean_lift(proj, tab)
    rows.append({"dataset_key": ds, "latent": latent, "Lift": lift})
    print(f"[{ds}] 0-layer linear AE: latent={latent} Lift={lift} -> cache/_lin0_{ds}/errors.csv", flush=True)


def main(only=None):
    rows = []
    for cite, ds in DATASETS:
        if only and ds != only:
            continue
        try:
            run(ds, rows)
        except Exception as e:  # noqa: BLE001
            print(f"[{ds}] FAILED: {type(e).__name__}: {e}", flush=True)
    pd.DataFrame(rows).to_csv("linear_ae_lift.csv", index=False)
    print(f"\nwrote linear_ae_lift.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
