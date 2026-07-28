"""
Regenerate the per-dataset RECONSTRUCTION Lift on the corrected preprocessing
(numeric vars with <20 distinct values kept categorical), for the four
reconstruction methods the paper averages: non-linear AE (p=100), non-linear AE
(p=85), the trained Linear AE, and PCA(10). Lift = per-variable reconstruction
accuracy / majority-class baseline, averaged over variables -- the same metric
as evaluate.evaluator.Evaluator, but without the per-variable plot files.

This exists because the binning fix changes the reconstruction target (more
Likert categories -> different majority baseline), so the Lift values feeding
the dataset-level correlation analysis must be recomputed on corrected data.

Run from the repo root:  python -m experiment_b.recon_grid
Out: recon_grid.csv  (dataset_key, AE_NL, AE_PL, AE_LIN, PCA  -> mean Lift each)
"""
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA

from utils import set_seed, define_necessary_elements, load_vectorizer
from dataset.loader import DataLoader
from main import (prepare_for_training, prepare_for_model,
                  _clean_for_saved_vectorizer)
from model.factory import get_model
from train.trainer import Trainer

# robinson2014 (sadc_2017) has no tuned AE; use a representative architecture
# (the reconstruction Lift is nearly architecture-invariant).
FALLBACK_AE = {
    "encoder_layers": 1, "encoder_units_1": 256, "encoder_activation_1": "relu",
    "encoder_dropout_1": 0.2, "encoder_l2_1": 0.0, "encoder_batch_norm_1": False,
    "latent_space_dim": 30, "latent_activation": "relu",
    "decoder_layers": 1, "decoder_units_1": 160, "decoder_activation_1": "relu",
    "decoder_dropout_1": 0.2, "decoder_l2_1": 0.01, "decoder_batch_norm_1": False,
    "learning_rate": 0.001,
}

# paper citation key -> internal dataset key
DATASETS = [
    ("robinson2014", "sadc_2017"), ("pennycook2020", "pennycook_1"),
    ("alvarez2019", "inattentive"), ("uhalt2020", "attention_check"),
    ("ogrady2019", "moral_data"), ("buchanan2018", "bot_bot_mturk"),
    ("moss2023", "mturk_ethics"), ("mastroianni2022", "public_opinion"),
    ("ivanov2021", "racial_data"),
]


def _loader(dataset):
    dc, rc, ic, adc, arc, aic = define_necessary_elements(dataset, None, None, None)
    return DataLoader(dc, rc, ic, additional_drop_columns=adc,
                      additional_rename_columns=arc, additional_columns_of_interest=aic)


def mean_lift(proj, tab):
    """Per-variable reconstruction accuracy / majority baseline, averaged over
    the modeled (categorical) variables. Predicted categories come from an
    argmax over each variable's one-hot block, so they are always in-vocabulary
    and the denominator is the full row count (matches Evaluator)."""
    n = len(proj)
    cols = [c for c in proj.columns if c in tab.columns]
    lifts = []
    for v in cols:
        t = proj[v].astype(str).values
        p = tab[v].astype(str).values
        acc = float((t == p).mean())
        base = proj[v].astype(str).value_counts().max() / n
        if base > 0:
            lifts.append(acc / base)
    return round(float(np.mean(lifts)), 3) if lifts else None


def _tab(vec, arr, like):
    pred = pd.DataFrame(np.asarray(arr), columns=like.columns, index=like.index)
    return vec.tabularize_vector(pred)


def ae_lift(dataset, p, best):
    set_seed(1)
    loader = _loader(dataset)
    data, meta = loader.load_data(dataset)
    vt = meta.get("variable_types", {})
    _, X_train, X_test, vec, card = prepare_for_training(data, vt, test_size=0.2)
    proj = _clean_for_saved_vectorizer(data.copy(), vec)
    Xfull = vec.transform(proj).astype("float32")
    cfg = dict(best)
    # 120 epochs: reconstruction Lift plateaus well before the 300 used for
    # detection tuning; a smaller, uniform budget keeps the grid fast and the
    # cross-dataset Lift comparison consistent.
    cfg.update(epochs=120, batch_size=64, test_size=0.2, percentile=p)
    model, _ = Trainer(get_model("AE", card), cfg).train(
        dataset=X_train, prior="gaussian", X_train=X_train, X_test=X_test)
    preds = model.predict(Xfull.values, verbose=0)
    if isinstance(preds, tuple):
        preds = preds[0]
    return mean_lift(proj, _tab(vec, preds, Xfull))


def lin_lift(dataset):
    mp = f"cache/_lin_{dataset}/autoencoder"
    if not os.path.exists(mp):
        return None
    loader = _loader(dataset)
    data, _ = loader.load_data(dataset)
    vec = load_vectorizer(mp)
    proj = _clean_for_saved_vectorizer(data.copy(), vec)
    X = vec.transform(proj).astype("float32")
    model = tf.keras.models.load_model(mp)
    xhat = model.predict(X.values, verbose=0)
    if isinstance(xhat, tuple):
        xhat = xhat[0]
    return mean_lift(proj, _tab(vec, xhat, X))


def pca_lift(dataset):
    loader = _loader(dataset)
    data, meta = loader.load_data(dataset)
    proj, Xv, vec, _ = prepare_for_model(data, meta.get("variable_types", {}))
    X = Xv.to_numpy(dtype=float)
    k = min(10, X.shape[1] - 1)
    pca = PCA(n_components=k).fit(X)
    recon = pca.inverse_transform(pca.transform(X))
    return mean_lift(proj, _tab(vec, recon, Xv))


def main(only=None):
    rows = []
    for cite, ds in DATASETS:
        if only and ds != only:
            continue
        best = FALLBACK_AE
        hp = f"cache/_tuned_{ds}_ae/best_hyperparameters.yaml"
        if os.path.exists(hp):
            best = yaml.safe_load(open(hp))
        rec = {"dataset_key": cite, "internal": ds}
        # The linear reconstruction is represented by PCA(10): the cached
        # cache/_lin_ model is the MSE-trained detection linear AE (raw,
        # non-softmax output) whose per-variable argmax accuracy is not a
        # faithful reconstruction measure. PCA is the canonical linear
        # reconstruction (used by experiment_b.final_table.linear_ae_scores).
        for name, fn in (("AE_NL", lambda: ae_lift(ds, 100, best)),
                         ("AE_PL", lambda: ae_lift(ds, 85, best)),
                         ("PCA", lambda: pca_lift(ds))):
            try:
                rec[name] = fn()
            except Exception as e:  # noqa: BLE001 - keep the grid going
                rec[name] = None
                print(f"[{ds}] {name} FAILED: {type(e).__name__}: {e}", flush=True)
        vals = [rec[k] for k in ("AE_NL", "AE_PL", "PCA") if rec.get(k) is not None]
        rec["MeanLift"] = round(float(np.mean(vals)), 3) if vals else None
        rows.append(rec)
        print(f"[{ds}] AE_NL={rec['AE_NL']} AE_PL={rec['AE_PL']} "
              f"PCA={rec['PCA']} -> MeanLift={rec['MeanLift']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("recon_grid.csv", index=False)
    print("\nwrote recon_grid.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
