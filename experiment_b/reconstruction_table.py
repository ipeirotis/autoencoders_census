"""
Turnkey, deterministic reproduction of the paper's RECONSTRUCTION table
(tab:reconstruction_perf): mean Accuracy, mean Baseline Accuracy, mean Lift, and
mean ORA (One-vs-All ROC AUC) for the three reconstruction models the paper
reports -- Non-Linear AE (p=100), Non-Linear AE (p=85), and the 0-layer Linear
AE -- across all nine datasets, on the CORRECTED preprocessing (numeric vars
with <20 distinct values kept categorical).

Why: the committed reconstruction table was hard-coded from a pre-binning-fix
run (run_stats.py) with no producing script, and recon_grid.py only recomputes
Lift. This script recomputes ALL FOUR columns with the SAME metric definitions
as evaluate.evaluator.Evaluator (accuracy = fraction correct, baseline =
majority-class share, lift = round(acc/base, 2) per variable, ORA = macro
one-vs-all ROC AUC on hard predictions), averaged over the modeled categorical
variables.

ONE CELL PER PROCESS. Each (dataset, method) is trained in its own python
invocation so TensorFlow graph/optimizer state never accumulates across builds
(the source of the Adam "Incompatible shapes" crash when many models are built
in one process). A cell writes reconstruction_cells/<ds>__<method>.csv; the
final combine step assembles reconstruction_table.csv + reconstruction_table.tex.

Usage (from repo root):
  python -m experiment_b.reconstruction_table <internal_ds> <nl100|nl85|lin>  # one cell
  python -m experiment_b.reconstruction_table combine                          # build table
Driver: experiment_b/run_reconstruction.sh runs all 27 cells then combines.
"""
import os
import sys
import glob
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from utils import set_seed, define_necessary_elements, training_batch_size
from dataset.loader import DataLoader
from main import prepare_for_training, _clean_for_saved_vectorizer
from model.factory import get_model
from train.trainer import Trainer

CELLS_DIR = "reconstruction_cells"

# robinson2014 (sadc_2017) has no tuned AE; representative architecture, as in
# recon_grid.py (reconstruction is ~architecture-invariant).
FALLBACK_AE = {
    "encoder_layers": 1, "encoder_units_1": 256, "encoder_activation_1": "relu",
    "encoder_dropout_1": 0.2, "encoder_l2_1": 0.0, "encoder_batch_norm_1": False,
    "latent_space_dim": 30, "latent_activation": "relu",
    "decoder_layers": 1, "decoder_units_1": 160, "decoder_activation_1": "relu",
    "decoder_dropout_1": 0.2, "decoder_l2_1": 0.01, "decoder_batch_norm_1": False,
    "learning_rate": 0.001,
}

# paper citation key -> internal dataset key (paper-table order)
DATASETS = [
    ("robinson2014", "sadc_2017"), ("pennycook2020", "pennycook_1"),
    ("alvarez2019", "inattentive"), ("uhalt2020", "attention_check"),
    ("ogrady2019", "moral_data"), ("buchanan2018", "bot_bot_mturk"),
    ("moss2023", "mturk_ethics"), ("mastroianni2022", "public_opinion"),
    ("ivanov2021", "racial_data"),
]
CITE = {ds: cite for cite, ds in DATASETS}
METHOD_ORDER = ["nl100", "nl85", "lin"]
METHOD_NAME = {"nl100": "Non-Linear AE (p=100)",
               "nl85": "Non-Linear AE (p=85)", "lin": "Linear AE"}


def _loader(ds):
    dc, rc, ic, adc, arc, aic = define_necessary_elements(ds, None, None, None)
    return DataLoader(dc, rc, ic, additional_drop_columns=adc,
                      additional_rename_columns=arc,
                      additional_columns_of_interest=aic)


def _cfg(ds, method):
    best = FALLBACK_AE
    hp = f"cache/_tuned_{ds}_ae/best_hyperparameters.yaml"
    if os.path.exists(hp):
        best = yaml.safe_load(open(hp))
    latent = best.get("latent_space_dim", 10)
    common = {"epochs": 120, "batch_size": training_batch_size(ds), "test_size": 0.2}
    if method == "nl100":
        return {**best, **common, "percentile": 100}
    if method == "nl85":
        return {**best, **common, "percentile": 85}
    if method == "lin":
        return {"encoder_layers": 0, "decoder_layers": 0, "latent_space_dim": latent,
                "latent_activation": "linear", "learning_rate": 1e-3,
                **common, "percentile": 100}
    raise ValueError(method)


def recon_metrics(proj, tab, variable_types):
    """Per-variable Accuracy/BaselineAcc/Lift/ORA averaged over the modeled
    categorical variables (evaluate.evaluator.Evaluator definitions, no plot)."""
    n = len(proj)
    accs, bases, lifts, oras = [], [], [], []
    cols = [v for v in variable_types if variable_types[v] == "categorical"
            and v in tab.columns and v in proj.columns]
    for v in cols:
        t = proj[v].astype(str).values
        p = tab[v].astype(str).values
        acc = float((t == p).mean()) * 100.0
        base = proj[v].astype(str).value_counts().max() / n * 100.0
        if base <= 0:
            continue
        accs.append(acc)
        bases.append(base)
        lifts.append(round(acc / base, 2))
        cats = np.unique(t)
        try:
            oras.append(roc_auc_score(label_binarize(t, classes=cats),
                                      label_binarize(p, classes=cats),
                                      multi_class="ovr", average="macro"))
        except Exception:  # noqa: BLE001 - degenerate/constant column
            pass
    if not accs:
        return None
    return {"Accuracy": round(float(np.mean(accs)), 2),
            "BaselineAcc": round(float(np.mean(bases)), 2),
            "Lift": round(float(np.mean(lifts)), 2),
            "ORA": round(float(np.mean(oras)), 2) if oras else None,
            "n_vars": len(accs)}


def run_cell(ds, method):
    tf.keras.backend.clear_session()
    set_seed(1)
    loader = _loader(ds)
    data, meta = loader.load_data(ds)
    vt = meta.get("variable_types", {})
    _, X_train, X_test, vec, card = prepare_for_training(data, vt, test_size=0.2)
    proj = _clean_for_saved_vectorizer(data.copy(), vec)
    Xfull = vec.transform(proj).astype("float32")

    model, _ = Trainer(get_model("AE", card), _cfg(ds, method)).train(
        dataset=X_train, prior="gaussian", X_train=X_train, X_test=X_test)
    preds = model.predict(Xfull.values, verbose=0)
    if isinstance(preds, tuple):
        preds = preds[0]
    tab = vec.tabularize_vector(pd.DataFrame(np.asarray(preds),
                                             columns=Xfull.columns, index=Xfull.index))
    m = recon_metrics(proj, tab, vt)
    if m is None:
        raise RuntimeError("no categorical variables scored")
    os.makedirs(CELLS_DIR, exist_ok=True)
    row = {"dataset_key": CITE.get(ds, ds), "internal": ds,
           "method": METHOD_NAME[method], "method_key": method, **m}
    pd.DataFrame([row]).to_csv(f"{CELLS_DIR}/{ds}__{method}.csv", index=False)
    print(f"[{ds}/{method}] Acc={m['Accuracy']} Base={m['BaselineAcc']} "
          f"Lift={m['Lift']} ORA={m['ORA']} (n={m['n_vars']}) -> "
          f"{CELLS_DIR}/{ds}__{method}.csv", flush=True)


def combine():
    files = glob.glob(f"{CELLS_DIR}/*.csv")
    if not files:
        print("no cells found; run the per-cell trainings first")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["_d"] = df.internal.map({ds: i for i, (_, ds) in enumerate(DATASETS)})
    df["_m"] = df.method_key.map({m: i for i, m in enumerate(METHOD_ORDER)})
    df = df.sort_values(["_d", "_m"]).drop(columns=["_d", "_m"])
    df.to_csv("reconstruction_table.csv", index=False)

    lines = []
    for cite, ds in DATASETS:
        sub = df[df.internal == ds]
        if sub.empty:
            continue
        lines.append(f"\\cite{{{cite}}} &&&& \\\\")
        for _, r in sub.iterrows():
            def f(x, pct=False):
                if pd.isna(x):
                    return "--"
                return f"{x:.2f}"
            nm = r.method.replace("p=100", "$p=100$").replace("p=85", "$p=85$")
            lines.append(f"\\quad {nm} & {f(r.Accuracy)} & {f(r.BaselineAcc)} "
                         f"& {f(r.Lift)} & {f(r.ORA)} \\\\")
        lines.append("\\addlinespace")
    open("reconstruction_table.tex", "w").write("\n".join(lines))
    print("wrote reconstruction_table.csv and reconstruction_table.tex\n")
    print(df.drop(columns=["method_key"]).to_string(index=False))


def main(argv):
    if not argv or argv[0] == "combine":
        combine()
        return
    ds, method = argv[0], argv[1]
    run_cell(ds, method)


if __name__ == "__main__":
    main(sys.argv[1:])
