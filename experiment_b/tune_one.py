"""
Per-dataset AE hyperparameter search + final training (paper's tuning protocol,
compute-reduced). Runs Bayesian optimization over the search space in
config/hp_autoencoder_real.yaml, retrains the best configuration to convergence,
saves the model + fitted vectorizer + chosen hyperparameters under
cache/_tuned_<dataset>_ae/, and writes errors.csv via the standard scoring path.

Usage:  python tune_one.py <dataset> [search_config.yaml]
"""
import os
import shutil
import sys
import yaml

from utils import set_seed, save_model, save_hyperparameters, define_necessary_elements
from dataset.loader import DataLoader
from main import (
    prepare_for_training, _clean_for_saved_vectorizer, _compute_attr_layout,
)
from model.factory import get_model
from train.trainer import Trainer
from evaluate.outliers import get_outliers_list
from utils import save_to_csv


def main(dataset, search_config="config/hp_autoencoder_real.yaml"):
    out = f"cache/_tuned_{dataset}_ae/"
    os.makedirs(out, exist_ok=True)
    set_seed(2)

    dc, rc, ic, adc, arc, aic = define_necessary_elements(dataset, None, None, None)
    loader = DataLoader(dc, rc, ic, additional_drop_columns=adc,
                        additional_rename_columns=arc, additional_columns_of_interest=aic)
    data, meta = loader.load_data(dataset)
    vt = meta.get("variable_types", {})

    # Split BEFORE vectorization (leak-free), exactly like the CLI train path.
    _, X_train, X_test, vectorizer, cardinalities = prepare_for_training(
        data, vt, test_size=0.2
    )

    # 1) Bayesian search. Unique project_name per dataset so KerasTuner does not
    #    reuse another dataset's oracle; wipe any stale tuner cache first.
    hp = yaml.safe_load(open(search_config))
    hp["run_name"] = f"tuned_{dataset}"
    shutil.rmtree(f"tuned_{dataset}", ignore_errors=True)
    best = Trainer(get_model("AE", cardinalities), hp).search_hyperparameters(
        dataset=X_train, prior="gaussian", X_train=X_train, X_test=X_test
    )
    print(f"[{dataset}] best HPs: {best}", flush=True)

    # 2) Retrain the winning configuration to convergence (EarlyStopping inside).
    cfg = dict(best)
    cfg.update(epochs=300, batch_size=64, test_size=0.2)
    model, _ = Trainer(get_model("AE", cardinalities), cfg).train(
        dataset=X_train, prior="gaussian", X_train=X_train, X_test=X_test
    )
    save_model(model, out, vectorizer=vectorizer)
    save_hyperparameters(best, out)

    # 3) Score every respondent through the TRAINING-fitted vectorizer
    #    (transductive), exactly like find_outliers. Using the saved vectorizer
    #    (not a refit on the full data) keeps the input width equal to what the
    #    model was trained on; unseen full-data categories map to all-zero
    #    one-hot blocks (handle_unknown='ignore') and are penalized by the
    #    unseen-category override inside compute_reconstruction_error.
    proj = _clean_for_saved_vectorizer(data.copy(), vectorizer)
    vectorized_df = vectorizer.transform(proj).astype("float32")
    attr_card, attr_is_cat, attr_names = _compute_attr_layout(vectorizer, proj.columns)
    err = get_outliers_list(
        vectorized_df, model, 1.0, attr_card, vectorizer, "gaussian",
        attr_is_categorical=attr_is_cat, attr_names=attr_names,
    )
    save_to_csv(err, out, "errors")
    print(f"[{dataset}] tuned+scored: {len(err)} rows, latent="
          f"{best.get('latent_space_dim')}, lr={best.get('learning_rate')}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "config/hp_autoencoder_real.yaml")
