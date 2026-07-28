"""
Train each dataset's tuned AE architecture under several random seeds (p=100 and
p=85) and score each, so the detection metrics can be reported as a MEDIAN across
seeds. This removes the single-run variance that can make a tuned model look
worse than the paper on one dataset (the search optimizes reconstruction, a noisy
proxy for detection).

Run from the repo root:
  python -m experiment_b.multiseed_ae <dataset> [n_seeds]        # all seeds x {100,85}
  python -m experiment_b.multiseed_ae <dataset> --cell <seed> <p>  # one build only
Outputs: cache/_ms_<dataset>_p{100,85}_s{seed}/errors.csv

Each build calls tf.keras.backend.clear_session() first, and the ``--cell`` mode
runs exactly one build per process, so TensorFlow graph/optimizer state never
accumulates across builds (the source of the Adam "Incompatible shapes" crash
when many models are trained in a single long-running process).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import tensorflow as tf

from utils import set_seed, save_to_csv, define_necessary_elements, training_batch_size
from dataset.loader import DataLoader
from main import prepare_for_training, _clean_for_saved_vectorizer, _compute_attr_layout
from model.factory import get_model
from train.trainer import Trainer
from evaluate.outliers import get_outliers_list


def _load(dataset):
    best_path = f"cache/_tuned_{dataset}_ae/best_hyperparameters.yaml"
    if not os.path.exists(best_path):
        print(f"[{dataset}] no best_hyperparameters.yaml")
        return None, None, None
    best = yaml.safe_load(open(best_path))
    dc, rc, ic, adc, arc, aic = define_necessary_elements(dataset, None, None, None)
    loader = DataLoader(dc, rc, ic, additional_drop_columns=adc,
                        additional_rename_columns=arc, additional_columns_of_interest=aic)
    data, meta = loader.load_data(dataset)
    return best, data, meta.get("variable_types", {})


def run_cell(dataset, best, data, vt, seed, p):
    tf.keras.backend.clear_session()
    set_seed(seed)  # drives both the weight init and the train/test split
    _, X_train, X_test, vec, card = prepare_for_training(data, vt, test_size=0.2)
    proj = _clean_for_saved_vectorizer(data.copy(), vec)
    vectorized = vec.transform(proj).astype("float32")
    ac, aic2, an = _compute_attr_layout(vec, proj.columns)
    cfg = dict(best)
    cfg.update(epochs=300, batch_size=training_batch_size(dataset), test_size=0.2, percentile=p)
    model, _ = Trainer(get_model("AE", card), cfg).train(
        dataset=X_train, prior="gaussian", X_train=X_train, X_test=X_test)
    err = get_outliers_list(vectorized, model, 1.0, ac, vec, "gaussian",
                            attr_is_categorical=aic2, attr_names=an)
    save_to_csv(err, f"cache/_ms_{dataset}_p{p}_s{seed}/", "errors")
    print(f"[{dataset}] seed {seed} p{p} done", flush=True)


def main(dataset, n_seeds=5):
    best, data, vt = _load(dataset)
    if best is None:
        return
    for seed in range(1, n_seeds + 1):
        for p in (100, 85):
            run_cell(dataset, best, data, vt, seed, p)


if __name__ == "__main__":
    ds = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--cell":
        seed, p = int(sys.argv[3]), int(sys.argv[4])
        best, data, vt = _load(ds)
        if best is not None:
            run_cell(ds, best, data, vt, seed, p)
    else:
        main(ds, int(sys.argv[2]) if len(sys.argv) > 2 else 5)
