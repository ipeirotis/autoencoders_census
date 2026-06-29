"""
Train each dataset's tuned AE architecture under several random seeds (p=100 and
p=85) and score each, so the detection metrics can be reported as a MEDIAN across
seeds. This removes the single-run variance that can make a tuned model look
worse than the paper on one dataset (the search optimizes reconstruction, a noisy
proxy for detection).

Run from the repo root:  python -m experiment_b.multiseed_ae <dataset> [n_seeds]
Outputs: cache/_ms_<dataset>_p{100,85}_s{seed}/errors.csv
"""
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import set_seed, save_to_csv, define_necessary_elements
from dataset.loader import DataLoader
from main import prepare_for_training, _clean_for_saved_vectorizer, _compute_attr_layout
from model.factory import get_model
from train.trainer import Trainer
from evaluate.outliers import get_outliers_list


def main(dataset, n_seeds=5):
    best_path = f"cache/_tuned_{dataset}_ae/best_hyperparameters.yaml"
    if not os.path.exists(best_path):
        print(f"[{dataset}] no best_hyperparameters.yaml")
        return
    best = yaml.safe_load(open(best_path))

    dc, rc, ic, adc, arc, aic = define_necessary_elements(dataset, None, None, None)
    loader = DataLoader(dc, rc, ic, additional_drop_columns=adc,
                        additional_rename_columns=arc, additional_columns_of_interest=aic)
    data, meta = loader.load_data(dataset)
    vt = meta.get("variable_types", {})

    for seed in range(1, n_seeds + 1):
        set_seed(seed)  # drives both the weight init and the train/test split
        _, X_train, X_test, vec, card = prepare_for_training(data, vt, test_size=0.2)
        proj = _clean_for_saved_vectorizer(data.copy(), vec)
        vectorized = vec.transform(proj).astype("float32")
        ac, aic2, an = _compute_attr_layout(vec, proj.columns)
        for p in (100, 85):
            cfg = dict(best)
            cfg.update(epochs=300, batch_size=64, test_size=0.2, percentile=p)
            model, _ = Trainer(get_model("AE", card), cfg).train(
                dataset=X_train, prior="gaussian", X_train=X_train, X_test=X_test)
            err = get_outliers_list(vectorized, model, 1.0, ac, vec, "gaussian",
                                    attr_is_categorical=aic2, attr_names=an)
            save_to_csv(err, f"cache/_ms_{dataset}_p{p}_s{seed}/", "errors")
        print(f"[{dataset}] seed {seed} done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
