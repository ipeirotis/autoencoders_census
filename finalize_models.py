"""
After tune_one.py has searched the architecture for a dataset (saving
best_hyperparameters.yaml), retrain that architecture as both the base AE
(p=100) and the Percentile-Loss AE (p=85) the paper reports, scoring each
through the training-fitted vectorizer.

Usage:  python finalize_models.py <dataset>
Outputs: cache/_tuned_<dataset>_ae_p100/errors.csv, _p85/errors.csv
"""
import os
import sys
import yaml

from utils import set_seed, save_model, save_to_csv, define_necessary_elements
from dataset.loader import DataLoader
from main import prepare_for_training, _clean_for_saved_vectorizer, _compute_attr_layout
from model.factory import get_model
from train.trainer import Trainer
from evaluate.outliers import get_outliers_list


def _train_and_score(dataset, best_hp, percentile, X_train, X_test, card, vec, data):
    cfg = dict(best_hp)
    cfg.update(epochs=300, batch_size=64, test_size=0.2, percentile=percentile)
    model, _ = Trainer(get_model("AE", card), cfg).train(
        dataset=X_train, prior="gaussian", X_train=X_train, X_test=X_test
    )
    out = f"cache/_tuned_{dataset}_ae_p{percentile}/"
    save_model(model, out, vectorizer=vec)
    proj = _clean_for_saved_vectorizer(data.copy(), vec)
    vectorized = vec.transform(proj).astype("float32")
    ac, aic, an = _compute_attr_layout(vec, proj.columns)
    err = get_outliers_list(vectorized, model, 1.0, ac, vec, "gaussian",
                            attr_is_categorical=aic, attr_names=an)
    save_to_csv(err, out, "errors")
    print(f"[{dataset}] p={percentile} scored {len(err)} rows -> {out}", flush=True)


def main(dataset):
    set_seed(2)
    best_path = f"cache/_tuned_{dataset}_ae/best_hyperparameters.yaml"
    if not os.path.exists(best_path):
        print(f"[{dataset}] no best_hyperparameters.yaml (run tune_one.py first)")
        return
    best_hp = yaml.safe_load(open(best_path))

    dc, rc, ic, adc, arc, aic = define_necessary_elements(dataset, None, None, None)
    loader = DataLoader(dc, rc, ic, additional_drop_columns=adc,
                        additional_rename_columns=arc, additional_columns_of_interest=aic)
    data, meta = loader.load_data(dataset)
    _, X_train, X_test, vec, card = prepare_for_training(
        data, meta.get("variable_types", {}), test_size=0.2
    )
    for p in (100, 85):
        _train_and_score(dataset, best_hp, p, X_train, X_test, card, vec, data)


if __name__ == "__main__":
    main(sys.argv[1])
