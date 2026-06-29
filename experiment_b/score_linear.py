"""
Score a trained LinearAutoencoder (model_name=PCA) by MEAN SQUARED reconstruction
error -- matching its MSE training loss -- and write errors.csv. The CLI
``find_outliers`` path instead uses categorical cross-entropy, which is the wrong
objective for the linear model's raw (non-softmax) reconstruction.

Run from the repo root:  python experiment_b/score_linear.py <dataset>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from utils import define_necessary_elements, load_vectorizer, save_to_csv
from dataset.loader import DataLoader
from main import _clean_for_saved_vectorizer


def main(dataset):
    out = f"cache/_lin_{dataset}/"
    model_path = f"{out}autoencoder"
    if not os.path.exists(model_path):
        print(f"[{dataset}] no trained linear model at {model_path}")
        return

    dc, rc, ic, adc, arc, aic = define_necessary_elements(dataset, None, None, None)
    loader = DataLoader(dc, rc, ic, additional_drop_columns=adc,
                        additional_rename_columns=arc, additional_columns_of_interest=aic)
    data, _ = loader.load_data(dataset)
    vec = load_vectorizer(model_path)
    proj = _clean_for_saved_vectorizer(data.copy(), vec)
    X = vec.transform(proj).astype("float32")

    model = tf.keras.models.load_model(model_path)
    x_hat = model.predict(X.values, verbose=0)
    mse = ((X.values - x_hat) ** 2).mean(axis=1)

    errors = vec.tabularize_vector(X).copy()
    errors["error"] = mse
    save_to_csv(errors, out, "errors")
    print(f"[{dataset}] linear AE MSE-scored {len(mse)} rows -> {out}errors.csv")


if __name__ == "__main__":
    main(sys.argv[1])
