import numpy as np
import pandas as pd
import tensorflow as tf

from model.base import VAE


def get_outliers_list(data, model, k, attr_cardinalities, vectorizer):

    predictions = model.predict(data)

    z_mean = None
    z_log_var = None

    if isinstance(predictions, tuple):
        predictions, z_mean, z_log_var = predictions

    predictions = pd.DataFrame(predictions, columns=data.columns)
    errors = pd.DataFrame()

    reconstruction_loss = VAE.reconstruction_loss(
        attr_cardinalities,
        data.to_numpy(),
        predictions.to_numpy(),
        find_outliers=True,
    )

    if z_mean is None:
        errors["error"] = reconstruction_loss.numpy()

    else:
        kl_loss = VAE.kl_loss(z_mean, z_log_var, find_outliers=True)

        custom_loss = reconstruction_loss + k * kl_loss
        errors["error"] = custom_loss.numpy()
        errors["reconstruction_loss"] = reconstruction_loss.numpy()
        errors["kl_loss"] = kl_loss.numpy()

    combined_df = pd.concat(
        [
            vectorizer.tabularize_vector(data),
            vectorizer.tabularize_vector(predictions),
            errors,
        ],
        axis=1,
    )
    combined_df = combined_df.sort_values("error", ascending=False)

    return combined_df
