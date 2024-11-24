import pandas as pd

from model.base import VAE


def get_outliers_list(data, model, k, attr_cardinalities, vectorizer, prior):

    predictions = model.predict(data)

    z1 = None
    z2 = None

    if isinstance(predictions, tuple):
        predictions, z1, z2 = predictions

    predictions = pd.DataFrame(predictions, columns=data.columns)
    errors = pd.DataFrame()

    reconstruction_loss = VAE.reconstruction_loss(
        attr_cardinalities,
        data.to_numpy(),
        predictions.to_numpy(),
    )

    if z1 is None:
        errors["error"] = reconstruction_loss.numpy()

    else:
        if prior == "gaussian":
            kl_loss = VAE.kl_loss_gaussian(z1, z2)

        elif prior == "gumbel":
            kl_loss = VAE.kl_loss_gumbel(
                z1, model.get_config()["temperature"], len(attr_cardinalities)
            )

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


    return combined_df
