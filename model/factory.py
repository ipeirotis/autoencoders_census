from model.autoencoder import AutoencoderModel
from model.variational_autoencoder import VariationalAutoencoderModel


def get_model(model: str, cardinalities: list):

    if model == "AE":
        return AutoencoderModel(cardinalities)

    if model == "VAE":
        return VariationalAutoencoderModel(cardinalities)

    raise ValueError("Model not found")
