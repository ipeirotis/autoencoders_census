import logging

import pandas as pd

logger = logging.Logger(__name__)


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def train(self, dataset: pd.DataFrame):
        X_train, X_test = self.model.split_train_test(dataset, self.config["test_size"])

        model = self.model.build_autoencoder(self.config)
        logger.info(model.summary())

        history = model.fit(
            X_train,
            X_train,
            epochs=self.config["epochs"],
            verbose=1,
            validation_data=(X_test, X_test),
        )

        return model, history

    def search_hyperparameters(self, dataset: pd.DataFrame):
        X_train, X_test = self.model.split_train_test(dataset, self.config["test_size"])

        tuner = self.model.define_tuner(self.config)
        tuner.search(
            X_train,
            X_train,
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            validation_data=(X_test, X_test),
        )

        best_hps = tuner.get_best_hyperparameters()[0]

        return best_hps.values
