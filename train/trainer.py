import logging

import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from model.variational_autoencoder import VariationalAutoencoderModel

logger = logging.Logger(__name__)


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def train(self, dataset: pd.DataFrame, prior):
        X_train, X_test = self.model.split_train_test(dataset, self.config["test_size"])

        if isinstance(self.model, VariationalAutoencoderModel):
            model = self.model.build_autoencoder(self.config, prior)

        else:
            model = self.model.build_autoencoder(self.config)

        logger.info(model.summary())

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        history = model.fit(
            X_train,
            X_train,
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            verbose=1,
            validation_data=(X_test, X_test),
            callbacks=[early_stopping],
        )

        return model, history

    def search_hyperparameters(self, dataset: pd.DataFrame, prior):
        X_train, X_test = self.model.split_train_test(dataset, self.config["test_size"])

        if isinstance(self.model, VariationalAutoencoderModel):
            tuner = self.model.define_tuner(self.config, prior)

        else:
            tuner = self.model.define_tuner(self.config)

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, X_test))

        train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(
            self.config["batch_size"]
        )

        test_dataset = test_dataset.batch(self.config["batch_size"])

        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        train_dataset = train_dataset.cache()

        tuner.search(
            train_dataset,
            epochs=self.config["epochs"],
            validation_data=test_dataset,
            callbacks=[early_stopping],
        )

        best_hps = tuner.get_best_hyperparameters()[0]

        return best_hps.values
