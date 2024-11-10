import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Concatenate, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras_tuner.tuners import BayesianOptimization
from sklearn.model_selection import train_test_split
from tensorflow import keras

from model.base import VAE
from model.layers import build_encoder, build_decoder

tf.config.run_functions_eagerly(True)


class VariationalAutoencoderModel:
    def __init__(self, attribute_cardinalities):
        self.INPUT_SHAPE = None
        self.attribute_cardinalities = attribute_cardinalities

        log_cardinalities = [
            np.log(cardinality) for cardinality in self.attribute_cardinalities
        ]
        log_cardinalities_tensor = tf.constant(log_cardinalities, dtype=tf.float32)
        self.log_cardinalities_expanded = tf.expand_dims(
            log_cardinalities_tensor, axis=-1
        )

    def get_config(self):
        return {"attribute_cardinalities": self.attribute_cardinalities}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def split_train_test(self, df, test_size=0.2):
        X_train, X_test = train_test_split(df.copy(), test_size=test_size)
        self.INPUT_SHAPE = X_train.shape[1:]
        return X_train.dropna(), X_test.dropna()

    def build_encoder_hp(self, hp, hp_limits, prior, num_classes):
        inputs = Input(shape=self.INPUT_SHAPE)
        x = inputs

        for i in range(
            hp.Int(
                "encoder_layers",
                min_value=hp_limits["min_encoder_layers"],
                max_value=hp_limits["max_encoder_layers"],
            )
        ):
            x = Dense(
                units=hp.Int(
                    f"encoder_units_{i + 1}",
                    min_value=hp_limits[f"min_encoder_units_{i + 1}"],
                    max_value=hp_limits[f"max_encoder_units_{i + 1}"],
                    step=hp_limits[f"step_encoder_units_{i + 1}"],
                ),
                activation=hp.Choice(
                    f"encoder_activation_{i + 1}",
                    hp_limits[f"encoder_activation_{i + 1}"],
                ),
                kernel_regularizer=l2(
                    hp.Choice(
                        f"encoder_l2_{i + 1}", hp_limits[f"encoder_regularizer_{i + 1}"]
                    )
                ),
            )(x)
            x = Dropout(
                hp.Float(
                    f"encoder_dropout_{i + 1}",
                    min_value=hp_limits[f"min_encoder_dropout_{i + 1}"],
                    max_value=hp_limits[f"max_encoder_dropout_{i + 1}"],
                    step=hp_limits[f"step_encoder_dropout_{i + 1}"],
                )
            )(x)

            if hp.Boolean(
                f"encoder_batch_norm_{i + 1}", hp_limits[f"encoder_batch_norm_{i + 1}"]
            ):
                x = BatchNormalization()(x)

        if prior == "gaussian":
            z_mean = Dense(
                units=hp.Int(
                    "latent_space_dim",
                    min_value=hp_limits["latent_space_min"],
                    max_value=hp_limits["latent_space_max"],
                    step=hp_limits["latent_space_step"],
                ),
                activation=hp.Choice(
                    "latent_activation", hp_limits["latent_activation"]
                ),
            )(x)

            z_log_var = Dense(
                units=hp.Int(
                    "latent_space_dim",
                    min_value=hp_limits["latent_space_min"],
                    max_value=hp_limits["latent_space_max"],
                    step=hp_limits["latent_space_step"],
                ),
                activation=hp.Choice(
                    "latent_activation", hp_limits["latent_activation"]
                ),
            )(x)

            return Model(inputs, [z_mean, z_log_var])

        elif prior == "gumbel":
            z = Dense(
                units=hp.Int(
                    "latent_space_dim",
                    min_value=hp_limits["latent_space_min"],
                    max_value=hp_limits["latent_space_max"],
                    step=hp_limits["latent_space_step"],
                )
                * num_classes,
                activation=hp.Choice(
                    "latent_activation", hp_limits["latent_activation"]
                ),
            )(x)

            return Model(inputs, z)

        else:
            raise ValueError("Invalid prior")

    def build_decoder_hp(self, hp, hp_limits, prior, num_classes):

        if prior == "gaussian":
            decoder_inputs = Input(
                shape=(
                    hp.Int(
                        "latent_space_dim",
                        min_value=hp_limits["latent_space_min"],
                        max_value=hp_limits["latent_space_max"],
                        step=hp_limits["latent_space_step"],
                    ),
                )
            )
        elif prior == "gumbel":
            decoder_inputs = Input(
                shape=(
                    hp.Int(
                        "latent_space_dim",
                        min_value=hp_limits["latent_space_min"],
                        max_value=hp_limits["latent_space_max"],
                        step=hp_limits["latent_space_step"],
                    )
                    * num_classes,
                )
            )
        else:
            raise ValueError("Invalid prior")

        x = decoder_inputs

        for i in range(
            hp.Int(
                "decoder_layers",
                min_value=hp_limits["min_decoder_layers"],
                max_value=hp_limits["max_decoder_layers"],
            )
        ):
            x = Dense(
                units=hp.Int(
                    f"decoder_units_{i + 1}",
                    min_value=hp_limits[f"min_decoder_units_{i + 1}"],
                    max_value=hp_limits[f"max_decoder_units_{i + 1}"],
                    step=hp_limits[f"step_decoder_units_{i + 1}"],
                ),
                activation=hp.Choice(
                    f"decoder_activation_{i + 1}",
                    hp_limits[f"decoder_activation_{i + 1}"],
                ),
                kernel_regularizer=l2(
                    hp.Choice(
                        f"decoder_l2_{i + 1}", hp_limits[f"decoder_regularizer_{i + 1}"]
                    )
                ),
            )(x)
            x = Dropout(
                hp.Float(
                    f"decoder_dropout_{i + 1}",
                    min_value=hp_limits[f"min_decoder_dropout_{i + 1}"],
                    max_value=hp_limits[f"max_decoder_dropout_{i + 1}"],
                    step=hp_limits[f"step_decoder_dropout_{i + 1}"],
                )
            )(x)

            if hp.Boolean(
                f"decoder_batch_norm_{i + 1}", hp_limits[f"decoder_batch_norm_{i + 1}"]
            ):
                x = BatchNormalization()(x)

        decoded_attrs = []
        for categories in self.attribute_cardinalities:
            decoder_softmax = Dense(categories, activation="softmax")(x)
            decoded_attrs.append(decoder_softmax)

        outputs = Concatenate()(decoded_attrs)

        return Model(decoder_inputs, outputs)

    @staticmethod
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def build_autoencoder_hp(self, hp, hp_limits, prior):
        learning_rate = hp.Choice("learning_rate", values=hp_limits["learning_rate"])
        kl_loss_weight = hp.Float(
            "kl_loss_weight",
            min_value=hp_limits["kl_loss_weight_min"],
            max_value=hp_limits["kl_loss_weight_max"],
        )

        encoder = self.build_encoder_hp(
            hp, hp_limits, prior, len(self.attribute_cardinalities)
        )
        decoder = self.build_decoder_hp(
            hp, hp_limits, prior, len(self.attribute_cardinalities)
        )

        autoencoder = VAE(
            encoder,
            decoder,
            self.attribute_cardinalities,
            kl_loss_weight,
            hp_limits,
            self.INPUT_SHAPE,
            hp_limits.get("temperature", 1),
            prior,
        )

        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
        )

        return autoencoder

    def build_autoencoder(self, config, prior):
        learning_rate = config.get("learning_rate", 1e-3)
        kl_loss_weight = config.get("kl_loss_weight", 1)

        encoder = build_encoder(
            self.INPUT_SHAPE, config, prior, len(self.attribute_cardinalities)
        )
        decoder = build_decoder(
            self.attribute_cardinalities,
            config,
            prior,
            len(self.attribute_cardinalities),
        )

        autoencoder = VAE(
            encoder,
            decoder,
            self.attribute_cardinalities,
            kl_loss_weight,
            config,
            self.INPUT_SHAPE,
            config.get("temperature", 1),
            prior,
        )

        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
        )

        return autoencoder

    def define_tuner(self, hp_limits, prior):

        build_fn = lambda hp: self.build_autoencoder_hp(hp, hp_limits, prior)

        tuner = BayesianOptimization(
            build_fn,
            objective="val_loss",
            max_trials=hp_limits["max_trials"],
            executions_per_trial=hp_limits["executions_per_trial"],
            project_name=hp_limits["run_name"],
        )
        return tuner
