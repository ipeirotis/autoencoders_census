import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, BatchNormalization, Concatenate, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras_tuner.tuners import BayesianOptimization
from sklearn.model_selection import train_test_split

from model.loss import CustomCategoricalCrossentropyAE


@keras.utils.register_keras_serializable()
class AutoencoderModel:
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
        """
        Split the dataset into train and test

        :param df: Dataframe to split
        :return:
            X_train: train data
            X_test: test data
        """
        X_train, X_test = train_test_split(df.copy(), test_size=test_size)
        self.INPUT_SHAPE = X_train.shape[1:]
        return X_train.dropna(), X_test.dropna()

    @staticmethod
    def masked_mse(y_true, y_pred):
        """
        Compute the mean squared error between the true and predicted values
        Mask the NaN values in the true values
        :param y_true: true values
        :param y_pred: predicted values
        :return:
            Return the mean squared error
        """
        mask = tf.math.is_finite(y_true)
        y_t = tf.where(tf.math.is_finite(y_true), y_true, 0.0)
        y_p = tf.where(tf.math.is_finite(y_pred), y_pred, 0.0)
        mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        return tf.reduce_mean(
            mse(y_t * tf.cast(mask, y_t.dtype), y_p * tf.cast(mask, y_p.dtype))
        )

    def build_encoder_hp(self, hp, hp_limits):
        inputs = Input(shape=self.INPUT_SHAPE)

        for i in range(
            hp.Int(
                "encoder_layers",
                min_value=hp_limits["min_encoder_layers"],
                max_value=hp_limits["max_encoder_layers"],
            )
        ):
            x = Dense(
                units=hp.Int(
                    f"encoder_units_{i+1}",
                    min_value=hp_limits[f"min_encoder_units_{i+1}"],
                    max_value=hp_limits[f"max_encoder_units_{i+1}"],
                    step=hp_limits[f"step_encoder_units_{i+1}"],
                ),
                activation=hp.Choice(
                    f"encoder_activation_{i+1}", hp_limits[f"encoder_activation_{i+1}"]
                ),
                kernel_regularizer=l2(
                    hp.Choice(
                        f"encoder_l2_{i+1}", hp_limits[f"encoder_regularizer_{i+1}"]
                    )
                ),
            )(inputs)
            x = Dropout(
                hp.Float(
                    f"encoder_dropout_{i+1}",
                    min_value=hp_limits[f"min_encoder_dropout_{i+1}"],
                    max_value=hp_limits[f"max_encoder_dropout_{i+1}"],
                    step=hp_limits[f"step_encoder_dropout_{i+1}"],
                )
            )(x)

            if hp.Boolean(
                f"encoder_batch_norm_{i+1}", hp_limits[f"encoder_batch_norm_{i+1}"]
            ):
                x = BatchNormalization()(x)

        latent_space = Dense(
            units=hp.Int(
                "latent_space_dim",
                min_value=hp_limits["latent_space_min"],
                max_value=hp_limits["latent_space_max"],
                step=hp_limits["latent_space_step"],
            ),
            activation=hp.Choice("latent_activation", hp_limits["latent_activation"]),
        )(x)

        return Model(inputs, latent_space)

    def build_encoder(self, config):
        inputs = Input(shape=self.INPUT_SHAPE)

        for i in range(config.get("encoder_layers", 2)):
            x = Dense(
                units=config.get(f"encoder_units_{i+1}", 160),
                activation=config.get(f"encoder_activation_{i+1}", "relu"),
                kernel_regularizer=l2(config.get(f"encoder_l2_{i+1}", 0.001)),
            )(inputs)
            x = Dropout(config.get(f"encoder_dropout_{i+1}", 0.1))(x)

            if config.get(f"encoder_batch_norm_{i+1}", True):
                x = BatchNormalization()(x)

        latent_space = Dense(
            units=config.get("latent_space_dim", 2),
            activation=config.get("latent_activation", "relu"),
        )(x)

        return Model(inputs, latent_space)

    def build_decoder_hp(self, hp, hp_limits):
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

        for i in range(
            hp.Int(
                "decoder_layers",
                min_value=hp_limits["min_decoder_layers"],
                max_value=hp_limits["max_decoder_layers"],
            )
        ):
            x = Dense(
                units=hp.Int(
                    f"decoder_units_{i+1}",
                    min_value=hp_limits[f"min_decoder_units_{i+1}"],
                    max_value=hp_limits[f"max_decoder_units_{i+1}"],
                    step=hp_limits[f"step_decoder_units_{i+1}"],
                ),
                activation=hp.Choice(
                    f"decoder_activation_{i+1}", hp_limits[f"decoder_activation_{i+1}"]
                ),
                kernel_regularizer=l2(
                    hp.Choice(
                        f"decoder_l2_{i+1}", hp_limits[f"decoder_regularizer_{i+1}"]
                    )
                ),
            )(decoder_inputs)
            x = Dropout(
                hp.Float(
                    f"decoder_dropout_{i+1}",
                    min_value=hp_limits[f"min_decoder_dropout_{i+1}"],
                    max_value=hp_limits[f"max_decoder_dropout_{i+1}"],
                    step=hp_limits[f"step_decoder_dropout_{i+1}"],
                )
            )(x)

            if hp.Boolean(
                f"decoder_batch_norm_{i+1}", hp_limits[f"decoder_batch_norm_{i+1}"]
            ):
                x = BatchNormalization()(x)

        decoded_attrs = []
        for categories in self.attribute_cardinalities:
            decoder_softmax = Dense(categories, activation="softmax")(x)
            decoded_attrs.append(decoder_softmax)

        outputs = Concatenate()(decoded_attrs)

        return Model(decoder_inputs, outputs)

    def build_decoder(self, config):
        decoder_inputs = Input(shape=(config.get("latent_space_dim", 2),))

        for i in range(config.get("decoder_layers", 2)):
            x = Dense(
                units=config.get(f"decoder_units_{i+1}", 160),
                activation=config.get(f"decoder_activation_{i+1}", "relu"),
                kernel_regularizer=l2(config.get(f"decoder_l2_{i+1}", 0.001)),
            )(decoder_inputs)
            x = Dropout(config.get(f"decoder_dropout_{i+1}", 0.1))(x)

            if config.get(f"decoder_batch_norm_{i+1}", True):
                x = BatchNormalization()(x)

        decoded_attrs = []
        for categories in self.attribute_cardinalities:
            decoder_softmax = Dense(categories, activation="softmax")(x)
            decoded_attrs.append(decoder_softmax)

        outputs = Concatenate()(decoded_attrs)

        return Model(decoder_inputs, outputs)

    def build_autoencoder_hp(self, hp, hp_limits):
        learning_rate = hp.Choice("learning_rate", values=hp_limits["learning_rate"])

        autoencoder_input = Input(shape=self.INPUT_SHAPE)
        encoder_output = self.build_encoder_hp(hp, hp_limits)(autoencoder_input)
        decoder_output = self.build_decoder_hp(hp, hp_limits)(encoder_output)
        autoencoder = Model(autoencoder_input, decoder_output)
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=CustomCategoricalCrossentropyAE(
                attribute_cardinalities=self.attribute_cardinalities
            ),
        )

        return autoencoder

    def build_autoencoder(self, config):
        learning_rate = config.get("learning_rate", 1e-3)

        autoencoder_input = Input(shape=self.INPUT_SHAPE)
        encoder_output = self.build_encoder(config)(autoencoder_input)
        decoder_output = self.build_decoder(config)(encoder_output)
        autoencoder = Model(autoencoder_input, decoder_output)
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=CustomCategoricalCrossentropyAE(
                attribute_cardinalities=self.attribute_cardinalities
            ),
        )

        return autoencoder

    def define_tuner(self, hp_limits):

        build_fn = lambda hp: self.build_autoencoder_hp(hp, hp_limits)

        tuner = BayesianOptimization(
            build_fn,
            objective="val_loss",
            max_trials=hp_limits["max_trials"],
            executions_per_trial=hp_limits["executions_per_trial"],
            project_name=hp_limits["run_name"]
        )
        return tuner
