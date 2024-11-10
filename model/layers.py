from keras.layers import Reshape, Flatten, Input, Dense, BatchNormalization, Concatenate, Dropout
from keras.models import Model
from keras.regularizers import l2


def build_decoder(attribute_cardinalities, config):

	decoder_inputs = Input(shape=(config.get("latent_space_dim", 2), len(attribute_cardinalities)), name="decoder_input")
	x = Flatten()(decoder_inputs)

	for i in range(config.get("decoder_layers", 2)):
		x = Dense(
		units=config.get(f"decoder_units_{i + 1}", 160),
		activation=config.get(f"decoder_activation_{i + 1}", "relu"),
		kernel_regularizer=l2(config.get(f"decoder_l2_{i + 1}", 0.001)),
		name=f"decoder_layer_{i+1}",
                )(x)
		x = Dropout(config.get(f"decoder_dropout_{i + 1}", 0.1), name=f"decoder_dropout_{i+1}")(x)

		if config.get(f"decoder_batch_norm_{i + 1}", True):
			x = BatchNormalization(name=f"decoder_batch_norm_{i+1}")(x)

	decoded_attrs = []
	for idx, categories in enumerate(attribute_cardinalities):
		decoder_softmax = Dense(categories, activation="softmax", name=f"decoder_output_{idx}")(x)
		decoded_attrs.append(decoder_softmax)

	outputs = Concatenate(name="decoder_concat")(decoded_attrs)

	return Model(decoder_inputs, outputs)


def build_encoder(input_shape, config, num_cat_var):
	inputs = Input(shape=input_shape, name="encoder_input")
	x = inputs

	for i in range(config.get("encoder_layers", 2)):
		x = Dense(
		units=config.get(f"encoder_units_{i+1}", 160),
		activation=config.get(f"encoder_activation_{i+1}", "relu"),
		kernel_regularizer=l2(config.get(f"encoder_l2_{i+1}", 0.001)),
		name=f"encoder_layer_{i+1}",
                )(x)
		x = Dropout(config.get(f"encoder_dropout_{i+1}", 0.1), name=f"encoder_dropout_{i+1}")(x)

		if config.get(f"encoder_batch_norm_{i+1}", True):
			x = BatchNormalization(name = f"encoder_batch_norm_{i+1}")(x)

	#z_mean = Dense(
	#    units=config.get("latent_space_dim", 2),
	#    activation=config.get("latent_activation", "linear"),
        #    name="encoder_z_mean",
	#)(x)

	#z_log_var = Dense(
	#    units=config.get("latent_space_dim", 2),
	#    activation=config.get("latent_activation", "linear"),
        #    name="encoder_z_logvar",
	#)(x)
	z = Dense(
            units=config.get("latent_space_dim", 2) * num_cat_var,
            activation=config.get("latent_activation", "linear"),
        )(x)

	z = Reshape((config.get("latent_space_dim", 2), num_cat_var))(z)

	return Model(inputs, z)
