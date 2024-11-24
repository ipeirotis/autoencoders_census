import pandas as pd
import tensorflow as tf


class Generator:
    def __init__(
        self,
        model,
        number_of_samples,
        prior,
        number_of_classes,
        temperature,
        columns,
        means,
        logvars,
    ):
        self.model = model
        self.number_of_samples = number_of_samples
        self.prior = prior
        self.number_of_classes = number_of_classes
        self.temperature = temperature
        self.latent_dim = model.input_shape[-1]
        self.columns = columns
        self.means = means
        self.log_vars = logvars

    # def generate_gaussian(self):
    #     z = tf.random.normal(shape=(self.number_of_samples, self.latent_dim))
    #
    #     return z

    def generate_specific_gaussians(self):
        """
        Samples one point from each Gaussian defined by the given means and log variances.

        Args:
            means (tf.Tensor): A tensor of shape (100,) representing the means of the Gaussians.
            log_vars (tf.Tensor): A tensor of shape (100,) representing the log variances of the Gaussians.

        Returns:
            tf.Tensor: A tensor of shape (100, latent_dim) containing sampled points.
        """
        # Ensure means and log_vars are tensors
        means = tf.convert_to_tensor(self.means, dtype=tf.float32)
        log_vars = tf.convert_to_tensor(self.log_vars, dtype=tf.float32)

        # Compute the standard deviation from the log variance
        stds = tf.exp(0.5 * log_vars)

        # Sample from a standard normal distribution
        epsilon = tf.random.normal(shape=(self.number_of_samples, tf.shape(means)[0]))

        # Reparameterize to sample from the desired Gaussians
        z = (
            means + epsilon * stds
        )  # Broadcasting (means, stds) over `number_of_samples`

        return z

    def generate_gumbel(self):
        logits = tf.random.uniform(
            shape=(self.number_of_samples, self.latent_dim),
            minval=-1,
            maxval=1,
        )
        z = self.gumbel_softmax_sample(logits)
        z = tf.reshape(z, (self.number_of_samples, -1))

        noise = tf.random.normal(shape=tf.shape(z), mean=0.0, stddev=0.1)
        z += noise

        return z

    @staticmethod
    def sample_gumbel(shape, eps=1e-20):
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits):
        logits = tf.reshape(
            logits,
            (-1, self.number_of_classes, int(logits.shape[1] / self.number_of_classes)),
        )

        y = logits + self.sample_gumbel(tf.shape(logits))
        return tf.nn.softmax(y / self.temperature, axis=-1)

    def generate_wo_condition(self):
        if self.prior == "gaussian":
            z = self.generate_specific_gaussians()

        elif self.prior == "gumbel":
            z = self.generate_gumbel()

        else:
            raise ValueError("Invalid prior")

        samples = self.model.predict(z)

        return samples

    def generate_w_condition(self, feature_indices, output_dimensionality):
        if self.prior == "gaussian":
            z = tf.Variable(self.generate_gaussian(), trainable=True)

        elif self.prior == "gumbel":
            z = tf.Variable(self.generate_gumbel(), trainable=True)

        else:
            raise ValueError("Invalid prior")

        optimizer = tf.optimizers.Adam(0.01)

        target_mask = tf.constant(
            [1 if i in feature_indices else 0 for i in range(output_dimensionality)],
            dtype=tf.float32,
        )
        target_mask = tf.tile(
            tf.reshape(target_mask, [1, output_dimensionality]),
            [self.number_of_samples, 1],
        )

        for step in range(70):
            with tf.GradientTape() as tape:
                tape.watch(z)
                generated_sample = self.model(z)

                masked_diff = (generated_sample - target_mask) * target_mask
                loss = tf.reduce_mean(tf.square(masked_diff))

            grads = tape.gradient(loss, [z])
            grads_and_vars = [
                (grad, var) for grad, var in zip(grads, [z]) if grad is not None
            ]
            optimizer.apply_gradients(grads_and_vars)

            print(f"Step {step}, Loss: {loss.numpy()}")

        final_samples = self.model(z)

        return final_samples

    def generate(self, vectorizer, target_features, output_dimensionality):
        if target_features is None:
            predictions = self.generate_wo_condition()

        else:
            predictions = self.generate_w_condition(
                target_features, output_dimensionality
            )

        predicted = pd.DataFrame(predictions)
        predicted.columns = self.columns

        tabular_from_predicted = vectorizer.tabularize_vector(predicted)

        return tabular_from_predicted


## add row to a df
# def add_row(df, row):
#     df.loc[-1] = row
#     df.index = df.index + 1
#     df = df.sort_index()
