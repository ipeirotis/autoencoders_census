import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Tuple

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


def compute_per_column_contributions(
    data: np.ndarray,
    predictions: np.ndarray,
    attr_cardinalities: List[int],
    column_names: List[str]
) -> List[Tuple[str, float]]:
    """
    Decompose reconstruction loss into per-column contribution percentages.

    This function helps researchers understand which survey questions (columns)
    contributed most to a row being flagged as an outlier. It decomposes the
    total reconstruction loss into per-attribute components and expresses them
    as percentages.

    Args:
        data: True values (one-hot encoded), shape [N, total_features]
        predictions: Model predictions (softmax outputs), shape [N, total_features]
        attr_cardinalities: List of category counts per attribute
        column_names: List of original column names (length = len(attr_cardinalities))

    Returns:
        List of (column_name, contribution_percentage) tuples, sorted descending by percentage

    Example:
        >>> data = np.array([[1, 0, 0, 0, 1, 0]])  # 2 columns with 3 categories each
        >>> predictions = np.array([[0.7, 0.2, 0.1, 0.1, 0.7, 0.2]])
        >>> attr_cardinalities = [3, 3]
        >>> column_names = ['Q1', 'Q2']
        >>> contributions = compute_per_column_contributions(data, predictions, attr_cardinalities, column_names)
        >>> # Returns: [('Q2', 60.5), ('Q1', 39.5)]  # Q2 contributed more to reconstruction error
    """
    per_attr_losses = []
    start_idx = 0

    for categories in attr_cardinalities:
        x_attr = data[:, start_idx : start_idx + categories]
        y_attr = predictions[:, start_idx : start_idx + categories]

        x_attr = tf.cast(x_attr, tf.float32)
        y_attr = tf.cast(y_attr, tf.float32)

        # Categorical crossentropy per attribute (same as VAE.reconstruction_loss)
        attr_loss = tf.keras.backend.categorical_crossentropy(x_attr, y_attr)
        attr_loss_normalized = attr_loss / np.log(categories)  # Normalize by cardinality

        # Mean loss for this attribute across batch (or single row)
        per_attr_losses.append(tf.reduce_mean(attr_loss_normalized).numpy())
        start_idx += categories

    # Convert to contribution percentages
    total_loss = sum(per_attr_losses)

    if total_loss == 0:
        # Fallback: equal contribution if no loss (avoid division by zero)
        contributions = [
            (column_names[i], 100.0 / len(per_attr_losses))
            for i in range(len(per_attr_losses))
        ]
    else:
        contributions = [
            (column_names[i], float((per_attr_losses[i] / total_loss) * 100.0))
            for i in range(len(per_attr_losses))
        ]

    # Normalize to exactly 100% (address potential floating-point rounding issues)
    total_pct = sum(pct for _, pct in contributions)
    contributions = [
        (col, float((pct / total_pct) * 100.0))
        for col, pct in contributions
    ]

    # Sort descending by contribution (highest contributors first)
    contributions.sort(key=lambda x: x[1], reverse=True)

    return contributions
