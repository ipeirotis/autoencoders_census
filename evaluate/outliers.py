import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Tuple, Union

from model.base import VAE


def compute_reconstruction_error(
    data: Union[pd.DataFrame, np.ndarray],
    predictions: Union[pd.DataFrame, np.ndarray],
    attr_cardinalities: List[int],
) -> np.ndarray:
    """
    Compute per-row reconstruction error using per-attribute categorical
    crossentropy normalized by ``log(K)``.

    This is the single source of truth for outlier scoring. It must be used by
    every code path that produces outlier rankings — the CLI ``find_outliers``
    command, the local worker in ``worker.py``, and the Vertex AI container in
    ``train/task.py`` — so that the web UI and the CLI produce identical
    rankings on the same model and the same data (TASKS.md 2.8).

    **Unseen-category handling.** Since TASKS.md 3.8, ``Table2Vector`` fits on
    the training split only with ``OneHotEncoder(handle_unknown='ignore')``.
    When the full dataset is later transformed for scoring, rows whose
    category values were not in the training split are encoded as an
    all-zero one-hot block for that attribute. Standard categorical
    crossentropy ``-sum(target * log(pred))`` returns ~0 on an all-zero
    target, which would artificially lower the error for exactly the rare
    rows we want to flag as outliers (Codex P1 on PR #46). To preserve the
    penalty, every all-zero attribute block is assigned the maximum
    normalized loss (``1.0``, i.e. ``log(K) / log(K)``) — matching the
    treatment a uniform wrong-category prediction would receive.

    For the normal case (observed categories, single active one-hot per
    attribute), this function is numerically equivalent to
    :meth:`model.base.VAE.reconstruction_loss`, so scoring stays consistent
    with the training loss the model optimized.

    Args:
        data: One-hot encoded inputs, shape ``(N, total_one_hot_dims)``.
            DataFrame or ndarray.
        predictions: Model reconstruction outputs with the same shape as
            ``data``. Expected to contain per-attribute softmax blocks
            (matching ``attr_cardinalities``).
        attr_cardinalities: List of category counts per original attribute.
            ``sum(attr_cardinalities)`` must equal ``data.shape[1]``.

    Returns:
        numpy array of shape ``(N,)`` containing per-row reconstruction error.
        Higher values are more anomalous.
    """
    data_np = data.to_numpy() if hasattr(data, "to_numpy") else np.asarray(data)
    predictions_np = (
        predictions.to_numpy()
        if hasattr(predictions, "to_numpy")
        else np.asarray(predictions)
    )

    data_t = tf.cast(data_np, tf.float32)
    predictions_t = tf.cast(predictions_np, tf.float32)

    per_attr_losses = []
    start_idx = 0
    for categories in attr_cardinalities:
        x_attr = data_t[:, start_idx : start_idx + categories]
        y_attr = predictions_t[:, start_idx : start_idx + categories]

        # Per-attribute CE normalized to [0, 1] by log(K). The max(..., 2)
        # guard matches VAE.reconstruction_loss and avoids log(1) = 0
        # blowing up on a degenerate single-category attribute.
        ce = tf.keras.backend.categorical_crossentropy(x_attr, y_attr)
        denom = np.log(max(int(categories), 2))
        ce_normalized = ce / denom

        # Unseen-category penalty: when the target one-hot block for this
        # attribute is all zeros (the row's original category was not in
        # the training split and got ignored by OneHotEncoder), standard
        # CE returns ~0 because -sum(0 * log(pred)) = 0. That would let
        # rare/unseen rows slip through the outlier ranking. Replace the
        # zero loss with the maximum normalized value (1.0) so these rows
        # are penalized at least as strongly as a uniformly wrong
        # prediction would be.
        #
        # Restriction: apply this override ONLY to categorical one-hot
        # blocks (``categories > 1``). Numeric features produced by
        # ``Table2Vector`` have ``categories == 1`` (a single
        # MinMax-scaled scalar), so a valid scaled numeric value in
        # ``[0, 0.5]`` would otherwise trip the "unseen" heuristic and
        # get forced to loss 1.0, silently distorting rankings on
        # datasets with numeric features (Codex P1 #2 on PR #46).
        if int(categories) > 1:
            attr_observed = tf.reduce_sum(x_attr, axis=1) > 0.5
            ce_final = tf.where(
                attr_observed, ce_normalized, tf.ones_like(ce_normalized)
            )
        else:
            ce_final = ce_normalized

        per_attr_losses.append(ce_final)
        start_idx += categories

    stacked = tf.stack(per_attr_losses, axis=0)  # (num_attrs, N)
    row_loss = tf.reduce_mean(stacked, axis=0)  # (N,)
    return np.asarray(row_loss.numpy())


def get_outliers_list(data, model, k, attr_cardinalities, vectorizer, prior):

    predictions = model.predict(data)

    z1 = None
    z2 = None

    if isinstance(predictions, tuple):
        predictions, z1, z2 = predictions

    predictions = pd.DataFrame(predictions, columns=data.columns, index=data.index)
    errors = pd.DataFrame(index=data.index)

    reconstruction_loss_np = compute_reconstruction_error(
        data, predictions, attr_cardinalities
    )

    if z1 is None:
        errors["error"] = reconstruction_loss_np

    else:
        if prior == "gaussian":
            kl_loss = VAE.kl_loss_gaussian(z1, z2)

        elif prior == "gumbel":
            kl_loss = VAE.kl_loss_gumbel(
                z1, model.get_config()["temperature"], len(attr_cardinalities)
            )

        kl_loss_np = kl_loss.numpy() if hasattr(kl_loss, "numpy") else np.asarray(kl_loss)
        errors["error"] = reconstruction_loss_np + k * kl_loss_np
        errors["reconstruction_loss"] = reconstruction_loss_np
        errors["kl_loss"] = kl_loss_np

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
        attr_loss_normalized = attr_loss / max(np.log(categories), 1e-10)  # Normalize by cardinality

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
