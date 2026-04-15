import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Optional, Tuple, Union

from model.base import VAE


def _compute_per_attr_losses_tf(
    data_t: "tf.Tensor",
    predictions_t: "tf.Tensor",
    attr_cardinalities: List[int],
    attr_is_categorical: List[bool],
) -> "tf.Tensor":
    """
    Internal helper: compute per-attribute CE losses entirely in TensorFlow.

    Returns a tensor of shape ``(num_attrs, N)`` where entry ``[j, i]`` is
    the normalized categorical crossentropy for row ``i`` on attribute ``j``.
    Keeping the tensor in TF allows callers that only need the row-level
    mean to call ``tf.reduce_mean(result, axis=0)`` and convert a small
    ``(N,)`` array to NumPy, rather than materializing the full
    ``(N, num_attrs)`` matrix unnecessarily.

    Args:
        data_t: Float32 TF tensor, shape ``(N, total_one_hot_dims)``.
        predictions_t: Float32 TF tensor, same shape as ``data_t``.
        attr_cardinalities: Per-attribute category counts.
        attr_is_categorical: Per-attribute categorical flag (must be the
            same length as ``attr_cardinalities``; no default here).

    Returns:
        TF tensor of shape ``(num_attrs, N)``.
    """
    per_attr_losses = []
    start_idx = 0
    for categories, is_categorical in zip(attr_cardinalities, attr_is_categorical):
        x_attr = data_t[:, start_idx : start_idx + categories]
        y_attr = predictions_t[:, start_idx : start_idx + categories]

        ce = tf.keras.backend.categorical_crossentropy(x_attr, y_attr)
        denom = np.log(max(int(categories), 2))
        ce_normalized = ce / denom

        if is_categorical:
            attr_observed = tf.reduce_sum(x_attr, axis=1) > 0.5
            ce_final = tf.where(
                attr_observed, ce_normalized, tf.ones_like(ce_normalized)
            )
        else:
            ce_final = ce_normalized

        per_attr_losses.append(ce_final)
        start_idx += categories

    return tf.stack(per_attr_losses, axis=0)  # (num_attrs, N)


def _prepare_inputs(
    data: Union[pd.DataFrame, np.ndarray],
    predictions: Union[pd.DataFrame, np.ndarray],
    attr_cardinalities: List[int],
    attr_is_categorical: Optional[List[bool]],
) -> tuple:
    """Validate inputs and cast to float32 TF tensors. Returns
    ``(data_t, predictions_t, attr_is_categorical)``."""
    data_np = data.to_numpy() if hasattr(data, "to_numpy") else np.asarray(data)
    predictions_np = (
        predictions.to_numpy()
        if hasattr(predictions, "to_numpy")
        else np.asarray(predictions)
    )
    if attr_is_categorical is None:
        attr_is_categorical = [True] * len(attr_cardinalities)
    if len(attr_is_categorical) != len(attr_cardinalities):
        raise ValueError(
            f"attr_is_categorical length ({len(attr_is_categorical)}) must "
            f"match attr_cardinalities length ({len(attr_cardinalities)})"
        )
    return (
        tf.cast(data_np, tf.float32),
        tf.cast(predictions_np, tf.float32),
        attr_is_categorical,
    )


def compute_per_column_errors(
    data: Union[pd.DataFrame, np.ndarray],
    predictions: Union[pd.DataFrame, np.ndarray],
    attr_cardinalities: List[int],
    attr_is_categorical: Optional[List[bool]] = None,
) -> np.ndarray:
    """
    Compute per-row, per-attribute reconstruction error.

    Returns an array of shape ``(N, num_attrs)`` where entry ``[i, j]`` is
    the normalized categorical crossentropy for row ``i`` on attribute ``j``.
    Taking ``result.mean(axis=1)`` recovers the aggregate per-row score
    returned by :func:`compute_reconstruction_error`.

    This function applies the same unseen-category handling and normalization
    as :func:`compute_reconstruction_error` — see that function's docstring
    for a full description of the algorithm.

    Args:
        data: One-hot encoded inputs, shape ``(N, total_one_hot_dims)``.
        predictions: Model reconstruction outputs, same shape as ``data``.
        attr_cardinalities: List of category counts per original attribute.
        attr_is_categorical: Optional list of booleans, same length as
            ``attr_cardinalities``. When ``None`` (the default), every
            attribute is treated as categorical.

    Returns:
        numpy array of shape ``(N, num_attrs)``. Higher values mean the
        model reconstructed that attribute more poorly for that row.
    """
    data_t, predictions_t, attr_is_categorical = _prepare_inputs(
        data, predictions, attr_cardinalities, attr_is_categorical
    )
    stacked = _compute_per_attr_losses_tf(
        data_t, predictions_t, attr_cardinalities, attr_is_categorical
    )  # (num_attrs, N)
    return np.asarray(stacked.numpy()).T  # (N, num_attrs)


def compute_reconstruction_error(
    data: Union[pd.DataFrame, np.ndarray],
    predictions: Union[pd.DataFrame, np.ndarray],
    attr_cardinalities: List[int],
    attr_is_categorical: Optional[List[bool]] = None,
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
    penalty, every all-zero block of a **categorical** attribute is
    assigned the maximum normalized loss (``1.0``, i.e. ``log(K) / log(K)``)
    — matching the treatment a uniform wrong-category prediction would
    receive.

    The override must *only* apply to categorical attributes. Numeric
    features produced by ``Table2Vector`` have cardinality 1 (a single
    MinMax-scaled scalar in ``[0, 1]``), so a valid scaled numeric value in
    ``[0, 0.5]`` would otherwise trip a naive sum-based heuristic and get
    forced to loss 1.0 regardless of reconstruction quality (Codex P1 #2).
    And critically, a **categorical** attribute whose fitted one-hot block
    happens to be width 1 (because only one category appeared in the
    training split of a small/imbalanced dataset) still needs the
    override, because test rows with the unseen minority value will be
    encoded as ``[0]`` and otherwise contribute zero CE (Codex P1 #3).
    The data alone cannot disambiguate "numeric value 0" from "unseen
    categorical" — the caller must supply the type hint via
    ``attr_is_categorical``.

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
        attr_is_categorical: Optional list of booleans the same length as
            ``attr_cardinalities`` indicating whether each attribute is a
            categorical one-hot block (``True``) or a numeric scaled
            scalar (``False``). The unseen-category override is applied
            only to attributes marked ``True``. When ``None`` (the default),
            every attribute is treated as categorical, which is correct
            for the worker and Vertex AI paths since both hard-code all
            uploaded columns to categorical. The CLI ``find_outliers``
            path must pass an explicit hint derived from the fitted
            ``Table2Vector.var_types`` when the dataset contains numeric
            columns.

    Returns:
        numpy array of shape ``(N,)`` containing per-row reconstruction error.
        Higher values are more anomalous.
    """
    # Reduce to (N,) inside TensorFlow before converting to NumPy so that
    # the full (N, num_attrs) matrix is never materialized — important for
    # large scoring runs where allocating that intermediate array would
    # waste memory and CPU time (Codex P2 review on PR #51).
    data_t, predictions_t, attr_is_categorical = _prepare_inputs(
        data, predictions, attr_cardinalities, attr_is_categorical
    )
    stacked = _compute_per_attr_losses_tf(
        data_t, predictions_t, attr_cardinalities, attr_is_categorical
    )  # (num_attrs, N)
    row_loss = tf.reduce_mean(stacked, axis=0)  # (N,) — stays in TF
    return np.asarray(row_loss.numpy())


def get_outliers_list(
    data,
    model,
    k,
    attr_cardinalities,
    vectorizer,
    prior,
    attr_is_categorical=None,
    attr_names=None,
):
    """
    Score every row in ``data`` and return a DataFrame combining the
    decoded original values, decoded reconstruction, and error columns.

    Args:
        data: One-hot encoded inputs (DataFrame or ndarray), shape
            ``(N, total_one_hot_dims)``.
        model: Trained Keras model. ``model.predict(data)`` must return
            either an ndarray of shape ``(N, total_one_hot_dims)`` (AE)
            or a tuple ``(reconstruction, z_mean, z_log_var)`` (VAE).
        k: KL-loss weight for VAE models. Unused for AE.
        attr_cardinalities: Per-attribute category counts in vectorized
            matrix slice order.
        vectorizer: Fitted :class:`features.transform.Table2Vector` used to
            decode one-hot vectors back to human-readable column values.
        prior: ``"gaussian"`` or ``"gumbel"`` for VAE models; ``None`` for AE.
        attr_is_categorical: Optional list of booleans (same length as
            ``attr_cardinalities``) for the unseen-category override.
            Defaults to all-True when ``None``.
        attr_names: Optional list of original column names in the same order
            as ``attr_cardinalities`` (i.e. the vectorized-matrix slice
            order). When provided, the returned DataFrame includes one
            ``col_error__{name}`` column per attribute showing the
            per-row reconstruction error for that specific attribute.
            These per-column scores enable users to see *which* survey
            questions a flagged respondent answered anomalously
            (TASKS.md 3.3).

    Returns:
        DataFrame with decoded original values, decoded reconstructions,
        and error columns (including ``col_error__{name}`` columns when
        ``attr_names`` is supplied).
    """
    predictions = model.predict(data)

    z1 = None
    z2 = None

    if isinstance(predictions, tuple):
        predictions, z1, z2 = predictions

    predictions = pd.DataFrame(predictions, columns=data.columns, index=data.index)
    errors = pd.DataFrame(index=data.index)

    # Compute per-column errors once; derive the aggregate from them to
    # avoid a second pass through the data (TASKS.md 3.3).
    per_col_errors = compute_per_column_errors(
        data,
        predictions,
        attr_cardinalities,
        attr_is_categorical=attr_is_categorical,
    )
    reconstruction_loss_np = per_col_errors.mean(axis=1)

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

    # Add per-column error columns for interpretability (TASKS.md 3.3).
    if attr_names is not None:
        for j, name in enumerate(attr_names):
            errors[f"col_error__{name}"] = per_col_errors[:, j]

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
    column_names: List[str],
    attr_is_categorical: Optional[List[bool]] = None,
) -> List[Tuple[str, float]]:
    """
    Decompose reconstruction loss into per-column contribution percentages.

    Averages the per-row, per-attribute errors across the batch and expresses
    each attribute's share of the total loss as a percentage. This is used by
    the web worker to show which survey questions contributed most to a
    respondent being flagged as an outlier.

    Args:
        data: True values (one-hot encoded), shape ``(N, total_features)``.
        predictions: Model predictions (softmax outputs), same shape.
        attr_cardinalities: List of category counts per attribute.
        column_names: Original column names, length ``len(attr_cardinalities)``.
        attr_is_categorical: Optional list of booleans for the unseen-category
            override (see :func:`compute_reconstruction_error`). Defaults to
            all-True when ``None``.

    Returns:
        List of ``(column_name, contribution_percentage)`` tuples, sorted
        descending by percentage. Percentages sum to 100.

    Example:
        >>> data = np.array([[1, 0, 0, 0, 1, 0]])  # 2 columns, 3 categories each
        >>> predictions = np.array([[0.7, 0.2, 0.1, 0.1, 0.7, 0.2]])
        >>> attr_cardinalities = [3, 3]
        >>> column_names = ['Q1', 'Q2']
        >>> contributions = compute_per_column_contributions(
        ...     data, predictions, attr_cardinalities, column_names)
        >>> # Returns: [('Q1', 55.2), ('Q2', 44.8)]  (approximate)
    """
    per_col = compute_per_column_errors(
        data, predictions, attr_cardinalities, attr_is_categorical
    )
    # Average across rows to get a single loss value per attribute.
    per_attr_mean = per_col.mean(axis=0)  # (num_attrs,)

    total_loss = float(per_attr_mean.sum())

    if total_loss < 1e-10:
        # Perfect reconstruction: assign equal contribution to every attribute.
        contributions = [
            (column_names[i], 100.0 / len(per_attr_mean))
            for i in range(len(per_attr_mean))
        ]
    else:
        contributions = [
            (column_names[i], float(per_attr_mean[i] / total_loss * 100.0))
            for i in range(len(per_attr_mean))
        ]

    # Renormalize to exactly 100% (guard against floating-point drift).
    total_pct = sum(pct for _, pct in contributions)
    contributions = [
        (col, float(pct / total_pct * 100.0))
        for col, pct in contributions
    ]

    contributions.sort(key=lambda x: x[1], reverse=True)
    return contributions
