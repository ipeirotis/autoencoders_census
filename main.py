"""
CLI Entry Point - Command-line interface for training and evaluating autoencoders.

Commands:
    train                  - Train an autoencoder (AE or VAE) on a dataset
    search_hyperparameters - Run Bayesian hyperparameter optimization
    evaluate               - Evaluate model reconstruction accuracy
    find_outliers          - Detect outliers using reconstruction error
    chow_liu_outliers      - Detect outliers using Chow-Liu tree log-likelihood
    generate               - Generate synthetic samples from a trained VAE

Example Usage:
    python main.py train --model_name AE --data sadc_2017
    python main.py find_outliers --model_path cache/simple_model/autoencoder
    python main.py chow_liu_outliers --data sadc_2017
    python main.py search_hyperparameters --model_name VAE

Pipeline Steps:
    1. Load data via DataLoader (handles multiple dataset formats)
    2. Clean data (fill NaN, apply "Rule of 9" filter for cardinality)
    3. Vectorize categorical data via Table2Vector (one-hot encoding) [AE only]
    4. Train/load autoencoder model OR fit Chow-Liu tree
    5. Calculate reconstruction error / log-likelihood for outlier detection
"""

import logging
import os
import sys

import click
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from dataset.loader import DataLoader, DEFAULT_MAX_UNIQUE_VALUES
from evaluate.evaluator import Evaluator
from evaluate.generator import Generator
from evaluate.outliers import get_outliers_list
from features.transform import Table2Vector
from model.factory import get_model
from chow_liu_rank import rank_rows_by_chow_liu
from train.trainer import Trainer
from utils import (
    set_seed,
    save_model,
    save_history,
    save_hyperparameters,
    model_analysis,
    load_model,
    load_vectorizer,
    save_to_csv,
    evaluate_errors,
    define_necessary_elements,
)


logger = logging.Logger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def _coerce_max_unique_values(value):
    """Normalize a user-supplied ``max_unique_values`` at config-ingestion
    boundaries (YAML files, CLI flags).

    - ``None`` (unset CLI flag, or ``max_unique_values: null`` in the
      YAML — common in templated configs) coerces to the built-in
      default instead of propagating downstream. Without this, a
      ``None`` would reach ``prepare_for_categorical`` and crash on
      ``None < 2`` with a confusing ``TypeError`` before data
      processing even starts (Codex P2 PR#49).
    - Non-integer values (e.g. a string "9" from an env-var leak into
      the YAML loader, or a float like ``9.0``) are cast to ``int``.
      Anything that can't be cast raises ``ValueError``.
    - Integers below 2 raise ``ValueError`` — a threshold below 2 would
      drop every column because the Rule-of-N filter also rejects
      ``n_unique <= 1``.

    Args:
        value: The raw value from a CLI flag or YAML config.

    Returns:
        int: A validated ``max_unique_values`` suitable for passing
        into ``DataLoader`` / ``prepare_for_*`` helpers.

    Raises:
        ValueError: If ``value`` is non-None but not a valid threshold.
    """
    if value is None:
        return DEFAULT_MAX_UNIQUE_VALUES
    try:
        coerced = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"max_unique_values must be an integer (got {value!r})"
        ) from e
    if coerced < 2:
        raise ValueError(
            f"max_unique_values must be >= 2 (got {coerced})"
        )
    return coerced


def prepare_for_categorical(project_data, max_unique_values=DEFAULT_MAX_UNIQUE_VALUES):
    """Clean data for categorical-only models (Chow-Liu tree, etc.).

    Steps:
        1. Fill NaN with "missing" and cast to string
        2. Apply Rule-of-N filter (keep columns with 2..max_unique_values
           unique values).

    Args:
        project_data: Raw DataFrame from DataLoader.
        max_unique_values: Upper bound on the number of unique values a
            column may have and still be kept. Defaults to
            ``DEFAULT_MAX_UNIQUE_VALUES`` (9). Must be >= 2.

    Returns:
        cleaned_df: DataFrame with only low-cardinality categorical columns.
    """
    if max_unique_values < 2:
        raise ValueError(
            f"max_unique_values must be >= 2 (got {max_unique_values})"
        )

    project_data = project_data.fillna("missing")
    project_data = project_data.astype(str)

    cols_to_keep = [
        col for col in project_data.columns
        if 1 < project_data[col].nunique() <= max_unique_values
    ]
    return project_data[cols_to_keep]


def _clean_for_saved_vectorizer(project_data, vectorizer):
    """Clean data for inference with a saved vectorizer.

    Applies the same fillna/astype cleaning as training, but does NOT
    re-apply the Rule-of-9 filter.  Instead, ensures exactly the columns
    the vectorizer was trained on are present.  Columns missing from the
    scoring data are filled with type-appropriate defaults (``"missing"``
    for categorical, ``NaN`` for numeric) so the transformed matrix
    always matches the model's expected input width.

    Args:
        project_data: Raw DataFrame from DataLoader.
        vectorizer: A fitted Table2Vector instance.

    Returns:
        cleaned_df with exactly the columns the vectorizer expects,
        in the same order as training.
    """
    import numpy as np

    # Derive the full ordered column list from the vectorizer's var_types
    # (covers categorical, numeric, and any passthrough columns)
    trained_cols = [
        col for col in vectorizer.var_types.get("categorical", [])
    ] + [
        col for col in vectorizer.var_types.get("numeric", [])
    ]
    # Also include any columns tracked by encoders/scalers but not in
    # var_types (defensive, shouldn't happen in practice)
    for col in list(vectorizer.one_hot_encoders.keys()) + list(vectorizer.min_max_scalers.keys()):
        if col not in trained_cols:
            trained_cols.append(col)

    # Clean categorical columns as strings; leave numeric columns numeric
    numeric_cols = set(vectorizer.var_types.get("numeric", []))
    for col in trained_cols:
        if col in project_data.columns:
            if col not in numeric_cols:
                project_data[col] = project_data[col].fillna("missing").astype(str)
        else:
            # Column missing from scoring data — fill with type-appropriate default
            if col in numeric_cols:
                project_data[col] = np.nan
            else:
                project_data[col] = "missing"

    # Ensure non-numeric columns that were present are also string-typed
    for col in trained_cols:
        if col not in numeric_cols and col in project_data.columns:
            project_data[col] = project_data[col].astype(str)

    return project_data[trained_cols]


def _clean_and_build_vectorizer(
    project_data,
    variable_types=None,
    max_unique_values=DEFAULT_MAX_UNIQUE_VALUES,
):
    """Clean data and create a (not-yet-fitted) Table2Vector.

    Shared first step for both ``prepare_for_model`` and
    ``prepare_for_training``.

    Args:
        project_data: Raw DataFrame from DataLoader.
        variable_types: Optional dict mapping column names to types.
        max_unique_values: Upper bound on the number of unique values a
            column may have and still be kept (Rule-of-N threshold).

    Returns:
        (cleaned_df, variable_types_dict, vectorizer)
    """
    if variable_types is None:
        variable_types = {}

    # 1-2. Clean data (fillna, Rule-of-N)
    project_data = prepare_for_categorical(
        project_data, max_unique_values=max_unique_values
    )

    # 3. Sync variable_types to surviving columns
    variable_types = {c: variable_types.get(c, "categorical") for c in project_data.columns}
    if not variable_types:
        variable_types = {col: "categorical" for col in project_data.columns}

    vectorizer = Table2Vector(variable_types)
    return project_data, variable_types, vectorizer


def _compute_attr_layout(vectorizer, cleaned_columns):
    """Derive ``(attr_cardinalities, attr_is_categorical)`` in the slice
    order that matches the vectorized matrix produced by ``Table2Vector``.

    ``Table2Vector._apply_transforms`` reorders columns during
    vectorization: numeric and untransformed columns stay in their
    original relative positions, while **categorical** columns are
    dropped from their original slot and their one-hot blocks are
    appended at the end (in original relative order). For a cleaned
    input ``[num1, cat1, num2, cat2]`` the vectorized matrix has columns
    ``[num1, num2, cat1__a, cat1__b, cat2__x, cat2__y]``.

    ``get_cardinalities(cleaned_columns)``, on the other hand, walks the
    raw (pre-vectorized) column order, so for the same input it returns
    ``[1, K_cat1, 1, K_cat2]``. Slicing the vectorized matrix with those
    cardinalities lines up with the wrong features for mixed
    numeric+categorical datasets — and every call into
    :meth:`model.base.VAE.reconstruction_loss` or
    :func:`evaluate.outliers.compute_reconstruction_error` silently
    operates on mis-aligned slices (Codex P1 #4 on PR #46).

    This helper rebuilds the per-attribute metadata in the actual
    vectorized-matrix slice order so the scoring path is correct even
    for mixed datasets:

        ordered_cols = [non_categoricals_in_orig_order]
                     + [categoricals_in_orig_order]

    The worker and Vertex AI paths (``worker.py``, ``train/task.py``)
    are all-categorical and don't need this helper — their vectorized
    layout happens to match ``process_df.columns`` by coincidence since
    every column is moved to the end in iteration order.

    Args:
        vectorizer: A fitted :class:`features.transform.Table2Vector`.
        cleaned_columns: The pre-vectorized (cleaned) column order,
            typically ``project_data.columns`` at the scoring step.

    Returns:
        Tuple ``(attr_cardinalities, attr_is_categorical)`` — two
        parallel lists in vectorized-matrix slice order.
    """
    categorical_set = set(vectorizer.var_types.get("categorical", []))
    non_categorical = [c for c in cleaned_columns if c not in categorical_set]
    categorical = [c for c in cleaned_columns if c in categorical_set]

    ordered = non_categorical + categorical
    attr_cardinalities = vectorizer.get_cardinalities(ordered)
    attr_is_categorical = [c in categorical_set for c in ordered]
    return attr_cardinalities, attr_is_categorical


def prepare_for_model(
    project_data,
    variable_types=None,
    max_unique_values=DEFAULT_MAX_UNIQUE_VALUES,
):
    """Shared data-cleaning and vectorization pipeline.

    Use this for **scoring / evaluation** where the entire dataset is
    transformed at once (no train/test split needed).

    For **training** use :func:`prepare_for_training` instead — it splits
    the data before fitting the vectorizer to prevent data leakage.

    Steps:
        1. Fill NaN with "missing" and cast to string
        2. Apply Rule-of-N filter (keep columns with 2..N unique values)
        3. Sync variable_types to surviving columns
        4. Vectorize via Table2Vector (one-hot encoding)
        5. Convert to float32
        6. Compute per-column cardinalities

    Args:
        project_data: Raw DataFrame from DataLoader.
        variable_types: Optional dict mapping column names to types.
            If None or empty, all columns are treated as categorical.
        max_unique_values: Upper bound on the number of unique values a
            column may have and still be kept (Rule-of-N threshold).

    Returns:
        (cleaned_df, vectorized_df, vectorizer, cardinalities)
    """
    project_data, _, vectorizer = _clean_and_build_vectorizer(
        project_data, variable_types, max_unique_values=max_unique_values
    )

    # 4. Vectorize (fit + transform on the full data — acceptable for
    #    scoring because there is no train/test distinction)
    vectorized_df = vectorizer.vectorize_table(project_data)

    # 5. Float32 conversion
    vectorized_df = vectorized_df.astype("float32")

    # 6. Cardinalities from the fitted encoder (consistent with one-hot width)
    cardinalities = vectorizer.get_cardinalities(project_data.columns)

    return project_data, vectorized_df, vectorizer, cardinalities


def prepare_for_training(
    project_data,
    variable_types=None,
    test_size=0.2,
    max_unique_values=DEFAULT_MAX_UNIQUE_VALUES,
):
    """Clean, split, then vectorize — preventing data leakage.

    Unlike :func:`prepare_for_model`, the vectorizer is fitted on the
    **training split only**, so test-set statistics never leak into the
    encoder categories or scaler ranges.

    Steps:
        1-3. Same cleaning as prepare_for_model
        4.   Train/test split on *cleaned* (pre-vectorized) data
        5.   Fit vectorizer on training split
        6.   Transform both splits
        7.   Float32 conversion
        8.   Compute per-column cardinalities

    Args:
        project_data: Raw DataFrame from DataLoader.
        variable_types: Optional dict mapping column names to types.
        test_size: Fraction of data for the test set (default 0.2).
        max_unique_values: Upper bound on the number of unique values a
            column may have and still be kept (Rule-of-N threshold).

    Returns:
        (cleaned_df, X_train, X_test, vectorizer, cardinalities)
    """
    project_data, _, vectorizer = _clean_and_build_vectorizer(
        project_data, variable_types, max_unique_values=max_unique_values
    )

    # 4. Split *before* vectorization
    train_df, test_df = train_test_split(
        project_data, test_size=test_size, random_state=None
    )

    # 5-6. Fit on training data, transform both
    vectorizer.fit(train_df)
    X_train = vectorizer.transform(train_df).astype("float32")
    X_test = vectorizer.transform(test_df).astype("float32")

    # 7. Cardinalities from the *fitted encoder* so model dimensions
    #    match the actual one-hot width (not the full dataset which may
    #    contain categories absent from the training split).
    cardinalities = vectorizer.get_cardinalities(project_data.columns)

    return project_data, X_train, X_test, vectorizer, cardinalities


def run_training_pipeline(df, config_path, output_path, model_name="AE", prior="gaussian"):
    """
    Reusable training logic that accepts a DataFrame directly.
    Used by both the CLI and the Cloud Worker.

    The optional ``max_unique_values`` YAML key in the config file
    overrides the default Rule-of-N threshold for data cleaning
    (TASKS.md 3.1). An explicit ``null`` in the YAML is coerced to
    the default via ``_coerce_max_unique_values``.
    """
    logger.info(f"loading config from {config_path}...")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    _, X_train, X_test, vectorizer, cardinalities = prepare_for_training(
        df,
        test_size=config.get("test_size", 0.2),
        max_unique_values=_coerce_max_unique_values(
            config.get("max_unique_values")
        ),
    )

    model = get_model(model_name, cardinalities)
    trainer = Trainer(model, config)

    logger.info("Training model....")
    model, history = trainer.train(
        dataset=X_train, prior=prior,
        X_train=X_train, X_test=X_test,
    )

    logger.info("Saving model....")
    save_model(model, output_path, vectorizer=vectorizer)
    save_history(history, output_path)

    return model, history

@click.group()
def cli():
    pass


@cli.command("train")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option(
    "--model_name", help="model to train between AE and VAE", type=str, default="AE"
)
@click.option("--prior", help="prior to use for VAE", type=str, default="gaussian")
@click.option("--data", help="data to train on", type=str, default="sadc_2017")
@click.option(
    "--drop_columns", help="columns to drop from the data", type=str, default=None
)
@click.option(
    "--rename_columns", help="columns to rename from the data", type=str, default=None
)
@click.option(
    "--interest_columns", help="columns to merge from the data", type=str, default=None
)
@click.option(
    "--config",
    help="config file for training",
    type=str,
    default="config/simple_variational_autoencoder.yaml",
)
@click.option(
    "--output",
    help="output path for saving the model",
    type=str,
    default="cache/simple_model/",
)
@click.option(
    "--max_unique_values",
    help=(
        "Upper bound on unique values per column for the Rule-of-N filter "
        "(default 9). Columns with <= 1 or > this many unique values are "
        "dropped. Overrides any ``max_unique_values`` key in the YAML config."
    ),
    type=int,
    default=None,
)
def train(
    seed,
    model_name,
    prior,
    data,
    drop_columns,
    rename_columns,
    interest_columns,
    config,
    output,
    max_unique_values,
):
    logger.debug("Starting train function")

    # 1. Set Seed
    set_seed(seed)



    # 2. Parse Column Arguments (Safe Split)
    (
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns,
        additional_rename_columns,
        additional_interest_columns,
    ) = define_necessary_elements(data, drop_columns, rename_columns, interest_columns)

    # 5. Clean, split, vectorize (leak-free)
    logger.info("Transforming the data....")
    with open(config, "r") as file:
        config_dict = yaml.safe_load(file)

    # CLI flag takes precedence over the YAML key, which takes precedence
    # over the built-in default. Both routes go through
    # ``_coerce_max_unique_values`` so ``null`` in a templated YAML
    # config doesn't crash downstream with ``None < 2`` (Codex P2 PR#49).
    if max_unique_values is None:
        max_unique_values = config_dict.get("max_unique_values")
    max_unique_values = _coerce_max_unique_values(max_unique_values)

    # 3. Initialize Loader
    logger.info("Loading data....")
    data_loader = DataLoader(
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns=additional_drop_columns,
        additional_rename_columns=additional_rename_columns,
        additional_columns_of_interest=additional_interest_columns,
        max_unique_values=max_unique_values,
    )

    # 4. Load data -- all loaders return (DataFrame, metadata_dict)
    project_data, metadata = data_loader.load_data(data)
    variable_types = metadata.get("variable_types", {})

    project_data, X_train, X_test, vectorizer, cardinalities = prepare_for_training(
        project_data, variable_types,
        test_size=config_dict.get("test_size", 0.2),
        max_unique_values=max_unique_values,
    )

    # 6. Build and Train model
    logger.info("Loading model....")
    model = get_model(model_name, cardinalities)

    trainer = Trainer(model, config_dict)

    logger.info("Training model....")
    model, history = trainer.train(
        dataset=X_train, prior=prior,
        X_train=X_train, X_test=X_test,
    )

    # 10. Save Results
    logger.info("Saving model....")
    save_model(model, output, vectorizer=vectorizer)
    logger.info("Saving history....")
    save_history(history, output)
    logger.info("Saving plots....")
    model_analysis(history, output, model_name)

    logger.info("Training pipeline finished successfully.")


@cli.command("search_hyperparameters")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option(
    "--model_name", help="model to train between AE and VAE", type=str, default="AE"
)
@click.option("--prior", help="prior to use for VAE", type=str, default="gaussian")
@click.option("--data", help="data to train on", type=str, default="sadc_2017")
@click.option(
    "--drop_columns", help="columns to drop from the data", type=str, default=None
)
@click.option(
    "--rename_columns", help="columns to rename from the data", type=str, default=None
)
@click.option(
    "--interest_columns", help="columns to merge from the data", type=str, default=None
)
@click.option(
    "--config",
    help="config file for training",
    type=str,
    default="config/hp_autoencoder.yaml",
)
@click.option(
    "--output",
    help="output path for saving the hyperparameters",
    type=str,
    default="cache/simple_model/",
)
@click.option(
    "--max_unique_values",
    help=(
        "Upper bound on unique values per column for the Rule-of-N filter "
        "(default 9). Overrides any ``max_unique_values`` key in the YAML config."
    ),
    type=int,
    default=None,
)
def search_hyperparameters(
    seed,
    model_name,
    prior,
    data,
    drop_columns,
    rename_columns,
    interest_columns,
    config,
    output,
    max_unique_values,
):

    set_seed(seed)

    (
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns,
        additional_rename_columns,
        additional_interest_columns,
    ) = define_necessary_elements(data, drop_columns, rename_columns, interest_columns)

    logger.info("Loading config from config file....")
    with open(config, "r") as file:
        config = yaml.safe_load(file)

    # CLI flag > YAML key > built-in default, coerced through
    # ``_coerce_max_unique_values`` so ``null`` in a templated YAML
    # doesn't crash downstream (Codex P2 PR#49).
    if max_unique_values is None:
        max_unique_values = config.get("max_unique_values")
    max_unique_values = _coerce_max_unique_values(max_unique_values)

    data_loader = DataLoader(
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns=additional_drop_columns,
        additional_rename_columns=additional_rename_columns,
        additional_columns_of_interest=additional_interest_columns,
        max_unique_values=max_unique_values,
    )
    project_data, metadata = data_loader.load_data(data)
    variable_types = metadata.get("variable_types", {})

    logger.info("Transforming the data....")
    project_data, X_train, X_test, vectorizer, cardinalities = prepare_for_training(
        project_data, variable_types,
        test_size=config.get("test_size", 0.2),
        max_unique_values=max_unique_values,
    )

    logger.info("Loading model....")
    model = get_model(model_name, cardinalities)

    trainer = Trainer(model, config)

    logger.info("Searching hyperparameters....")
    best_hps = trainer.search_hyperparameters(
        dataset=X_train, prior=prior,
        X_train=X_train, X_test=X_test,
    )

    logger.info(f"Best hyperparameters found: {best_hps}")

    logger.info("Saving hyperparameters....")
    save_hyperparameters(best_hps, output)


@cli.command("evaluate")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option(
    "--model_path",
    help="model to evaluate",
    type=str,
    default="cache/simple_model/autoencoder",
)
@click.option("--data", help="data to train on", type=str, default="sadc_2017")
@click.option(
    "--drop_columns", help="columns to drop from the data", type=str, default=None
)
@click.option(
    "--rename_columns", help="columns to rename from the data", type=str, default=None
)
@click.option(
    "--interest_columns", help="columns to merge from the data", type=str, default=None
)
@click.option(
    "--output",
    help="output path for saving the predictions",
    type=str,
    default="cache/predictions/",
)
@click.option(
    "--max_unique_values",
    help=(
        "Upper bound on unique values per column for the Rule-of-N filter "
        "(default 9). Only applies when no saved vectorizer is present; "
        "when a saved vectorizer exists the Rule-of-N filter is bypassed "
        "entirely so the training-fitted vectorizer is the authoritative "
        "source of truth on kept columns."
    ),
    type=int,
    default=None,
)
def evaluate(
    seed, model_path, data, drop_columns, rename_columns, interest_columns, output,
    max_unique_values,
):

    (
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns,
        additional_rename_columns,
        additional_interest_columns,
    ) = define_necessary_elements(data, drop_columns, rename_columns, interest_columns)

    effective_max_unique = _coerce_max_unique_values(max_unique_values)

    set_seed(seed)

    logger.info("Loading model....")
    model = load_model(model_path)

    # Load the training-fitted vectorizer *before* constructing the
    # DataLoader so that we can disable the Rule-of-N filter when a
    # saved vectorizer is present. Otherwise columns whose training-time
    # cardinality exceeded the current loader threshold would be
    # silently dropped here and backfilled as constant ``"missing"``
    # values by ``_clean_for_saved_vectorizer`` below, corrupting model
    # inputs (Codex P1 PR#49).
    saved_vectorizer = load_vectorizer(model_path)
    apply_rule_of_n = saved_vectorizer is None

    data_loader = DataLoader(
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns=additional_drop_columns,
        additional_rename_columns=additional_rename_columns,
        additional_columns_of_interest=additional_interest_columns,
        max_unique_values=effective_max_unique,
        apply_rule_of_n=apply_rule_of_n,
    )

    logger.info("Loading data....")
    project_data, metadata = data_loader.load_data(data)
    variable_types = metadata.get("variable_types", {})

    logger.info("Transforming the data....")
    if saved_vectorizer is not None:
        # Use the training-fitted vectorizer so one-hot width matches the model
        project_data = _clean_for_saved_vectorizer(project_data, saved_vectorizer)
        vectorized_df = saved_vectorizer.transform(project_data).astype("float32")
        vectorizer = saved_vectorizer
    else:
        project_data, vectorized_df, vectorizer, _ = prepare_for_model(
            project_data, variable_types,
            max_unique_values=effective_max_unique,
        )
    variable_types = {c: "categorical" for c in project_data.columns}

    evaluator = Evaluator(model)

    if not os.path.exists(output):
        os.makedirs(output)

    predictions_df = evaluator.evaluate(
        vectorized_df, vectorizer, project_data, variable_types, output
    )

    logger.info("Saving metrics....")
    save_to_csv(predictions_df, output)

    averages = predictions_df[
        ["Accuracy", "Baseline Accuracy", "Lift", "OVA ROC AUC"]
    ].mean()
    average_df = pd.DataFrame(averages).transpose()

    logger.info("Saving average metrics....")
    save_to_csv(average_df, output, "averages")


@cli.command("find_outliers")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option(
    "--model_path",
    help="model to evaluate",
    type=str,
    default="cache/simple_model/autoencoder",
)
@click.option("--prior", help="prior to use for VAE", type=str, default="gaussian")
@click.option("--data", help="data to train on", type=str, default="sadc_2017")
@click.option(
    "--drop_columns", help="columns to drop from the data", type=str, default=None
)
@click.option(
    "--rename_columns", help="columns to rename from the data", type=str, default=None
)
@click.option(
    "--interest_columns", help="columns to merge from the data", type=str, default=None
)
@click.option("--k", help="k to rely on the kl_loss if VAE", type=float, default=1.0)
@click.option(
    "--output",
    help="output path for saving the predictions",
    type=str,
    default="cache/predictions/",
)
@click.option(
    "--max_unique_values",
    help=(
        "Upper bound on unique values per column for the Rule-of-N filter "
        "(default 9). Only applies when no saved vectorizer is present; "
        "when a saved vectorizer exists the Rule-of-N filter is bypassed "
        "entirely so the training-fitted vectorizer is the authoritative "
        "source of truth on kept columns."
    ),
    type=int,
    default=None,
)
def find_outliers(
    seed,
    model_path,
    prior,
    data,
    drop_columns,
    rename_columns,
    interest_columns,
    k,
    output,
    max_unique_values,
):
    logger.debug("Starting find_outliers")
    set_seed(seed)

    # 1. Parse Column Arguments
    (
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns,
        additional_rename_columns,
        additional_interest_columns,
    ) = define_necessary_elements(data, drop_columns, rename_columns, interest_columns)

    effective_max_unique = _coerce_max_unique_values(max_unique_values)

    # Load the training-fitted vectorizer *before* constructing the
    # DataLoader so that we can disable the Rule-of-N filter when a
    # saved vectorizer is present (Codex P1 PR#49). See ``evaluate`` for
    # the full rationale.
    saved_vectorizer = load_vectorizer(model_path)
    apply_rule_of_n = saved_vectorizer is None

    # 2. Initialize Loader
    logger.info("Loading data...")
    data_loader = DataLoader(
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns=additional_drop_columns,
        additional_rename_columns=additional_rename_columns,
        additional_columns_of_interest=additional_interest_columns,
        max_unique_values=effective_max_unique,
        apply_rule_of_n=apply_rule_of_n,
    )

    # 3. Load data -- all loaders return (DataFrame, metadata_dict)
    project_data, metadata = data_loader.load_data(data)
    variable_types = metadata.get("variable_types", {})

    # 4. Clean and vectorize
    logger.info("Transforming the data....")
    if saved_vectorizer is not None:
        project_data = _clean_for_saved_vectorizer(project_data, saved_vectorizer)
        vectorized_df = saved_vectorizer.transform(project_data).astype("float32")
        vectorizer = saved_vectorizer
    else:
        project_data, vectorized_df, vectorizer, _ = prepare_for_model(
            project_data, variable_types,
            max_unique_values=effective_max_unique,
        )

    # Derive attr_cardinalities and attr_is_categorical in the actual
    # vectorized-matrix slice order. ``Table2Vector._apply_transforms``
    # reorders columns (numerics/untransformed keep their original
    # positions, categoricals move to the end as one-hot blocks), so
    # calling ``get_cardinalities(project_data.columns)`` directly would
    # give cardinalities in raw column order and slicing the vectorized
    # matrix by those cardinalities would line up with the wrong
    # features on mixed numeric+categorical datasets (Codex P1 #4).
    # The explicit categorical hint also fixes Codex P1 #2/#3: without it,
    # numeric MinMax values in [0, 0.5] get clamped as "unseen" and
    # cardinality-1 categorical columns with unseen values don't.
    attr_cardinalities, attr_is_categorical = _compute_attr_layout(
        vectorizer, project_data.columns
    )

    # 5. Load model
    logger.info(f"Loading model from {model_path}")
    try:
        from tensorflow.keras.models import load_model as keras_load_model
        model = keras_load_model(model_path)
    except Exception:
        model = load_model(model_path)

    # 6. Get Outliers
    logger.info("Calculating outliers...")
    error_df = get_outliers_list(
        vectorized_df,
        model,
        k,
        attr_cardinalities,
        vectorizer,
        prior,
        attr_is_categorical=attr_is_categorical,
    )

    # 10. Save
    logger.info("Saving outliers....")
    if not os.path.exists(output):
        os.makedirs(output) 
    save_to_csv(error_df, output, "errors")
    logger.info("Outliers identified successfully.")


@cli.command("chow_liu_outliers")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option("--data", help="data to train on", type=str, default="sadc_2017")
@click.option(
    "--drop_columns", help="columns to drop from the data", type=str, default=None
)
@click.option(
    "--rename_columns", help="columns to rename from the data", type=str, default=None
)
@click.option(
    "--interest_columns", help="columns to merge from the data", type=str, default=None
)
@click.option("--alpha", help="Laplace smoothing parameter for CLTree", type=float, default=1.0)
@click.option("--mi_subsample", help="Subsample size for mutual information computation (speeds up large datasets)", type=int, default=None)
@click.option(
    "--output",
    help="output path for saving the outlier scores",
    type=str,
    default="cache/predictions/",
)
@click.option(
    "--max_unique_values",
    help=(
        "Upper bound on unique values per column for the Rule-of-N filter "
        "(default 9)."
    ),
    type=int,
    default=None,
)
def chow_liu_outliers(
    seed,
    data,
    drop_columns,
    rename_columns,
    interest_columns,
    alpha,
    mi_subsample,
    output,
    max_unique_values,
):
    """Detect outliers using Chow-Liu tree log-likelihood scoring.

    Fits a maximum spanning tree of pairwise mutual information on
    categorical data and scores each row by its log-likelihood under
    the learned tree distribution. Rows with low log-likelihood
    (high error) are outliers.
    """
    set_seed(seed)

    # 1. Parse Column Arguments
    (
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns,
        additional_rename_columns,
        additional_interest_columns,
    ) = define_necessary_elements(data, drop_columns, rename_columns, interest_columns)

    effective_max_unique = _coerce_max_unique_values(max_unique_values)

    # 2. Initialize Loader
    logger.info("Loading data...")
    data_loader = DataLoader(
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns=additional_drop_columns,
        additional_rename_columns=additional_rename_columns,
        additional_columns_of_interest=additional_interest_columns,
        max_unique_values=effective_max_unique,
    )

    # 3. Load data
    project_data, metadata = data_loader.load_data(data)

    # 4. Clean data (fillna + Rule-of-N, no vectorization needed)
    logger.info("Cleaning the data...")
    cleaned_df = prepare_for_categorical(
        project_data, max_unique_values=effective_max_unique
    )
    logger.info(f"Cleaned data: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")

    if cleaned_df.shape[1] == 0:
        logger.error(
            "No columns survived cleaning (Rule-of-N, N=%d). Cannot fit "
            "Chow-Liu tree.",
            effective_max_unique,
        )
        return

    # 5. Fit Chow-Liu tree and score rows
    logger.info("Fitting Chow-Liu tree...")
    ranked_df, cl_model = rank_rows_by_chow_liu(
        cleaned_df, alpha=alpha, mi_subsample=mi_subsample, random_state=seed
    )

    # 6. Map to error column (1 - pct: higher = more anomalous)
    ranked_df["error"] = 1.0 - ranked_df["pct"]

    # 7. Log tree edges
    edges = cl_model.edges()
    logger.info(f"Chow-Liu tree has {len(edges)} edges")
    for u, v, mi in sorted(edges, key=lambda e: -e[2])[:5]:
        logger.info(f"  {u} -> {v}  (MI={mi:.4f})")

    # 8. Save
    if not os.path.exists(output):
        os.makedirs(output)
    save_to_csv(ranked_df, output, "errors")
    logger.info(f"Outlier scores saved to {os.path.join(output, 'errors.csv')}")


@cli.command("generate")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option("--prior", help="prior to use for VAE", type=str, default="gaussian")
@click.option(
    "--model_path",
    help="model to generate samples from",
    type=str,
    default="cache/simple_model/autoencoder",
)
@click.option(
    "--number_samples", help="number of samples to generate", type=int, default=10
)
@click.option(
    "--output",
    help="output path for saving the samples",
    type=str,
    default="cache/samples/",
)
@click.option("--data", help="data to take columns from", type=str, default="sadc_2017")
@click.option(
    "--drop_columns", help="columns to drop from the data", type=str, default=None
)
@click.option(
    "--rename_columns", help="columns to rename from the data", type=str, default=None
)
@click.option(
    "--interest_columns", help="columns to merge from the data", type=str, default=None
)
@click.option(
    "--target_features",
    help="features to condition samples on (comma separated)",
    type=str,
    default=None,
)
@click.option(
    "--max_unique_values",
    help=(
        "Upper bound on unique values per column for the Rule-of-N filter "
        "(default 9). Only applies when no saved vectorizer is present; "
        "when a saved vectorizer exists the Rule-of-N filter is bypassed "
        "entirely so the training-fitted vectorizer is the authoritative "
        "source of truth on kept columns."
    ),
    type=int,
    default=None,
)
def generate(
    seed,
    prior,
    model_path,
    number_samples,
    output,
    data,
    drop_columns,
    rename_columns,
    interest_columns,
    target_features,
    max_unique_values,
):

    set_seed(seed)

    logger.info("Loading model....")
    model = load_model(model_path)
    decoder = model.decoder

    # Reuse the same dataset configuration as `train` so the feature schema
    # (drops, renames, columns-of-interest) matches what the model was
    # trained on. Otherwise the encoder receives a mismatched input shape.
    (
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns,
        additional_rename_columns,
        additional_interest_columns,
    ) = define_necessary_elements(data, drop_columns, rename_columns, interest_columns)

    effective_max_unique = _coerce_max_unique_values(max_unique_values)

    # Load the training-fitted vectorizer *before* constructing the
    # DataLoader so that we can disable the Rule-of-N filter when a
    # saved vectorizer is present (Codex P1 PR#49). See ``evaluate`` for
    # the full rationale.
    saved_vectorizer = load_vectorizer(model_path)
    apply_rule_of_n = saved_vectorizer is None

    logger.info("Loading data....")
    data_loader = DataLoader(
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns=additional_drop_columns,
        additional_rename_columns=additional_rename_columns,
        additional_columns_of_interest=additional_interest_columns,
        max_unique_values=effective_max_unique,
        apply_rule_of_n=apply_rule_of_n,
    )
    project_data, metadata = data_loader.load_data(data)
    variable_types = metadata.get("variable_types", {})

    logger.info("Creating the vectorizer....")
    if saved_vectorizer is not None:
        project_data = _clean_for_saved_vectorizer(project_data, saved_vectorizer)
        vectorized_df = saved_vectorizer.transform(project_data).astype("float32")
        vectorizer = saved_vectorizer
        attr_cardinalities = saved_vectorizer.get_cardinalities(project_data.columns)
    else:
        project_data, vectorized_df, vectorizer, attr_cardinalities = prepare_for_model(
            project_data, variable_types,
            max_unique_values=effective_max_unique,
        )

    if target_features is not None:
        possible_features = list(project_data.columns)

        features = target_features.split(",")
        desired_feature_names = [x.split("_")[0] for x in features]
        desired_values = [int(x.split("_")[1]) for x in features]

        features_indices = [possible_features.index(f) for f in desired_feature_names]
        target_features = [
            sum(attr_cardinalities[:ind]) + val
            for ind, val in zip(features_indices, desired_values)
        ]

    logger.info("Generating samples....")

    # Compute latent-space statistics from the encoder on training data
    # (the VAE's get_config() does not store prior_means / prior_log_vars)
    import tensorflow as tf
    if prior == "gaussian":
        z_mean, z_log_var = model.encoder(vectorized_df.values.astype("float32"))
        prior_means = tf.reduce_mean(z_mean, axis=0)
        prior_log_vars = tf.reduce_mean(z_log_var, axis=0)
    else:
        prior_means = None
        prior_log_vars = None

    generator = Generator(
        decoder,
        number_samples,
        prior,
        len(attr_cardinalities),
        model.get_config()["temperature"],
        vectorized_df.columns,
        prior_means,
        prior_log_vars,
    )
    samples = generator.generate(
        vectorizer, target_features, len(list(vectorized_df.columns))
    )

    logger.info("Saving samples....")
    save_to_csv(samples, output, "samples_conditional")


@cli.command("evaluate_on_condition")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option("--data", help="data to train on", type=str, default="sadc_2017")
@click.option(
    "--error_data_path", help="data path of the error dataset", type=str, default=""
)
@click.option(
    "--rename_columns", help="columns to rename from the data", type=str, default=None
)
@click.option(
    "--output",
    help="output path for saving the predictions",
    type=str,
    default="cache/predictions/",
)
@click.option(
    "--column_to_condition",
    help="column to take the gold labels",
    type=str,
    default=None,
)
@click.option(
    "--outlier_value", help="outlier_value of the column", type=str, default=None
)
def evaluate_on_condition(
    seed,
    data,
    error_data_path,
    rename_columns,
    output,
    column_to_condition,
    outlier_value,
):

    (
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns,
        additional_rename_columns,
        additional_interest_columns,
    ) = define_necessary_elements(data, None, rename_columns, None)

    set_seed(seed)

    data_loader = DataLoader(
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns=additional_drop_columns,
        additional_rename_columns=additional_rename_columns,
        additional_columns_of_interest=additional_interest_columns,
    )

    if column_to_condition is None or outlier_value is None:
        logger.error("--column_to_condition and --outlier_value are required for evaluate_on_condition")
        return

    column_to_condition = column_to_condition.split(",")
    outlier_values = outlier_value.split(",")

    if data == "sadc_2017" or data == "sadc_2015":
        outlier_dataset = data_loader.find_outlier_data_sadc_2017(data, column_to_condition)

    else:
        outlier_dataset = data_loader.find_outlier_data(data, column_to_condition)

    error_data = pd.read_csv(error_data_path)

    df_combined = pd.concat([error_data, outlier_dataset], axis=1)

    metrics = evaluate_errors(df_combined, column_to_condition, outlier_values)

    logger.info("Saving metrics....")
    save_to_csv(metrics, output, "metrics_error")


@cli.command("pca_baseline")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option("--data", help="data to train on", type=str, default="sadc_2017")
@click.option(
    "--drop_columns", help="columns to drop from the data", type=str, default=None
)
@click.option(
    "--rename_columns", help="columns to rename from the data", type=str, default=None
)
@click.option(
    "--interest_columns", help="columns to merge from the data", type=str, default=None
)
@click.option(
    "--column_to_condition",
    help="column to take the gold labels",
    type=str,
    default=None,
)
@click.option(
    "--outlier_value", help="outlier_value of the column", type=str, default=None
)
@click.option(
    "--max_unique_values",
    help=(
        "Upper bound on unique values per column for the Rule-of-N filter "
        "(default 9)."
    ),
    type=int,
    default=None,
)
def pca_baseline(
    seed,
    data,
    drop_columns,
    rename_columns,
    interest_columns,
    column_to_condition,
    outlier_value,
    max_unique_values,
):

    set_seed(seed)

    (
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns,
        additional_rename_columns,
        additional_interest_columns,
    ) = define_necessary_elements(data, drop_columns, rename_columns, interest_columns)

    effective_max_unique = _coerce_max_unique_values(max_unique_values)

    data_loader = DataLoader(
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns=additional_drop_columns,
        additional_rename_columns=additional_rename_columns,
        additional_columns_of_interest=additional_interest_columns,
        max_unique_values=effective_max_unique,
    )

    if column_to_condition is None or outlier_value is None:
        logger.error("--column_to_condition and --outlier_value are required for pca_baseline")
        return

    column_to_condition = column_to_condition.split(",")
    outlier_values = outlier_value.split(",")

    if data == "sadc_2017" or data == "sadc_2015":
        outlier_dataset = data_loader.find_outlier_data_sadc_2017(data, column_to_condition)

    else:
        outlier_dataset = data_loader.find_outlier_data(data, column_to_condition)

    project_data, metadata = data_loader.load_data(data)
    variable_types = metadata.get("variable_types", {})

    # drop conditioning column before cleaning/vectorization
    project_data = project_data.drop(columns=column_to_condition, errors="ignore")
    variable_types = {
        k: v for k, v in variable_types.items() if k not in column_to_condition
    }

    logger.info("Transforming the data....")
    project_data, vectorized_df, vectorizer, _ = prepare_for_model(
        project_data, variable_types,
        max_unique_values=effective_max_unique,
    )
    variable_types = {c: "categorical" for c in project_data.columns}

    from sklearn.decomposition import PCA
    import numpy as np

    # Step 1: Fit PCA
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(vectorized_df)
    X_reconstructed = pca.inverse_transform(X_pca)

    evaluator = Evaluator("fake_model")

    predictions = pd.DataFrame(X_reconstructed)
    predictions.columns = vectorized_df.columns

    predictions_df = evaluator.evaluate_with_predictions(
        predictions, vectorizer, project_data, variable_types, "cache/temp/"
    )

    averages = predictions_df[
        ["Accuracy", "Baseline Accuracy", "Lift", "OVA ROC AUC"]
    ].mean()
    average_df = pd.DataFrame(averages).transpose()

    print(average_df)

    reconstruction_errors = np.mean((X_reconstructed - vectorized_df) ** 2, axis=1)
    reconstruction_errors = pd.DataFrame(reconstruction_errors, columns=["error"])


    df_combined = pd.concat([reconstruction_errors, outlier_dataset], axis=1)

    evaluate_errors(df_combined, column_to_condition, outlier_values)


if __name__ == "__main__":
    cli()
