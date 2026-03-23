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

from google.cloud import storage
from dataset.loader import DataLoader
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
    save_to_csv,
    evaluate_errors,
    define_necessary_elements,
)


logger = logging.Logger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def prepare_for_categorical(project_data):
    """Clean data for categorical-only models (Chow-Liu tree, etc.).

    Steps:
        1. Fill NaN with "missing" and cast to string
        2. Apply Rule-of-9 filter (keep columns with 2-9 unique values)

    Args:
        project_data: Raw DataFrame from DataLoader.

    Returns:
        cleaned_df: DataFrame with only low-cardinality categorical columns.
    """
    project_data = project_data.fillna("missing")
    project_data = project_data.astype(str)

    cols_to_keep = [
        col for col in project_data.columns
        if 1 < project_data[col].nunique() <= 9
    ]
    return project_data[cols_to_keep]


def prepare_for_model(project_data, variable_types=None):
    """Shared data-cleaning and vectorization pipeline.

    Steps:
        1. Fill NaN with "missing" and cast to string
        2. Apply Rule-of-9 filter (keep columns with 2-9 unique values)
        3. Sync variable_types to surviving columns
        4. Vectorize via Table2Vector (one-hot encoding)
        5. Convert to float32
        6. Compute per-column cardinalities

    Args:
        project_data: Raw DataFrame from DataLoader.
        variable_types: Optional dict mapping column names to types.
            If None or empty, all columns are treated as categorical.

    Returns:
        (cleaned_df, vectorized_df, vectorizer, cardinalities)
    """
    if variable_types is None:
        variable_types = {}

    # 1-2. Clean data (fillna, Rule-of-9)
    project_data = prepare_for_categorical(project_data)

    # 3. Sync variable_types to surviving columns
    variable_types = {c: variable_types.get(c, "categorical") for c in project_data.columns}
    if not variable_types:
        variable_types = {col: "categorical" for col in project_data.columns}

    # 4. Vectorize
    vectorizer = Table2Vector(variable_types)
    vectorized_df = vectorizer.vectorize_table(project_data)

    # 5. Float32 conversion
    vectorized_df = vectorized_df.astype("float32")

    # 6. Cardinalities
    cardinalities = [project_data[c].nunique() for c in project_data.columns]

    return project_data, vectorized_df, vectorizer, cardinalities


def run_training_pipeline(df, config_path, output_path, model_name="AE", prior="gaussian"):
    """
    Reusable training logic that accepts a DataFrame directly.
    Used by both the CLI and the Cloud Worker
    """
    _, vectorized_df, _, cardinalities = prepare_for_model(df)

    model = get_model(model_name, cardinalities)

    logger.info(f"loading config from {config_path}...")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    trainer = Trainer(model, config)

    logger.info(f"Training model....")
    model, history = trainer.train(vectorized_df, prior)

    logger.info("Saving model....")
    save_model(model, output_path)
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

    # 3. Initialize Loader
    logger.info(f"Loading data....")
    data_loader = DataLoader(
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns=additional_drop_columns,
        additional_rename_columns=additional_rename_columns,
        additional_columns_of_interest=additional_interest_columns,
    )
    
    # 4. Load data -- all loaders return (DataFrame, metadata_dict)
    project_data, metadata = data_loader.load_data(data)
    variable_types = metadata.get("variable_types", {})

    # 5. Clean, vectorize, and compute cardinalities
    logger.info("Transforming the data....")
    project_data, vectorized_df, vectorizer, cardinalities = prepare_for_model(
        project_data, variable_types
    )

    # 6. Build and Train model
    logger.info(f"Loading model....")
    model = get_model(model_name, cardinalities)

    logger.info(f"Loading config from config file....")
    with open(config, "r") as file:
        config_dict = yaml.safe_load(file)

    trainer = Trainer(model, config_dict)

    logger.info(f"Training model....")
    model, history = trainer.train(vectorized_df, prior)

    # 10. Save Results
    logger.info("Saving model....")
    save_model(model, output)
    logger.info(f"Saving history....")
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

    data_loader = DataLoader(
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns=additional_drop_columns,
        additional_rename_columns=additional_rename_columns,
        additional_columns_of_interest=additional_interest_columns,
    )
    project_data, metadata = data_loader.load_data(data)
    variable_types = metadata.get("variable_types", {})

    logger.info("Transforming the data....")
    project_data, vectorized_df, vectorizer, cardinalities = prepare_for_model(
        project_data, variable_types
    )

    logger.info(f"Loading model....")
    model = get_model(model_name, cardinalities)

    logger.info(f"Loading config from config file....")
    with open(config, "r") as file:
        config = yaml.safe_load(file)

    trainer = Trainer(model, config)

    logger.info(f"Searching hyperparameters....")
    best_hps = trainer.search_hyperparameters(vectorized_df, prior)

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
def evaluate(
    seed, model_path, data, drop_columns, rename_columns, interest_columns, output
):

    (
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns,
        additional_rename_columns,
        additional_interest_columns,
    ) = define_necessary_elements(data, drop_columns, rename_columns, interest_columns)

    data_loader = DataLoader(
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns=additional_drop_columns,
        additional_rename_columns=additional_rename_columns,
        additional_columns_of_interest=additional_interest_columns,
    )

    set_seed(seed)

    logger.info(f"Loading model....")
    model = load_model(model_path)

    logger.info(f"Loading data....")
    project_data, metadata = data_loader.load_data(data)
    variable_types = metadata.get("variable_types", {})

    logger.info(f"Transforming the data....")
    project_data, vectorized_df, vectorizer, _ = prepare_for_model(
        project_data, variable_types
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

    # 2. Initialize Loader
    logger.info(f"Loading data...")
    data_loader = DataLoader(
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns=additional_drop_columns,
        additional_rename_columns=additional_rename_columns,
        additional_columns_of_interest=additional_interest_columns,
    )

    # 3. Load data -- all loaders return (DataFrame, metadata_dict)
    project_data, metadata = data_loader.load_data(data)
    variable_types = metadata.get("variable_types", {})

    # 4. Clean, vectorize, and compute cardinalities
    logger.info("Transforming the data....")
    project_data, vectorized_df, vectorizer, attr_cardinalities = prepare_for_model(
        project_data, variable_types
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
        vectorized_df, model, k, attr_cardinalities, vectorizer, prior
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
def chow_liu_outliers(
    seed,
    data,
    drop_columns,
    rename_columns,
    interest_columns,
    alpha,
    mi_subsample,
    output,
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

    # 2. Initialize Loader
    logger.info("Loading data...")
    data_loader = DataLoader(
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns=additional_drop_columns,
        additional_rename_columns=additional_rename_columns,
        additional_columns_of_interest=additional_interest_columns,
    )

    # 3. Load data
    project_data, metadata = data_loader.load_data(data)

    # 4. Clean data (fillna + Rule-of-9, no vectorization needed)
    logger.info("Cleaning the data...")
    cleaned_df = prepare_for_categorical(project_data)
    logger.info(f"Cleaned data: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")

    # 5. Fit Chow-Liu tree and score rows
    logger.info("Fitting Chow-Liu tree...")
    ranked_df, cl_model = rank_rows_by_chow_liu(cleaned_df, alpha=alpha, mi_subsample=mi_subsample)

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
    logger.info(f"Outlier scores saved to {output}errors.csv")


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
    "--target_features",
    help="features to condition samples on (comma separated)",
    type=str,
    default=None,
)
def generate(seed, prior, model_path, number_samples, output, data, target_features):

    set_seed(seed)

    logger.info(f"Loading model....")
    model = load_model(model_path)
    decoder = model.decoder

    logger.info(f"Loading data....")
    data_loader = DataLoader(drop_columns=[], rename_columns={}, columns_of_interest=[])
    project_data, metadata = data_loader.load_data(data)
    variable_types = metadata.get("variable_types", {})

    logger.info(f"Creating the vectorizer....")
    project_data, vectorized_df, vectorizer, attr_cardinalities = prepare_for_model(
        project_data, variable_types
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

    logger.info(f"Generating samples....")

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
def pca_baseline(
    seed,
    data,
    drop_columns,
    rename_columns,
    interest_columns,
    column_to_condition,
    outlier_value,
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

    data_loader = DataLoader(
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns=additional_drop_columns,
        additional_rename_columns=additional_rename_columns,
        additional_columns_of_interest=additional_interest_columns,
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
        project_data, variable_types
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

    metrics = evaluate_errors(df_combined, column_to_condition, outlier_values)


if __name__ == "__main__":
    cli()
