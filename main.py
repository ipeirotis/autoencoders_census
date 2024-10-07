import logging
import os
import sys

import click
import yaml

from dataset.loader import DataLoader
from evaluate.evaluator import Evaluator
from features.transform import Table2Vector
from model.factory import get_model
from train.trainer import Trainer
from utils import (
    set_seed,
    save_model,
    save_history,
    save_hyperparameters,
    model_analysis,
    load_model,
    save_to_csv,
)

logger = logging.Logger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


@click.group()
def cli():
    pass


@cli.command("train")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option(
    "--model_name", help="model to train between AE and VAE", type=str, default="AE"
)
@click.option("--data", help="data to train on", type=str, default="sadc_2017")
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
def train(seed, model_name, data, config, output):

    set_seed(seed)

    logger.info(f"Loading data....")
    data_loader = DataLoader()
    project_data, variable_types = data_loader.load_data(data)

    logger.info(f"Transforming the data....")
    vectorizer = Table2Vector(variable_types)
    vectorized_df = vectorizer.vectorize_table(project_data)

    cardinalities = list(project_data.describe().T["unique"].values)

    logger.info(f"Looading model....")
    model = get_model(model_name, cardinalities)

    logger.info(f"Loading config from config file....")
    with open(config, "r") as file:
        config = yaml.safe_load(file)

    trainer = Trainer(model, config)

    logger.info(f"Training model....")
    model, history = trainer.train(vectorized_df)

    logger.info("Saving model....")
    save_model(model, output)

    logger.info(f"Saving history....")
    save_history(history, output)

    logger.info("Saving plots....")
    model_analysis(history, output, model_name)


@cli.command("search_hyperparameters")
@click.option("--seed", help="seed for reproducibility", type=int, default=2)
@click.option(
    "--model_name", help="model to train between AE and VAE", type=str, default="AE"
)
@click.option("--data", help="data to train on", type=str, default="sadc_2017")
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
    default="cache/hp_model_1/",
)
def search_hyperparameters(seed, model_name, data, config, output):

    set_seed(seed)

    logger.info(f"Loading data....")
    data_loader = DataLoader()
    project_data, variable_types = data_loader.load_data(data)

    logger.info(f"Transforming the data....")
    vectorizer = Table2Vector(variable_types)
    vectorized_df = vectorizer.vectorize_table(project_data)

    cardinalities = list(project_data.describe().T["unique"].values)

    logger.info(f"Looading model....")
    model = get_model(model_name, cardinalities)

    logger.info(f"Loading config from config file....")
    with open(config, "r") as file:
        config = yaml.safe_load(file)

    trainer = Trainer(model, config)

    logger.info(f"Searching hyperparameters....")
    best_hps = trainer.search_hyperparameters(vectorized_df)

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
    "--output",
    help="output path for saving the predictions",
    type=str,
    default="cache/predictions/",
)
def evaluate(seed, model_path, data, output):

    set_seed(seed)

    logger.info(f"Looading model....")
    model = load_model(model_path)

    logger.info(f"Loading data....")
    data_loader = DataLoader()
    project_data, variable_types = data_loader.load_data(data)

    logger.info(f"Transforming the data....")
    vectorizer = Table2Vector(variable_types)
    vectorized_df = vectorizer.vectorize_table(project_data)

    evaluator = Evaluator(model)

    if not os.path.exists(output):
        os.makedirs(output)

    predictions_df = evaluator.evaluate(
        vectorized_df, vectorizer, project_data, variable_types, output
    )

    logger.info("Saving metrics....")
    save_to_csv(predictions_df, output)


if __name__ == "__main__":
    cli()
