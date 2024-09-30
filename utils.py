import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import yaml


def set_seed(seed):
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def save_model(model, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filename = output_path + "autoencoder.h5"
    model.save(filename)


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def save_history(history, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filename = output_path + "history.npy"
    np.save(filename, history.history)


def save_hyperparameters(hyperparameters, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filename = output_path + "best_hyperparameters.yaml"
    with open(filename, 'w') as file:
        yaml.dump(hyperparameters, file)


def model_analysis(history, output_path):
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 6))
    plt.title("Cross categorical entropy loss")
    sns.lineplot(x=epochs, y=train_loss, label="Train", linewidth=3)
    sns.lineplot(x=epochs, y=val_loss, label="Validation", linewidth=3)
    plt.xlabel("Epochs")

    plt.legend()
    plt.savefig(output_path + "loss_plot.png")


def save_to_csv(df: pd.DataFrame, output_path: str):
    df.to_csv(output_path + "metrics.csv", index=False)
