import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import yaml
from ranx import Qrels, Run, evaluate


def set_seed(seed):
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def save_model(model, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filename = output_path + "autoencoder"
    model.save(filename, save_format="tf")


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
    with open(filename, "w") as file:
        yaml.dump(hyperparameters, file)


def model_analysis(history, output_path, model):

    if model == "AE":
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

    if model == "VAE":
        train_loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        epochs = range(1, len(train_loss) + 1)

        plt.figure(figsize=(8, 6))
        plt.title("Total loss")
        sns.lineplot(x=epochs, y=train_loss, label="Train", linewidth=3)
        sns.lineplot(x=epochs, y=val_loss, label="Validation", linewidth=3)
        plt.xlabel("Epochs")

        plt.legend()
        plt.savefig(output_path + "total_loss_plot.png")

        train_loss = history.history["reconstruction_loss"]
        val_loss = history.history["val_reconstruction_loss"]

        epochs = range(1, len(train_loss) + 1)

        plt.figure(figsize=(8, 6))
        plt.title("Reconstruction loss")
        sns.lineplot(x=epochs, y=train_loss, label="Train", linewidth=3)
        sns.lineplot(x=epochs, y=val_loss, label="Validation", linewidth=3)
        plt.xlabel("Epochs")

        plt.legend()
        plt.savefig(output_path + "reconstruction_loss_plot.png")

        train_loss = history.history["kl_loss"]
        val_loss = history.history["val_kl_loss"]

        epochs = range(1, len(train_loss) + 1)

        plt.figure(figsize=(8, 6))
        plt.title("KL loss")
        sns.lineplot(x=epochs, y=train_loss, label="Train", linewidth=3)
        sns.lineplot(x=epochs, y=val_loss, label="Validation", linewidth=3)
        plt.xlabel("Epochs")

        plt.legend()
        plt.savefig(output_path + "kl_loss_plot.png")


def save_to_csv(df: pd.DataFrame, output_path: str, suffix: str = "metrics"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df.to_csv(output_path + f"{suffix}.csv", index=False)


# Precision@k
def precision_at_k(relevance, k):
    return np.sum(relevance[:k]) / k


# Recall@k
def recall_at_k(relevance, k, total_relevant):
    return np.sum(relevance[:k]) / total_relevant


# DCG
def dcg(relevance, k):
    relevance = relevance[:k]
    return np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))


# nDCG
def ndcg(relevance, k):
    ideal_relevance = np.sort(relevance)[::-1]
    return dcg(relevance, k) / dcg(ideal_relevance, k)


def evaluate_errors(error_data, column, values):

    error_data = error_data.replace("<NA>", "0")
    error_data = error_data.fillna("0")

    print(len(error_data))

    min_error_data = error_data["error"].min()
    max_error_data = error_data["error"].max()

    error_data["error"] = (error_data["error"] - min_error_data)/(max_error_data - min_error_data)

    # error_data["error"] = 1 - error_data["error"]

    if len(column) == 1:
        column = column[0]

    if len(values) == 1:
        value = values[0]

        total_w_error = len(error_data) - len(error_data[error_data[column] == value])

        if total_w_error == 0:
            total_w_error = len(error_data) - len(
                error_data[error_data[column] == str(value)]
            )

        # sort df by "error" column
        error_data = error_data.sort_values("error", ascending=False)
        errors = len(error_data.iloc[:total_w_error, :][error_data[column] == value])
        str_detection = False

        if errors == 0:
            errors = len(
                error_data.iloc[:total_w_error, :][error_data[column] != str(value)]
            )
            str_detection = True

        relevant = []
        for i, row in error_data.iterrows():
            if not str_detection:
                relevant.append(1 if row[column] != value else 0)

            else:
                relevant.append(1 if str(row[column]) != str(value) else 0)

    else:
        temp_data = error_data.copy()
        for c, v in zip(column, values):
            temp_data = temp_data[temp_data[c] == str(v)]

        total_w_error = len(error_data) - len(temp_data)

        error_data = error_data.sort_values("error", ascending=False)

        relevant = []
        for i, row in error_data.iterrows():
            correct = True
            for c, v in zip(column, values):
                if row[c] == str(v):
                    correct = False
                    break

            relevant.append(0 if not correct else 1)
    total_w_error = sum(relevant)
    errors = len([x for x in relevant[:total_w_error] if x == 1])

    accuracy = errors / total_w_error

    errors_prob = error_data["error"].tolist()

    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, thresholds = roc_curve(relevant, errors_prob)
    auc_score = roc_auc_score(relevant, errors_prob)

    # Step 3: Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    probs_0 = [prob for rel, prob in zip(relevant, errors_prob) if rel == 0]
    probs_1 = [prob for rel, prob in zip(relevant, errors_prob) if rel == 1]

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(probs_0, bins=30, alpha=0.5, label='Class 0 (Negatives)', color='blue', density=True)
    plt.hist(probs_1, bins=30, alpha=0.5, label='Class 1 (Positives)', color='red', density=True)

    # Labels and title
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.title("Histogram of Predicted Probabilities for 0s and 1s")
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()

    qrels = {
        "query_1": {f"doc_{i + 1}": rel for i, rel in enumerate(relevant) if rel == 1}
    }

    results = {
        "query_1": {f"doc_{i + 1}": len(relevant) - i for i in range(len(relevant))}
    }

    qrels = Qrels(qrels)
    run = Run(results)

    # Evaluate metrics
    metrics = [
        f"hits@{total_w_error}",
        "precision@10",
        "precision@50",
        "precision@100",
        f"map",
        f"map@68",
        f"ndcg@{total_w_error}",
        f"mrr",
    ]
    evaluation = evaluate(qrels, run, metrics)

    # Print results
    print(evaluation)
    print(f"Total errors: {total_w_error}")

    metrics = pd.DataFrame(
        {"Total": [total_w_error], "Total_errors": [errors], "Accuracy": [accuracy]}
    )
    print(f"Accuracy: {round(accuracy, 4)}")

    y_true = [1] * (total_w_error) + [0] * (len(error_data) - total_w_error)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, relevant)

    print(auc)

    k_values = range(1, len(error_data))
    recall_at_k = []
    best_recall_at_k = []
    random_recall_at_k = []
    total_positives = np.sum(y_true)

    for k in k_values:
        # True positives within top k
        tp_at_k = np.sum(relevant[:k])

        # Recall@k
        recall = tp_at_k / total_positives if total_positives > 0 else 0
        recall_at_k.append(recall)

        # Best Recall@k (all top k are true positives if possible)
        best_recall = min(k, total_positives) / total_positives if total_positives > 0 else 0
        best_recall_at_k.append(best_recall)

        # Random Recall@k (proportional to k/total samples)
        random_recall = (k * total_positives / (len(error_data) * total_positives))
        random_recall_at_k.append(random_recall)

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, recall_at_k, label="Recall@k", marker='o')
    plt.plot(k_values, best_recall_at_k, label="Best Recall@k (Ideal)", linestyle='--')
    plt.plot(k_values, random_recall_at_k, label="Random Recall@k (Baseline)", linestyle=':')
    plt.xlabel("k (Number of Samples)")
    plt.ylabel("Recall")
    plt.title("Recall@k vs. Best Recall@k and Random Recall@k")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    # recall_figure_path = "recall_at_k_comparison.png"
    plt.show()
    metrics = pd.DataFrame(
        {"Total": [total_w_error], "Total_errors": [errors], "Accuracy": [accuracy]}
    )
    print(f"Accuracy: {round(accuracy, 4)}")

    return metrics
