import logging
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Evaluator:
    def __init__(self, model):
        self.model = model

    def impute_missing_values(self, vectorized_df):
        """
        Impute missing values in the dataframe using the model.

        :param df: Dataframe with missing values
        :return: Dataframe with imputed missing values
        """
        filled = vectorized_df.fillna(vectorized_df.median())

        # Define a condition for stopping the iteration
        epsilon = 1e-5
        old_filled = None

        while old_filled is None or np.abs(filled - old_filled).sum().sum() > epsilon:
            # Save the old filled DataFrame for convergence check
            old_filled = filled.copy()

            # Run the data through the autoencoder, which will return a complete version of the data.
            predicted = self.model.predict(filled)

            # Replace the initially guessed values in the original data with the corresponding values from the autoencoder's output. But keep the observed values unchanged.
            mask = vectorized_df.isna()
            filled[mask] = np.where(mask, predicted, filled)

        return filled

    def predict(self, data):
        """
        Predict the target variable using the model.

        :param df: Dataframe with features
        :return: Predictions
        """
        predicted = pd.DataFrame(self.model.predict(data))
        predicted.columns = data.columns
        return predicted

    def create_scatterplot_for_categorical(
        self, original_df, predicted_df, categ_attr, output_path
    ):
        # Create a list of all unique categories present in the original data
        all_categories = original_df[categ_attr].unique()

        # Create the confusion matrix using crosstab
        confusion_matrix = pd.crosstab(
            original_df[categ_attr], predicted_df[categ_attr]
        )

        # Reindex the confusion matrix to include all categories in the original data
        confusion_matrix = confusion_matrix.reindex(
            index=all_categories, columns=all_categories, fill_value=0
        )

        # Calculate accuracy
        diagonal_sum = np.trace(confusion_matrix.values)
        total_sum = np.sum(confusion_matrix.values)
        accuracy = diagonal_sum / total_sum * 100

        # Calculate baseline accuracy
        baseline_accuracy = (
            original_df[categ_attr].value_counts().max() / total_sum * 100
        )

        plt.figure(figsize=(5, 5))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="YlGnBu")
        plt.title(
            f"Accuracy: {accuracy:.2f}%, Baseline accuracy: {baseline_accuracy:.2f}%"
        )

        plt.savefig(output_path + f"/{categ_attr.replace('/', '_')}_conf_matrix.png")

        y_true = label_binarize(original_df[categ_attr], classes=all_categories)
        y_pred = label_binarize(predicted_df[categ_attr], classes=all_categories)
        ova_roc_auc = roc_auc_score(y_true, y_pred, multi_class="ovr", average="macro")

        return accuracy, baseline_accuracy, ova_roc_auc

    def evaluate(self, data, vectorizer, project_data, variable_types, output_path):

        logger.info(f"Imputing missing values....")
        modified_data = self.impute_missing_values(data)

        logger.info("Predicting....")
        predictions = self.predict(modified_data)

        tabular_from_predicted = vectorizer.tabularize_vector(predictions)

        logger.info("Evaluating....")
        variables = []
        accuracies = []
        baseline_accuracies = []
        ova = []
        lift = []
        for v in variable_types.keys():
            if variable_types[v] == "categorical":
                acc, base_acc, ova_roc_auc = self.create_scatterplot_for_categorical(
                    project_data, tabular_from_predicted, v, output_path
                )
                variables.append(v)
                accuracies.append(acc)
                baseline_accuracies.append(base_acc)
                ova.append(ova_roc_auc)
                lift.append(round(acc / base_acc, 2))

        logger.info(f"Mean lift of the classifier: {np.mean(lift)}")

        return pd.DataFrame(
            {
                "Variable": variables,
                "Accuracy": accuracies,
                "Baseline Accuracy": baseline_accuracies,
                "Lift": lift,
                "OVA ROC AUC": ova,
            }
        )
