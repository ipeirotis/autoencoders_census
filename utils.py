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
                relevant.append(1 if str(int(row[column])) != str(int(value)) else 0)

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


def define_necessary_elements(data, drop_columns, rename_columns, interest_columns):

    additional_drop_columns = None
    additional_rename_columns = None
    additional_interest_columns = None

    if data == "sadc_2017" or data == "sadc_2015":
        drop_columns = [
            "sitecode",
            "sitename",
            "sitetype",
            "sitetypenum",
            "year",
            "survyear",
            "record",
            "stratum",
            "PSU",
            "q14",
            "q20",
            "q31",
            "q36",
            "q37",
            "q39",
            "q44",
            "q56",
            "q84",
        ]

        rename_columns = {
            "age": "age",
            "sex": "sex",
            "grade": "grade",
            "race4": "Hispanic_or_Latino",
            "race7": "race",
            "qnobese": "obese",
            "qnowt": "overweight",
            "q67": "sexual_identity",
            "q66": "sex/sexual_contacts",
            "sexid": "sexid",
            "sexid2": "sexid2",
            "sexpart": "sexpart",
            "sexpart2": "sexpart2",
            "q8": "seat_belt_use",
            "q9": "riding_with_a_drinking_driver",
            "q10": "drinking_and_driving",
            "q11": "texting_and_driving",
            "q12": "weapon_carrying",
            "q13": "weapon_carrying_at_school",
            "q15": "safety_concerns_at_school",
            "q16": "threatened_at_school",
            "q17": "physical_fighting",
            "q18": "physical_fighting_at_school",
            "q19": "forced_sexual_intercourse",
            "q21": "sexual_dating_violence",
            "q22": "physical_dating_violence",
            "q23": "bullying_at_school",
            "q24": "electronic_bullying",
            "q25": "sad_or_hopeless",
            "q26": "considered_suicide",
            "q27": "made_a_suicide_plan",
            "q28": "attempted_suicide",
            "q29": "injurious_suicide_attempt",
            "q30": "ever_cigarette_use",
            "q32": "current_cigarette_use",
            "q33": "smoking_amounts_per_day",
            "q34": "electronic_vapor_product_use",
            "q35": "current_electronic_vapor_product_use",
            "q38": "current_cigar_use",
            "q40": "ever_alcohol_use",
            "q41": "initiation_of_alcohol_use",
            "q42": "current_alcohol_use",
            "q43": "source_of_alcohol",
            "q45": "largest_number_of_drinks",
            "q46": "ever_marijuana_use",
            "q47": "initiation_of_marijuana_use",
            "q48": "current_marijuana_use",
            "q49": "ever_cocaine_use",
            "q50": "ever_inhalant_use",
            "q51": "ever_heroin_use",
            "q52": "ever_methamphetamine_use",
            "q53": "ever_ecstasy_use",
            "q54": "ever_synthetic_marijuana_use",
            "q55": "ever_steroid_use",
            "q57": "illegal_injected_drug_use",
            "q58": "illegal_drugs_at_school",
            "q59": "ever_sexual_intercourse",
            "q60": "first_sex_intercourse",
            "q61": "multiple_sex_partners",
            "q62": "current_sexual_activity",
            "q63": "alcohol/drugs_at_sex",
            "q64": "condom_use",
            "q65": "birth_control_pill_use",
            "q68": "perception_of_weight",
            "q69": "weight_loss",
            "q70": "fruit_juice_drinking",
            "q71": "fruit_eating",
            "q72": "green _salad_eating",
            "q73": "potato_eating",
            "q74": "carrot_eating",
            "q75": "other_vegetable_eating",
            "q76": "soda_drinking",
            "q77": "milk_drinking",
            "q78": "breakfast_eating",
            "q79": "physical_activity",
            "q80": "television_watching",
            "q81": "computer_not_school_work_use",
            "q82": "PE_attendance",
            "q83": "sports_team_participation",
            "q85": "HIV_testing",
            "q86": "oral_health_care",
            "q87": "asthma",
            "q88": "sleep_on_school_night",
            "q89": "grades_in_school",
            "qhallucdrug": "ever_used_LSD",
            "qsportsdrink": "sports_drinks",
            "qwater": "plain_water",
            "qfoodallergy": "food_allergies",
            "qmusclestrength": "muscle_stregthening",
            "qindoortanning": "indoor_tanning",
            "qsunburn": "sunburn",
            "qconcentrating": "difficulty_concentrating",
            "qspeakenglish": "how_well_speak_English",
        }

        # The dataframe contains separate questionnaire questions, here we merge these columns to our project dataframe
        interest_columns = [x for x in range(89)] + [
            221,
            231,
            234,
            236,
            238,
            240,
            241,
            242,
            245,
        ]

    elif data == "pennycook_1" or data == "pennycook":
        drop_columns = []
        rename_columns = {
            "COVID_concern_1": "COVID_concern",
            "Media1.0": "news_side",
            "Media1": "news_criticism",
            "Media3_1": "trust_national_news_org",
            "Media3_2": "trust_local_news_org",
            "Media3_3": "trust_friends_family",
            "Media3_11": "trust_social",
            "Media3_12": "trust_fact_checkers",
            "SharingType_1": "sharing_political",
            "SharingType_2": "sharing_sports",
            "SharingType_3": "sharing_celebrity",
            "SharingType_4": "sharing_science",
            "SharingType_6": "sharing_business",
            "SharingType_7": "sharing_other",
            "SocialMedia_1": "facebook",
            "SocialMedia_2": "twitter",
            "SocialMedia_3": "snapchat",
            "SocialMedia_4": "instagram",
            "SocialMedia_5": "whatsapp",
            "SocialMedia_6": "other",
        }
        # interest_columns = [7,8,9,11,12,13,14,27,28,30,31,32,33,47,48,49,50.51,52,54,55,56,57,58,59,
        #                     73,74,75,76,77,78,79] + [x for x in range(79,193)] + [314,315,316,317,318,319] + [
        #     x for x in range(328, 345)
        # ] + [x for x in range(346, 356)] + [363,364,365,366,367,368,369,370,375]
        interest_columns = ([
            7,
            8,
            9,
            11,
            12,
            13,
            14,
            15,
            27,
            28,
            30,
            31,
            32,
            33,
            47,
            48,
            49,
            50,
            51,
            52,
            54,
            55,
            56,
            57,
            58,
            59,
            363,
            364,
            365,
            366,
            367,
            368,
            369,
            370,
            375,
        ] +
        # [
        #     x for x in range(73, 103) #cond1
        # ] +
        # [
        #     x for x in range(103, 133) #cond2
        # ] +
        # [
        #     x for x in range(133, 163) #cond3
        # ]
        #    +
        [x for x in range(163, 193)] +  #cond4
                            [
            x for x in range(314, 321) #crt
        ] + [
            x for x in range(328, 345) #sci
        ] + [
            x for x in range(346, 356) #mms
        ])
        # interest_columns = []


    elif data == "pennycook_2":
        drop_columns = []
        rename_columns = {
            "COVID_concern_1": "COVID_concern",
            "Media1.0": "news_side",
            "Media1": "news_criticism",
            "Media3_1": "trust_national_news_org",
            "Media3_2": "trust_local_news_org",
            "Media3_3": "trust_friends_family",
            "Media3_11": "trust_social",
            "Media3_12": "trust_fact_checkers",
            "SharingType_1": "sharing_political",
            "SharingType_2": "sharing_sports",
            "SharingType_3": "sharing_celebrity",
            "SharingType_4": "sharing_science",
            "SharingType_6": "sharing_business",
            "SharingType_7": "sharing_other",
            "SocialMedia_1": "facebook",
            "SocialMedia_2": "twitter",
            "SocialMedia_3": "snapchat",
            "SocialMedia_4": "instagram",
            "SocialMedia_5": "whatsapp",
            "SocialMedia_6": "other",
        }
        interest_columns = [
            1,
            2,
            4,
            5,
            6,
            51,
            52,
            310,
            312,
            313,
            314,
            315,
            18,
            19,
            21,
            22,
            23,
            24,
            37,
            38,
            39,
            40,
            41,
            42,
            44,
            45,
            46,
            47,
            48,
            49,
            289,
            290,
            291,
            292,
            293,
            294,
            295,
            296,
            301,
        ]

    elif data == "bot_bot_mturk":
        drop_columns = []
        rename_columns = {}
        interest_columns = (
            [11, 12, 13, 14, 16, 17, 18, 19]
            + [x for x in range(20, 34)]
            + [35, 36, 37, 38, 39]
        )

    elif data == "inattentive":
        drop_columns = []
        rename_columns = {}
        interest_columns = [x for x in range(10, 16)] + [
            x for x in range(18, 24)
        ] + [
            25, 27, 29, 30, 31, 32, 33, 35
        ] + [
            x for x in range(36, 55)
        ]

    elif data == "attention_check":
        drop_columns = []
        rename_columns = {}
        interest_columns = [x for x in range(4, 64)]

    elif data == "moral_data":
        drop_columns = []
        rename_columns = {}
        interest_columns = [x for x in range(2, 10)] + [
            x for x in range(12, 77)
        ]

    elif data == "mturk_ethics":
        drop_columns = []
        rename_columns = {}
        interest_columns = [13, 14] + [
            x for x in range(17, 52)
        ] + [53, 55, 58, 61, 63, 65, 68, 69, 70, 72, 73, 74, 76, 77]

    elif data == "public_opinion":
        drop_columns = []
        rename_columns = {}
        interest_columns = [19, 4] + [
            x for x in range(21, 175)
        ]

    elif data == "racial_data":
        drop_columns = []
        rename_columns = {}
        interest_columns = [5, 6] + [
            x for x in range(7, 73)
        ]

    else:
        drop_columns = drop_columns.split(",")
        rename_columns = {
            x.split(":")[0]: x.split(":")[1] for x in rename_columns.split(",")
        }
        interest_columns = [int(x) for x in interest_columns.split(",")]

    if data == "pennycook":
        additional_drop_columns = []
        additional_rename_columns = {
            "COVID_concern_1": "COVID_concern",
            "Media1.0": "news_side",
            "Media1": "news_criticism",
            "Media3_1": "trust_national_news_org",
            "Media3_2": "trust_local_news_org",
            "Media3_3": "trust_friends_family",
            "Media3_11": "trust_social",
            "Media3_12": "trust_fact_checkers",
            "SharingType_1": "sharing_political",
            "SharingType_2": "sharing_sports",
            "SharingType_3": "sharing_celebrity",
            "SharingType_4": "sharing_science",
            "SharingType_6": "sharing_business",
            "SharingType_7": "sharing_other",
            "SocialMedia_1": "facebook",
            "SocialMedia_2": "twitter",
            "SocialMedia_3": "snapchat",
            "SocialMedia_4": "instagram",
            "SocialMedia_5": "whatsapp",
            "SocialMedia_6": "other",
        }

        additional_interest_columns = [
            1,
            2,
            4,
            5,
            6,
            51,
            52,
            310,
            312,
            313,
            314,
            315,
            18,
            19,
            21,
            22,
            23,
            24,
            37,
            38,
            39,
            40,
            41,
            42,
            44,
            45,
            46,
            47,
            48,
            49,
            289,
            290,
            291,
            292,
            293,
            294,
            295,
            296,
            301,
        ]

    return (
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns,
        additional_rename_columns,
        additional_interest_columns,
    )
