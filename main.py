import logging
import os
import sys

import click
import pandas as pd
import yaml

from dataset.loader import DataLoader
from evaluate.evaluator import Evaluator
from evaluate.generator import Generator
from evaluate.outliers import get_outliers_list
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
    evaluate_errors,
    define_necessary_elements,
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

    set_seed(seed)

    logger.info(f"Loading data....")

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
    model, history = trainer.train(vectorized_df, prior)

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

    logger.info(f"Loading model....")
    model = load_model(model_path)

    logger.info(f"Loading data....")
    project_data, variable_types = data_loader.load_data(data)

    base_df = None
    if isinstance(project_data, tuple):
        project_data, base_df = project_data

    logger.info(f"Transforming the data....")
    vectorizer = Table2Vector(variable_types)
    vectorized_df = vectorizer.vectorize_table(project_data, base_df)

    attr_cardinalities = list(project_data.describe().T["unique"].values)

    error_df = get_outliers_list(
        vectorized_df, model, k, attr_cardinalities, vectorizer, prior
    )

    logger.info("Saving outliers....")
    save_to_csv(error_df, output, "errors")


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
    data_loader = DataLoader()
    project_data, variable_types = data_loader.load_data(data)

    attr_cardinalities = list(project_data.describe().T["unique"].values)

    logger.info(f"Creating the vectorizer....")
    vectorizer = Table2Vector(variable_types)
    vectorized_df = vectorizer.vectorize_table(project_data)

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
    generator = Generator(
        decoder,
        number_samples,
        prior,
        len(attr_cardinalities),
        model.get_config()["temperature"],
        vectorized_df.columns,
        model.get_config()["prior_means"][2],
        model.get_config()["prior_log_vars"][2],
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

    additional_drop_columns = None
    additional_rename_columns = None
    additional_interest_columns = None

    if data == "sadc_2017" or data == "sadc_2015":
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
        interest_columns = []

    elif data == "pennycook_1" or data == "pennycook":
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
        interest_columns = []

    elif data == "pennycook_2":
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
        rename_columns = {}
        interest_columns = (
            [11, 12, 13, 14, 16, 17, 18, 19]
            + [x for x in range(20, 35)]
            + [35, 36, 37, 38, 39]
        )

    elif data == "inattentive":
        rename_columns = {}
        interest_columns = (
            [x for x in range(10, 16)]
            + [x for x in range(18, 24)]
            + [25, 27, 29, 30, 31, 32, 33, 35, 3]
            + [x for x in range(36, 55)]
        )

    elif data == "attention_check":
        rename_columns = {}
        interest_columns = [x for x in range(4, 64)] + [2]

    elif data == "moral_data":
        drop_columns = []
        rename_columns = {}
        interest_columns = [x for x in range(2, 10)] + [
            x for x in range(12, 78)
        ]

    elif data == "mturk_ethics":
        drop_columns = []
        rename_columns = {}
        interest_columns = [13, 14] + [
            x for x in range(17, 52)
        ] + [53, 55, 58, 61, 63, 65, 68, 69, 70, 72, 73, 74, 76, 77, 107, 108]

    elif data == "public_opinion":
        drop_columns = []
        rename_columns = {}
        interest_columns = [19, 4] + [
            x for x in range(21, 176)
        ]

    elif data == "racial_data":
        drop_columns = []
        rename_columns = {}
        interest_columns = [5, 6] + [
            x for x in range(7, 73)
        ] + [74,75,76]

    else:
        rename_columns = {
            x.split(":")[0]: x.split(":")[1] for x in rename_columns.split(",")
        }
        interest_columns = []

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

    set_seed(seed)

    data_loader = DataLoader(
        [],
        rename_columns,
        interest_columns,
        additional_drop_columns=additional_drop_columns,
        additional_rename_columns=additional_rename_columns,
        additional_columns_of_interest=additional_interest_columns,
    )

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
    outlier_value
):

    set_seed(seed)

    additional_drop_columns = None
    additional_rename_columns = None
    additional_interest_columns = None

    if data == "sadc_2017" or data == "sadc_2015":
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
        interest_columns = []

    elif data == "pennycook_1" or data == "pennycook":
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
        interest_columns = []

    elif data == "pennycook_2":
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
        rename_columns = {}
        interest_columns = (
                [11, 12, 13, 14, 16, 17, 18, 19]
                + [x for x in range(20, 35)]
                + [35, 36, 37, 38, 39]
        )

    elif data == "inattentive":
        rename_columns = {}
        interest_columns = (
                [x for x in range(10, 16)]
                + [x for x in range(18, 24)]
                + [25, 27, 29, 30, 31, 32, 33, 35, 3]
                + [x for x in range(36, 55)]
        )

    elif data == "attention_check":
        rename_columns = {}
        interest_columns = [x for x in range(4, 64)] + [2]

    elif data == "moral_data":
        drop_columns = []
        rename_columns = {}
        interest_columns = [x for x in range(2, 10)] + [
            x for x in range(12, 78)
        ]

    elif data == "mturk_ethics":
        drop_columns = []
        rename_columns = {}
        interest_columns = [13, 14] + [
            x for x in range(17, 52)
        ] + [53, 55, 58, 61, 63, 65, 68, 69, 70, 72, 73, 74, 76, 77, 107, 108]

    elif data == "public_opinion":
        drop_columns = []
        rename_columns = {}
        interest_columns = [19, 4] + [
            x for x in range(21, 176)
        ]

    elif data == "racial_data":
        drop_columns = []
        rename_columns = {}
        interest_columns = [5, 6] + [
            x for x in range(7, 73)
        ] + [74, 75, 76]

    else:
        rename_columns = {
            x.split(":")[0]: x.split(":")[1] for x in rename_columns.split(",")
        }
        interest_columns = []

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

    set_seed(seed)

    data_loader = DataLoader(
        [],
        rename_columns,
        interest_columns,
        additional_drop_columns=additional_drop_columns,
        additional_rename_columns=additional_rename_columns,
        additional_columns_of_interest=additional_interest_columns,
    )

    column_to_condition = column_to_condition.split(",")
    outlier_values = outlier_value.split(",")

    if data == "sadc_2017" or data == "sadc_2015":
        outlier_dataset = data_loader.find_outlier_data_sadc_2017(data, column_to_condition)

    else:
        outlier_dataset = data_loader.find_outlier_data(data, column_to_condition)

    project_data, variable_types = data_loader.load_data(data)

    logger.info(f"Transforming the data....")

    # drop column of column_to_condition from project_data
    if column_to_condition is not None:
        column_to_condition = column_to_condition[0]
        project_data = project_data.drop(columns=column_to_condition)#["attn1", "attn2", "attnB"])
        variable_types = {
            k: v for k, v in variable_types.items() if k != column_to_condition#!= "attn1" and k != "attn2" and k != "attnB"
        }

    vectorizer = Table2Vector(variable_types)
    vectorized_df = vectorizer.vectorize_table(project_data)

    cardinalities = list(project_data.describe().T["unique"].values)

    from sklearn.decomposition import PCA
    from sklearn.metrics import mean_squared_error
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
