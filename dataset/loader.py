import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """
    Class to handle data loading and preprocessing for the project.
    """

    DROP_COLUMNS = [
        "sitecode",
        "sitename",
        "sitetype",
        "sitetypenum",
        "year",
        "survyear",
        "record",
        "stratum",
        "PSU",
    ]

    RENAME_COLUMNS = {
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
        "q14": "gun_carrying_past_12_mos",
        "q15": "safety_concerns_at_school",
        "q16": "threatened_at_school",
        "q17": "physical_fighting",
        "q18": "physical_fighting_at_school",
        "q19": "forced_sexual_intercourse",
        "q20": "sexual_violence",
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
        "q31": "initation_of_cigarette_smoking",
        "q32": "current_cigarette_use",
        "q33": "smoking_amounts_per_day",
        "q34": "electronic_vapor_product_use",
        "q35": "current_electronic_vapor_product_use",
        "q36": "EVP_from_store",
        "q37": "current_smokeless_tobacco_use",
        "q38": "current_cigar_use",
        "q39": "all_tobacco_product_cessation",
        "q40": "ever_alcohol_use",
        "q41": "initiation_of_alcohol_use",
        "q42": "current_alcohol_use",
        "q43": "source_of_alcohol",
        "q44": "current_binge_drinking",
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
        "q56": "ever_prescription_pain_medicine_use",
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
        "q84": "concussion_in_last_12_mos",
        "q85": "HIV_testing",
        "q86": "oral_health_care",
        "q87": "asthma",
        "q88": "sleep_on_school_night",
        "q89": "grades_in_school",
        "qdrivemarijuana": "drive_when_using_marijuana",
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
    COLUMNS_OF_INTEREST = [214, 230, 240, 243, 245, 247, 249, 250, 251, 254]

    @staticmethod
    def load_2015():
        url = "data/sadc_2015only_national.csv"
        df2015 = DataLoader.load_original_data(url)
        return DataLoader.prepare_original_dataset(df2015)

    @staticmethod
    def load_2017():
        url = "data/sadc_2017only_national_full.csv"
        df2017 = DataLoader.load_original_data(url)
        return DataLoader.prepare_original_dataset(df2017)

    def load_data(self, dataset: str):
        if dataset == "sadc_2015":
            return DataLoader.load_2015()

        if dataset == "sadc_2017":
            return DataLoader.load_2017()

        raise ValueError(f"Dataset {dataset} not found")

    @staticmethod
    def load_original_data(dataset_url, columns_to_drop=DROP_COLUMNS):
        original_df = pd.read_csv(dataset_url).drop(columns=columns_to_drop)
        return original_df

    @staticmethod
    def convert_to_categorical(df, numeric_vars):
        df_copy = df.copy()

        for column in numeric_vars:
            missing_mask = df_copy[column].isna()

            scaler = StandardScaler()
            df_copy[column] = scaler.fit_transform(df_copy[[column]])

            conditions = [
                (df_copy[column] > 1.8),
                (df_copy[column] < -1.8),
                (missing_mask),
                (df_copy[column] >= -1.8) & (df_copy[column] <= 1.8),
            ]
            choices = ["top-extreme", "bottom-extreme", "missing", "normal"]
            df_copy[column + "_cat"] = np.select(conditions, choices, default="unknown")

        df_copy = df_copy.drop(columns=numeric_vars)

        return df_copy

    @staticmethod
    def prepare_original_dataset(
        original_df, column_lst=COLUMNS_OF_INTEREST, rename_columns=RENAME_COLUMNS
    ):
        """
        Prepare the dataset for the project, this includes selecting specific columns, renaming them
        and categorizing them as numeric or categorical.
        """
        project_data = original_df.iloc[:, :98]
        project_data = pd.concat(
            [project_data, original_df.iloc[:, column_lst]], axis=1
        )
        project_data.rename(columns=rename_columns, inplace=True)

        project_data["obese"] = project_data["obese"].replace(
            {1: "obese", 2: "not obese"}
        )
        project_data["overweight"] = project_data["overweight"].replace(
            {1: "overweight", 2: "not overweight"}
        )

        numeric_vars = ["weight", "stheight", "stweight", "bmi", "bmipct"]
        project_data = DataLoader.convert_to_categorical(project_data, numeric_vars)

        # We drop these variables because in 2015 they have zero values
        problematic = [
            "gun_carrying_past_12_mos",
            "sexual_violence",
            "initation_of_cigarette_smoking",
            "EVP_from_store",
            "current_smokeless_tobacco_use",
            "all_tobacco_product_cessation",
            "current_binge_drinking",
            "ever_prescription_pain_medicine_use",
            "concussion_in_last_12_mos",
            "drive_when_using_marijuana",
        ]
        project_data = project_data.drop(columns=problematic)

        categorical_vars = [c for c in project_data.columns if c not in numeric_vars]
        for c in categorical_vars:
            project_data[c] = project_data[c].astype("str")

        variable_types = {
            column: ("numeric" if column in numeric_vars else "categorical")
            for column in project_data.columns
        }

        return project_data, variable_types
