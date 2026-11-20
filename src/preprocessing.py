# Data Cleaning, categorical encoding, and missing value handling pipeline

import logging
from typing import Tuple 

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from src.config import (
    CONTRACT_MAP,
    DAYS_EMPLOYED_ANOMALY_VAL,
    EDUCATION_ORDINAL_MAP,
    GENDER_MAP,
    INCOME_CLIP_QUANTILE,
    MNAR_EXT_COLS,
    ORG_MIN_FREQ,
    ORG_SECTOR_PREFIXES,
    OWN_CAR_AGE_DECADE_CAP,
    OWN_CAR_AGE_MAX_VALID,
    RANDOM_STATE,
    RARE_INCOME_TYPES,
    RARE_OFFICE_OCCUPATIONS,
    RARE_SERVICE_OCCUPATIONS,
    SELECTED_FEATURES,
    TARGET_COL,
    TARGET_ENCODE_COLS,
    TEST_PROCESSED_PATH,
    TEST_RAW_PATH,
    TRAIN_PROCESSED_PATH,
    TRAIN_RAW_PATH,
)

logger = logging.getLogger(__name__)
# -----------------------------------------------------------------------------------------------------
def clean_data(train_df, test_df):
    """
    Applies Section 4.1 data cleaning steps strictly fitting thresholds on train_df:
    - Winsorizes AMT_INCOME_TOTAL at the 99.9th percentile.
    - Replaces DAYS_EMPLOYED anomaly (365243) and OWN_CAR_AGE (> 60) with NaN.
    - Replaces residual CODE_GENDER 'XNA' with mode ('F').
    - Consolidates rare categories in NAME_INCOME_TYPE, OCCUPATION_TYPE, and ORGANIZATION_TYPE.
    """

    train_clean = train_df.copy()
    test_clean = test_df.copy()

    # 4.1.1 AMT_INCOME_TOTAL Winsorization (99.9th percentile fitted on train)
    income_upper_limit = train_clean["AMT_INCOME_TOTAL"].quantile(INCOME_CLIP_QUANTILE)
    train_clean["AMT_INCOME_TOTAL"] = train_clean["AMT_INCOME_TOTAL"].clip(upper=income_upper_limit)
    test_clean["AMT_INCOME_TOTAL"] = test_clean["AMT_INCOME_TOTAL"].clip(upper=income_upper_limit)

    # 4.1.2 DAYS_EMPLOYED placeholder replacement
    train_clean["DAYS_EMPLOYED"] = train_clean["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_ANOMALY_VAL, np.nan)
    test_clean["DAYS_EMPLOYED"] = test_clean["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_ANOMALY_VAL, np.nan)

    # 4.1.3 OWN_CAR_AGE outliers (> 60) replacement
    train_clean.loc[train_clean["OWN_CAR_AGE"] > OWN_CAR_AGE_MAX_VALID, "OWN_CAR_AGE"] = np.nan
    test_clean.loc[test_clean["OWN_CAR_AGE"] > OWN_CAR_AGE_MAX_VALID, "OWN_CAR_AGE"] = np.nan

    # 4.1.4 CODE_GENDER residual 'XNA' replacement
    train_clean.loc[train_clean["CODE_GENDER"] == "XNA", "CODE_GENDER"] = "F"
    test_clean.loc[test_clean["CODE_GENDER"] == "XNA", "CODE_GENDER"] = "F"

    # 4.1.5 NAME_INCOME_TYPE rare category consolidation
    train_clean["NAME_INCOME_TYPE"] = train_clean["NAME_INCOME_TYPE"].replace(RARE_INCOME_TYPES, "Other")
    test_clean["NAME_INCOME_TYPE"] = test_clean["NAME_INCOME_TYPE"].replace(RARE_INCOME_TYPES, "Other")

    # 4.1.6 OCCUPATION_TYPE rare profession grouping
    for df in (train_clean, test_clean):
        df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].replace(RARE_OFFICE_OCCUPATIONS, "Other_Office_Staff")
        df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].replace(RARE_SERVICE_OCCUPATIONS, "Other_Service_Labor")

    # 4.1.7 ORGANIZATION_TYPE domain consolidation
    for df in (train_clean, test_clean):
        df["ORGANIZATION_TYPE"] = df["ORGANIZATION_TYPE"].replace("XNA", "Unemployed_Pensioner")
        for prefix in ORG_SECTOR_PREFIXES:
            df.loc[df["ORGANIZATION_TYPE"].str.startswith(prefix, na=False), "ORGANIZATION_TYPE"] = prefix

    org_counts = train_clean["ORGANIZATION_TYPE"].value_counts()
    rare_orgs = org_counts[org_counts < ORG_MIN_FREQ].index

    train_clean["ORGANIZATION_TYPE"] = train_clean["ORGANIZATION_TYPE"].replace(rare_orgs, "Rare_Organization")
    test_clean["ORGANIZATION_TYPE"] = test_clean["ORGANIZATION_TYPE"].replace(rare_orgs, "Rare_Organization")

    return train_clean, test_clean
# ------------------------------------------------------------------------------------------------------------------------
def encode_categorical_features(train_df, test_df):
    """
    Applies Section 4.2 categorical encodings (fitted strictly on train_df):
    - Binary mapping for CODE_GENDER and NAME_CONTRACT_TYPE.
    - Ordinal mapping for NAME_EDUCATION_TYPE.
    - One-Hot Encoding for NAME_INCOME_TYPE.
    - Out-of-Fold Target Encoding for OCCUPATION_TYPE and ORGANIZATION_TYPE.
    """

    train_enc = train_df.copy()
    test_enc = test_df.copy()

    # Binary & Ordinal Mappings
    for df in (train_enc, test_enc):
        df["CODE_GENDER"] = df["CODE_GENDER"].map(GENDER_MAP)
        df["NAME_CONTRACT_TYPE"] = df["NAME_CONTRACT_TYPE"].map(CONTRACT_MAP)
        df["NAME_EDUCATION_TYPE"] = df["NAME_EDUCATION_TYPE"].map(EDUCATION_ORDINAL_MAP)

    # One-Hot Encoding (NAME_INCOME_TYPE)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    train_income_ohe = pd.DataFrame(
        ohe.fit_transform(train_enc[["NAME_INCOME_TYPE"]]),
        columns=ohe.get_feature_names_out(["NAME_INCOME_TYPE"]),
        index=train_enc.index
    )
    test_income_ohe = pd.DataFrame(
        ohe.transform(test_enc[["NAME_INCOME_TYPE"]]),
        columns=ohe.get_feature_names_out(["NAME_INCOME_TYPE"]),
        index=test_enc.index
    )

    train_enc = pd.concat([train_enc.drop(columns=["NAME_INCOME_TYPE"]), train_income_ohe], axis=1)
    test_enc = pd.concat([test_enc.drop(columns=["NAME_INCOME_TYPE"]), test_income_ohe], axis=1)

    # Out-of-Fold Target Encoding (OCCUPATION_TYPE and ORGANIZATION_TYPE)
    te = TargetEncoder(cv=5, smooth="auto", random_state=RANDOM_STATE)
    train_enc[TARGET_ENCODE_COLS] = te.fit_transform(train_enc[TARGET_ENCODE_COLS], train_enc[TARGET_COL])
    test_enc[TARGET_ENCODE_COLS] = te.transform(test_enc[TARGET_ENCODE_COLS])

    return train_enc, test_enc
# ----------------------------------------------------------------------------------------------------------------
def handle_missing_values(train_df, test_df):
    """
    Applies Section 4.3 missing value strategies (fitted strictly on train_df):
    - EXT_SOURCE_2: median imputation.
    - EXT_SOURCE_1 & EXT_SOURCE_3: binary MNAR flag + median imputation.
    - DAYS_EMPLOYED: anomaly flag + zero fill.
    - OWN_CAR_AGE: decade binning (capped at 3, NaN -> -1) + One-Hot Encoding.
    - Sanitizes spaces in column names.
    """

    train_hmv = train_df.copy()
    test_hmv = test_df.copy()

    # EXT_SOURCE_2
    ext2_median = train_hmv["EXT_SOURCE_2"].median()
    train_hmv["EXT_SOURCE_2"] = train_hmv["EXT_SOURCE_2"].fillna(ext2_median)
    test_hmv["EXT_SOURCE_2"] = test_hmv["EXT_SOURCE_2"].fillna(ext2_median)

    # EXT_SOURCE_1 and EXT_SOURCE_3
    for col in MNAR_EXT_COLS:
        median_val = train_hmv[col].median()
        for df in (train_hmv, test_hmv):
            df[f"{col}_IS_MISSING"] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(median_val)

    # DAYS_EMPLOYED
    for df in (train_hmv, test_hmv):
        df["DAYS_EMPLOYED_ANOM"] = df["DAYS_EMPLOYED"].isnull().astype(int)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].fillna(0)

    # OWN_CAR_AGE decade binning + One-Hot Encoding
    for df in (train_hmv, test_hmv):
        df["OWN_CAR_AGE"] = (df["OWN_CAR_AGE"] // 10).clip(upper=OWN_CAR_AGE_DECADE_CAP).fillna(-1).astype(int)

    ohe_car = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    train_car_ohe = pd.DataFrame(
        ohe_car.fit_transform(train_hmv[["OWN_CAR_AGE"]]),
        columns=ohe_car.get_feature_names_out(["OWN_CAR_AGE"]),
        index=train_hmv.index
    )
    test_car_ohe = pd.DataFrame(
        ohe_car.transform(test_hmv[["OWN_CAR_AGE"]]),
        columns=ohe_car.get_feature_names_out(["OWN_CAR_AGE"]),
        index=test_hmv.index
    )

    train_hmv = pd.concat([train_hmv.drop(columns=["OWN_CAR_AGE"]), train_car_ohe], axis=1)
    test_hmv = pd.concat([test_hmv.drop(columns=["OWN_CAR_AGE"]), test_car_ohe], axis=1)

    # Sanitize spaces in column names
    train_hmv.columns = train_hmv.columns.str.replace(" ", "_", regex=False)
    test_hmv.columns = test_hmv.columns.str.replace(" ", "_", regex=False)

    return train_hmv, test_hmv
# ----------------------------------------------------------------------------------------------------------------
def run_preprocessing(train = None, test = None, save_outputs = True):
    """
    End-to-end preprocessing orchestrator mirroring Section 4 of the EDA notebook.
    Loads raw datasets if DataFrames are not provided, executes cleaning, encoding,
    and missing value handling, validates integrity, and saves processed CSVs.
    """

    if train is None or test is None:
        logger.info("Loading raw datasets from disk...")
        train = pd.read_csv(TRAIN_RAW_PATH)
        test = pd.read_csv(TEST_RAW_PATH)

    # Select the 19 core domain features (+ TARGET for train)
    train_selected = train[SELECTED_FEATURES + [TARGET_COL]].copy()
    test_selected = test[SELECTED_FEATURES].copy()

    logger.info("Step 1/3: Running data cleaning (Section 4.1)...")
    train_clean, test_clean = clean_data(train_selected, test_selected)

    logger.info("Step 2/3: Running categorical encoding (Section 4.2)...")
    train_enc, test_enc = encode_categorical_features(train_clean, test_clean)

    logger.info("Step 3/3: Handling missing values (Section 4.3)...")
    train_final, test_final = handle_missing_values(train_enc, test_enc)

    # Data Quality Gates
    assert train_final.isnull().sum().sum() == 0, "Null values remain in train_final!"
    assert test_final.isnull().sum().sum() == 0, "Null values remain in test_final!"
    assert list(train_final.drop(columns=[TARGET_COL]).columns) == list(test_final.columns), ("Feature schema mismatch between train_final and test_final!")

    if save_outputs:
        train_final.to_csv(TRAIN_PROCESSED_PATH, index=False)
        test_final.to_csv(TEST_PROCESSED_PATH, index=False)
        logger.info(f"Processed datasets saved -> train_final: {train_final.shape} | test_final: {test_final.shape}")

    return train_final, test_final