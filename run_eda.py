"""
Orchestrator script for Exploratory Data Analysis (EDA) — Section 3 of the notebook.
Performs missing-value auditing, logs summary statistics, and automatically generates
and saves all exploratory visualizations to reports/figures/.
"""

import logging

import matplotlib  
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    FIGURES_DIR,
    REPORTS_DIR,
    SELECTED_FEATURES,
    TARGET_COL,
    TEST_RAW_PATH,
    TRAIN_RAW_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RunEDA")
# ------------------------------------------------------------------------------------------------
def audit_missing_values(train, test):
    # Computes and saves a missing-value audit table for train and test sets.

    train_missing = train[SELECTED_FEATURES].isnull().sum()
    test_missing = test[SELECTED_FEATURES].isnull().sum()

    audit_df = pd.DataFrame({
        "Train_Missing_Count": train_missing,
        "Train_Missing_Pct": (train_missing / len(train) * 100).round(2),
        "Test_Missing_Count": test_missing,
        "Test_Missing_Pct": (test_missing / len(test) * 100).round(2),
    })
    audit_df = audit_df[
        (audit_df["Train_Missing_Count"] > 0) | (audit_df["Test_Missing_Count"] > 0)
    ].sort_values(by="Train_Missing_Pct", ascending=False)

    audit_path = REPORTS_DIR / "missing_values_audit.csv"
    audit_df.to_csv(audit_path)
    logger.info(f"Missing values audit saved to: {audit_path}")
    logger.info(f"{audit_df.to_string()}")

    return audit_df
# -------------------------------------------------------------------------------------------------
def save_figure(filename):
    """Saves the current matplotlib figure in high resolution and frees memory."""
    path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved figure: {path.name}")
# -------------------------------------------------------------------------------------------------
def generate_eda_plots(train) -> None:
    """Generates all exploratory plots from Section 3.2 of the EDA notebook."""

    # 3.2.1 TARGET
    plt.figure(figsize=(6, 4))
    sns.countplot(data=train, x=TARGET_COL)
    plt.title("Target Class Distribution (TARGET)")
    save_figure("01_target_distribution.png")

    # 3.2.2 EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]):
        sns.histplot(data=train, x=col, ax=ax)
        ax.set_title(col)
    save_figure("02_ext_sources_distributions.png")

    # 3.2.3 AMT_CREDIT, AMT_ANNUITY, AMT_INCOME_TOTAL (raw + log10), AMT_GOODS_PRICE
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    sns.histplot(data=train, x="AMT_CREDIT", ax=axes[0, 0])
    axes[0, 0].set_title("AMT_CREDIT")

    sns.histplot(data=train, x="AMT_ANNUITY", ax=axes[0, 1])
    axes[0, 1].set_title("AMT_ANNUITY")

    sns.histplot(data=train, x="AMT_GOODS_PRICE", ax=axes[0, 2])
    axes[0, 2].set_title("AMT_GOODS_PRICE")

    sns.histplot(data=train, x="AMT_INCOME_TOTAL", ax=axes[1, 0])
    axes[1, 0].set_title("AMT_INCOME_TOTAL (Raw)")

    sns.histplot(data=train, x="AMT_INCOME_TOTAL", log_scale=10, ax=axes[1, 1])
    axes[1, 1].set_title("AMT_INCOME_TOTAL (Log10 Scale)")

    axes[1, 2].axis("off")
    save_figure("03_amount_features_distributions.png")

    # 3.2.4 DAYS_BIRTH, DAYS_EMPLOYED, DAYS_ID_PUBLISH, DAYS_LAST_PHONE_CHANGE
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    day_cols = ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"]
    for ax, col in zip(axes.flat, day_cols):
        sns.histplot(data=train, x=col, ax=ax)
        ax.set_title(col)
    save_figure("04_days_features_distributions.png")

    # 3.2.5 NAME_EDUCATION_TYPE
    plt.figure(figsize=(12, 5))
    sns.countplot(data=train, x="NAME_EDUCATION_TYPE")
    plt.title("NAME_EDUCATION_TYPE")
    plt.xticks(rotation=20, ha="right")
    save_figure("05_education_type.png")

    # 3.2.6 CODE_GENDER
    plt.figure(figsize=(6, 4))
    sns.countplot(data=train, x="CODE_GENDER")
    plt.title("CODE_GENDER")
    save_figure("06_code_gender.png")

    # 3.2.7 NAME_INCOME_TYPE
    plt.figure(figsize=(12, 5))
    sns.countplot(data=train, x="NAME_INCOME_TYPE")
    plt.title("NAME_INCOME_TYPE")
    plt.xticks(rotation=25, ha="right")
    save_figure("07_income_type.png")

    # 3.2.8 NAME_CONTRACT_TYPE
    plt.figure(figsize=(6, 4))
    sns.countplot(data=train, x="NAME_CONTRACT_TYPE")
    plt.title("NAME_CONTRACT_TYPE")
    save_figure("08_contract_type.png")

    # 3.2.9 OCCUPATION_TYPE
    plt.figure(figsize=(14, 5))
    sns.countplot(data=train, x="OCCUPATION_TYPE")
    plt.title("OCCUPATION_TYPE")
    plt.xticks(rotation=45, ha="right")
    save_figure("09_occupation_type.png")

    # 3.2.10 OWN_CAR_AGE
    plt.figure(figsize=(8, 4))
    sns.histplot(data=train, x="OWN_CAR_AGE", bins=40)
    plt.title("OWN_CAR_AGE")
    save_figure("10_own_car_age.png")

    # 3.2.11 ORGANIZATION_TYPE
    plt.figure(figsize=(20, 6))
    sns.countplot(data=train, x="ORGANIZATION_TYPE")
    plt.title("ORGANIZATION_TYPE")
    plt.xticks(rotation=90)
    save_figure("11_organization_type.png")

    # 3.2.12 REGION_RATING_CLIENT
    plt.figure(figsize=(6, 4))
    sns.countplot(data=train, x="REGION_RATING_CLIENT")
    plt.title("REGION_RATING_CLIENT")
    save_figure("12_region_rating_client.png")
# -------------------------------------------------------------------------------------------------
def main():
    logger.info("=== Starting Exploratory Data Analysis (Section 3) ===\n")

    df_train = pd.read_csv(TRAIN_RAW_PATH)
    df_test = pd.read_csv(TEST_RAW_PATH)

    logger.info(f"\nRaw train shape: {df_train.shape} | Raw test shape: {df_test.shape}")

    train = df_train[SELECTED_FEATURES + [TARGET_COL]].copy()
    test = df_test[SELECTED_FEATURES].copy()

    # Target balance log
    target_pct = train[TARGET_COL].value_counts(normalize=True) * 100
    logger.info(
        f"Target Distribution -> Class 0: {target_pct.get(0, 0):.2f}% | "
        f"Class 1 (Default): {target_pct.get(1, 0):.2f}%"
    )

    # Income discreteness check (from Section 3.2.3)
    unique_incomes = train["AMT_INCOME_TOTAL"].nunique()
    logger.info(f"\nUnique AMT_INCOME_TOTAL values: {unique_incomes:,}")

    # 3.1 Missing value audit
    audit_missing_values(train, test)

    # 3.2 Feature visualizations
    logger.info(f"\nGenerating EDA visualizations in {FIGURES_DIR}...")
    generate_eda_plots(train)

    logger.info("\n=== EDA Completed Successfully ===")

if __name__ == "__main__":
    main()
