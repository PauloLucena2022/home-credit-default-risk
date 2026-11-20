from pathlib import Path

# 1. PROJECT DIRECTORY PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Input file paths
TRAIN_RAW_PATH = RAW_DATA_DIR / "train.csv"
TEST_RAW_PATH = RAW_DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = RAW_DATA_DIR / "sample_submission.csv"

# Processed file paths
TRAIN_PROCESSED_PATH = PROCESSED_DATA_DIR / "train_final.csv"
TEST_PROCESSED_PATH = PROCESSED_DATA_DIR / "test_final.csv"

# Final submission path
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

# Ensure output directories exist automatically upon import
for directory in [PROCESSED_DATA_DIR, OUTPUT_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# 2. FEATURE SCHEMA & CLEANING THRESHOLDS
ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"

SELECTED_FEATURES = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_INCOME_TOTAL",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
    "NAME_EDUCATION_TYPE",
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_CONTRACT_TYPE",
    "OCCUPATION_TYPE",
    "OWN_CAR_AGE",
    "ORGANIZATION_TYPE",
    "REGION_RATING_CLIENT"
]

# Cleaning thresholds
INCOME_CLIP_QUANTILE = 0.999
DAYS_EMPLOYED_ANOMALY_VAL = 365243
OWN_CAR_AGE_MAX_VALID = 60
OWN_CAR_AGE_DECADE_CAP = 3
ORG_MIN_FREQ = 1000

# Rare category groupings
RARE_INCOME_TYPES = [
    "Unemployed",
    "Student",
    "Businessman",
    "Maternity leave"
]

RARE_OFFICE_OCCUPATIONS = [
    "Secretaries",
    "Realty agents",
    "HR staff",
    "IT staff"
]

RARE_SERVICE_OCCUPATIONS = [
    "Private service staff",
    "Low-skill Laborers",
    "Waiters/barmen staff"
]

ORG_SECTOR_PREFIXES = [
    "Trade",
    "Industry",
    "Transport",
    "Business Entity"
]

# 3. CATEGORICAL ENCODING MAPS & GROUPS
GENDER_MAP = {
    "F": 0,
    "M": 1
}

CONTRACT_MAP = {
    "Cash loans": 0,
    "Revolving loans": 1
}

EDUCATION_ORDINAL_MAP = {
    "Lower secondary": 0,
    "Secondary / secondary special": 1,
    "Incomplete higher": 2,
    "Higher education": 3,
    "Academic degree": 4
}

TARGET_ENCODE_COLS = [
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE"
]

MNAR_EXT_COLS = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_3"
]

# 4. VALIDATION & MODEL HYPERPARAMETERS
RANDOM_STATE = 42
CV_RANDOM_STATE = 99
N_SPLITS = 5

KNN_PARAMS = {
    "n_neighbors": 15,
    "n_jobs": -1
}

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 12,
    "min_samples_leaf": 20,
    "n_jobs": -1,
    "random_state": RANDOM_STATE
}

CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "eval_metric": "AUC",
    "random_seed": RANDOM_STATE,
    "thread_count": -1,
    "verbose": 0
}