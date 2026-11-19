# Home Credit Default Risk — Credit Scoring Pipeline

Project developed for the **Home Credit Default Risk** competition, part of the **GCI — Data Science & Deep Learning** course offered by the **University of Tokyo** (via the Omnicampus platform).

The goal of this challenge is to predict the probability of default (`TARGET`) for loan applicants using imbalanced tabular financial data, evaluated with **ROC-AUC** as the primary metric.

---

## Project Overview

* **Dataset:** 171,202 training samples with a binary target (`0` = repaid on time, `1` = payment difficulties / default).
* **Validation Strategy:** `5-Fold Stratified Cross-Validation` evaluated via **ROC-AUC**, generating combined **Out-of-Fold (OOF)** ROC curves for a fair comparison across models.

---

## Preprocessing & Feature Engineering

In credit risk modeling, missing values frequently carry meaningful domain signals (**MNAR — Missing Not At Random**). Instead of applying a one-size-fits-all imputation, tailored strategies were designed per feature:

| Feature | Applied Strategy | Rationale |
| :--- | :--- | :--- |
| **`EXT_SOURCE_2`** | Median imputation | Negligible missing rate (~0.2%). |
| **`EXT_SOURCE_1` / `EXT_SOURCE_3`** | Median imputation + Binary indicator (`_IS_MISSING`) | High missingness (~32% to ~70%) strongly associated with *thin-file* applicants (no external bureau history). |
| **`DAYS_EMPLOYED`** | Imputed with `0` + Anomaly flag (`DAYS_EMPLOYED_ANOM`) | Anomalous placeholder values (`365243`) correspond to pensioners and unemployed applicants. |
| **`OWN_CAR_AGE`** | Decade binning (`// 10`, capped at 30+ years, `-1` for no car) + One-Hot Encoding | ~67% missingness (applicants without a car); binning captures non-linear vehicle depreciation cleanly. |

---

## Model Benchmark (5-Fold Stratified CV)

All models were evaluated under the exact same cross-validation split (`random_state=99`). `StandardScaler` was applied exclusively to scale-sensitive models (KNN), leaving tree-based models on their original scale.

| Model | Mean ROC-AUC | Std ROC-AUC | Notes |
| :--- | :---: | :---: | :--- |
| **CatBoost** | *Best Model* | — | Selected for final submission |
| **XGBoost** | `0.74864` | `± 0.00309` | Strong tabular baseline |
| **Random Forest** | `0.73817` | `± 0.00213` | Solid non-linear baseline |
| **KNN** | `0.65405` | `± 0.00449` | Distance-based baseline |

---

## Tech Stack

* **Language:** Python 3
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `xgboost`, `catboost`, `lightgbm`
* **Visualization:** `matplotlib`


