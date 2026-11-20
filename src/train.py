# Model benchmarking, 5-fold stratified cross-validation, ROC curve visualization, and CatBoost ensemble inference for the Home Credit Default Risk project.

import logging

from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    CATBOOST_PARAMS,
    CV_RANDOM_STATE,
    FIGURES_DIR,
    ID_COL,
    KNN_PARAMS,
    N_SPLITS,
    REPORTS_DIR,
    RF_PARAMS,
    SUBMISSION_PATH,
    TARGET_COL,
    TEST_PROCESSED_PATH,
    TEST_RAW_PATH,
    TRAIN_PROCESSED_PATH,
)

logger = logging.getLogger(__name__)
# --------------------------------------------------------------------------------------------------------------------------------
def get_models():
    """
    Initializes the three benchmarked model families using hyperparameters
    defined in src/config.py.
    """
    return {
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(**KNN_PARAMS)),
        ]),
        "Random Forest": RandomForestClassifier(**RF_PARAMS),
        "CatBoost": CatBoostClassifier(**CATBOOST_PARAMS),
    }
# --------------------------------------------------------------------------------------------------------------------------------
def plot_roc_comparison(y_true: pd.Series, oof_predictions: Dict[str, np.ndarray], results_df: pd.DataFrame, save_fig = True, show_fig = True):
    """
    Plots and optionally saves the 5-Fold Out-of-Fold ROC Curve comparison.
    """
    plt.figure(figsize=(8, 6))

    for name in results_df["Model"]:
        fpr, tpr, _ = roc_curve(y_true, oof_predictions[name])
        auc_score = roc_auc_score(y_true, oof_predictions[name])
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc_score:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random (AUC = 0.5000)")
    plt.title("ROC Curve Comparison (5-Fold Out-of-Fold)", fontsize=13)
    plt.xlabel("False Positive Rate (FPR)", fontsize=11)
    plt.ylabel("True Positive Rate (TPR)", fontsize=11)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_fig:
        save_path = FIGURES_DIR / "roc_curve_comparison.png"
        plt.savefig(save_path, dpi=300)
        logger.info(f"ROC curve comparison saved to: {save_path}")

    if show_fig:
        plt.show()
    else:
        plt.close()
# --------------------------------------------------------------------------------------------------------------------------------
def run_training(train_df, test_df, save_artifacts = True):
    """
    Executes 5-fold stratified cross-validation across all models,
    generates final test predictions using the 5-fold CatBoost ensemble,
    validates outputs, and persists reports/submission artifacts.
    """

    if train_df is None or test_df is None:
        logger.info("Loading processed datasets from disk...")
        train_df = pd.read_csv(TRAIN_PROCESSED_PATH)
        test_df = pd.read_csv(TEST_PROCESSED_PATH)

    X = train_df.drop(columns=[TARGET_COL]).copy()
    y = train_df[TARGET_COL].copy()
    X_test = test_df.copy()

    models = get_models()
    kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)

    results = []
    oof_predictions = {}
    catboost_test_preds = np.zeros(len(X_test))

    logger.info(f"Starting {N_SPLITS}-Fold Stratified Cross-Validation...")

    for name, model in models.items():
        fold_aucs = []
        oof_preds = np.zeros(len(X))

        for train_idx, val_idx in kfold.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_tr, y_tr)
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            oof_preds[val_idx] = y_pred_proba
            fold_aucs.append(roc_auc_score(y_val, y_pred_proba))

            # Accumulate Out-of-Fold ensemble test predictions for champion model (CatBoost)
            if name == "CatBoost":
                catboost_test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))

        oof_predictions[name] = oof_preds
        results.append({
            "Model": name,
            "Mean ROC-AUC": mean_auc,
            "Std ROC-AUC": std_auc
        })
        logger.info(f"{name:20s} | ROC-AUC: {mean_auc:.5f} ± {std_auc:.5f}")

    results_df = pd.DataFrame(results).sort_values(by="Mean ROC-AUC", ascending=False)

    # Assemble final submission matching original test IDs
    raw_test_ids = pd.read_csv(TEST_RAW_PATH, usecols=[ID_COL])
    submission = pd.DataFrame({
        ID_COL: raw_test_ids[ID_COL],
        TARGET_COL: catboost_test_preds,
    })

    # Post-inference Quality Gates
    assert len(submission) == len(raw_test_ids), "Submission row count mismatch!"
    assert submission[TARGET_COL].isnull().sum() == 0, "NaN values detected in predictions!"
    assert submission[TARGET_COL].between(0, 1).all(), "Predicted probabilities out of [0, 1] bounds!"

    logger.info(
        f"CatBoost Test Predictions ({len(catboost_test_preds):,} rows) -> "
        f"Min: {catboost_test_preds.min():.4f} | "
        f"Mean: {catboost_test_preds.mean():.4f} | "
        f"Max: {catboost_test_preds.max():.4f}"
    )

    if save_artifacts:
        benchmark_path = REPORTS_DIR / "model_benchmark.csv"
        results_df.to_csv(benchmark_path, index=False)
        logger.info(f"Model benchmark saved to: {benchmark_path}")

        plot_roc_comparison(y, oof_predictions, results_df, save_fig=True, show_fig=False)

        submission.to_csv(SUBMISSION_PATH, index=False)
        logger.info(f"Final submission saved to: {SUBMISSION_PATH}")

    return results_df, submission