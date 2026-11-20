"""
Main orchestrator script for the end-to-end Home Credit Default Risk pipeline:
1. Data cleaning, categorical encoding, and missing value handling (src/preprocessing.py)
2. 5-Fold Stratified CV benchmarking, ROC plotting, and CatBoost submission (src/train.py)
"""

import logging

from src.preprocessing import run_preprocessing
from src.train import run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MainPipeline")

def main():
    
    logger.info("=== Step 1/2: Running Preprocessing Pipeline ===")
    train_final, test_final = run_preprocessing(save_outputs=True)

    logger.info("=== Step 2/2: Running Model Training & Ensemble Inference ===")
    run_training(train_df=train_final, test_df=test_final, save_artifacts=True)

    logger.info("=== End-to-End Pipeline Completed Successfully ===")

if __name__ == '__main__':
    main()

