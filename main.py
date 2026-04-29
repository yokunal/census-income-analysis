import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(steps: list[str]) -> None:
    """Run selected pipeline steps in order."""
    if "eda" in steps:
        logger.info("=== Step 1: EDA Pipeline ===")
        from main.eda_pipeline import main as run_eda
        run_eda()

    if "hypothesis" in steps:
        logger.info("=== Step 2: Hypothesis Testing ===")
        from main.eda_pipeline import load_and_clean_data
        from main.hypothesis_testing import run_hypothesis_tests
        df = load_and_clean_data()
        run_hypothesis_tests(df)

    if "model" in steps:
        logger.info("=== Step 3: ML Modeling ===")
        from main.ml_modeling import load_features, train_and_evaluate_models, tune_best_model, save_model, save_confusion_matrix
        from sklearn.model_selection import train_test_split
        from config import TEST_SIZE, RANDOM_STATE
        X, y = load_features()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
        results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        best = tune_best_model(X_train, y_train)
        save_model(best)
        save_confusion_matrix(y_test, best.predict(X_test))

    if "visualize" in steps:
        logger.info("=== Step 4: Predictive Visualization ===")
        from main.predictive_visualization import main as run_viz
        run_viz()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Census Income Analysis Pipeline")
    parser.add_argument("--steps", nargs="+",
                        choices=["eda", "hypothesis", "model", "visualize", "all"],
                        default=["all"],
                        help="Pipeline steps to run")
    args = parser.parse_args()
    steps = ["eda", "hypothesis", "model", "visualize"] if "all" in args.steps else args.steps
    run_pipeline(steps)
