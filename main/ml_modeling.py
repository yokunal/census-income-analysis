import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, f1_score)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (CLEANED_DATA_PATH, MODEL_PATH, CONFUSION_MATRIX_PATH,
                    FEATURE_IMPORTANCE_PATH, CATEGORICAL_FEATURES,
                    NUMERICAL_FEATURES, TARGET_COLUMN, TEST_SIZE,
                    RANDOM_STATE, CV_FOLDS, N_ITER, SVM_PARAM_GRID)

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def load_features(filepath: Path = CLEANED_DATA_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """Load cleaned data and return X, y."""
    if not filepath.exists():
        raise FileNotFoundError(f"Cleaned data not found at {filepath}. Run eda_pipeline.py first.")
    df = pd.read_csv(filepath)
    missing = [c for c in CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")
    X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = df[TARGET_COLUMN]
    logger.info(f"Loaded features: {X.shape[0]} rows, {X.shape[1]} columns")
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build sklearn ColumnTransformer for numeric + categorical features."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def train_and_evaluate_models(X_train: pd.DataFrame, X_test: pd.DataFrame,
                               y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
    """Train multiple models, evaluate with CV, return results dict."""
    preprocessor = build_preprocessor()
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(random_state=RANDOM_STATE, probability=True),
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, clf in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_weighted")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_f1 = f1_score(y_test, y_pred, average="weighted")
        logger.info(f"{name} — CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} | Test F1: {test_f1:.4f}")
        results[name] = {"pipeline": pipeline, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(), "test_f1": test_f1, "y_pred": y_pred}
    return results


def tune_best_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Run RandomizedSearchCV on SVM and return best pipeline."""
    preprocessor = build_preprocessor()
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", SVC(probability=True))])
    search = RandomizedSearchCV(pipeline, SVM_PARAM_GRID, n_iter=N_ITER,
                                 cv=CV_FOLDS, scoring="f1_weighted",
                                 random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(X_train, y_train)
    logger.info(f"Best SVM params: {search.best_params_}")
    logger.info(f"Best CV F1: {search.best_score_:.4f}")
    return search.best_estimator_


def save_model(pipeline: Pipeline, path: Path = MODEL_PATH) -> None:
    """Save trained pipeline to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info(f"Model saved to {path}")


def save_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray,
                           path: Path = CONFUSION_MATRIX_PATH) -> None:
    """Save confusion matrix plot."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    ax.set_title("Confusion Matrix — Best SVM Model")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {path}")


def predict_new(model: Pipeline, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run inference on a single new record.

    Args:
        model: Trained sklearn Pipeline
        input_data: Dict with keys matching CATEGORICAL_FEATURES + NUMERICAL_FEATURES

    Returns:
        Dict with keys: prediction, probability_low, probability_high
    """
    missing = [c for c in CATEGORICAL_FEATURES + NUMERICAL_FEATURES if c not in input_data]
    if missing:
        raise ValueError(f"Input missing required fields: {missing}")
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    return {"prediction": prediction, "probability_<=50K": round(proba[0], 4), "probability_>50K": round(proba[1], 4)}


if __name__ == "__main__":
    X, y = load_features()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,
                                                          random_state=RANDOM_STATE, stratify=y)
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    best_pipeline = tune_best_model(X_train, y_train)
    y_pred_best = best_pipeline.predict(X_test)
    logger.info("\n" + classification_report(y_test, y_pred_best))
    save_model(best_pipeline)
    save_confusion_matrix(y_test, y_pred_best)
